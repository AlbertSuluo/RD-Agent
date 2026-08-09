from __future__ import annotations

import math
import tempfile
import uuid
from pathlib import Path

import pandas as pd

from rdagent.components.runner import CachedRunner
from rdagent.log import rdagent_logger as logger
from rdagent.scenarios.qlib.experiment.factor_experiment import QlibFactorExperiment

from .config import get_crypto_factor_settings
from .data import filter_universe, load_candles, select_top_universe, write_json_atomic
from .evaluation_adapter import EvaluationAdapter, apply_strict_gate
from .qlib_analysis import CryptoQlibAnalyzer
from .registry import BaselineRegistry
from .store import (
    AcceptedFactorStore,
    Top100FeatureStore,
    max_cross_sectional_correlation,
    merge_factor_frames,
)


class CryptoFactorRunner(CachedRunner[QlibFactorExperiment]):
    """Gate candidates on all symbols, then run non-blocking Top100 Qlib analysis."""

    def __init__(self, scen) -> None:
        super().__init__(scen)
        self.settings = get_crypto_factor_settings()
        self.registry = BaselineRegistry.load(self.settings.baseline_dirs)
        self.adapter = EvaluationAdapter(self.settings)
        self.accepted_store = AcceptedFactorStore(self.settings.cache_dir)

    @staticmethod
    def _candidate_sources(exp: QlibFactorExperiment, stage: Path) -> tuple[dict[str, Path], list[str]]:
        sources: dict[str, Path] = {}
        errors: list[str] = []
        (stage / "__init__.py").write_text('"""Staged RD-Agent crypto candidates."""\n')
        for task, workspace in zip(exp.sub_tasks, exp.sub_workspace_list):
            name = task.factor_name
            if not isinstance(name, str) or not name.isidentifier():
                errors.append(f"{name!r}: factor name must be a valid Python identifier")
                continue
            code = workspace.file_dict.get("factor.py") if workspace is not None else None
            if not code:
                errors.append(f"{name}: coder produced no factor.py")
                continue
            path = stage / f"{name}.py"
            path.write_text(code)
            sources[name] = path
        return sources, errors

    @staticmethod
    def _discard_full_workspace_values(exp: QlibFactorExperiment) -> None:
        for workspace in exp.sub_workspace_list:
            if workspace is None:
                continue
            result_path = Path(workspace.workspace_path) / "result.h5"
            if result_path.exists():
                result_path.unlink()

    def _compute_frames(
        self,
        feature_store: Top100FeatureStore,
        factors: list[tuple[str, Path]],
        errors: list[str],
    ) -> dict[str, pd.DataFrame]:
        frames = {}
        for name, source_dir in factors:
            try:
                frames[name] = feature_store.get_or_compute(source_dir, name)
            except Exception as exc:
                errors.append(f"{name}: Top100 value calculation failed: {type(exc).__name__}: {exc}")
        return frames

    @staticmethod
    def _filter_experiment(exp: QlibFactorExperiment, accepted_names: set[str]) -> None:
        exp.generated_sub_tasks = list(exp.sub_tasks)
        exp.generated_sub_workspace_list = list(exp.sub_workspace_list)
        keep = [index for index, task in enumerate(exp.sub_tasks) if task.factor_name in accepted_names]
        exp.sub_tasks = [exp.sub_tasks[index] for index in keep]
        exp.sub_workspace_list = [exp.sub_workspace_list[index] for index in keep]
        feedback = exp.prop_dev_feedback
        if feedback is not None and hasattr(feedback, "__getitem__"):
            try:
                exp.prop_dev_feedback = type(feedback)([feedback[index] for index in keep])
            except Exception:
                pass

    def develop(self, exp: QlibFactorExperiment) -> QlibFactorExperiment:
        cache_root = Path(self.settings.cache_dir).expanduser().resolve()
        artifact_dir = cache_root / "runs" / uuid.uuid4().hex
        artifact_dir.mkdir(parents=True, exist_ok=True)
        runtime_errors: list[str] = []
        previous_accepted = self.accepted_store.names
        reserved_names = set(self.registry.names) | set(previous_accepted)

        with tempfile.TemporaryDirectory(prefix="candidates-", dir=cache_root) as temp:
            stage = Path(temp)
            sources, source_errors = self._candidate_sources(exp, stage)
            self._discard_full_workspace_values(exp)
            runtime_errors.extend(source_errors)
            candidate_names = [task.factor_name for task in exp.sub_tasks]
            try:
                comparison, gate_results = (
                    self.adapter.evaluate(stage, list(sources)) if sources else (pd.DataFrame(), [])
                )
            except (Exception, SystemExit) as exc:
                logger.error(f"Crypto factor metric evaluation failed: {type(exc).__name__}: {exc}")
                comparison = pd.DataFrame()
                gate_results = [
                    apply_strict_gate(name, math.nan, math.nan, math.nan, math.nan, self.settings) for name in sources
                ]
                runtime_errors.append(f"metric evaluation failed: {type(exc).__name__}: {exc}")
            gate_by_name = {result.factor: result for result in gate_results}
            for name in candidate_names:
                gate_by_name.setdefault(
                    name, apply_strict_gate(name, math.nan, math.nan, math.nan, math.nan, self.settings)
                )

            candles = load_candles(self.settings.data_path)
            universe = select_top_universe(
                candles, self.settings.train_start, self.settings.train_end, self.settings.universe_size
            )
            top_candles = filter_universe(candles, universe)
            write_json_atomic(artifact_dir / "top100_universe.json", universe)
            feature_store = Top100FeatureStore(self.settings, self.adapter.module, top_candles, universe)

            baseline_pairs = [(factor.name, factor.source_dir) for factor in self.registry.factors]
            history_pairs = [(name, self.accepted_store.source_dir(name)) for name in previous_accepted]
            baseline_frames = self._compute_frames(feature_store, baseline_pairs + history_pairs, runtime_errors)
            existing = merge_factor_frames(baseline_frames)

            accepted_names: list[str] = []
            candidate_frames: dict[str, pd.DataFrame] = {}
            decision_rows: list[dict[str, object]] = []
            for name in candidate_names:
                gate = gate_by_name[name]
                reasons = list(gate.reasons)
                if name in reserved_names:
                    reasons.append(f"factor name {name!r} already exists in the trusted/accepted factor library")
                max_corr, duplicate_of = math.nan, None
                frame = None
                if gate.passed and name in sources:
                    try:
                        # Keep rejected values in memory only. Persistence happens only
                        # after both the metric gate and correlation de-duplication pass.
                        frame = feature_store.compute_ephemeral(stage, name)
                        max_corr, duplicate_of = max_cross_sectional_correlation(existing, frame, name)
                        if max_corr >= self.settings.dedup_correlation_threshold:
                            reasons.append(
                                f"correlation dedup failed: mean absolute cross-sectional corr {max_corr:.6f} "
                                f">= {self.settings.dedup_correlation_threshold} against {duplicate_of}"
                            )
                    except Exception as exc:
                        reasons.append(f"Top100/dedup calculation failed: {type(exc).__name__}: {exc}")
                accepted = gate.passed and not reasons and frame is not None
                if accepted:
                    accepted_names.append(name)
                    candidate_frames[name] = frame
                    existing = existing.merge(frame, on=["candle_begin_time", "symbol"], how="outer")
                    source_dir = self.accepted_store.accept(name, sources[name], gate.to_dict())
                    # Re-key the persistent cache to the immutable accepted source copy.
                    candidate_frames[name] = feature_store.get_or_compute(source_dir, name)
                decision_rows.append(
                    {
                        **gate.to_dict(),
                        "passed_metric_gate": gate.passed,
                        "max_existing_correlation": max_corr,
                        "duplicate_of": duplicate_of,
                        "accepted": accepted,
                        "reasons": reasons,
                    }
                )

            decision_table = pd.DataFrame(decision_rows)
            decision_table.to_csv(artifact_dir / "candidate_metrics.csv", index=False)
            if not comparison.empty:
                comparison.to_csv(artifact_dir / "evaluation_comparison.csv", index=False)

        baseline_metrics: dict[str, object] = {}
        current_metrics: dict[str, object] = {}
        analyzer = CryptoQlibAnalyzer(self.settings)
        try:
            baseline_metrics = analyzer.run(baseline_frames, top_candles, artifact_dir / "qlib_baseline")
        except Exception as exc:
            runtime_errors.append(f"Qlib baseline analysis failed (non-gating): {type(exc).__name__}: {exc}")
        try:
            current_frames = {**baseline_frames, **candidate_frames}
            current_metrics = analyzer.run(current_frames, top_candles, artifact_dir / "qlib_current")
        except Exception as exc:
            runtime_errors.append(f"Qlib current analysis failed (non-gating): {type(exc).__name__}: {exc}")

        exp.crypto_candidate_metrics = decision_table
        exp.accepted_factor_names = accepted_names
        exp.rejected_factor_details = (
            decision_table.loc[~decision_table["accepted"]].to_dict("records") if not decision_table.empty else []
        )
        exp.crypto_baseline_metrics = baseline_metrics
        exp.crypto_current_metrics = current_metrics
        exp.crypto_runtime_errors = runtime_errors
        exp.crypto_artifact_dir = str(artifact_dir)
        result_values = {key: value for key, value in current_metrics.items() if isinstance(value, (int, float))}
        result_values["accepted_factor_count"] = len(accepted_names)
        exp.result = pd.Series(result_values, dtype="float64")
        self._filter_experiment(exp, set(accepted_names))
        write_json_atomic(
            artifact_dir / "run_summary.json",
            {
                "accepted": accepted_names,
                "rejected": exp.rejected_factor_details,
                "baseline_metrics": baseline_metrics,
                "current_metrics": current_metrics,
                "runtime_errors": runtime_errors,
            },
        )
        return exp
