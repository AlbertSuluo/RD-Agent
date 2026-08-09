from __future__ import annotations

import importlib.util
import math
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from types import ModuleType

import pandas as pd

from .config import CryptoFactorPropSetting


REQUIRED_EVALUATION_API = (
    "load_candle_data",
    "discover_factors",
    "_split_factors",
    "_evaluate_factor_batches",
    "build_comparison_table",
)


@dataclass(frozen=True)
class FactorGateResult:
    factor: str
    pfs: float
    rre: float
    best_ic: float
    best_ir: float
    passed: bool
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def apply_strict_gate(
    factor: str,
    pfs: float,
    rre: float,
    best_ic: float,
    best_ir: float,
    settings: CryptoFactorPropSetting,
) -> FactorGateResult:
    def numeric(value: object) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return math.nan

    pfs, rre, best_ic, best_ir = map(numeric, (pfs, rre, best_ic, best_ir))
    metrics = {
        "PFS": (pfs, settings.pfs_threshold),
        "RRE": (rre, settings.rre_threshold),
        "Best_IC": (best_ic, settings.best_ic_threshold),
        "Best_IR": (best_ir, settings.best_ir_threshold),
    }
    reasons = []
    for name, (value, threshold) in metrics.items():
        if not math.isfinite(value):
            reasons.append(f"{name} is not finite")
        elif not value > float(threshold):
            reasons.append(f"{name}={value:.10g} must be strictly > {float(threshold):.10g}")
    return FactorGateResult(
        factor=factor,
        pfs=float(pfs),
        rre=float(rre),
        best_ic=float(best_ic),
        best_ir=float(best_ir),
        passed=not reasons,
        reasons=tuple(reasons),
    )


def load_evaluation_module(path: Path) -> ModuleType:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    module_name = f"_rdagent_crypto_evaluation_{abs(hash(path))}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load evaluation module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        try:
            sys.path.remove(str(path.parent))
        except ValueError:
            pass
    missing = [name for name in REQUIRED_EVALUATION_API if not hasattr(module, name)]
    if missing:
        raise AttributeError(f"evaluation adapter API missing {missing}: {path}")
    return module


class EvaluationAdapter:
    """Reuse the supplied evaluation.py algorithms with RD-Agent policy gates."""

    def __init__(self, settings: CryptoFactorPropSetting):
        settings.require_input_paths()
        self.settings = settings
        self.module = load_evaluation_module(settings.evaluation_path)

    def evaluate(self, factors_dir: Path, factor_names: list[str]) -> tuple[pd.DataFrame, list[FactorGateResult]]:
        data_path, candles = self.module.load_candle_data(self.settings.data_path)
        usable = self.module.discover_factors(factors_dir, configured=factor_names)
        coin_factors, panel_factors = self.module._split_factors(usable, factors_dir)
        cache_root = Path(self.settings.cache_dir).expanduser().resolve()
        cache_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="gate-", dir=cache_root) as temp:
            work = Path(temp)
            factor_cols, quality, multi_results, _ = self.module._evaluate_factor_batches(
                candles,
                usable,
                factors_dir,
                data_path,
                work / "ephemeral-cache",
                work,
                list(self.settings.hold_periods),
                self.settings.min_cross_section_size,
                self.settings.entry_lag,
                self.settings.pfs_sigma,
                self.settings.pfs_trials,
                self.settings.evaluation_batch_size,
                self.settings.evaluation_n_jobs,
                False,
                bool(coin_factors),
                bool(panel_factors),
            )
            comparison = self.module.build_comparison_table(
                factor_cols, quality, multi_results, list(self.settings.hold_periods)
            )
        indexed = comparison.set_index("Factor", drop=False) if not comparison.empty else comparison
        results = []
        for name in factor_names:
            if comparison.empty or name not in indexed.index:
                results.append(apply_strict_gate(name, math.nan, math.nan, math.nan, math.nan, self.settings))
                continue
            row = indexed.loc[name]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            results.append(
                apply_strict_gate(
                    name,
                    row.get("PFS", math.nan),
                    row.get("RRE", math.nan),
                    row.get("Best_IC", math.nan),
                    row.get("Best_IR", math.nan),
                    self.settings,
                )
            )
        return comparison, results
