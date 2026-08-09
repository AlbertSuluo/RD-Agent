from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from .config import CryptoFactorPropSetting
from .data import cache_key, write_json_atomic


KEY_COLUMNS = ["candle_begin_time", "symbol"]


class Top100FeatureStore:
    """Content-addressed cache containing only the fixed Top100 universe."""

    def __init__(
        self,
        settings: CryptoFactorPropSetting,
        evaluation_module,
        candles: list[pd.DataFrame],
        universe: list[str],
    ):
        self.settings = settings
        self.evaluation = evaluation_module
        self.candles = candles
        self.universe = universe
        self.root = Path(settings.cache_dir).expanduser().resolve() / "top100_features"
        self.root.mkdir(parents=True, exist_ok=True)

    def get_or_compute(self, source_dir: Path, factor_name: str) -> pd.DataFrame:
        key = cache_key(self.settings.data_path, source_dir, factor_name, self.universe)
        factor_dir = self.root / factor_name
        path = factor_dir / f"{key}.parquet"
        if path.exists():
            return pd.read_parquet(path)
        factor_dir.mkdir(parents=True, exist_ok=True)
        values = self.compute_ephemeral(source_dir, factor_name)
        values.to_parquet(path, index=False)
        write_json_atomic(
            factor_dir / f"{key}.json",
            {
                "factor": factor_name,
                "source_dir": str(Path(source_dir).resolve()),
                "data_path": str(Path(self.settings.data_path).resolve()),
                "universe": self.universe,
                "cache_key": key,
            },
        )
        return values

    def compute_ephemeral(self, source_dir: Path, factor_name: str) -> pd.DataFrame:
        values = self.evaluation._run_factors(
            self.candles,
            [factor_name],
            source_dir,
            n_jobs=self.settings.evaluation_n_jobs,
        )
        if values.empty or factor_name not in values.columns:
            raise ValueError(f"factor {factor_name} produced no Top100 values")
        return values[KEY_COLUMNS + [factor_name]].sort_values(KEY_COLUMNS)


class AcceptedFactorStore:
    """Source-of-truth library for candidates that passed all deterministic checks."""

    def __init__(self, cache_dir: Path):
        self.root = Path(cache_dir).expanduser().resolve() / "accepted_factors"
        self.root.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.root / "manifest.json"

    def manifest(self) -> dict[str, dict[str, object]]:
        if not self.manifest_path.exists():
            return {}
        return json.loads(self.manifest_path.read_text())

    @property
    def names(self) -> list[str]:
        return sorted(self.manifest())

    def source_dir(self, factor_name: str) -> Path:
        source = self.root / factor_name / f"{factor_name}.py"
        if not source.exists():
            raise FileNotFoundError(source)
        return source.parent

    def accept(self, factor_name: str, source: Path, metrics: dict[str, object]) -> Path:
        destination_dir = self.root / factor_name
        destination_dir.mkdir(parents=True, exist_ok=True)
        destination = destination_dir / f"{factor_name}.py"
        shutil.copy2(source, destination)
        init_path = destination_dir / "__init__.py"
        if not init_path.exists():
            init_path.write_text('"""RD-Agent accepted crypto factor."""\n')
        manifest = self.manifest()
        manifest[factor_name] = {"source": str(destination), "metrics": metrics}
        write_json_atomic(self.manifest_path, manifest)
        return destination_dir


def merge_factor_frames(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    merged = None
    for name, frame in frames.items():
        current = frame[KEY_COLUMNS + [name]].copy()
        merged = current if merged is None else merged.merge(current, on=KEY_COLUMNS, how="outer")
    return merged if merged is not None else pd.DataFrame(columns=KEY_COLUMNS)


def max_cross_sectional_correlation(
    existing: pd.DataFrame, candidate: pd.DataFrame, candidate_name: str
) -> tuple[float, str | None]:
    """Match RD-Agent's per-time cross-sectional factor de-duplication."""

    if existing.empty or candidate.empty:
        return 0.0, None
    joined = existing.merge(candidate[KEY_COLUMNS + [candidate_name]], on=KEY_COLUMNS, how="inner")
    best_value, best_name = 0.0, None
    for column in [c for c in existing.columns if c not in KEY_COLUMNS]:
        pair = joined[["candle_begin_time", column, candidate_name]].dropna()
        if pair.empty:
            continue
        correlations = pair.groupby("candle_begin_time")[[column, candidate_name]].corr().iloc[0::2, -1]
        value = float(correlations.replace([np.inf, -np.inf], np.nan).abs().mean())
        if np.isfinite(value) and value > best_value:
            best_value, best_name = value, column
    return best_value, best_name
