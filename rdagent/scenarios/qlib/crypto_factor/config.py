from __future__ import annotations

from pathlib import Path

from pydantic import Field, model_validator
from pydantic_settings import SettingsConfigDict
from typing_extensions import Self

from rdagent.app.qlib_rd_loop.conf import FactorBasePropSetting


class CryptoFactorPropSetting(FactorBasePropSetting):
    """Configuration for the independent hourly crypto factor loop.

    Every path is deliberately externalized.  CLI values take precedence over
    ``QLIB_CRYPTO_FACTOR_*`` environment variables.
    """

    model_config = SettingsConfigDict(env_prefix="QLIB_CRYPTO_FACTOR_", protected_namespaces=())

    scen: str = "rdagent.scenarios.qlib.crypto_factor.scenario.CryptoFactorScenario"
    runner: str = "rdagent.scenarios.qlib.crypto_factor.runner.CryptoFactorRunner"
    summarizer: str = "rdagent.scenarios.qlib.crypto_factor.feedback.CryptoFactorExperiment2Feedback"

    data_path: Path | None = None
    evaluation_path: Path | None = None
    baseline_dirs: list[Path] = Field(default_factory=list)
    cache_dir: Path = Path("git_ignore_folder/crypto_factor_cache")

    pfs_threshold: float = 0.8
    rre_threshold: float = 0.3
    best_ic_threshold: float = 0.03
    best_ir_threshold: float = 0.2
    dedup_correlation_threshold: float = 0.99

    hold_periods: tuple[str, ...] = ("1H", "2H", "4H", "8H", "1D")
    min_cross_section_size: int = 10
    entry_lag: int = 0
    pfs_sigma: float = 1.0
    pfs_trials: int = 2
    evaluation_batch_size: int = 1
    evaluation_n_jobs: int | None = None

    universe_size: int = 100
    train_start: str = "2021-01-01"
    train_end: str = "2023-12-31 23:59:59"
    valid_start: str = "2024-01-01"
    valid_end: str = "2024-12-31 23:59:59"
    test_start: str = "2025-01-01"
    test_end: str = "2025-07-06 22:00:00"

    freq: str = "60min"
    topk: int = 50
    n_drop: int = 5
    open_cost: float = 0.0004
    close_cost: float = 0.0004
    min_cost: float = 0.0
    annualization_hours: int = 8760

    @model_validator(mode="after")
    def validate_crypto_configuration(self) -> Self:
        if not (0 < self.universe_size and 0 < self.topk <= self.universe_size):
            raise ValueError("topk must be positive and no larger than universe_size")
        if not 0 <= self.n_drop <= self.topk:
            raise ValueError("n_drop must be between zero and topk")
        if self.freq != "60min":
            raise ValueError("the crypto scenario currently requires Qlib frequency '60min'")
        return self

    def require_input_paths(self) -> None:
        missing = []
        if self.data_path is None:
            missing.append("--data-path / QLIB_CRYPTO_FACTOR_DATA_PATH")
        if self.evaluation_path is None:
            missing.append("--evaluation-path / QLIB_CRYPTO_FACTOR_EVALUATION_PATH")
        if not self.baseline_dirs:
            missing.append("--baseline-dir / QLIB_CRYPTO_FACTOR_BASELINE_DIRS")
        if missing:
            raise ValueError("missing required crypto factor settings: " + ", ".join(missing))
        paths = [self.data_path, self.evaluation_path, *self.baseline_dirs]
        absent = [str(p) for p in paths if p is not None and not Path(p).expanduser().exists()]
        if absent:
            raise FileNotFoundError("configured crypto factor inputs do not exist: " + ", ".join(absent))


_CURRENT_SETTINGS = CryptoFactorPropSetting()


def set_crypto_factor_settings(settings: CryptoFactorPropSetting) -> None:
    global _CURRENT_SETTINGS
    _CURRENT_SETTINGS = settings


def get_crypto_factor_settings() -> CryptoFactorPropSetting:
    return _CURRENT_SETTINGS
