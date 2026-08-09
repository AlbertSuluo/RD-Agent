"""Independent RD-Agent entry point for hourly cryptocurrency factor mining."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import fire

from rdagent.components.coder.factor_coder.config import FACTOR_COSTEER_SETTINGS
from rdagent.components.workflow.rd_loop import RDLoop
from rdagent.core.conf import RD_AGENT_SETTINGS
from rdagent.core.exception import CoderError
from rdagent.log import rdagent_logger as logger
from rdagent.scenarios.qlib.crypto_factor.config import (
    CryptoFactorPropSetting,
    set_crypto_factor_settings,
)
from rdagent.scenarios.qlib.crypto_factor.data import prepare_data_links
from rdagent.scenarios.qlib.crypto_factor.registry import BaselineRegistry


class CryptoFactorRDLoop(RDLoop):
    skip_loop_error = (CoderError,)
    skip_loop_error_stepname = "feedback"

    def running(self, prev_out: dict[str, Any]):
        exp = self.runner.develop(prev_out["coding"])
        logger.log_object(exp, tag="crypto runner result")
        return exp


def main(
    data_path: str | None = None,
    evaluation_path: str | None = None,
    baseline_dirs: list[str] | None = None,
    cache_dir: str | None = None,
    pfs_threshold: float | None = None,
    rre_threshold: float | None = None,
    best_ic_threshold: float | None = None,
    best_ir_threshold: float | None = None,
    path: str | None = None,
    step_n: int | None = None,
    loop_n: int | None = None,
    all_duration: str | None = None,
    checkout: bool = True,
) -> None:
    settings = CryptoFactorPropSetting()
    updates = {
        "data_path": Path(data_path).expanduser() if data_path else None,
        "evaluation_path": Path(evaluation_path).expanduser() if evaluation_path else None,
        "baseline_dirs": [Path(item).expanduser() for item in baseline_dirs] if baseline_dirs else None,
        "cache_dir": Path(cache_dir).expanduser() if cache_dir else None,
        "pfs_threshold": pfs_threshold,
        "rre_threshold": rre_threshold,
        "best_ic_threshold": best_ic_threshold,
        "best_ir_threshold": best_ir_threshold,
    }
    settings = CryptoFactorPropSetting(
        **{**settings.model_dump(), **{key: value for key, value in updates.items() if value is not None}}
    )
    settings.require_input_paths()
    set_crypto_factor_settings(settings)

    runtime_data = prepare_data_links(settings.data_path, settings.cache_dir)
    FACTOR_COSTEER_SETTINGS.data_folder = str(runtime_data)
    FACTOR_COSTEER_SETTINGS.data_folder_debug = str(runtime_data)
    # FactorFBWorkspace's generic pickle cache would otherwise retain full-universe
    # result.h5 values for rejected factors. Session checkpoints remain enabled.
    RD_AGENT_SETTINGS.cache_with_pickle = False
    registry = BaselineRegistry.load(settings.baseline_dirs)
    logger.info(f"Loaded {len(registry.factors)} trusted crypto baseline factors.")

    factor_loop = CryptoFactorRDLoop(settings) if path is None else CryptoFactorRDLoop.load(path, checkout=checkout)
    factor_loop.plan["features"] = {}
    factor_loop.plan["feature_codes"] = {}
    factor_loop.plan["user_instruction"] = (
        f"The following {len(registry.factors)} trusted baseline factors are already registered; "
        "do not duplicate them: "
        + registry.prompt_summary()
    )
    asyncio.run(factor_loop.run(step_n=step_n, loop_n=loop_n, all_duration=all_duration))


if __name__ == "__main__":
    fire.Fire(main)
