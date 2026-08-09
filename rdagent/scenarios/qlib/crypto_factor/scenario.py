from __future__ import annotations

from rdagent.core.experiment import Task
from rdagent.core.scenario import Scenario

from .config import get_crypto_factor_settings


class CryptoFactorScenario(Scenario):
    """Prompt contract for no-lookahead, hourly cryptocurrency factors."""

    @property
    def background(self) -> str:
        settings = get_crypto_factor_settings()
        return f"""We mine cross-sectional cryptocurrency factors from hourly OHLCV candles.
The dataset is a list of per-symbol DataFrames with candle_begin_time, symbol, open, high, low,
close, volume, quote_volume, trade_num and taker_buy_base_volume.  A value at time t may use only
information available at or before t.  Any negative shift, centered window, backward fill from a
future observation, or full-sample statistic that leaks future data is forbidden.

New factors are admitted only by deterministic strict gates: PFS > {settings.pfs_threshold},
RRE > {settings.rre_threshold}, absolute Best_IC > {settings.best_ic_threshold}, and absolute
Best_IR > {settings.best_ir_threshold}.  Qlib model/backtest results are diagnostic feedback only."""

    def get_source_data_desc(self, task: Task | None = None) -> str:
        settings = get_crypto_factor_settings()
        return (
            f"The configured candle pickle is {settings.data_path}. "
            "It is hourly and contains the full 509-symbol universe."
        )

    @property
    def interface(self) -> str:
        return """Return a self-contained factor.py. It MUST define exactly one of:

1. signal(df, param, factor_name): copy/transform one symbol DataFrame, put the factor in
   df[factor_name], and return that DataFrame; or
2. signal_panel(panel, param, factor_name): consume a dict of wide OHLCV matrices and return one
   wide DataFrame indexed by candle_begin_time with symbols as columns.

The file must also be directly executable. Its main routine must locate the linked candle pickle,
calculate the same factor without lookahead, convert it to a DataFrame indexed by the two named
levels (datetime, instrument), and write result.h5. Do not download data or use absolute paths."""

    @property
    def output_format(self) -> str:
        return "Python source implementing signal/signal_panel and a standalone result.h5 entry point."

    @property
    def simulator(self) -> str:
        return (
            "The supplied evaluation.py provides OHLCV perturbation PFS, KL-RRE and "
            "non-overlapping holding-period RankIC/ICIR."
        )

    @property
    def rich_style_description(self) -> str:
        return self.background

    @property
    def experiment_setting(self) -> str:
        settings = get_crypto_factor_settings()
        return (
            f"train={settings.train_start}..{settings.train_end}; valid={settings.valid_start}..{settings.valid_end}; "
            f"test={settings.test_start}..{settings.test_end}; freq=60min; liquidity universe=training-only Top100"
        )

    def get_scenario_all_desc(
        self, task: Task | None = None, filtered_tag: str | None = None, simple_background: bool | None = None
    ) -> str:
        if simple_background:
            return self.background
        return "\n\n".join(
            [
                "Background:\n" + self.background,
                "Source data:\n" + self.get_source_data_desc(task),
                "Implementation interface:\n" + self.interface,
                "Output:\n" + self.output_format,
                "Evaluation and simulator:\n" + self.simulator,
                "Experiment setting:\n" + self.experiment_setting,
            ]
        )
