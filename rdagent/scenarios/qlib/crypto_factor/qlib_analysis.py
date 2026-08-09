from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import CryptoFactorPropSetting
from .data import safe_instrument, write_json_atomic
from .store import KEY_COLUMNS, merge_factor_frames


def build_backtest_config(settings: CryptoFactorPropSetting) -> dict[str, object]:
    return {
        "freq": settings.freq,
        "benchmark": None,
        "strategy": {
            "class": "TopkDropoutStrategy",
            "module_path": "qlib.contrib.strategy.signal_strategy",
            "kwargs": {"topk": settings.topk, "n_drop": settings.n_drop},
        },
        "executor": {
            "class": "SimulatorExecutor",
            "module_path": "qlib.backtest.executor",
            "kwargs": {"time_per_step": settings.freq, "generate_portfolio_metrics": True},
        },
        "exchange_kwargs": {
            "freq": settings.freq,
            "deal_price": "$close",
            "open_cost": settings.open_cost,
            "close_cost": settings.close_cost,
            "min_cost": settings.min_cost,
            "limit_threshold": None,
            "trade_unit": None,
        },
        "annualization_hours": settings.annualization_hours,
        "segments": {
            "train": (settings.train_start, settings.train_end),
            "valid": (settings.valid_start, settings.valid_end),
            "test": (settings.test_start, settings.test_end),
        },
    }


def _cross_sectional_correlation(values: pd.DataFrame, a: str, b: str, method: str) -> pd.Series:
    return pd.Series(
        {
            timestamp: group[a].corr(group[b], method=method)
            for timestamp, group in values.groupby("datetime", sort=True)
        },
        dtype="float64",
    )


def signal_metrics(prediction: pd.Series, label: pd.Series) -> dict[str, float]:
    joined = pd.concat([prediction.rename("score"), label.rename("label")], axis=1).dropna().reset_index()
    if joined.empty:
        return {key: math.nan for key in ("IC", "ICIR", "RankIC", "RankICIR")}
    ic = _cross_sectional_correlation(joined, "score", "label", "pearson")
    rank_ic = _cross_sectional_correlation(joined, "score", "label", "spearman")
    return {
        "IC": float(ic.mean()),
        "ICIR": float(ic.mean() / ic.std()) if ic.std() else math.nan,
        "RankIC": float(rank_ic.mean()),
        "RankICIR": float(rank_ic.mean() / rank_ic.std()) if rank_ic.std() else math.nan,
    }


class CryptoRiskRecorder:
    """Risk calculations with a crypto-specific 8,760-hour year."""

    def __init__(self, annualization_hours: int = 8760):
        self.annualization_hours = annualization_hours

    def calculate(self, report: pd.DataFrame, turnover: float = math.nan) -> dict[str, float]:
        returns = pd.to_numeric(report.get("return", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
        costs = pd.to_numeric(report.get("cost", pd.Series(0.0, index=report.index)), errors="coerce").fillna(0.0)
        net = returns - costs
        wealth = (1.0 + net).cumprod()
        years = len(net) / self.annualization_hours
        annualized = float(wealth.iloc[-1] ** (1.0 / years) - 1.0) if len(wealth) and years > 0 else math.nan
        volatility = float(net.std() * math.sqrt(self.annualization_hours))
        information_ratio = (
            float(net.mean() / net.std() * math.sqrt(self.annualization_hours)) if net.std() else math.nan
        )
        drawdown = wealth / wealth.cummax() - 1.0 if len(wealth) else pd.Series(dtype=float)
        return {
            "annualized_return": annualized,
            "information_ratio": information_ratio,
            "annualized_volatility": volatility,
            "max_drawdown": float(drawdown.min()) if not drawdown.empty else math.nan,
            "total_cost": float(costs.sum()),
            "turnover": float(turnover),
        }

    def save(self, report: pd.DataFrame, output_dir: Path, turnover: float = math.nan) -> dict[str, float]:
        metrics = self.calculate(report, turnover)
        output_dir.mkdir(parents=True, exist_ok=True)
        report.to_csv(output_dir / "portfolio_report.csv")
        write_json_atomic(output_dir / "crypto_risk_metrics.json", metrics)
        return metrics


def _feature_label_frames(
    factor_frames: dict[str, pd.DataFrame], candles: list[pd.DataFrame]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    features = merge_factor_frames(factor_frames)
    prices = pd.concat(
        [frame[["candle_begin_time", "symbol", "close", "volume"]] for frame in candles], ignore_index=True
    )
    prices = prices.sort_values(["symbol", "candle_begin_time"])
    prices["LABEL0"] = prices.groupby("symbol")["close"].shift(-1) / prices["close"] - 1.0
    full = features.merge(prices, on=KEY_COLUMNS, how="inner")
    full["datetime"] = pd.to_datetime(full.pop("candle_begin_time"))
    full["instrument"] = full.pop("symbol").map(safe_instrument)
    full = full.dropna(subset=["LABEL0"])
    full = full.set_index(["datetime", "instrument"]).sort_index()
    feature_columns = list(factor_frames)
    return full[feature_columns], full[["LABEL0"]], full[["close", "volume"]]


def _write_partitioned(frame: pd.DataFrame, root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    flat = frame.reset_index()
    flat["year"] = pd.to_datetime(flat["datetime"]).dt.year
    flat.to_parquet(root, index=False, partition_cols=["year"])
    return root


def _read_partitioned(root: Path) -> pd.DataFrame:
    frame = pd.read_parquet(root).drop(columns=["year"], errors="ignore")
    return frame.set_index(["datetime", "instrument"]).sort_index()


def materialize_test_provider(market: pd.DataFrame, output: Path, start: str, end: str) -> Path:
    """Create a minimal Qlib provider containing only Top100 test-period quotes."""

    test = market.loc[pd.Timestamp(start) : pd.Timestamp(end)].reset_index()
    calendars = sorted(pd.to_datetime(test["datetime"].unique()))
    calendar_index = {timestamp: position for position, timestamp in enumerate(calendars)}
    (output / "calendars").mkdir(parents=True, exist_ok=True)
    (output / "instruments").mkdir(parents=True, exist_ok=True)
    (output / "features").mkdir(parents=True, exist_ok=True)
    (output / "calendars" / "60min.txt").write_text("\n".join(ts.strftime("%Y-%m-%d %H:%M:%S") for ts in calendars))
    instrument_rows = []
    for instrument, frame in test.groupby("instrument"):
        frame = frame.sort_values("datetime")
        first, last = pd.Timestamp(frame["datetime"].iloc[0]), pd.Timestamp(frame["datetime"].iloc[-1])
        instrument_rows.append(f"{instrument}\t{first}\t{last}")
        frame = frame.set_index("datetime").reindex([ts for ts in calendars if first <= ts <= last]).reset_index()
        frame["instrument"] = instrument
        directory = output / "features" / str(instrument).lower()
        directory.mkdir(parents=True, exist_ok=True)
        start_index = calendar_index[first]
        for field in ("close", "volume"):
            values = pd.to_numeric(frame[field], errors="coerce").to_numpy(dtype="float32")
            payload = np.concatenate([np.asarray([start_index], dtype="float32"), values])
            payload.astype("<f4").tofile(directory / f"{str(instrument).lower()}.{field}.60min.bin")
        factor = np.ones(len(frame), dtype="float32")
        np.concatenate([np.asarray([start_index], dtype="float32"), factor]).astype("<f4").tofile(
            directory / f"{str(instrument).lower()}.factor.60min.bin"
        )
    (output / "instruments" / "all.txt").write_text("\n".join(instrument_rows))
    return output


@dataclass
class CryptoQlibAnalyzer:
    settings: CryptoFactorPropSetting

    def run(
        self, factor_frames: dict[str, pd.DataFrame], candles: list[pd.DataFrame], output_dir: Path
    ) -> dict[str, object]:
        """Train Qlib LightGBM, predict next hour and execute the crypto backtest."""

        if not factor_frames:
            raise ValueError("Qlib analysis requires at least one factor")
        output_dir.mkdir(parents=True, exist_ok=True)
        features, labels, market = _feature_label_frames(factor_frames, candles)
        feature_path = _write_partitioned(features, output_dir / "model_features.parquet")
        label_path = _write_partitioned(labels, output_dir / "model_labels.parquet")
        write_json_atomic(output_dir / "backtest_config.json", build_backtest_config(self.settings))

        from qlib.backtest import backtest
        from qlib.contrib.model.gbdt import LGBModel
        from qlib.data.dataset import DatasetH
        from qlib.data.dataset.handler import DataHandlerLP
        from qlib.data.dataset.loader import StaticDataLoader
        from qlib.workflow import R
        import qlib
        from qlib.constant import REG_US

        provider = materialize_test_provider(
            market, output_dir / "qlib_provider", self.settings.test_start, self.settings.test_end
        )
        qlib.init(provider_uri=str(provider), region=REG_US)
        loader = StaticDataLoader({"feature": _read_partitioned(feature_path), "label": _read_partitioned(label_path)})
        handler = DataHandlerLP(data_loader=loader)
        dataset = DatasetH(
            handler=handler,
            segments={
                "train": (self.settings.train_start, self.settings.train_end),
                "valid": (self.settings.valid_start, self.settings.valid_end),
                "test": (self.settings.test_start, self.settings.test_end),
            },
        )
        model = LGBModel(loss="mse", colsample_bytree=0.8879, learning_rate=0.0421, subsample=0.8789)
        with R.start(experiment_name="crypto_factor_lightgbm"):
            model.fit(dataset)
            prediction = model.predict(dataset, "test")
        label = dataset.prepare("test", col_set="label", data_key=DataHandlerLP.DK_I).iloc[:, 0]
        metrics: dict[str, object] = signal_metrics(prediction, label)
        prediction.rename("score").to_frame().to_parquet(output_dir / "predictions.parquet")

        portfolio, indicators = backtest(
            start_time=self.settings.test_start,
            end_time=self.settings.test_end,
            strategy={
                "class": "TopkDropoutStrategy",
                "module_path": "qlib.contrib.strategy.signal_strategy",
                "kwargs": {"signal": prediction, "topk": self.settings.topk, "n_drop": self.settings.n_drop},
            },
            executor={
                "class": "SimulatorExecutor",
                "module_path": "qlib.backtest.executor",
                "kwargs": {"time_per_step": self.settings.freq, "generate_portfolio_metrics": True},
            },
            benchmark=None,
            exchange_kwargs={
                "freq": self.settings.freq,
                "deal_price": "$close",
                "open_cost": self.settings.open_cost,
                "close_cost": self.settings.close_cost,
                "min_cost": self.settings.min_cost,
                "limit_threshold": None,
                "trade_unit": None,
            },
        )
        report = portfolio[self.settings.freq][0]
        indicator_frame = indicators.get(self.settings.freq, (pd.DataFrame(),))[0]
        turnover = float(indicator_frame["turnover"].mean()) if "turnover" in indicator_frame else math.nan
        metrics.update(CryptoRiskRecorder(self.settings.annualization_hours).save(report, output_dir, turnover))
        write_json_atomic(output_dir / "metrics.json", metrics)
        return metrics
