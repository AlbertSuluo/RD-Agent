import pandas as pd

from rdagent.scenarios.qlib.crypto_factor.config import CryptoFactorPropSetting
from rdagent.scenarios.qlib.crypto_factor.qlib_analysis import (
    CryptoRiskRecorder,
    build_backtest_config,
    materialize_test_provider,
)


def test_hourly_qlib_configuration_is_fixed_to_crypto_contract():
    config = build_backtest_config(CryptoFactorPropSetting())
    assert config["freq"] == "60min"
    assert config["benchmark"] is None
    assert config["strategy"]["kwargs"] == {"topk": 50, "n_drop": 5}
    assert config["exchange_kwargs"] == {
        "freq": "60min",
        "deal_price": "$close",
        "open_cost": 0.0004,
        "close_cost": 0.0004,
        "min_cost": 0.0,
        "limit_threshold": None,
        "trade_unit": None,
    }
    assert config["annualization_hours"] == 8760
    assert config["segments"]["train"] == ("2021-01-01", "2023-12-31 23:59:59")
    assert config["segments"]["valid"] == ("2024-01-01", "2024-12-31 23:59:59")
    assert config["segments"]["test"] == ("2025-01-01", "2025-07-06 22:00:00")


def test_crypto_risk_recorder_uses_8760_hours():
    report = pd.DataFrame({"return": [0.001] * 8760, "cost": [0.0] * 8760})
    metrics = CryptoRiskRecorder().calculate(report)
    assert metrics["annualized_return"] == (1.001**8760) - 1


def test_minimal_provider_contains_only_test_period_quotes(tmp_path):
    index = pd.MultiIndex.from_tuples(
        [
            (pd.Timestamp("2024-12-31 23:00:00"), "BTCUSDT"),
            (pd.Timestamp("2025-01-01 00:00:00"), "BTCUSDT"),
            (pd.Timestamp("2025-01-01 01:00:00"), "BTCUSDT"),
        ],
        names=["datetime", "instrument"],
    )
    market = pd.DataFrame({"close": [99.0, 100.0, 101.0], "volume": [1.0, 2.0, 3.0]}, index=index)
    provider = materialize_test_provider(market, tmp_path / "provider", "2025-01-01", "2025-01-01 01:00:00")
    calendar = (provider / "calendars" / "60min.txt").read_text().splitlines()
    assert calendar == ["2025-01-01 00:00:00", "2025-01-01 01:00:00"]
    assert (provider / "features" / "btcusdt" / "btcusdt.close.60min.bin").exists()
