import os
from pathlib import Path

import pandas as pd
import pytest

from rdagent.scenarios.qlib.crypto_factor.data import cache_key, select_top_universe, validate_candle_list


def _candles(symbol: str, train_volume: float, future_volume: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "candle_begin_time": pd.to_datetime(["2023-12-31", "2024-01-01"]),
            "symbol": [symbol, symbol],
            "open": [1.0, 1.0],
            "high": [1.1, 1.1],
            "low": [0.9, 0.9],
            "close": [1.0, 1.0],
            "volume": [1.0, 1.0],
            "quote_volume": [train_volume, future_volume],
        }
    )


def test_top100_selection_ignores_post_training_liquidity():
    candles = [_candles("TRAIN_WINNER", 100.0, 0.0), _candles("FUTURE_WINNER", 1.0, 1e12)]
    selected = select_top_universe(candles, "2021-01-01", "2023-12-31 23:59:59", size=1)
    assert selected == ["TRAIN_WINNER"]


def test_data_structure_validation_rejects_missing_ohlcv():
    with pytest.raises(ValueError, match="missing columns"):
        validate_candle_list([pd.DataFrame({"symbol": ["BTC"], "close": [1.0]})])


def test_cache_invalidates_on_source_and_data_changes(tmp_path):
    data_path = tmp_path / "candles.pkl"
    pd.to_pickle([_candles("BTC", 1.0, 1.0)], data_path)
    source = tmp_path / "factors"
    source.mkdir()
    factor = source / "Factor.py"
    factor.write_text("def signal(df, param, factor_name): return df\n")
    first = cache_key(data_path, source, "Factor", ["BTC"])

    factor.write_text("def signal(df, param, factor_name):\n    df[factor_name] = 1\n    return df\n")
    second = cache_key(data_path, source, "Factor", ["BTC"])
    assert first != second

    with data_path.open("ab") as stream:
        stream.write(b"changed")
    os.utime(data_path, None)
    third = cache_key(data_path, source, "Factor", ["BTC"])
    assert second != third
