from __future__ import annotations

import hashlib
import json
import os
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_CANDLE_COLUMNS = {
    "candle_begin_time",
    "symbol",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quote_volume",
}


def validate_candle_list(candles: object) -> list[pd.DataFrame]:
    if not isinstance(candles, list) or not candles:
        raise ValueError("crypto candles must be a non-empty list[pandas.DataFrame]")
    valid = []
    for position, frame in enumerate(candles):
        if not isinstance(frame, pd.DataFrame):
            raise TypeError(f"candles[{position}] is {type(frame).__name__}, expected DataFrame")
        if frame.empty:
            continue
        missing = REQUIRED_CANDLE_COLUMNS - set(frame.columns)
        if missing:
            raise ValueError(f"candles[{position}] missing columns: {sorted(missing)}")
        valid.append(frame)
    if not valid:
        raise ValueError("crypto candle list contains no non-empty instruments")
    return valid


def load_candles(path: Path) -> list[pd.DataFrame]:
    return validate_candle_list(pd.read_pickle(Path(path).expanduser().resolve()))


def select_top_universe(
    candles: list[pd.DataFrame], train_start: str, train_end: str, size: int = 100
) -> list[str]:
    """Select liquidity universe using training-period information only."""

    start, end = pd.Timestamp(train_start), pd.Timestamp(train_end)
    rows = []
    for frame in candles:
        dates = pd.to_datetime(frame["candle_begin_time"])
        sample = frame.loc[dates.between(start, end), ["symbol", "quote_volume"]]
        if sample.empty:
            continue
        symbol = str(sample["symbol"].iloc[0])
        rows.append((symbol, float(pd.to_numeric(sample["quote_volume"], errors="coerce").mean())))
    ranked = sorted(rows, key=lambda item: (-np.nan_to_num(item[1], nan=-np.inf), item[0]))
    return [symbol for symbol, _ in ranked[:size]]


def filter_universe(candles: list[pd.DataFrame], symbols: list[str]) -> list[pd.DataFrame]:
    selected = set(symbols)
    return [frame for frame in candles if not frame.empty and str(frame["symbol"].iloc[0]) in selected]


def source_tree_hash(source_dir: Path, factor_name: str) -> str:
    root = Path(source_dir).resolve()
    digest = hashlib.sha256()
    candidates = sorted({root / f"{factor_name}.py", *root.rglob("*.py")})
    for path in candidates:
        if path.is_file():
            digest.update(str(path.relative_to(root)).encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


@lru_cache(maxsize=8)
def _content_hash(path_text: str, size: int, modified_ns: int) -> str:
    del size, modified_ns  # They are cache-key components and intentionally force invalidation.
    digest = hashlib.sha256()
    with Path(path_text).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def data_source_hash(data_path: Path) -> str:
    path = Path(data_path).expanduser().resolve()
    stat = path.stat()
    return _content_hash(str(path), stat.st_size, stat.st_mtime_ns)


def cache_key(data_path: Path, source_dir: Path, factor_name: str, universe: list[str]) -> str:
    value = "|".join(
        [data_source_hash(data_path), source_tree_hash(source_dir, factor_name), factor_name, *sorted(universe)]
    )
    return hashlib.sha256(value.encode()).hexdigest()[:20]


def safe_instrument(symbol: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9_]+", "_", str(symbol)).strip("_")
    if not clean:
        raise ValueError(f"cannot convert symbol to Qlib instrument: {symbol!r}")
    return clean.upper()


def prepare_data_links(data_path: Path, cache_dir: Path) -> Path:
    """Expose the 582 MB pickle by symlink; never copy it."""

    data_path = Path(data_path).expanduser().resolve()
    runtime_dir = Path(cache_dir).expanduser().resolve() / "runtime_data"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    link = runtime_dir / data_path.name
    if link.is_symlink() and link.resolve() == data_path:
        return runtime_dir
    if link.exists() or link.is_symlink():
        link.unlink()
    os.symlink(data_path, link)
    return runtime_dir


def write_json_atomic(path: Path, value: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str))
    temporary.replace(path)
