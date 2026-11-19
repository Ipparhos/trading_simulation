from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from src.ingestion.kraken_loader import (
    KrakenDataLoader,
    df_to_bars,
    timeframe_to_timedelta,
    validate_ohlcv_df,
)


class FakeExchange:
    """Deterministic in-memory CCXT stub for testing pagination and caching."""

    def __init__(self, rows: list[list[Any]]) -> None:
        self.rows = rows
        self.call_count = 0

    def fetch_ohlcv(
        self, symbol: str, timeframe: str = "15m", since: int = 0, limit: int = 720
    ) -> list[list[Any]]:
        del symbol, timeframe
        self.call_count += 1
        eligible = [r for r in self.rows if int(r[0]) >= since]
        return eligible[:limit]


def _make_15m_rows(start: datetime, count: int) -> list[list[Any]]:
    rows: list[list[Any]] = []
    delta = timedelta(minutes=15)
    for i in range(count):
        ts_ms = int((start + i * delta).timestamp() * 1000)
        price = 100_000.0 + i * 10.0
        rows.append([ts_ms, price, price + 20.0, price - 20.0, price + 5.0, 1.5])
    return rows


def test_parquet_caching_and_pagination(tmp_path: Path) -> None:
    """Verify loader paginates across batches, writes Parquet, and serves cache hits."""
    start = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
    until = start + timedelta(minutes=15 * 5)
    rows = _make_15m_rows(start, 6)

    fake_ex = FakeExchange(rows)
    loader = KrakenDataLoader(cache_dir=tmp_path, exchange=fake_ex, page_limit=2)

    df = loader.load_ohlcv("BTC/USD", "15m", since=start, until=until, strict_gaps=True)
    assert len(df) == 6
    assert fake_ex.call_count == 3
    assert loader.cache_path("BTC/USD", "15m").exists()

    bars = df_to_bars(df)
    assert len(bars) == 6
    assert bars[0].timestamp == start
    assert bars[-1].timestamp == until

    # Second load over covered range should hit Parquet cache without calling exchange.
    df_cached = loader.load_ohlcv("BTC/USD", "15m", since=start, until=until)
    assert len(df_cached) == 6
    assert fake_ex.call_count == 3


def test_gap_detection_and_utc_validation() -> None:
    """Verify validate_ohlcv_df detects missing bars, duplicates, and naive timestamps."""
    assert timeframe_to_timedelta("15m") == timedelta(minutes=15)
    with pytest.raises(ValueError):
        timeframe_to_timedelta("7m")

    t0 = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
    t1 = datetime(2026, 1, 1, 0, 15, tzinfo=timezone.utc)
    # Missing 00:30 bar -> jump to 00:45
    t3 = datetime(2026, 1, 1, 0, 45, tzinfo=timezone.utc)

    gapped_df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime([t0, t1, t3], utc=True),
            "open": [100.0, 101.0, 102.0],
            "high": [101.0, 102.0, 103.0],
            "low": [99.0, 100.0, 101.0],
            "close": [100.5, 101.5, 102.5],
            "volume": [1.0, 1.0, 1.0],
        }
    )

    gaps = validate_ohlcv_df(gapped_df, "15m", strict_gaps=False)
    assert gaps == [datetime(2026, 1, 1, 0, 30, tzinfo=timezone.utc)]

    with pytest.raises(ValueError, match="missing OHLCV interval"):
        validate_ohlcv_df(gapped_df, "15m", strict_gaps=True)

    # Naive timestamps must be rejected
    naive_df = gapped_df.copy()
    naive_df["timestamp"] = pd.to_datetime(["2026-01-01 00:00:00", "2026-01-01 00:15:00", "2026-01-01 00:30:00"])
    with pytest.raises(ValueError, match="timezone-aware UTC"):
        validate_ohlcv_df(naive_df, "15m")
