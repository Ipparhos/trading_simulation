from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

import ccxt
import pandas as pd

from src.schemas.orders import ensure_utc_timestamp
from src.schemas.telemetry import OhlcBar

DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "parquet"
REQUIRED_OHLCV_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]

SUPPORTED_TIMEFRAMES: dict[str, timedelta] = {
    "1m": timedelta(minutes=1),
    "5m": timedelta(minutes=5),
    "15m": timedelta(minutes=15),
    "1h": timedelta(hours=1),
    "4h": timedelta(hours=4),
    "1d": timedelta(days=1),
}


def timeframe_to_timedelta(timeframe: str) -> timedelta:
    """Map a CCXT timeframe string to a Python timedelta."""
    if timeframe not in SUPPORTED_TIMEFRAMES:
        raise ValueError(f"Unsupported timeframe: '{timeframe}'")
    return SUPPORTED_TIMEFRAMES[timeframe]


def validate_ohlcv_df(
    df: pd.DataFrame,
    timeframe: str,
    strict_gaps: bool = False,
) -> list[datetime]:
    """Validate OHLCV columns, UTC awareness, ordering, uniqueness, and bar gaps."""
    missing_cols = [c for c in REQUIRED_OHLCV_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required OHLCV columns: {missing_cols}")
    if df.empty:
        return []

    ts_dtype = df["timestamp"].dtype
    if not isinstance(ts_dtype, pd.DatetimeTZDtype) or str(ts_dtype.tz) != "UTC":
        raise ValueError("DataFrame 'timestamp' column must be timezone-aware UTC.")

    if df["timestamp"].duplicated().any():
        raise ValueError("DataFrame contains duplicate timestamps.")
    if not df["timestamp"].is_monotonic_increasing:
        raise ValueError("DataFrame timestamps must be sorted in ascending order.")

    expected_delta = timeframe_to_timedelta(timeframe)
    diffs = df["timestamp"].diff()
    gap_indices = df.index[diffs > expected_delta].tolist()

    gaps: list[datetime] = []
    for idx in gap_indices:
        pos = df.index.get_loc(idx)
        prev_ts = df["timestamp"].iloc[pos - 1].to_pydatetime()
        gaps.append(prev_ts + expected_delta)

    if strict_gaps and gaps:
        raise ValueError(
            f"Detected {len(gaps)} missing OHLCV interval(s); first gap at {gaps[0].isoformat()}"
        )
    return gaps


def df_to_bars(df: pd.DataFrame) -> list[OhlcBar]:
    """Convert a validated OHLCV DataFrame into a sequence of OhlcBar objects."""
    bars: list[OhlcBar] = []
    for row in df.itertuples(index=False):
        ts = row.timestamp.to_pydatetime()
        bars.append(
            OhlcBar(
                timestamp=ts,
                open=float(row.open),
                high=float(row.high),
                low=float(row.low),
                close=float(row.close),
                volume=float(row.volume),
            )
        )
    return bars


class KrakenDataLoader:
    """CCXT historical OHLCV fetcher with local Parquet caching and gap validation."""

    def __init__(
        self,
        cache_dir: Path = DEFAULT_CACHE_DIR,
        exchange: Optional[Any] = None,
        page_limit: int = 720,
    ) -> None:
        self.cache_dir = cache_dir
        self._exchange = exchange
        self.page_limit = page_limit

    @property
    def exchange(self) -> Any:
        if self._exchange is None:
            self._exchange = ccxt.kraken({"enableRateLimit": True})
        return self._exchange

    def cache_path(self, symbol: str, timeframe: str) -> Path:
        slug = symbol.lower().replace("/", "_").replace("-", "_")
        return self.cache_dir / f"{slug}_{timeframe}.parquet"

    def _fetch_range(
        self, symbol: str, timeframe: str, since_utc: datetime, until_utc: datetime
    ) -> pd.DataFrame:
        delta = timeframe_to_timedelta(timeframe)
        step_ms = int(delta.total_seconds() * 1000)
        cursor_ms = int(since_utc.timestamp() * 1000)
        until_ms = int(until_utc.timestamp() * 1000)

        rows: list[list[Any]] = []
        while cursor_ms <= until_ms:
            batch = self.exchange.fetch_ohlcv(
                symbol, timeframe=timeframe, since=cursor_ms, limit=self.page_limit
            )
            if not batch:
                break
            rows.extend(batch)
            last_ms = int(batch[-1][0])
            next_ms = last_ms + step_ms
            if next_ms <= cursor_ms or (
                len(batch) < self.page_limit and last_ms >= until_ms
            ):
                break
            cursor_ms = next_ms

        if not rows:
            return pd.DataFrame(
                {
                    "timestamp": pd.Series(dtype="datetime64[ns, UTC]"),
                    "open": pd.Series(dtype="float64"),
                    "high": pd.Series(dtype="float64"),
                    "low": pd.Series(dtype="float64"),
                    "close": pd.Series(dtype="float64"),
                    "volume": pd.Series(dtype="float64"),
                }
            )

        raw_df = pd.DataFrame(rows, columns=REQUIRED_OHLCV_COLUMNS)
        raw_df["timestamp"] = pd.to_datetime(raw_df["timestamp"], unit="ms", utc=True)
        for col in ("open", "high", "low", "close", "volume"):
            raw_df[col] = raw_df[col].astype(float)
        return raw_df

    def load_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        since: datetime,
        until: Optional[datetime] = None,
        force_refresh: bool = False,
        strict_gaps: bool = False,
    ) -> pd.DataFrame:
        """Load OHLCV bars from Parquet cache or CCXT Kraken, returning validated UTC DataFrame."""
        since_utc = ensure_utc_timestamp(since)
        until_utc = (
            ensure_utc_timestamp(until)
            if until is not None
            else datetime.now(timezone.utc)
        )
        if until_utc < since_utc:
            raise ValueError("'until' timestamp must be >= 'since' timestamp.")

        delta = timeframe_to_timedelta(timeframe)
        path = self.cache_path(symbol, timeframe)
        cached_df: Optional[pd.DataFrame] = None

        if path.exists() and not force_refresh:
            cached_df = pd.read_parquet(path)
            if not cached_df.empty:
                cached_min = cached_df["timestamp"].min().to_pydatetime()
                cached_max = cached_df["timestamp"].max().to_pydatetime()
                if cached_min <= since_utc and cached_max >= until_utc - delta:
                    mask = (cached_df["timestamp"] >= since_utc) & (
                        cached_df["timestamp"] <= until_utc
                    )
                    result_df = cached_df.loc[mask].reset_index(drop=True)
                    validate_ohlcv_df(result_df, timeframe, strict_gaps=strict_gaps)
                    return result_df

        fetched_df = self._fetch_range(symbol, timeframe, since_utc, until_utc)
        if cached_df is not None and not cached_df.empty:
            combined = pd.concat([cached_df, fetched_df], ignore_index=True)
        else:
            combined = fetched_df

        if not combined.empty:
            combined = (
                combined.drop_duplicates(subset=["timestamp"], keep="last")
                .sort_values("timestamp")
                .reset_index(drop=True)
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            combined.to_parquet(path, index=False)
            mask = (combined["timestamp"] >= since_utc) & (
                combined["timestamp"] <= until_utc
            )
            result_df = combined.loc[mask].reset_index(drop=True)
        else:
            result_df = combined

        validate_ohlcv_df(result_df, timeframe, strict_gaps=strict_gaps)
        return result_df
