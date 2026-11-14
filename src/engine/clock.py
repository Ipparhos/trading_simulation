from datetime import datetime, timedelta
from typing import Optional

from src.schemas.orders import ensure_utc_timestamp

ROLLOVER_INTERVAL_HOURS = 4


def check_4h_rollovers(prev_time: datetime, curr_time: datetime) -> list[datetime]:
    """Return all UTC 4-hour rollover boundaries strictly inside (prev_time, curr_time]."""
    prev_utc = ensure_utc_timestamp(prev_time)
    curr_utc = ensure_utc_timestamp(curr_time)
    if curr_utc <= prev_utc:
        return []

    base_hour = (prev_utc.hour // ROLLOVER_INTERVAL_HOURS) * ROLLOVER_INTERVAL_HOURS
    candidate = prev_utc.replace(
        hour=base_hour, minute=0, second=0, microsecond=0
    )
    if candidate <= prev_utc:
        candidate += timedelta(hours=ROLLOVER_INTERVAL_HOURS)

    rollovers: list[datetime] = []
    while candidate <= curr_utc:
        rollovers.append(candidate)
        candidate += timedelta(hours=ROLLOVER_INTERVAL_HOURS)
    return rollovers


def next_daily_reset_after(
    prev_time: datetime,
    reset_hour: int = 0,
    reset_minute: int = 30,
) -> datetime:
    """Return the first UTC daily reset timestamp strictly after prev_time."""
    prev_utc = ensure_utc_timestamp(prev_time)
    candidate = prev_utc.replace(
        hour=reset_hour, minute=reset_minute, second=0, microsecond=0
    )
    if candidate <= prev_utc:
        candidate += timedelta(days=1)
    return candidate


def crossed_daily_reset(
    prev_time: datetime,
    curr_time: datetime,
    reset_hour: int = 0,
    reset_minute: int = 30,
) -> bool:
    """Return True if (prev_time, curr_time] crosses the UTC daily reset boundary."""
    prev_utc = ensure_utc_timestamp(prev_time)
    curr_utc = ensure_utc_timestamp(curr_time)
    if curr_utc <= prev_utc:
        return False

    candidate = next_daily_reset_after(
        prev_utc, reset_hour=reset_hour, reset_minute=reset_minute
    )
    return candidate <= curr_utc


class EngineClock:
    """Sequential UTC clock tracking 4-hour swap boundaries and daily reset triggers."""

    def __init__(self, reset_hour: int = 0, reset_minute: int = 30) -> None:
        self.reset_hour = reset_hour
        self.reset_minute = reset_minute
        self.last_processed_time: Optional[datetime] = None

    def advance(self, curr_time: datetime) -> tuple[list[datetime], bool]:
        """Advance clock to curr_time and return (4h_rollovers, crossed_daily_reset)."""
        curr_utc = ensure_utc_timestamp(curr_time)
        if self.last_processed_time is None:
            self.last_processed_time = curr_utc
            return [], False
        if curr_utc < self.last_processed_time:
            raise ValueError("EngineClock received out-of-order timestamp.")

        rollovers = check_4h_rollovers(self.last_processed_time, curr_utc)
        did_reset = crossed_daily_reset(
            self.last_processed_time,
            curr_utc,
            reset_hour=self.reset_hour,
            reset_minute=self.reset_minute,
        )
        self.last_processed_time = curr_utc
        return rollovers, did_reset
