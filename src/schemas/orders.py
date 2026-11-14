from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


def ensure_utc_timestamp(value: datetime) -> datetime:
    """Validate that a datetime is timezone-aware and in UTC."""
    if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
        raise ValueError("Timestamp must be timezone-aware (UTC).")
    if value.utcoffset() != timezone.utc.utcoffset(value):
        raise ValueError("Timestamp must be in UTC timezone.")
    return value.astimezone(timezone.utc)


class Side(str, Enum):
    """Directional side of an order or open position."""

    LONG = "LONG"
    SHORT = "SHORT"


class OrderType(str, Enum):
    """Supported execution order types."""

    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP_MARKET = "STOP_MARKET"


class ExitReason(str, Enum):
    """Reason a position was closed or force-liquidated."""

    STOP_LOSS = "STOP_LOSS"
    TAKE_PROFIT = "TAKE_PROFIT"
    SIGNAL_EXIT = "SIGNAL_EXIT"
    TARGET_AUTO_LIQUIDATE = "TARGET_AUTO_LIQUIDATE"
    BREACH_LIQUIDATE = "BREACH_LIQUIDATE"


class LedgerEntryType(str, Enum):
    """Categories of realized account balance adjustments."""

    ENTRY_FEE = "ENTRY_FEE"
    EXIT_FEE = "EXIT_FEE"
    SWAP_FEE = "SWAP_FEE"
    REALIZED_PNL = "REALIZED_PNL"


class Position(BaseModel):
    """Active open position state tracked by the execution engine."""

    model_config = ConfigDict(extra="forbid")

    symbol: str
    side: Side
    qty: float = Field(gt=0)
    entry_price: float = Field(gt=0)
    entry_time: datetime
    stop_loss: Optional[float] = Field(default=None, gt=0)
    take_profit: Optional[float] = Field(default=None, gt=0)
    realized_entry_fee: float = 0.0
    accumulated_swap_fees: float = 0.0

    @field_validator("entry_time")
    @classmethod
    def _validate_entry_time(cls, value: datetime) -> datetime:
        return ensure_utc_timestamp(value)


class TradeRecord(BaseModel):
    """Immutable record of a completed round-trip trade."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    symbol: str
    side: Side
    qty: float = Field(gt=0)
    entry_price: float = Field(gt=0)
    exit_price: float = Field(gt=0)
    entry_time: datetime
    exit_time: datetime
    exit_reason: ExitReason
    gross_pnl: float
    entry_fee: float
    exit_fee: float
    swap_fees: float
    net_pnl: float

    @field_validator("entry_time", "exit_time")
    @classmethod
    def _validate_times(cls, value: datetime) -> datetime:
        return ensure_utc_timestamp(value)


Fill = TradeRecord


class LedgerEntry(BaseModel):
    """Immutable chronological entry describing every balance change."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    timestamp: datetime
    entry_type: LedgerEntryType
    amount: float
    balance_after: float

    @field_validator("timestamp")
    @classmethod
    def _validate_timestamp(cls, value: datetime) -> datetime:
        return ensure_utc_timestamp(value)
