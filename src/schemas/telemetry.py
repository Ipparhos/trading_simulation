from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.schemas.orders import LedgerEntry, TradeRecord, ensure_utc_timestamp


class AccountStatus(str, Enum):
    """Current evaluation or breach state of the simulated account."""

    ACTIVE = "ACTIVE"
    PASSED = "PASSED"
    BREACHED_MDL = "BREACHED_MDL"
    BREACHED_MDD = "BREACHED_MDD"


class OhlcBar(BaseModel):
    """Single historical OHLCV candlestick with UTC timestamp."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    timestamp: datetime
    open: float = Field(gt=0)
    high: float = Field(gt=0)
    low: float = Field(gt=0)
    close: float = Field(gt=0)
    volume: float = Field(default=0.0, ge=0)

    @field_validator("timestamp")
    @classmethod
    def _validate_timestamp(cls, value: datetime) -> datetime:
        return ensure_utc_timestamp(value)


class BarSnapshot(BaseModel):
    """Bar-level telemetry snapshot emitted after processing a single bar."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    timestamp: datetime
    closed_balance: float
    worst_intrabar_equity: float
    best_intrabar_equity: float
    close_equity: float
    active_mdd_floor: float
    active_mdl_floor: Optional[float] = None
    status: AccountStatus

    @field_validator("timestamp")
    @classmethod
    def _validate_timestamp(cls, value: datetime) -> datetime:
        return ensure_utc_timestamp(value)


class SimulationResult(BaseModel):
    """Complete simulation output including status, trades, ledger, and bar snapshots."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    status: AccountStatus
    final_balance: float
    final_equity: float
    breach_timestamp: Optional[datetime] = None
    breach_reason: Optional[str] = None
    trades: list[TradeRecord] = Field(default_factory=list)
    ledger: list[LedgerEntry] = Field(default_factory=list)
    snapshots: list[BarSnapshot] = Field(default_factory=list)

    @field_validator("breach_timestamp")
    @classmethod
    def _validate_breach_timestamp(
        cls, value: Optional[datetime]
    ) -> Optional[datetime]:
        if value is not None:
            return ensure_utc_timestamp(value)
        return None
