from abc import ABC, abstractmethod
from typing import Optional, Sequence, Union

from pydantic import BaseModel, ConfigDict, Field

from src.schemas.orders import OrderType, Position, Side
from src.schemas.telemetry import OhlcBar


class OrderIntent(BaseModel):
    """Signal emitted at the close of bar i to open a position on bar i+1."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    symbol: str = "BTC/USD"
    side: Side
    qty: float = Field(gt=0)
    order_type: OrderType = OrderType.MARKET
    limit_price: Optional[float] = Field(default=None, gt=0)
    stop_loss: Optional[float] = Field(default=None, gt=0)
    take_profit: Optional[float] = Field(default=None, gt=0)


class CloseIntent(BaseModel):
    """Signal emitted at the close of bar i to exit the open position at bar i+1 open."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    reason: str = "SIGNAL_EXIT"


StrategySignal = Union[OrderIntent, CloseIntent]


class BaseStrategy(ABC):
    """Abstract interface for zero-lookahead event-driven strategies."""

    @abstractmethod
    def on_bar(
        self,
        index: int,
        bar: OhlcBar,
        history_view: Sequence[OhlcBar],
        open_position: Optional[Position],
        equity: float,
    ) -> Optional[StrategySignal]:
        """Evaluate closed bar at `index` using only `history_view` (bars 0..index)."""
