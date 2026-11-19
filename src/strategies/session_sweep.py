from typing import Optional, Sequence

from pydantic import BaseModel, ConfigDict, Field

from src.schemas.orders import OrderType, Position, Side
from src.schemas.telemetry import OhlcBar
from src.strategies.base import BaseStrategy, OrderIntent, StrategySignal


class SessionSweepConfig(BaseModel):
    """Configuration parameters for the Session Liquidity Sweep strategy."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    symbol: str = "BTC/USD"
    lookback_bars: int = Field(default=20, ge=2)
    risk_per_trade_pct: float = Field(default=0.005, gt=0, le=0.05)
    stop_buffer_pct: float = Field(default=0.002, gt=0)
    reward_risk_ratio: float = Field(default=2.0, ge=1.0)
    max_leverage: float = Field(default=5.0, gt=0)


class SessionSweepStrategy(BaseStrategy):
    """Reference liquidity-sweep strategy trading false breakouts of prior range highs/lows."""

    def __init__(self, config: Optional[SessionSweepConfig] = None) -> None:
        self.config = config or SessionSweepConfig()

    def on_bar(
        self,
        index: int,
        bar: OhlcBar,
        history_view: Sequence[OhlcBar],
        open_position: Optional[Position],
        equity: float,
    ) -> Optional[StrategySignal]:
        if open_position is not None or equity <= 0:
            return None
        if (
            index < self.config.lookback_bars
            or len(history_view) <= self.config.lookback_bars
        ):
            return None

        prior_window = history_view[index - self.config.lookback_bars : index]
        prior_high = max(b.high for b in prior_window)
        prior_low = min(b.low for b in prior_window)

        # Bullish liquidity sweep: wick sweeps below prior_low and closes back above prior_low
        if bar.low < prior_low and bar.close > prior_low:
            stop_loss = min(bar.low, prior_low) * (1.0 - self.config.stop_buffer_pct)
            risk_per_unit = bar.close - stop_loss
            if risk_per_unit <= 0:
                return None
            take_profit = bar.close + self.config.reward_risk_ratio * risk_per_unit
            qty = self._compute_qty(equity, bar.close, risk_per_unit)
            if qty <= 0:
                return None
            return OrderIntent(
                symbol=self.config.symbol,
                side=Side.LONG,
                qty=qty,
                order_type=OrderType.MARKET,
                stop_loss=stop_loss,
                take_profit=take_profit,
            )

        # Bearish liquidity sweep: wick sweeps above prior_high and closes back below prior_high
        if bar.high > prior_high and bar.close < prior_high:
            stop_loss = max(bar.high, prior_high) * (1.0 + self.config.stop_buffer_pct)
            risk_per_unit = stop_loss - bar.close
            if risk_per_unit <= 0:
                return None
            take_profit = bar.close - self.config.reward_risk_ratio * risk_per_unit
            if take_profit <= 0:
                return None
            qty = self._compute_qty(equity, bar.close, risk_per_unit)
            if qty <= 0:
                return None
            return OrderIntent(
                symbol=self.config.symbol,
                side=Side.SHORT,
                qty=qty,
                order_type=OrderType.MARKET,
                stop_loss=stop_loss,
                take_profit=take_profit,
            )

        return None

    def _compute_qty(
        self, equity: float, reference_price: float, risk_per_unit: float
    ) -> float:
        risk_usd = equity * self.config.risk_per_trade_pct
        raw_qty = risk_usd / risk_per_unit
        max_notional_qty = (equity * self.config.max_leverage) / reference_price
        return max(0.0, min(raw_qty, max_notional_qty))
