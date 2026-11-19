from typing import Optional

from src.engine.fee_calculator import compute_commission, effective_fill_price
from src.schemas.config import ChallengeConfig, DrawdownType, MdlResetBasis
from src.schemas.orders import Position, Side
from src.schemas.telemetry import AccountStatus


class RulesGuard:
    """Real-time balance, live equity, drawdown floor, and profit target guard."""

    def __init__(self, config: ChallengeConfig) -> None:
        self.config = config
        self.closed_balance: float = config.initial_balance
        self.highest_closed_balance: float = config.initial_balance
        self.active_mdd_floor: float = config.initial_balance * (
            1.0 - config.max_drawdown_pct
        )
        self.active_mdl_floor: Optional[float] = (
            config.initial_balance * (1.0 - config.max_daily_loss_pct)
            if config.max_daily_loss_pct is not None
            else None
        )
        self.profit_target_equity: float = config.initial_balance * (
            1.0 + config.profit_target_pct
        )
        self.status: AccountStatus = AccountStatus.ACTIVE

    def record_balance_delta(self, delta: float, update_hwm: bool = True) -> float:
        """Apply a realized balance change and optionally update closed high-water mark."""
        self.closed_balance += delta
        if update_hwm and self.closed_balance > self.highest_closed_balance:
            self.highest_closed_balance = self.closed_balance
        return self.closed_balance

    def live_equity(
        self,
        position: Optional[Position],
        raw_market_price: float,
        slippage_bps: float = 0.0,
    ) -> float:
        """Compute mark-to-market equity net of estimated close spread and commission."""
        if position is None:
            return self.closed_balance

        effective_exit = effective_fill_price(
            raw_price=raw_market_price,
            side=position.side,
            is_entry=False,
            spread_markup_bps_per_side=self.config.spread_markup_bps_per_side,
            slippage_bps=slippage_bps,
        )
        if position.side == Side.LONG:
            unrealized_pnl = position.qty * (effective_exit - position.entry_price)
        else:
            unrealized_pnl = position.qty * (position.entry_price - effective_exit)

        estimated_exit_comm = compute_commission(
            notional=position.qty * effective_exit,
            commission_bps_per_side=self.config.commission_bps_per_side,
        )
        return self.closed_balance + unrealized_pnl - estimated_exit_comm

    def on_daily_reset(self, current_close_equity: float) -> None:
        """Recalculate active MDL floor and trailing-closed MDD floor at daily reset."""
        if self.config.max_daily_loss_pct is not None:
            if self.config.mdl_reset_basis == MdlResetBasis.BALANCE:
                basis = self.closed_balance
            elif self.config.mdl_reset_basis == MdlResetBasis.EQUITY:
                basis = current_close_equity
            else:
                basis = max(self.closed_balance, current_close_equity)
            self.active_mdl_floor = basis * (1.0 - self.config.max_daily_loss_pct)

        if self.config.drawdown_type == DrawdownType.TRAILING_CLOSED:
            max_dd_dollars = self.config.initial_balance * self.config.max_drawdown_pct
            self.active_mdd_floor = self.highest_closed_balance - max_dd_dollars

    def check_breach(self, equity: float) -> Optional[AccountStatus]:
        """Pure query: return breach status if equity hits MDD or MDL floor."""
        if equity <= self.active_mdd_floor:
            return AccountStatus.BREACHED_MDD
        if self.active_mdl_floor is not None and equity <= self.active_mdl_floor:
            return AccountStatus.BREACHED_MDL
        return None

    def check_target_reached(self, equity: float) -> bool:
        """Pure query: return True if equity meets or exceeds profit target."""
        return equity >= self.profit_target_equity

    def transition_status(self, status: AccountStatus) -> None:
        """Explicitly transition the account status after accounting is finalized."""
        self.status = status
