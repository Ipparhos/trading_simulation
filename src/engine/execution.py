from datetime import datetime
from typing import Optional

from src.engine.fee_calculator import (
    BPS_DIVISOR,
    compute_commission,
    effective_fill_price,
)
from src.engine.rules_guard import RulesGuard
from src.schemas.config import ChallengeConfig
from src.schemas.orders import (
    ExitReason,
    LedgerEntry,
    LedgerEntryType,
    Position,
    Side,
    TradeRecord,
    ensure_utc_timestamp,
)


class ExecutionEngine:
    """Handles order fills, ledger accounting, and pessimistic intrabar price resolution."""

    def __init__(self, config: ChallengeConfig) -> None:
        self.config = config

    def open_position(
        self,
        guard: RulesGuard,
        ledger: list[LedgerEntry],
        symbol: str,
        side: Side,
        qty: float,
        raw_price: float,
        entry_time: datetime,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        slippage_bps: float = 0.0,
    ) -> Position:
        """Open a position at raw_price, deducting entry commission immediately."""
        entry_time_utc = ensure_utc_timestamp(entry_time)
        fill_price = effective_fill_price(
            raw_price=raw_price,
            side=side,
            is_entry=True,
            spread_markup_bps_per_side=self.config.spread_markup_bps_per_side,
            slippage_bps=slippage_bps,
        )
        entry_fee = compute_commission(
            notional=qty * fill_price,
            commission_bps_per_side=self.config.commission_bps_per_side,
        )
        balance_after = guard.record_balance_delta(-entry_fee, update_hwm=False)
        ledger.append(
            LedgerEntry(
                timestamp=entry_time_utc,
                entry_type=LedgerEntryType.ENTRY_FEE,
                amount=-entry_fee,
                balance_after=balance_after,
            )
        )
        return Position(
            symbol=symbol,
            side=side,
            qty=qty,
            entry_price=fill_price,
            entry_time=entry_time_utc,
            stop_loss=stop_loss,
            take_profit=take_profit,
            realized_entry_fee=entry_fee,
            accumulated_swap_fees=0.0,
        )

    def close_position(
        self,
        guard: RulesGuard,
        ledger: list[LedgerEntry],
        trades: list[TradeRecord],
        position: Position,
        raw_price: float,
        exit_time: datetime,
        reason: ExitReason,
        slippage_bps: float = 0.0,
    ) -> TradeRecord:
        """Close an open position, update closed balance, and record ledger/trade entries."""
        exit_time_utc = ensure_utc_timestamp(exit_time)
        exit_fill = effective_fill_price(
            raw_price=raw_price,
            side=position.side,
            is_entry=False,
            spread_markup_bps_per_side=self.config.spread_markup_bps_per_side,
            slippage_bps=slippage_bps,
        )
        if position.side == Side.LONG:
            gross_pnl = position.qty * (exit_fill - position.entry_price)
        else:
            gross_pnl = position.qty * (position.entry_price - exit_fill)

        exit_fee = compute_commission(
            notional=position.qty * exit_fill,
            commission_bps_per_side=self.config.commission_bps_per_side,
        )

        balance_after_pnl = guard.record_balance_delta(gross_pnl, update_hwm=False)
        ledger.append(
            LedgerEntry(
                timestamp=exit_time_utc,
                entry_type=LedgerEntryType.REALIZED_PNL,
                amount=gross_pnl,
                balance_after=balance_after_pnl,
            )
        )
        balance_after_fee = guard.record_balance_delta(-exit_fee, update_hwm=True)
        ledger.append(
            LedgerEntry(
                timestamp=exit_time_utc,
                entry_type=LedgerEntryType.EXIT_FEE,
                amount=-exit_fee,
                balance_after=balance_after_fee,
            )
        )

        net_pnl = (
            gross_pnl
            - position.realized_entry_fee
            - exit_fee
            - position.accumulated_swap_fees
        )
        trade = TradeRecord(
            symbol=position.symbol,
            side=position.side,
            qty=position.qty,
            entry_price=position.entry_price,
            exit_price=exit_fill,
            entry_time=position.entry_time,
            exit_time=exit_time_utc,
            exit_reason=reason,
            gross_pnl=gross_pnl,
            entry_fee=position.realized_entry_fee,
            exit_fee=exit_fee,
            swap_fees=position.accumulated_swap_fees,
            net_pnl=net_pnl,
        )
        trades.append(trade)
        return trade

    @staticmethod
    def adverse_raw_price(
        position: Position,
        open_price: float,
        high_price: float,
        low_price: float,
    ) -> tuple[float, bool]:
        """Return (adverse_raw_price, stop_loss_touched) for the current bar."""
        if position.side == Side.LONG:
            if position.stop_loss is not None and low_price <= position.stop_loss:
                if open_price <= position.stop_loss:
                    return open_price, True
                return max(low_price, position.stop_loss), True
            return low_price, False

        if position.stop_loss is not None and high_price >= position.stop_loss:
            if open_price >= position.stop_loss:
                return open_price, True
            return min(high_price, position.stop_loss), True
        return high_price, False

    @staticmethod
    def favorable_raw_price(
        position: Position,
        open_price: float,
        high_price: float,
        low_price: float,
    ) -> tuple[float, bool]:
        """Return (favorable_raw_price, take_profit_touched) for the current bar."""
        del open_price
        if position.side == Side.LONG:
            tp_hit = (
                position.take_profit is not None and high_price >= position.take_profit
            )
            return high_price, tp_hit

        tp_hit = (
            position.take_profit is not None and low_price <= position.take_profit
        )
        return low_price, tp_hit

    def solve_target_raw_price(
        self,
        position: Position,
        guard: RulesGuard,
        slippage_bps: float = 0.0,
    ) -> float:
        """Solve for the exact raw market price where live_equity hits profit_target_equity."""
        comm_rate = self.config.commission_bps_per_side / BPS_DIVISOR
        spread_rate = (
            self.config.spread_markup_bps_per_side + slippage_bps
        ) / BPS_DIVISOR
        target = guard.profit_target_equity
        balance = guard.closed_balance

        if position.side == Side.LONG:
            effective_exit = (
                target - balance + position.qty * position.entry_price
            ) / (position.qty * (1.0 - comm_rate))
            return effective_exit / (1.0 - spread_rate)

        effective_exit = (
            balance + position.qty * position.entry_price - target
        ) / (position.qty * (1.0 + comm_rate))
        return effective_exit / (1.0 + spread_rate)
