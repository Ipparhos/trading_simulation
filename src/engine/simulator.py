from collections.abc import Sequence
from datetime import datetime
from typing import Iterable, Optional, overload

from src.engine.clock import EngineClock, next_daily_reset_after
from src.engine.execution import ExecutionEngine
from src.engine.fee_calculator import compute_4h_swap
from src.engine.rules_guard import RulesGuard
from src.schemas.config import ChallengeConfig
from src.schemas.orders import (
    ExitReason,
    LedgerEntry,
    LedgerEntryType,
    OrderType,
    Position,
    Side,
    TradeRecord,
    ensure_utc_timestamp,
)
from src.schemas.telemetry import (
    AccountStatus,
    BarSnapshot,
    OhlcBar,
    SimulationResult,
)
from src.strategies.base import BaseStrategy, CloseIntent, OrderIntent, StrategySignal


class _BoundedHistoryView(Sequence[OhlcBar]):
    """Read-only view of bars[0..visible_len-1] preventing lookahead access."""

    def __init__(self, source: Sequence[OhlcBar], visible_len: int) -> None:
        self._source = source
        self._visible_len = visible_len

    def __len__(self) -> int:
        return self._visible_len

    @overload
    def __getitem__(self, index: int) -> OhlcBar: ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[OhlcBar]: ...

    def __getitem__(self, index: int | slice) -> OhlcBar | Sequence[OhlcBar]:
        if isinstance(index, slice):
            start, stop, step = index.indices(self._visible_len)
            return [self._source[i] for i in range(start, stop, step)]
        if index < 0:
            index += self._visible_len
        if index < 0 or index >= self._visible_len:
            raise IndexError("Lookahead violation or out-of-bounds history access.")
        return self._source[index]


class Simulator:
    """Sequential event-driven bar simulator enforcing Kraken Prop & Funded rules.

    Assumes sub-4-hour bar granularity (e.g., 1m/5m/15m) so 4h swap rollovers
    align with bar opens. Bars spanning multiple 4h boundaries value each rollover
    at that bar's ``open`` price.
    """

    def __init__(self, config: ChallengeConfig, slippage_bps: float = 0.0) -> None:
        self.config = config
        self.slippage_bps = slippage_bps
        self.clock = EngineClock(
            reset_hour=config.daily_reset_utc_hour,
            reset_minute=config.daily_reset_utc_minute,
        )
        self.guard = RulesGuard(config)
        self.execution = ExecutionEngine(config)
        self.position: Optional[Position] = None
        self.trades: list[TradeRecord] = []
        self.ledger: list[LedgerEntry] = []
        self.snapshots: list[BarSnapshot] = []
        self.breach_timestamp: Optional[datetime] = None
        self.breach_reason: Optional[str] = None

    def open_position(
        self,
        symbol: str,
        side: Side,
        qty: float,
        raw_price: float,
        entry_time: datetime,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        slippage_bps: Optional[float] = None,
    ) -> Position:
        """Open a position at entry_time, synchronizing the engine clock if flat."""
        if self.guard.status != AccountStatus.ACTIVE:
            raise RuntimeError(
                f"Cannot open position when account status is {self.guard.status.value}."
            )
        if self.position is not None:
            raise RuntimeError("A position is already open.")

        entry_utc = ensure_utc_timestamp(entry_time)
        if self.clock.last_processed_time is None:
            self.clock.last_processed_time = entry_utc
        elif entry_utc < self.clock.last_processed_time:
            raise ValueError("Position entry_time precedes clock last_processed_time.")
        elif entry_utc > self.clock.last_processed_time:
            _, did_reset = self.clock.advance(entry_utc)
            if did_reset:
                self.guard.on_daily_reset(self.guard.closed_balance)

        effective_slippage = (
            self.slippage_bps if slippage_bps is None else slippage_bps
        )
        self.position = self.execution.open_position(
            guard=self.guard,
            ledger=self.ledger,
            symbol=symbol,
            side=side,
            qty=qty,
            raw_price=raw_price,
            entry_time=entry_utc,
            stop_loss=stop_loss,
            take_profit=take_profit,
            slippage_bps=effective_slippage,
        )
        return self.position

    def close_position_signal(
        self,
        raw_price: float,
        exit_time: datetime,
        slippage_bps: Optional[float] = None,
    ) -> Optional[TradeRecord]:
        """Close the active position manually due to a strategy signal."""
        if self.position is None:
            return None
        exit_utc = ensure_utc_timestamp(exit_time)
        effective_slippage = (
            self.slippage_bps if slippage_bps is None else slippage_bps
        )
        trade = self.execution.close_position(
            guard=self.guard,
            ledger=self.ledger,
            trades=self.trades,
            position=self.position,
            raw_price=raw_price,
            exit_time=exit_utc,
            reason=ExitReason.SIGNAL_EXIT,
            slippage_bps=effective_slippage,
        )
        self.position = None
        return trade

    def _force_liquidate_on_breach(
        self,
        raw_price: float,
        exit_time: datetime,
        breach_status: AccountStatus,
        reason_text: str,
        slippage_bps: float = 0.0,
    ) -> None:
        if self.position is not None:
            self.execution.close_position(
                guard=self.guard,
                ledger=self.ledger,
                trades=self.trades,
                position=self.position,
                raw_price=raw_price,
                exit_time=exit_time,
                reason=ExitReason.BREACH_LIQUIDATE,
                slippage_bps=slippage_bps,
            )
            self.position = None
        self.guard.transition_status(breach_status)
        self.breach_timestamp = exit_time
        self.breach_reason = reason_text

    def _record_snapshot(
        self,
        timestamp: datetime,
        worst_intrabar_equity: float,
        best_intrabar_equity: float,
        close_equity: float,
    ) -> BarSnapshot:
        snapshot = BarSnapshot(
            timestamp=timestamp,
            closed_balance=self.guard.closed_balance,
            worst_intrabar_equity=worst_intrabar_equity,
            best_intrabar_equity=best_intrabar_equity,
            close_equity=close_equity,
            active_mdd_floor=self.guard.active_mdd_floor,
            active_mdl_floor=self.guard.active_mdl_floor,
            status=self.guard.status,
        )
        self.snapshots.append(snapshot)
        return snapshot

    def _process_step_a_clock_events(self, bar: OhlcBar) -> Optional[BarSnapshot]:
        """Process 4h swap rollovers and daily 00:30 UTC reset in chronological order.

        Note: Swap notional is valued at ``bar.open`` for every 4h boundary crossed
        in ``(prev_time, bar.timestamp]``.
        """
        prev_time = self.clock.last_processed_time
        rollovers, did_reset = self.clock.advance(bar.timestamp)
        if prev_time is None:
            return None

        events: list[tuple[datetime, int]] = [(rt, 0) for rt in rollovers]
        if did_reset:
            reset_time = next_daily_reset_after(
                prev_time,
                reset_hour=self.clock.reset_hour,
                reset_minute=self.clock.reset_minute,
            )
            events.append((reset_time, 1))
        events.sort(key=lambda item: (item[0], item[1]))

        for event_time, event_kind in events:
            if event_kind == 0:
                if self.position is None:
                    continue
                notional = self.position.qty * bar.open
                swap_fee = compute_4h_swap(notional, self.config.daily_swap_rate_pct)
                balance_after = self.guard.record_balance_delta(
                    -swap_fee, update_hwm=False
                )
                self.position.accumulated_swap_fees += swap_fee
                self.ledger.append(
                    LedgerEntry(
                        timestamp=event_time,
                        entry_type=LedgerEntryType.SWAP_FEE,
                        amount=-swap_fee,
                        balance_after=balance_after,
                    )
                )
                open_equity = self.guard.live_equity(self.position, bar.open)
                breach = self.guard.check_breach(open_equity)
                if breach is not None:
                    self._force_liquidate_on_breach(
                        raw_price=bar.open,
                        exit_time=bar.timestamp,
                        breach_status=breach,
                        reason_text=f"Swap rollover breach at {event_time.isoformat()}",
                    )
                    return self._record_snapshot(
                        timestamp=bar.timestamp,
                        worst_intrabar_equity=open_equity,
                        best_intrabar_equity=open_equity,
                        close_equity=self.guard.closed_balance,
                    )
            else:
                open_equity = self.guard.live_equity(self.position, bar.open)
                self.guard.on_daily_reset(current_close_equity=open_equity)
        return None

    def _apply_pending_signal_at_open(
        self, bar: OhlcBar, signal: StrategySignal
    ) -> None:
        if isinstance(signal, CloseIntent):
            if self.position is not None:
                self.close_position_signal(
                    raw_price=bar.open, exit_time=bar.timestamp
                )
            return

        if isinstance(signal, OrderIntent) and self.position is None:
            if signal.order_type == OrderType.MARKET:
                self.open_position(
                    symbol=signal.symbol,
                    side=signal.side,
                    qty=signal.qty,
                    raw_price=bar.open,
                    entry_time=bar.timestamp,
                    stop_loss=signal.stop_loss,
                    take_profit=signal.take_profit,
                )
            elif (
                signal.order_type == OrderType.LIMIT
                and signal.limit_price is not None
            ):
                if signal.side == Side.LONG and bar.low <= signal.limit_price:
                    fill_raw = min(bar.open, signal.limit_price)
                    self.open_position(
                        symbol=signal.symbol,
                        side=signal.side,
                        qty=signal.qty,
                        raw_price=fill_raw,
                        entry_time=bar.timestamp,
                        stop_loss=signal.stop_loss,
                        take_profit=signal.take_profit,
                        slippage_bps=0.0,
                    )
                elif signal.side == Side.SHORT and bar.high >= signal.limit_price:
                    fill_raw = max(bar.open, signal.limit_price)
                    self.open_position(
                        symbol=signal.symbol,
                        side=signal.side,
                        qty=signal.qty,
                        raw_price=fill_raw,
                        entry_time=bar.timestamp,
                        stop_loss=signal.stop_loss,
                        take_profit=signal.take_profit,
                        slippage_bps=0.0,
                    )

    def process_bar(
        self, bar: OhlcBar, pending_signal: Optional[StrategySignal] = None
    ) -> Optional[BarSnapshot]:
        """Process a single OHLC bar strictly in order: Clock -> Adverse -> Favorable."""
        if self.guard.status != AccountStatus.ACTIVE:
            return None

        # Step A: Clock events (4h rollovers & daily 00:30 UTC reset)
        step_a_halt = self._process_step_a_clock_events(bar)
        if step_a_halt is not None:
            return step_a_halt

        if pending_signal is not None:
            self._apply_pending_signal_at_open(bar, pending_signal)

        if self.position is None:
            balance = self.guard.closed_balance
            if self.guard.check_target_reached(balance):
                self.guard.transition_status(AccountStatus.PASSED)
            return self._record_snapshot(
                timestamp=bar.timestamp,
                worst_intrabar_equity=balance,
                best_intrabar_equity=balance,
                close_equity=balance,
            )

        # Step B: Intrabar adverse path first (Pessimistic execution)
        adverse_price, sl_touched = self.execution.adverse_raw_price(
            self.position, bar.open, bar.high, bar.low
        )
        sl_slippage = self.slippage_bps if sl_touched else 0.0
        worst_intrabar_equity = self.guard.live_equity(
            self.position, adverse_price, slippage_bps=sl_slippage
        )
        breach = self.guard.check_breach(worst_intrabar_equity)
        if breach is not None:
            self._force_liquidate_on_breach(
                raw_price=adverse_price,
                exit_time=bar.timestamp,
                breach_status=breach,
                reason_text=f"Intrabar {breach.value} at raw price {adverse_price}",
                slippage_bps=sl_slippage,
            )
            return self._record_snapshot(
                timestamp=bar.timestamp,
                worst_intrabar_equity=worst_intrabar_equity,
                best_intrabar_equity=worst_intrabar_equity,
                close_equity=self.guard.closed_balance,
            )

        if sl_touched:
            self.execution.close_position(
                guard=self.guard,
                ledger=self.ledger,
                trades=self.trades,
                position=self.position,
                raw_price=adverse_price,
                exit_time=bar.timestamp,
                reason=ExitReason.STOP_LOSS,
                slippage_bps=self.slippage_bps,
            )
            self.position = None
            close_equity = self.guard.closed_balance
            if self.guard.check_target_reached(close_equity):
                self.guard.transition_status(AccountStatus.PASSED)
            return self._record_snapshot(
                timestamp=bar.timestamp,
                worst_intrabar_equity=worst_intrabar_equity,
                best_intrabar_equity=close_equity,
                close_equity=close_equity,
            )

        # Step C: Intrabar favorable path second
        best_raw_price, tp_touched = self.execution.favorable_raw_price(
            self.position, bar.open, bar.high, bar.low
        )
        best_intrabar_equity = self.guard.live_equity(self.position, best_raw_price)
        if self.guard.check_target_reached(best_intrabar_equity):
            open_equity = self.guard.live_equity(self.position, bar.open)
            if open_equity >= self.guard.profit_target_equity:
                target_raw = bar.open
            else:
                solved = self.execution.solve_target_raw_price(
                    self.position, self.guard, slippage_bps=0.0
                )
                target_raw = min(bar.high, max(bar.low, solved))
                if (
                    self.guard.live_equity(self.position, target_raw)
                    < self.guard.profit_target_equity - 1e-9
                ):
                    target_raw = best_raw_price
            self.execution.close_position(
                guard=self.guard,
                ledger=self.ledger,
                trades=self.trades,
                position=self.position,
                raw_price=target_raw,
                exit_time=bar.timestamp,
                reason=ExitReason.TARGET_AUTO_LIQUIDATE,
                slippage_bps=0.0,
            )
            self.position = None
            self.guard.transition_status(AccountStatus.PASSED)
            return self._record_snapshot(
                timestamp=bar.timestamp,
                worst_intrabar_equity=worst_intrabar_equity,
                best_intrabar_equity=best_intrabar_equity,
                close_equity=self.guard.closed_balance,
            )

        if tp_touched and self.position.take_profit is not None:
            self.execution.close_position(
                guard=self.guard,
                ledger=self.ledger,
                trades=self.trades,
                position=self.position,
                raw_price=self.position.take_profit,
                exit_time=bar.timestamp,
                reason=ExitReason.TAKE_PROFIT,
                slippage_bps=0.0,
            )
            self.position = None
            close_equity = self.guard.closed_balance
            return self._record_snapshot(
                timestamp=bar.timestamp,
                worst_intrabar_equity=worst_intrabar_equity,
                best_intrabar_equity=best_intrabar_equity,
                close_equity=close_equity,
            )

        close_equity = self.guard.live_equity(self.position, bar.close)
        return self._record_snapshot(
            timestamp=bar.timestamp,
            worst_intrabar_equity=worst_intrabar_equity,
            best_intrabar_equity=best_intrabar_equity,
            close_equity=close_equity,
        )

    def run(self, bars: Iterable[OhlcBar]) -> SimulationResult:
        """Process a stream of bars sequentially until exhausted or halted."""
        for bar in bars:
            if self.guard.status != AccountStatus.ACTIVE:
                break
            self.process_bar(bar)
        return self.get_result()

    def get_result(self) -> SimulationResult:
        """Return the immutable SimulationResult summarizing the run."""
        final_equity = (
            self.snapshots[-1].close_equity
            if self.snapshots
            else self.guard.closed_balance
        )
        return SimulationResult(
            status=self.guard.status,
            final_balance=self.guard.closed_balance,
            final_equity=final_equity,
            breach_timestamp=self.breach_timestamp,
            breach_reason=self.breach_reason,
            trades=list(self.trades),
            ledger=list(self.ledger),
            snapshots=list(self.snapshots),
        )


def run_strategy_simulation(
    bars: Sequence[OhlcBar],
    strategy: BaseStrategy,
    config: ChallengeConfig,
    slippage_bps: float = 0.0,
) -> SimulationResult:
    """Run a strategy sequentially across bars with strict next-bar execution."""
    sim = Simulator(config=config, slippage_bps=slippage_bps)
    pending_signal: Optional[StrategySignal] = None

    for index, bar in enumerate(bars):
        if sim.guard.status != AccountStatus.ACTIVE:
            break
        snapshot = sim.process_bar(bar, pending_signal=pending_signal)
        pending_signal = None
        if snapshot is None or sim.guard.status != AccountStatus.ACTIVE:
            break
        history_view = _BoundedHistoryView(bars, index + 1)
        pending_signal = strategy.on_bar(
            index=index,
            bar=bar,
            history_view=history_view,
            open_position=sim.position,
            equity=snapshot.close_equity,
        )
    return sim.get_result()
