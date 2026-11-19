from collections.abc import Sequence
from datetime import datetime, timedelta, timezone
from typing import Optional

import pytest

from src.engine.simulator import run_strategy_simulation
from src.schemas.config import ChallengeConfig
from src.schemas.orders import Position, Side
from src.schemas.telemetry import OhlcBar
from src.strategies.base import BaseStrategy, OrderIntent, StrategySignal
from src.strategies.session_sweep import SessionSweepConfig, SessionSweepStrategy


class LookaheadProbeStrategy(BaseStrategy):
    """Test strategy attempting to read future bars and emitting a market order on bar 0."""

    def __init__(self) -> None:
        self.lookahead_blocked = False
        self.observed_lengths: list[int] = []

    def on_bar(
        self,
        index: int,
        bar: OhlcBar,
        history_view: Sequence[OhlcBar],
        open_position: Optional[Position],
        equity: float,
    ) -> Optional[StrategySignal]:
        del bar, equity
        self.observed_lengths.append(len(history_view))

        # Attempting to access index + 1 must raise IndexError.
        try:
            _ = history_view[index + 1]
        except IndexError:
            self.lookahead_blocked = True

        if index == 0 and open_position is None:
            return OrderIntent(
                symbol="BTC/USD",
                side=Side.LONG,
                qty=0.1,
                stop_loss=98_000.0,
                take_profit=103_000.0,
            )
        return None


def test_history_view_blocks_lookahead_and_executes_on_next_bar_open() -> None:
    """Verify signals on bar i never fill at bar[i].close and execute at bar[i+1].open."""
    config = ChallengeConfig.load_preset("kraken_prop_5k_starter")
    t0 = datetime(2026, 1, 2, 10, 0, tzinfo=timezone.utc)

    bars = [
        OhlcBar(
            timestamp=t0,
            open=100_000.0,
            high=100_500.0,
            low=99_500.0,
            close=100_200.0,
        ),
        OhlcBar(
            timestamp=t0 + timedelta(minutes=15),
            open=101_000.0,
            high=103_500.0,
            low=100_800.0,
            close=103_200.0,
        ),
    ]

    probe = LookaheadProbeStrategy()
    result = run_strategy_simulation(bars=bars, strategy=probe, config=config)

    assert probe.lookahead_blocked is True
    assert probe.observed_lengths == [1, 2]
    assert len(result.trades) == 1

    trade = result.trades[0]
    # Signal emitted at bar[0] (close=100,200) must fill at bar[1].open (101,000), not 100,200.
    assert trade.entry_time == bars[1].timestamp
    assert trade.entry_price == pytest.approx(101_000.0)
    assert trade.exit_price == pytest.approx(103_000.0)


def test_session_sweep_strategy_generates_valid_order_on_liquidity_sweep() -> None:
    """Verify SessionSweepStrategy triggers a LONG intent after sweeping below prior range low."""
    config = ChallengeConfig.load_preset("kraken_prop_5k_starter")
    sweep_cfg = SessionSweepConfig(
        symbol="BTC/USD",
        lookback_bars=3,
        risk_per_trade_pct=0.01,
        stop_buffer_pct=0.002,
        reward_risk_ratio=2.0,
    )
    strategy = SessionSweepStrategy(sweep_cfg)

    t0 = datetime(2026, 1, 2, 12, 0, tzinfo=timezone.utc)
    bars = [
        # 3 range bars establishing prior_low = 99,800, prior_high = 100,400
        OhlcBar(timestamp=t0, open=100_000.0, high=100_300.0, low=99_800.0, close=100_100.0),
        OhlcBar(timestamp=t0 + timedelta(minutes=15), open=100_100.0, high=100_400.0, low=99_900.0, close=100_200.0),
        OhlcBar(timestamp=t0 + timedelta(minutes=30), open=100_200.0, high=100_350.0, low=99_850.0, close=100_000.0),
        # Bar 3: sweeps below 99,800 (low=99,600) and closes back inside range (close=99,950)
        OhlcBar(timestamp=t0 + timedelta(minutes=45), open=100_000.0, high=100_050.0, low=99_600.0, close=99_950.0),
        # Bar 4: next bar opens at 99,960 and rallies through take profit
        OhlcBar(timestamp=t0 + timedelta(minutes=60), open=99_960.0, high=101_500.0, low=99_900.0, close=101_200.0),
    ]

    result = run_strategy_simulation(bars=bars, strategy=strategy, config=config)
    assert len(result.trades) == 1
    trade = result.trades[0]
    assert trade.side == Side.LONG
    assert trade.entry_time == bars[4].timestamp
    assert trade.entry_price == pytest.approx(99_960.0)
    assert trade.net_pnl > 0.0
