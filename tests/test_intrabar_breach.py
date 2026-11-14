from datetime import datetime, timezone

import pytest

from src.engine.clock import EngineClock, check_4h_rollovers, crossed_daily_reset
from src.engine.simulator import Simulator
from src.schemas.config import ChallengeConfig
from src.schemas.orders import ExitReason, Position, Side
from src.schemas.telemetry import AccountStatus, OhlcBar


def test_pessimistic_conflict_sl_vs_tp_breaches_before_tp() -> None:
    """Test 1A: On a single bar touching both SL and TP, adverse SL path evaluates first."""
    config = ChallengeConfig.load_preset("kraken_prop_5k_advanced")
    sim = Simulator(config)

    t_entry = datetime(2026, 1, 2, 10, 0, tzinfo=timezone.utc)
    sim.open_position(
        symbol="BTC/USD",
        side=Side.LONG,
        qty=1.0,
        raw_price=100_000.0,
        entry_time=t_entry,
        stop_loss=99_000.0,
        take_profit=102_000.0,
    )

    bar = OhlcBar(
        timestamp=datetime(2026, 1, 2, 10, 5, tzinfo=timezone.utc),
        open=100_000.0,
        high=102_500.0,
        low=98_500.0,
        close=102_000.0,
    )
    result = sim.run([bar])

    assert result.status == AccountStatus.BREACHED_MDD
    assert len(result.trades) == 1
    trade = result.trades[0]
    assert trade.exit_reason == ExitReason.BREACH_LIQUIDATE
    assert trade.exit_price == pytest.approx(99_000.0)
    assert trade.net_pnl < 0.0
    assert result.final_balance == pytest.approx(3_920.40)


def test_pessimistic_conflict_sl_fills_without_evaluating_tp() -> None:
    """When SL and TP are both inside one bar without breaching, SL executes and TP is skipped."""
    config = ChallengeConfig.load_preset("kraken_prop_5k_advanced")
    sim = Simulator(config)

    t_entry = datetime(2026, 1, 2, 10, 0, tzinfo=timezone.utc)
    sim.open_position(
        symbol="BTC/USD",
        side=Side.LONG,
        qty=0.05,
        raw_price=100_000.0,
        entry_time=t_entry,
        stop_loss=99_000.0,
        take_profit=102_000.0,
    )

    bar = OhlcBar(
        timestamp=datetime(2026, 1, 2, 10, 5, tzinfo=timezone.utc),
        open=100_000.0,
        high=102_500.0,
        low=98_500.0,
        close=102_000.0,
    )
    result = sim.run([bar])

    assert result.status == AccountStatus.ACTIVE
    assert len(result.trades) == 1
    trade = result.trades[0]
    assert trade.exit_reason == ExitReason.STOP_LOSS
    assert trade.exit_price == pytest.approx(99_000.0)
    assert trade.net_pnl < 0.0
    assert result.final_balance == pytest.approx(4_946.02)


def test_wick_breach_without_stop_loss_even_on_green_close() -> None:
    """Test 1B: Intrabar low wick breaches MDD floor even when candle closes green."""
    config = ChallengeConfig.load_preset("kraken_prop_5k_advanced")
    sim = Simulator(config)

    t_entry = datetime(2026, 1, 2, 11, 0, tzinfo=timezone.utc)
    sim.open_position(
        symbol="BTC/USD",
        side=Side.LONG,
        qty=0.5,
        raw_price=100_000.0,
        entry_time=t_entry,
        stop_loss=None,
        take_profit=None,
    )

    bar = OhlcBar(
        timestamp=datetime(2026, 1, 2, 11, 5, tzinfo=timezone.utc),
        open=100_000.0,
        high=100_500.0,
        low=99_700.0,
        close=100_400.0,
    )
    result = sim.run([bar])

    assert result.status == AccountStatus.BREACHED_MDD
    assert len(result.snapshots) == 1
    assert result.snapshots[0].worst_intrabar_equity == pytest.approx(4_810.06)
    assert result.final_balance == pytest.approx(4_810.06)


def test_naive_timestamps_raise_value_error() -> None:
    """All engine clock and schema entry points must reject naive datetimes."""
    naive = datetime(2026, 1, 2, 12, 0)
    utc_time = datetime(2026, 1, 2, 12, 0, tzinfo=timezone.utc)

    with pytest.raises(ValueError):
        OhlcBar(timestamp=naive, open=100.0, high=101.0, low=99.0, close=100.0)

    with pytest.raises(ValueError):
        Position(
            symbol="BTC/USD",
            side=Side.LONG,
            qty=1.0,
            entry_price=100.0,
            entry_time=naive,
        )

    with pytest.raises(ValueError):
        check_4h_rollovers(naive, utc_time)

    with pytest.raises(ValueError):
        crossed_daily_reset(utc_time, naive)

    clock = EngineClock()
    with pytest.raises(ValueError):
        clock.advance(naive)
