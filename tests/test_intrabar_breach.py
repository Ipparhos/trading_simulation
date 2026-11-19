from datetime import datetime, timedelta, timezone

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
    """All engine clock and schema entry points must reject naive and non-UTC datetimes."""
    naive = datetime(2026, 1, 2, 12, 0)
    non_utc = datetime(2026, 1, 2, 14, 0, tzinfo=timezone(timedelta(hours=2)))
    utc_time = datetime(2026, 1, 2, 12, 0, tzinfo=timezone.utc)

    for invalid_dt in (naive, non_utc):
        with pytest.raises(ValueError):
            OhlcBar(timestamp=invalid_dt, open=100.0, high=101.0, low=99.0, close=100.0)

        with pytest.raises(ValueError):
            Position(
                symbol="BTC/USD",
                side=Side.LONG,
                qty=1.0,
                entry_price=100.0,
                entry_time=invalid_dt,
            )

        with pytest.raises(ValueError):
            check_4h_rollovers(invalid_dt, utc_time)

        with pytest.raises(ValueError):
            crossed_daily_reset(utc_time, invalid_dt)

        clock = EngineClock()
        with pytest.raises(ValueError):
            clock.advance(invalid_dt)


def test_target_auto_liquidate_requires_equity_after_exit_fees() -> None:
    """Target auto-liquidation only triggers when live_equity after exit fees reaches target."""
    config = ChallengeConfig.load_preset("kraken_prop_5k_advanced")
    sim = Simulator(config)

    # Target is $5,450 (+9%). Open 0.5 BTC LONG @ $100,000 ($20 entry fee -> balance $4,980).
    sim.open_position(
        symbol="BTC/USD",
        side=Side.LONG,
        qty=0.5,
        raw_price=100_000.0,
        entry_time=datetime(2026, 1, 2, 13, 0, tzinfo=timezone.utc),
    )

    # At high=$100,940: gross PnL = +$470 -> balance+PnL = $5,450, but est. exit fee ~$20.19
    # leaves live_equity ~$5,429.81 (< $5,450), so account must stay ACTIVE.
    near_miss_bar = OhlcBar(
        timestamp=datetime(2026, 1, 2, 13, 5, tzinfo=timezone.utc),
        open=100_000.0,
        high=100_940.0,
        low=100_000.0,
        close=100_900.0,
    )
    snap_near_miss = sim.process_bar(near_miss_bar)
    assert snap_near_miss is not None
    assert snap_near_miss.status == AccountStatus.ACTIVE
    assert sim.position is not None

    # Next bar reaches high=$101,100 where net equity after exit fees exceeds $5,450.
    pass_bar = OhlcBar(
        timestamp=datetime(2026, 1, 2, 13, 10, tzinfo=timezone.utc),
        open=100_900.0,
        high=101_100.0,
        low=100_900.0,
        close=101_050.0,
    )
    snap_pass = sim.process_bar(pass_bar)
    assert snap_pass is not None
    assert snap_pass.status == AccountStatus.PASSED
    assert sim.position is None

    result = sim.get_result()
    assert result.status == AccountStatus.PASSED
    assert len(result.trades) == 1
    assert result.trades[0].exit_reason == ExitReason.TARGET_AUTO_LIQUIDATE
    assert result.final_balance == pytest.approx(5_450.0)
    assert result.final_equity == pytest.approx(result.final_balance)
