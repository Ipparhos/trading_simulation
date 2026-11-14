from datetime import datetime, timezone

import pytest

from src.engine.simulator import Simulator
from src.schemas.config import ChallengeConfig, MdlResetBasis
from src.schemas.orders import Side
from src.schemas.telemetry import AccountStatus, OhlcBar


def test_closed_balance_reset_updates_mdl_while_static_mdd_unchanged() -> None:
    """Test 2A: 00:30 UTC updates MDL floor from closed balance; static MDD stays fixed."""
    config = ChallengeConfig.load_preset("kraken_prop_5k_starter")
    sim = Simulator(config)

    t_entry = datetime(2026, 1, 2, 0, 10, tzinfo=timezone.utc)
    sim.open_position(
        symbol="BTC/USD",
        side=Side.LONG,
        qty=0.1,
        raw_price=100_000.0,
        entry_time=t_entry,
    )
    # Solve raw exit price for exact +$200 net profit after 4 bps/side entry & exit commissions.
    exact_exit_raw = 10_204.0 / 0.09996
    t_exit = datetime(2026, 1, 2, 0, 20, tzinfo=timezone.utc)
    sim.close_position_signal(raw_price=exact_exit_raw, exit_time=t_exit)

    assert sim.guard.closed_balance == pytest.approx(5_200.0)

    bar_after_reset = OhlcBar(
        timestamp=datetime(2026, 1, 2, 0, 35, tzinfo=timezone.utc),
        open=102_000.0,
        high=102_000.0,
        low=102_000.0,
        close=102_000.0,
    )
    snapshot = sim.process_bar(bar_after_reset)
    assert snapshot is not None
    assert snapshot.active_mdl_floor == pytest.approx(5_044.0)
    assert snapshot.active_mdd_floor == pytest.approx(4_700.0)
    assert snapshot.status == AccountStatus.ACTIVE


def test_open_winning_trade_pullback_trap_under_equity_basis() -> None:
    """Test 2B: Under EQUITY basis, open profits at 00:30 UTC lock in a higher MDL floor."""
    base_config = ChallengeConfig.load_preset("kraken_prop_5k_starter")
    config = base_config.model_copy(
        update={
            "mdl_reset_basis": MdlResetBasis.EQUITY,
            "commission_bps_per_side": 0.0,
        }
    )
    sim = Simulator(config)

    sim.open_position(
        symbol="BTC/USD",
        side=Side.LONG,
        qty=0.1,
        raw_price=100_000.0,
        entry_time=datetime(2026, 1, 2, 0, 15, tzinfo=timezone.utc),
    )

    reset_bar = OhlcBar(
        timestamp=datetime(2026, 1, 2, 0, 30, tzinfo=timezone.utc),
        open=103_000.0,
        high=103_000.0,
        low=103_000.0,
        close=103_000.0,
    )
    snap_reset = sim.process_bar(reset_bar)
    assert snap_reset is not None
    assert snap_reset.closed_balance == pytest.approx(5_000.0)
    assert snap_reset.active_mdl_floor == pytest.approx(5_141.0)
    assert snap_reset.status == AccountStatus.ACTIVE

    pullback_bar = OhlcBar(
        timestamp=datetime(2026, 1, 2, 0, 45, tzinfo=timezone.utc),
        open=103_000.0,
        high=103_000.0,
        low=101_350.0,
        close=101_350.0,
    )
    snap_pullback = sim.process_bar(pullback_bar)
    assert snap_pullback is not None
    assert snap_pullback.worst_intrabar_equity == pytest.approx(5_135.0)
    assert snap_pullback.status == AccountStatus.BREACHED_MDL


def test_kraken_prop_lite_trailing_closed_mdd_updates_only_at_0030_utc() -> None:
    """Test 2C: TRAILING_CLOSED MDD floor stays fixed intraday and updates at 00:30 UTC."""
    config = ChallengeConfig.load_preset("kraken_prop_50k_lite")
    sim = Simulator(config)

    assert sim.guard.active_mdd_floor == pytest.approx(48_500.0)

    sim.open_position(
        symbol="BTC/USD",
        side=Side.LONG,
        qty=1.0,
        raw_price=100_000.0,
        entry_time=datetime(2026, 1, 2, 0, 5, tzinfo=timezone.utc),
    )
    # Solve raw exit price for exact +$1,000 net profit after 4 bps/side commissions.
    exact_exit_raw = 101_040.0 / 0.9996
    sim.close_position_signal(
        raw_price=exact_exit_raw,
        exit_time=datetime(2026, 1, 2, 0, 15, tzinfo=timezone.utc),
    )
    assert sim.guard.closed_balance == pytest.approx(51_000.0)

    intraday_bar = OhlcBar(
        timestamp=datetime(2026, 1, 2, 0, 25, tzinfo=timezone.utc),
        open=101_000.0,
        high=101_000.0,
        low=101_000.0,
        close=101_000.0,
    )
    snap_intraday = sim.process_bar(intraday_bar)
    assert snap_intraday is not None
    assert snap_intraday.active_mdd_floor == pytest.approx(48_500.0)

    post_reset_bar = OhlcBar(
        timestamp=datetime(2026, 1, 2, 0, 30, tzinfo=timezone.utc),
        open=101_000.0,
        high=101_000.0,
        low=101_000.0,
        close=101_000.0,
    )
    snap_reset = sim.process_bar(post_reset_bar)
    assert snap_reset is not None
    assert snap_reset.active_mdd_floor == pytest.approx(49_500.0)
