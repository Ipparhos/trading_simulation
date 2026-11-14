from datetime import datetime, timezone

import pytest

from src.engine.simulator import Simulator
from src.schemas.config import ChallengeConfig
from src.schemas.orders import LedgerEntryType, Side
from src.schemas.telemetry import AccountStatus, OhlcBar


def test_exact_4h_swap_rollover_deductions_and_fee_induced_breach() -> None:
    """Test 3A: Verify 6 exact 4h swap deductions over 24h and rollover-triggered breach."""
    starter_config = ChallengeConfig.load_preset("kraken_prop_5k_starter")
    sim = Simulator(starter_config)

    sim.open_position(
        symbol="BTC/USD",
        side=Side.LONG,
        qty=0.25,
        raw_price=100_000.0,
        entry_time=datetime(2026, 1, 1, 23, 55, tzinfo=timezone.utc),
    )

    rollover_hours = [0, 4, 8, 12, 16, 20]
    bars = [
        OhlcBar(
            timestamp=datetime(2026, 1, 2, hour, 0, tzinfo=timezone.utc),
            open=100_000.0,
            high=100_000.0,
            low=100_000.0,
            close=100_000.0,
        )
        for hour in rollover_hours
    ]
    result = sim.run(bars)

    assert result.status == AccountStatus.ACTIVE
    swap_entries = [
        entry for entry in result.ledger if entry.entry_type == LedgerEntryType.SWAP_FEE
    ]
    assert len(swap_entries) == 6
    for entry in swap_entries:
        assert entry.amount == pytest.approx(-1.375)
    assert sum(entry.amount for entry in swap_entries) == pytest.approx(-8.25)

    # Part 2: Place live equity $1.00 above active_mdd_floor ($4,850) at 03:59 UTC.
    adv_config = ChallengeConfig.load_preset("kraken_prop_5k_advanced")
    breach_sim = Simulator(adv_config)

    breach_sim.open_position(
        symbol="BTC/USD",
        side=Side.LONG,
        qty=1.0,
        raw_price=100_000.0,
        entry_time=datetime(2026, 1, 2, 3, 0, tzinfo=timezone.utc),
    )
    # Close at exact price leaving closed_balance = $4,871.00.
    loss_exit_raw = 99_911.0 / 0.9996
    breach_sim.close_position_signal(
        raw_price=loss_exit_raw,
        exit_time=datetime(2026, 1, 2, 3, 30, tzinfo=timezone.utc),
    )
    assert breach_sim.guard.closed_balance == pytest.approx(4_871.0)

    # Open $25k notional position at 03:59 UTC: $10 entry fee + $10 est. exit fee -> equity $4,851.00.
    breach_sim.open_position(
        symbol="BTC/USD",
        side=Side.LONG,
        qty=0.25,
        raw_price=100_000.0,
        entry_time=datetime(2026, 1, 2, 3, 59, tzinfo=timezone.utc),
    )

    pre_rollover_bar = OhlcBar(
        timestamp=datetime(2026, 1, 2, 3, 59, tzinfo=timezone.utc),
        open=100_000.0,
        high=100_000.0,
        low=100_000.0,
        close=100_000.0,
    )
    snap_pre = breach_sim.process_bar(pre_rollover_bar)
    assert snap_pre is not None
    assert snap_pre.close_equity == pytest.approx(4_851.0)
    assert snap_pre.status == AccountStatus.ACTIVE

    rollover_bar = OhlcBar(
        timestamp=datetime(2026, 1, 2, 4, 0, tzinfo=timezone.utc),
        open=100_000.0,
        high=100_000.0,
        low=100_000.0,
        close=100_000.0,
    )
    snap_post = breach_sim.process_bar(rollover_bar)
    assert snap_post is not None
    assert snap_post.status == AccountStatus.BREACHED_MDD
    assert snap_post.worst_intrabar_equity == pytest.approx(4_849.625)
