from datetime import datetime, timedelta, timezone

import pytest

from src.analytics.ev_calculator import (
    EvInputs,
    calculate_challenge_ev,
    calculate_ev_from_config,
)
from src.analytics.monte_carlo import (
    run_friction_sensitivity,
    run_trade_bootstrap,
)
from src.schemas.config import ChallengeConfig
from src.schemas.orders import ExitReason, Side, TradeRecord
from src.schemas.telemetry import AccountStatus, OhlcBar
from src.strategies.session_sweep import SessionSweepConfig, SessionSweepStrategy


def _make_trade(
    idx: int,
    net_pnl: float,
    base_time: datetime,
    hours_apart: int = 8,
) -> TradeRecord:
    entry_t = base_time + timedelta(hours=idx * hours_apart)
    exit_t = entry_t + timedelta(hours=2)
    return TradeRecord(
        symbol="BTC/USD",
        side=Side.LONG,
        qty=0.1,
        entry_price=100_000.0,
        exit_price=100_000.0 + net_pnl * 10.0,
        entry_time=entry_t,
        exit_time=exit_t,
        exit_reason=ExitReason.TAKE_PROFIT if net_pnl >= 0 else ExitReason.STOP_LOSS,
        gross_pnl=net_pnl,
        entry_fee=0.0,
        exit_fee=0.0,
        swap_fees=0.0,
        net_pnl=net_pnl,
    )


def test_monte_carlo_bootstrap_is_deterministic_and_enforces_rules() -> None:
    """Verify seeded bootstrap produces identical reports and respects target/breach limits."""
    config = ChallengeConfig.load_preset("kraken_prop_5k_starter")
    t0 = datetime(2026, 1, 2, 1, 0, tzinfo=timezone.utc)

    trades = [
        _make_trade(0, 120.0, t0),
        _make_trade(1, -60.0, t0),
        _make_trade(2, 140.0, t0),
        _make_trade(3, -80.0, t0),
        _make_trade(4, 110.0, t0),
    ]

    report_a = run_trade_bootstrap(
        trades=trades, config=config, n_simulations=500, seed=42, max_trades_per_run=50
    )
    report_b = run_trade_bootstrap(
        trades=trades, config=config, n_simulations=500, seed=42, max_trades_per_run=50
    )

    assert report_a == report_b
    total_pct = (
        report_a.pass_probability_pct
        + report_a.breach_mdl_pct
        + report_a.breach_mdd_pct
        + report_a.truncated_pct
    )
    assert total_pct == pytest.approx(100.0)
    assert report_a.pass_probability_pct > 50.0
    assert report_a.median_trades_to_pass is not None


def test_monte_carlo_trailing_closed_mdd_across_daily_reset() -> None:
    """Verify 50k Lite TRAILING_CLOSED MDD ratchets up after 00:30 UTC in Monte Carlo."""
    config = ChallengeConfig.load_preset("kraken_prop_50k_lite")
    t0 = datetime(2026, 1, 2, 1, 0, tzinfo=timezone.utc)

    # Single trade losing -$800 every 12h inevitably breaches the $1,500 MDD floor.
    losing_trades = [_make_trade(0, -800.0, t0, hours_apart=12)]
    report = run_trade_bootstrap(
        trades=losing_trades,
        config=config,
        n_simulations=50,
        seed=42,
        max_trades_per_run=10,
    )
    assert report.breach_mdd_pct == pytest.approx(100.0)
    assert report.pass_probability_pct == pytest.approx(0.0)


def test_ev_calculator_and_friction_sensitivity() -> None:
    """Verify ChallengeEvReport formula and friction sensitivity degradation across slippage."""
    inputs = EvInputs(
        entry_fee_usd=45.0,
        profit_target_usd=500.0,
        profit_split_pct=0.80,
        pass_probability=0.25,
        p_first_payout_given_pass=0.70,
        avg_commissions_usd=18.0,
        avg_swap_fees_usd=4.5,
    )
    report = calculate_challenge_ev(inputs)

    # Trader payout = 500 * 0.80 = $400. Effective prob = 0.25 * 0.70 = 0.175.
    # EV = 0.175 * 400 - 45 = 70 - 45 = +$25.00.
    assert report.expected_trader_payout_usd == pytest.approx(400.0)
    assert report.effective_payout_probability == pytest.approx(0.175)
    assert report.challenge_ev_usd == pytest.approx(25.0)
    assert report.breakeven_pass_probability == pytest.approx(45.0 / (0.70 * 400.0))
    assert report.total_fee_drag_usd == pytest.approx(22.5)

    config = ChallengeConfig.load_preset("kraken_prop_5k_starter")
    cfg_report = calculate_ev_from_config(config, pass_probability=0.25)
    assert cfg_report.challenge_ev_usd == pytest.approx(25.0)

    # Friction sensitivity test: higher slippage reduces final balance on executed trades.
    t0 = datetime(2026, 1, 2, 12, 0, tzinfo=timezone.utc)
    bars = [
        OhlcBar(timestamp=t0, open=100_000.0, high=100_300.0, low=99_800.0, close=100_100.0),
        OhlcBar(timestamp=t0 + timedelta(minutes=15), open=100_100.0, high=100_400.0, low=99_900.0, close=100_200.0),
        OhlcBar(timestamp=t0 + timedelta(minutes=30), open=100_200.0, high=100_350.0, low=99_850.0, close=100_000.0),
        OhlcBar(timestamp=t0 + timedelta(minutes=45), open=100_000.0, high=100_050.0, low=99_600.0, close=99_950.0),
        OhlcBar(timestamp=t0 + timedelta(minutes=60), open=99_960.0, high=101_500.0, low=99_900.0, close=101_200.0),
    ]
    points = run_friction_sensitivity(
        bars=bars,
        strategy_factory=lambda: SessionSweepStrategy(
            SessionSweepConfig(lookback_bars=3, risk_per_trade_pct=0.01)
        ),
        config=config,
        slippage_bps_grid=(0.0, 4.0, 8.0),
    )
    assert len(points) == 3
    assert points[0].trade_count == 1
    assert points[0].final_balance > points[1].final_balance > points[2].final_balance
    assert points[0].status == AccountStatus.ACTIVE
