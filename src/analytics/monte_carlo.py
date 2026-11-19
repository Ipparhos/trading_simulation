from collections.abc import Callable, Sequence
from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from src.engine.clock import crossed_daily_reset
from src.engine.rules_guard import RulesGuard
from src.engine.simulator import run_strategy_simulation
from src.schemas.config import ChallengeConfig
from src.schemas.orders import TradeRecord
from src.schemas.telemetry import AccountStatus, OhlcBar
from src.strategies.base import BaseStrategy


class MonteCarloReport(BaseModel):
    """Aggregated statistics from a seeded Monte Carlo trade-sequence bootstrap."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    preset_name: str
    n_simulations: int = Field(gt=0)
    seed: int
    pass_probability_pct: float
    breach_mdl_pct: float
    breach_mdd_pct: float
    truncated_pct: float
    median_trades_to_pass: Optional[float] = None
    p10_final_balance: float
    p50_final_balance: float
    p90_final_balance: float


class FrictionSensitivityPoint(BaseModel):
    """Result of evaluating a strategy at a specific slippage setting."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    slippage_bps: float
    status: AccountStatus
    final_balance: float
    net_pnl: float
    total_fees_usd: float
    trade_count: int


def _sample_trade_indices(
    rng: np.random.Generator,
    n_trades: int,
    max_trades_per_run: int,
    block_size: int,
) -> np.ndarray:
    if block_size <= 1:
        return rng.integers(0, n_trades, size=max_trades_per_run)
    n_blocks = (max_trades_per_run + block_size - 1) // block_size
    starts = rng.integers(0, n_trades, size=n_blocks)
    offsets = np.arange(block_size)
    indices = (starts[:, None] + offsets[None, :]) % n_trades
    return indices.ravel()[:max_trades_per_run]


def run_trade_bootstrap(
    trades: Sequence[TradeRecord],
    config: ChallengeConfig,
    n_simulations: int = 10_000,
    seed: int = 42,
    max_trades_per_run: int = 500,
    block_size: int = 1,
) -> MonteCarloReport:
    """Run seeded Monte Carlo bootstrap over historical TradeRecords through RulesGuard."""
    if not trades:
        raise ValueError("Cannot run Monte Carlo bootstrap on an empty trade sequence.")
    if n_simulations <= 0:
        raise ValueError("n_simulations must be positive.")
    if max_trades_per_run <= 0:
        raise ValueError("max_trades_per_run must be positive.")

    rng = np.random.default_rng(seed)
    n_trades = len(trades)

    # Infer typical time advance per trade so daily 00:30 UTC resets occur realistically.
    durations: list[timedelta] = []
    for i, trade in enumerate(trades):
        if i > 0 and trades[i].exit_time > trades[i - 1].exit_time:
            durations.append(trades[i].exit_time - trades[i - 1].exit_time)
        elif trade.exit_time > trade.entry_time:
            durations.append(trade.exit_time - trade.entry_time)
        else:
            durations.append(timedelta(hours=6))

    pass_count = 0
    mdl_count = 0
    mdd_count = 0
    truncated_count = 0
    trades_to_pass: list[int] = []
    final_balances = np.empty(n_simulations, dtype=float)

    base_start = datetime(2026, 1, 1, 1, 0, tzinfo=timezone.utc)

    for sim_idx in range(n_simulations):
        guard = RulesGuard(config)
        curr_time = base_start
        sampled_indices = _sample_trade_indices(
            rng, n_trades, max_trades_per_run, block_size
        )

        for step_num, t_idx in enumerate(sampled_indices, start=1):
            trade = trades[int(t_idx)]
            next_time = curr_time + durations[int(t_idx)]
            if crossed_daily_reset(
                curr_time,
                next_time,
                reset_hour=config.daily_reset_utc_hour,
                reset_minute=config.daily_reset_utc_minute,
            ):
                guard.on_daily_reset(current_close_equity=guard.closed_balance)
            curr_time = next_time

            projected_balance = guard.closed_balance + trade.net_pnl
            breach = guard.check_breach(projected_balance)
            if breach is not None:
                guard.record_balance_delta(trade.net_pnl, update_hwm=False)
                guard.transition_status(breach)
                break

            guard.record_balance_delta(trade.net_pnl, update_hwm=True)
            if guard.check_target_reached(guard.closed_balance):
                guard.closed_balance = min(
                    guard.closed_balance, guard.profit_target_equity
                )
                guard.transition_status(AccountStatus.PASSED)
                trades_to_pass.append(step_num)
                break

        if guard.status == AccountStatus.PASSED:
            pass_count += 1
        elif guard.status == AccountStatus.BREACHED_MDL:
            mdl_count += 1
        elif guard.status == AccountStatus.BREACHED_MDD:
            mdd_count += 1
        else:
            truncated_count += 1
        final_balances[sim_idx] = guard.closed_balance

    p10, p50, p90 = np.percentile(final_balances, [10, 50, 90])
    median_pass = float(np.median(trades_to_pass)) if trades_to_pass else None

    return MonteCarloReport(
        preset_name=config.name,
        n_simulations=n_simulations,
        seed=seed,
        pass_probability_pct=(pass_count / n_simulations) * 100.0,
        breach_mdl_pct=(mdl_count / n_simulations) * 100.0,
        breach_mdd_pct=(mdd_count / n_simulations) * 100.0,
        truncated_pct=(truncated_count / n_simulations) * 100.0,
        median_trades_to_pass=median_pass,
        p10_final_balance=float(p10),
        p50_final_balance=float(p50),
        p90_final_balance=float(p90),
    )


def run_friction_sensitivity(
    bars: Sequence[OhlcBar],
    strategy_factory: Callable[[], BaseStrategy],
    config: ChallengeConfig,
    slippage_bps_grid: Sequence[float] = (0.0, 1.0, 2.0, 4.0, 6.0, 8.0),
) -> list[FrictionSensitivityPoint]:
    """Evaluate strategy across a slippage grid to quantify execution drag."""
    points: list[FrictionSensitivityPoint] = []
    for slip_bps in slippage_bps_grid:
        strategy = strategy_factory()
        res = run_strategy_simulation(
            bars=bars, strategy=strategy, config=config, slippage_bps=float(slip_bps)
        )
        total_fees = sum(
            t.entry_fee + t.exit_fee + t.swap_fees for t in res.trades
        )
        points.append(
            FrictionSensitivityPoint(
                slippage_bps=float(slip_bps),
                status=res.status,
                final_balance=res.final_balance,
                net_pnl=res.final_balance - config.initial_balance,
                total_fees_usd=total_fees,
                trade_count=len(res.trades),
            )
        )
    return points
