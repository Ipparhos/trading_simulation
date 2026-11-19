import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

from src.analytics.ev_calculator import calculate_ev_from_config
from src.analytics.monte_carlo import run_trade_bootstrap
from src.engine.simulator import run_strategy_simulation
from src.ingestion.kraken_loader import KrakenDataLoader, df_to_bars
from src.schemas.config import ChallengeConfig
from src.strategies.session_sweep import SessionSweepConfig, SessionSweepStrategy

FIVE_K_TIERS = (
    "kraken_prop_5k_starter",
    "kraken_prop_5k_intermediate",
    "kraken_prop_5k_advanced",
)


def _parse_iso_utc(raw: str) -> datetime:
    dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _resolve_date_window(
    days: int, since_str: Optional[str], until_str: Optional[str]
) -> tuple[datetime, datetime]:
    until_utc = (
        _parse_iso_utc(until_str)
        if until_str is not None
        else datetime.now(timezone.utc)
    )
    since_utc = (
        _parse_iso_utc(since_str)
        if since_str is not None
        else until_utc - timedelta(days=days)
    )
    return since_utc, until_utc


def cmd_fetch(args: argparse.Namespace) -> int:
    since_utc, until_utc = _resolve_date_window(args.days, args.since, args.until)
    loader = KrakenDataLoader()
    df = loader.load_ohlcv(
        symbol=args.symbol,
        timeframe=args.timeframe,
        since=since_utc,
        until=until_utc,
        force_refresh=args.force_refresh,
    )
    print(
        f"Fetched {len(df)} bars for {args.symbol} ({args.timeframe}) "
        f"cached at {loader.cache_path(args.symbol, args.timeframe)}"
    )
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    preset_names = (
        list(FIVE_K_TIERS)
        if args.all_5k_tiers
        else [args.preset or "kraken_prop_5k_starter"]
    )
    since_utc, until_utc = _resolve_date_window(args.days, args.since, args.until)
    loader = KrakenDataLoader()
    df = loader.load_ohlcv(
        symbol=args.symbol,
        timeframe=args.timeframe,
        since=since_utc,
        until=until_utc,
    )
    bars = df_to_bars(df)

    reports: list[dict[str, Any]] = []
    header = (
        f"{'Preset':<28} | {'Status':<12} | {'Trades':>6} | "
        f"{'MC Pass %':>9} | {'MC MDD %':>8} | {'EV ($)':>9}"
    )
    print(header)
    print("-" * len(header))

    for name in preset_names:
        config = ChallengeConfig.load_preset(name)
        strategy = SessionSweepStrategy(
            SessionSweepConfig(symbol=args.symbol, max_leverage=config.max_leverage)
        )
        sim_result = run_strategy_simulation(bars=bars, strategy=strategy, config=config)

        if sim_result.trades:
            mc_report = run_trade_bootstrap(
                trades=sim_result.trades,
                config=config,
                n_simulations=args.mc_runs,
                seed=args.seed,
            )
            pass_prob = mc_report.pass_probability_pct / 100.0
            mdd_pct = mc_report.breach_mdd_pct
        else:
            mc_report = None
            pass_prob = 1.0 if sim_result.status.value == "PASSED" else 0.0
            mdd_pct = 100.0 if sim_result.status.value == "BREACHED_MDD" else 0.0

        total_comm = sum(t.entry_fee + t.exit_fee for t in sim_result.trades)
        total_swap = sum(t.swap_fees for t in sim_result.trades)
        ev_report = calculate_ev_from_config(
            config=config,
            pass_probability=pass_prob,
            avg_commissions_usd=total_comm,
            avg_swap_fees_usd=total_swap,
        )

        print(
            f"{name:<28} | {sim_result.status.value:<12} | {len(sim_result.trades):>6} | "
            f"{pass_prob * 100.0:>8.2f}% | {mdd_pct:>7.2f}% | "
            f"{ev_report.challenge_ev_usd:>+9.2f}"
        )

        reports.append(
            {
                "preset": name,
                "deterministic_status": sim_result.status.value,
                "final_balance": sim_result.final_balance,
                "trade_count": len(sim_result.trades),
                "monte_carlo": mc_report.model_dump() if mc_report else None,
                "ev": ev_report.model_dump(),
            }
        )

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(reports, indent=2), encoding="utf-8")

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kraken_sim",
        description="Kraken Prop & Funded Scenario Simulation Engine CLI",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    fetch_p = subparsers.add_parser("fetch", help="Fetch and cache Kraken OHLCV data")
    fetch_p.add_argument("--symbol", default="BTC/USD")
    fetch_p.add_argument("--timeframe", default="15m")
    fetch_p.add_argument("--days", type=int, default=30)
    fetch_p.add_argument("--since", default=None)
    fetch_p.add_argument("--until", default=None)
    fetch_p.add_argument("--force-refresh", action="store_true")

    run_p = subparsers.add_parser("run", help="Run strategy simulation, Monte Carlo & EV")
    run_p.add_argument("--preset", default="kraken_prop_5k_starter")
    run_p.add_argument("--all-5k-tiers", action="store_true")
    run_p.add_argument("--symbol", default="BTC/USD")
    run_p.add_argument("--timeframe", default="15m")
    run_p.add_argument("--days", type=int, default=30)
    run_p.add_argument("--since", default=None)
    run_p.add_argument("--until", default=None)
    run_p.add_argument("--mc-runs", type=int, default=2000)
    run_p.add_argument("--seed", type=int, default=42)
    run_p.add_argument("--output-json", default=None)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "fetch":
        return cmd_fetch(args)
    if args.command == "run":
        return cmd_run(args)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
