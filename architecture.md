# ARCHITECTURE.md — Kraken Prop & Funded Scenario Simulation Engine

## 1. System Objective & Core Philosophy
You are tasked with building a high-precision, event-driven quantitative simulation and Monte Carlo engine in Python (`3.12+`) designed to stress-test trading strategies against **Kraken Prop (Kraken Pro)** and **Kraken Funded** evaluation rules.

### Non-Negotiable Engineering Constraints (Do Not Violate)
1. **Zero Lookahead Bias:** Strategies may only access closed historical bars up to index `i`. Orders generated at the close of bar `i` execute at the open of bar `i+1` (market) or intrabar during `i+1..n` (limit/stop).
2. **Pessimistic Intrabar Wick Resolution:** Never evaluate Drawdown or Stop-Loss/Take-Profit solely on candle closes. During any open position on bar `i`, the engine MUST evaluate the bar's `Low` (for Longs) and `High` (for Shorts) against both the position Stop-Loss and the account's real-time Equity Breach Floor. If a candle's `High` and `Low` touch both the Stop-Loss and Take-Profit within the same 1m/5m bar, **the engine must assume the Stop-Loss was hit first (Pessimistic Execution).**
3. **Exact Fee & Rollover State Machine:** Fees reduce account `balance` immediately and count directly against both Daily Loss and Maximum Drawdown limits.
4. **Vectorized Signal Prep, Sequential State Machine:** Indicators may be pre-computed vectorially, but account equity, 00:30 UTC daily resets, 4-hour swap rollovers, and breach checks MUST run sequentially bar-by-bar.

---

## 2. Directory & Module Structure

```text
kraken_sim/
├── pyproject.toml
├── config/
│   └── presets.yaml               # Exact Kraken Prop & Kraken Funded tier definitions
├── data/
│   └── parquet/                   # Local cache for CCXT Kraken OHLCV data
├── src/
│   ├── __init__.py
│   ├── schemas/
│   │   ├── config.py              # Pydantic v2 models for Challenge Rules & Fees
│   │   ├── orders.py              # Order, Position, Fill, and LedgerEntry models
│   │   └── telemetry.py           # SimulationResult, EquityPoint, BreachReport
│   ├── ingestion/
│   │   └── kraken_loader.py       # CCXT historical fetcher + Parquet cache + gap validator
│   ├── engine/
│   │   ├── clock.py               # 00:30 UTC Daily Reset & 4h Swap Rollover triggers
│   │   ├── fee_calculator.py      # Commission (bps), spread markup, & swap fee logic
│   │   ├── execution.py           # Intrabar OHLC fill simulator with slippage & pessimistic wicks
│   │   ├── rules_guard.py         # Real-time Static MDD, Trailing MDD, and MDL breach detector
│   │   └── simulator.py           # Main event loop coordinating Data -> Strategy -> Execution -> Guard
│   ├── strategies/
│   │   ├── base.py                # Abstract Strategy interface
│   │   └── session_sweep.py       # Reference strategy (PDH/PDL liquidity sweep + volatility filter)
│   └── analytics/
│       ├── monte_carlo.py         # Trade-sequence bootstrap & noisy-entry resampler
│       └── ev_calculator.py       # Expected Value ($ EV) per challenge attempt & ruin probability
└── tests/
    ├── test_intrabar_breach.py    # Proves wicks trigger MDL/MDD before TP
    ├── test_utc_0030_reset.py     # Proves open PnL pullback after 00:30 UTC triggers breach
    └── test_swap_rollover.py      # Proves 0.0055% fee deduction every 4 hours