from src.schemas.orders import Side

BPS_DIVISOR = 10_000.0
SWAP_INTERVALS_PER_DAY = 6.0


def effective_fill_price(
    raw_price: float,
    side: Side,
    is_entry: bool,
    spread_markup_bps_per_side: float,
    slippage_bps: float = 0.0,
) -> float:
    """Compute effective execution price after spread markup and slippage."""
    total_adjustment = (spread_markup_bps_per_side + slippage_bps) / BPS_DIVISOR
    is_buying = (side == Side.LONG and is_entry) or (
        side == Side.SHORT and not is_entry
    )
    if is_buying:
        return raw_price * (1.0 + total_adjustment)
    return raw_price * (1.0 - total_adjustment)


def compute_commission(notional: float, commission_bps_per_side: float) -> float:
    """Compute trading commission in USD for a given fill notional."""
    return notional * (commission_bps_per_side / BPS_DIVISOR)


def compute_4h_swap(notional: float, daily_swap_rate_pct: float) -> float:
    """Compute a single 4-hour rollover swap fee (one-sixth of the daily rate)."""
    return notional * (daily_swap_rate_pct / 100.0 / SWAP_INTERVALS_PER_DAY)
