from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

from src.schemas.config import ChallengeConfig


class EvInputs(BaseModel):
    """Inputs required to compute the Expected Value ($ EV) of a prop challenge attempt."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    entry_fee_usd: float = Field(ge=0)
    profit_target_usd: float = Field(gt=0)
    profit_split_pct: float = Field(gt=0, le=1.0)
    pass_probability: float = Field(ge=0.0, le=1.0)
    p_first_payout_given_pass: float = Field(default=0.70, gt=0.0, le=1.0)
    avg_gross_payout_usd: Optional[float] = Field(default=None, gt=0)
    avg_commissions_usd: float = Field(default=0.0, ge=0)
    avg_swap_fees_usd: float = Field(default=0.0, ge=0)


class ChallengeEvReport(BaseModel):
    """Expected Value ($ EV) and breakeven pass-rate summary for a challenge tier."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    expected_gross_payout_usd: float
    expected_trader_payout_usd: float
    effective_payout_probability: float
    challenge_ev_usd: float
    breakeven_pass_probability: float
    total_fee_drag_usd: float


def calculate_challenge_ev(inputs: EvInputs) -> ChallengeEvReport:
    """Compute challenge Expected Value ($ EV), breakeven pass probability, and fee drag."""
    gross_payout = (
        inputs.avg_gross_payout_usd
        if inputs.avg_gross_payout_usd is not None
        else inputs.profit_target_usd
    )
    trader_payout = gross_payout * inputs.profit_split_pct
    effective_prob = inputs.pass_probability * inputs.p_first_payout_given_pass
    ev_usd = (effective_prob * trader_payout) - inputs.entry_fee_usd

    denom = inputs.p_first_payout_given_pass * trader_payout
    breakeven_pass_prob = inputs.entry_fee_usd / denom if denom > 0 else 0.0
    total_fee_drag = inputs.avg_commissions_usd + inputs.avg_swap_fees_usd

    return ChallengeEvReport(
        expected_gross_payout_usd=gross_payout,
        expected_trader_payout_usd=trader_payout,
        effective_payout_probability=effective_prob,
        challenge_ev_usd=ev_usd,
        breakeven_pass_probability=breakeven_pass_prob,
        total_fee_drag_usd=total_fee_drag,
    )


def calculate_ev_from_config(
    config: ChallengeConfig,
    pass_probability: float,
    p_first_payout_given_pass: float = 0.70,
    avg_gross_payout_usd: Optional[float] = None,
    avg_commissions_usd: float = 0.0,
    avg_swap_fees_usd: float = 0.0,
) -> ChallengeEvReport:
    """Convenience wrapper building EvInputs directly from a ChallengeConfig."""
    profit_target_usd = config.initial_balance * config.profit_target_pct
    return calculate_challenge_ev(
        EvInputs(
            entry_fee_usd=config.entry_fee_usd,
            profit_target_usd=profit_target_usd,
            profit_split_pct=config.profit_split_pct,
            pass_probability=pass_probability,
            p_first_payout_given_pass=p_first_payout_given_pass,
            avg_gross_payout_usd=avg_gross_payout_usd,
            avg_commissions_usd=avg_commissions_usd,
            avg_swap_fees_usd=avg_swap_fees_usd,
        )
    )
