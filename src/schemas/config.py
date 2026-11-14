from enum import Enum
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field

DEFAULT_PRESETS_PATH = Path(__file__).resolve().parents[2] / "config" / "presets.yaml"


class DrawdownType(str, Enum):
    """Drawdown calculation modes supported across Kraken Prop and Funded accounts."""

    STATIC = "STATIC"
    TRAILING_CLOSED = "TRAILING_CLOSED"


class MdlResetBasis(str, Enum):
    """Reference basis used at the daily reset to anchor the Daily Loss floor."""

    BALANCE = "BALANCE"
    EQUITY = "EQUITY"
    HIGHER_OF_BOTH = "HIGHER_OF_BOTH"


class ChallengeConfig(BaseModel):
    """Immutable configuration describing challenge rules, fees, and reset schedules."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    initial_balance: float = Field(gt=0)
    entry_fee_usd: float = Field(ge=0)
    profit_target_pct: float = Field(gt=0)
    max_drawdown_pct: float = Field(gt=0)
    drawdown_type: DrawdownType
    max_daily_loss_pct: Optional[float] = Field(default=None, gt=0)
    mdl_reset_basis: MdlResetBasis = MdlResetBasis.BALANCE
    daily_reset_utc_hour: int = Field(default=0, ge=0, le=23)
    daily_reset_utc_minute: int = Field(default=30, ge=0, le=59)
    commission_bps_per_side: float = Field(default=4.0, ge=0)
    spread_markup_bps_per_side: float = Field(default=0.0, ge=0)
    daily_swap_rate_pct: float = Field(default=0.033, ge=0)
    max_leverage: float = Field(default=5.0, gt=0)
    profit_split_pct: float = Field(default=0.80, gt=0, le=1.0)

    @classmethod
    def load_preset(
        cls, preset_name: str, path: Path = DEFAULT_PRESETS_PATH
    ) -> "ChallengeConfig":
        """Load and validate a named challenge preset from YAML."""
        with path.open("r", encoding="utf-8") as handle:
            raw_data = yaml.safe_load(handle) or {}
        presets = raw_data.get("presets", raw_data)
        if preset_name not in presets:
            raise KeyError(f"Unknown challenge preset: '{preset_name}'")
        preset_dict = dict(presets[preset_name])
        preset_dict.setdefault("name", preset_name)
        return cls.model_validate(preset_dict)
