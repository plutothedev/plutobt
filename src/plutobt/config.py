"""Backtest configuration value type (the single home of BacktestConfig)."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, fields


@dataclass
class BacktestConfig:
    commission_per_side: float = 0.0
    slippage_ticks: float | None = None   # None => use instrument default
    risk_free: float = 0.0
    periods_per_year: float | None = None  # None => derived from data/session
    carry_overnight: bool = False
    allow_short: bool = True
    break_on: str = "gap"   # "gap" (default): a gap longer than the usual bar interval
                            # is a break; "session": only a change of the frame's
                            # `session` column is


def load_config(path: str) -> BacktestConfig:
    """Load a BacktestConfig from a TOML file (stdlib tomllib).

    Omitted keys fall back to the BacktestConfig dataclass defaults.
    Unknown keys raise ValueError so a typo never silently no-ops.
    """
    with open(path, "rb") as fh:
        raw = tomllib.load(fh)
    allowed = {f.name for f in fields(BacktestConfig)}
    for key in raw:
        if key not in allowed:
            raise ValueError(f"unknown config key: {key}")
    return BacktestConfig(**raw)
