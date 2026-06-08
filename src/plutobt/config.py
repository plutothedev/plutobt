"""Backtest configuration value type (the single home of BacktestConfig)."""

from __future__ import annotations

import tomllib  # noqa: F401  (used by M9 TOML loading)
from dataclasses import dataclass  # noqa: F401  (fields re-exported)


@dataclass
class BacktestConfig:
    commission_per_side: float = 0.0
    slippage_ticks: float | None = None   # None => use instrument default
    risk_free: float = 0.0
    periods_per_year: float | None = None  # None => derived from data/session
    carry_overnight: bool = False
    allow_short: bool = True
