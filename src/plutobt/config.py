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
    allow_short: bool = True   # False: an order that would leave a net short is refused
    # What is a break, where a pending order is cancelled (unless carry_overnight):
    # "gap" (default): a gap longer than the usual bar interval, or any change of the
    #   frame's `session` column (RTH/ETH labels work here); exits are cancelled too.
    # "session": only a change of `session`; an exit survives it and fills at the next
    #   session's open. `session` must then hold a trading-session ID on every bar that
    #   never decreases (e.g. the CME trading day as an int YYYYMMDD), not RTH/ETH labels.
    break_on: str = "gap"
    # What a margin call does (spec §5.2): at a bar's close, equity below the
    # maintenance margin of the held contracts records a MarginCall (the call restores
    # to INITIAL margin; no deposit is ever invented), then:
    # "liquidate" (default): a forced market exit of the whole position fills at the
    #   next bar's open; its Trade carries margin_liquidation=True.
    # "record": the event only; the position is kept, so the breach can be studied.
    on_margin_call: str = "liquidate"


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
