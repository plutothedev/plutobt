"""Verified futures contract registry and price/tick math (spec §5.3, §10.1)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Instrument:
    symbol: str
    point_value: float
    tick_size: float
    tick_value: float
    currency: str
    session_open_ct: str        # "08:30"
    session_close_ct: str       # "15:15"
    slippage_ticks: float
    initial_margin: float
    maintenance_margin: float
    settlement: str             # "cash" | "physical"
    price_format: str           # "decimal" | "thirty_seconds"
