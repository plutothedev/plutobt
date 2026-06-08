"""Futures performance-bond accounting: MTM equity + margin (spec §5.2)."""

from __future__ import annotations

from dataclasses import dataclass

from plutobt.instruments import Instrument, pnl_points


@dataclass
class Position:
    qty: int = 0                       # signed; >0 long, <0 short, 0 flat
    avg_price: float | None = None
    initial_stop: float | None = None


@dataclass
class Trade:
    entry_ts: int
    exit_ts: int | None
    side: str
    contracts: int
    entry_price: float
    exit_price: float | None
    pnl: float
    r_multiple: float | None = None
    open_at_end: bool = False
