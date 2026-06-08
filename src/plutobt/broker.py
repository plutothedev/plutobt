"""Order execution: directional slippage, gap-through stops, lifecycle (spec 5.4, 5.4a)."""

from __future__ import annotations

from dataclasses import dataclass

from plutobt.config import BacktestConfig
from plutobt.instruments import Instrument, ticks_to_price
from plutobt.portfolio import Fill, Portfolio


@dataclass
class Order:
    side: str                    # "buy" | "sell"
    contracts: int
    kind: str                    # "market" | "stop" | "limit"
    price: float | None = None   # trigger for stop/limit
    attached_stop: float | None = None
    status: str = "pending"      # "pending"|"filled"|"cancelled"|"rejected"
