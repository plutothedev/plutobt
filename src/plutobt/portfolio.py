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


@dataclass
class Fill:
    ts: int
    side: str                          # "buy" | "sell"
    contracts: int
    price: float
    commission: float


class Portfolio:
    """Performance-bond MTM accounting for a single instrument (spec §5.2).

    equity = cash + open MTM, where open MTM is measured on a *fill basis*
    (from the position's VWAP entry/avg price, not the prior close).
    """

    def __init__(self, starting_cash: float, instrument: Instrument) -> None:
        self.cash: float = float(starting_cash)
        self.instrument: Instrument = instrument
        self.position: Position = Position()
        self.trades: list[Trade] = []
        self._mark_price: float | None = None
        self._open_entry_ts: int | None = None

    def equity(self, mark_price: float) -> float:
        """cash + open MTM (fill-basis). Flat => cash."""
        if self.position.qty == 0 or self.position.avg_price is None:
            return self.cash
        open_mtm = pnl_points(
            self.position.avg_price,
            mark_price,
            self.position.qty,
            self.instrument,
        )
        return self.cash + open_mtm
