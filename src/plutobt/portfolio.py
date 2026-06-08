"""Futures performance-bond accounting: MTM equity + margin (spec §5.2)."""

from __future__ import annotations

from dataclasses import dataclass

from plutobt.instruments import Instrument, pnl_points


@dataclass
class Position:
    qty: int = 0  # signed; >0 long, <0 short, 0 flat
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
    side: str  # "buy" | "sell"
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

    def apply_fill(self, fill: Fill) -> None:
        """Update position (VWAP) + cash on a fill (spec §5.2).

        Commission is always deducted from cash. Opening/scaling into a
        position does NOT subtract notional (performance-bond, not a loan).
        Closing/reducing realizes pro-rata P&L into cash and books Trades.
        """
        self.cash -= fill.commission
        signed = fill.contracts if fill.side == "buy" else -fill.contracts
        cur_qty = self.position.qty

        # Opening from flat, or scaling in the same direction.
        if cur_qty == 0 or (cur_qty > 0) == (signed > 0):
            self._scale_in(fill, signed)
            return

        # Opposite direction: reduce / close / flip.
        self._reduce_or_flip(fill, signed)

    def _scale_in(self, fill: Fill, signed: int) -> None:
        cur_qty = self.position.qty
        if cur_qty == 0:
            self.position.qty = signed
            self.position.avg_price = fill.price
            self._open_entry_ts = fill.ts
            return
        new_qty = cur_qty + signed
        # VWAP on absolute sizes (sign is uniform here).
        prior_avg = self.position.avg_price
        self.position.avg_price = (prior_avg * abs(cur_qty) + fill.price * abs(signed)) / abs(
            new_qty
        )
        self.position.qty = new_qty

    def _reduce_or_flip(self, fill: Fill, signed: int) -> None:
        cur_qty = self.position.qty
        avg = self.position.avg_price
        closing = min(abs(signed), abs(cur_qty))  # contracts being closed
        side_str = "long" if cur_qty > 0 else "short"

        # Pro-rata realized P&L on the closed contracts, fill-basis.
        # Long close: (exit - entry); short close: (entry - exit). pnl_points
        # handles the sign via the position's signed qty direction.
        closed_signed = closing if cur_qty > 0 else -closing
        realized = pnl_points(avg, fill.price, closed_signed, self.instrument)
        self.cash += realized

        # §0.4 Van Tharp R: r_multiple when initial_stop is set.
        r_mult: float | None = None
        initial_stop = self.position.initial_stop
        if initial_stop is not None and avg is not None:
            risk_per_contract = abs(avg - initial_stop) * self.instrument.point_value
            if risk_per_contract != 0.0:
                r_mult = realized / (risk_per_contract * closing)

        self.trades.append(
            Trade(
                entry_ts=self._open_entry_ts,
                exit_ts=fill.ts,
                side=side_str,
                contracts=closing,
                entry_price=avg,
                exit_price=fill.price,
                pnl=realized,
                r_multiple=r_mult,
            )
        )

        remaining = cur_qty + signed  # signed residual after applying the fill
        if remaining == 0:
            self._go_flat()
            return
        if (remaining > 0) == (cur_qty > 0):
            # Partial close: same direction remains, VWAP unchanged.
            self.position.qty = remaining
            return
        # Flip: residual opens a new position in the opposite direction at fill.
        self.position.qty = remaining
        self.position.avg_price = fill.price
        self.position.initial_stop = None
        self._open_entry_ts = fill.ts

    def _go_flat(self) -> None:
        self.position = Position()
        self._open_entry_ts = None
        self._mark_price = None

    def mark(self, price: float) -> None:
        """Record the per-bar mark price for MTM (spec §5.2 settlement proxy).

        Does not move cash: unrealized MTM is reflected only through equity().
        """
        self._mark_price = float(price)

    def margin_ok(self, contracts: int) -> bool:
        """Entry gate (spec §5.2): cash >= initial_margin * contracts."""
        return self.cash >= self.instrument.initial_margin * contracts

    def margin_call_amount(self) -> float:
        """Variation-margin call on a maintenance breach (spec §5.2).

        Breach when equity < maintenance_margin * |qty|.
        On breach the call restores to INITIAL:
            initial_margin * |qty| - equity   (>= 0)
        Otherwise 0. Flat => 0. Uses the last mark() price.
        """
        qty = abs(self.position.qty)
        if qty == 0:
            return 0.0
        mark = self._mark_price
        if mark is None:
            mark = self.position.avg_price
        eq = self.equity(mark_price=mark)
        if eq >= self.instrument.maintenance_margin * qty:
            return 0.0
        return max(0.0, self.instrument.initial_margin * qty - eq)

    def mark_open_at_end(self, ts: int) -> None:
        """Book a residual open position as a flagged open-at-end Trade (§5.9).

        Marks the open position to the last mark() price and appends a
        Trade with open_at_end=True (exit_ts/exit_price=None, pnl=unrealized).
        The position is NOT closed (no cash movement). Idempotent: an
        open-at-end Trade is booked at most once. No-op when flat.
        """
        if self.position.qty == 0 or self.position.avg_price is None:
            return
        if any(t.open_at_end for t in self.trades):
            return
        mark = self._mark_price
        if mark is None:
            mark = self.position.avg_price
        unrealized = pnl_points(self.position.avg_price, mark, self.position.qty, self.instrument)

        # §0.4 Van Tharp R: r_multiple when initial_stop is set.
        r_mult: float | None = None
        initial_stop = self.position.initial_stop
        avg = self.position.avg_price
        if initial_stop is not None and avg is not None:
            risk_per_contract = abs(avg - initial_stop) * self.instrument.point_value
            contracts = abs(self.position.qty)
            if risk_per_contract != 0.0:
                r_mult = unrealized / (risk_per_contract * contracts)

        self.trades.append(
            Trade(
                entry_ts=self._open_entry_ts if self._open_entry_ts is not None else ts,
                exit_ts=None,
                side="long" if self.position.qty > 0 else "short",
                contracts=abs(self.position.qty),
                entry_price=self.position.avg_price,
                exit_price=None,
                pnl=unrealized,
                r_multiple=r_mult,
                open_at_end=True,
            )
        )
