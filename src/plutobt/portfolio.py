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
    """One closed slice of a position, or the position still open at the end.

    ``pnl`` is GROSS price P&L on the fill basis, with no fees in it. ``commission`` is
    the fees attributable to this trade's contracts, and ``net_pnl`` is ``pnl`` minus
    them: every trade statistic reads ``net_pnl``. ``r_multiple`` is net of those fees,
    measured against a price-risk 1R (no fees in the risk), so a stop-out exactly at the
    stop reads below -1R.
    """

    entry_ts: int
    exit_ts: int | None
    side: str
    contracts: int
    entry_price: float
    exit_price: float | None
    pnl: float
    r_multiple: float | None = None
    open_at_end: bool = False
    # Fees attributable to this trade's contracts: their pro-rata share of the
    # entry fees still pooled on the position, plus the closing fill's share.
    # pnl stays PRICE P&L so every existing reader keeps its meaning.
    commission: float = 0.0
    # True when the engine closed these contracts on a margin call
    # (on_margin_call="liquidate"), not the strategy or its stop.
    margin_liquidation: bool = False

    @property
    def net_pnl(self) -> float:
        """P&L after the fees attributable to this trade."""
        return self.pnl - self.commission


@dataclass
class Fill:
    ts: int
    side: str  # "buy" | "sell"
    contracts: int
    price: float
    commission: float
    margin_liquidation: bool = False   # the engine's forced exit on a margin call


@dataclass
class MarginCall:
    """A maintenance breach at a bar's close (spec §5.2).

    ``equity`` is account equity (cash + open MTM) at the close; ``maintenance`` and
    ``initial`` are the requirements for the ``contracts`` held. The call restores the
    account to INITIAL, so ``amount = initial - equity``, never ``maintenance - equity``.
    """

    ts: int
    contracts: int
    equity: float
    maintenance: float
    initial: float
    amount: float


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
        self._open_commission: float = 0.0

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
        A fill of fewer than 1 contract raises ValueError before anything moves.
        """
        if fill.contracts < 1:
            # Last line of defence (the broker refuses these first): 0 contracts would
            # split the fee over zero contracts, and a negative count reverses the side.
            raise ValueError(f"a fill needs 1 or more contracts, got {fill.contracts!r}")
        self.cash -= fill.commission
        signed = fill.contracts if fill.side == "buy" else -fill.contracts
        cur_qty = self.position.qty

        # Opening from flat, or scaling in the same direction.
        if cur_qty == 0 or (cur_qty > 0) == (signed > 0):
            # Entry fees pool on the open position until contracts close.
            self._open_commission += fill.commission
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
        # Fees for the closed contracts: their share of the pooled entry fees,
        # plus this fill's share for them (the rest of a flip fill opens the new
        # position and becomes ITS entry fee, at the end of this method).
        entry_share = self._open_commission * closing / abs(cur_qty)
        exit_share = fill.commission * closing / fill.contracts
        self._open_commission -= entry_share
        trade_commission = entry_share + exit_share
        side_str = "long" if cur_qty > 0 else "short"

        # Pro-rata realized P&L on the closed contracts, fill-basis.
        # Long close: (exit - entry); short close: (entry - exit). pnl_points
        # handles the sign via the position's signed qty direction.
        closed_signed = closing if cur_qty > 0 else -closing
        realized = pnl_points(avg, fill.price, closed_signed, self.instrument)
        self.cash += realized

        # §0.4 Van Tharp R: r_multiple when initial_stop is set.
        r_mult = self._calc_r_multiple(realized - trade_commission, closing)

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
                commission=trade_commission,
                margin_liquidation=fill.margin_liquidation,
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
        self._open_commission = fill.commission * abs(remaining) / fill.contracts

    def _calc_r_multiple(self, pnl: float, contracts: int) -> float | None:
        """Van Tharp R = pnl / (risk_per_contract * contracts).

        ``pnl`` is NET of the fees attributable to ``contracts`` (the pooled entry
        share plus the closing fill's share; at open-at-end: entry fees only). Risk is
        price risk only, with no fees in it, so a stop-out exactly at the stop reads
        below -1R by fees/risk.

        None when there is no attached stop or the per-contract risk is zero
        (§0.4). ``contracts`` is the unsigned number of contracts the ``pnl``
        is measured over.

        Convention: after a scale-in, risk is |blended avg_price - the FIRST entry's
        initial_stop|; a scale-in's own attached stop moves the resting stop, not R.
        """
        avg = self.position.avg_price
        initial_stop = self.position.initial_stop
        if initial_stop is None or avg is None:
            return None
        risk_per_contract = abs(avg - initial_stop) * self.instrument.point_value
        if risk_per_contract == 0.0:
            return None
        return pnl / (risk_per_contract * contracts)

    def _go_flat(self) -> None:
        self.position = Position()
        self._open_entry_ts = None
        self._mark_price = None
        # exact reset: x * n / n can leave a ~1e-17 residue after a full close
        self._open_commission = 0.0

    def mark(self, price: float) -> None:
        """Record the per-bar mark price for MTM (spec §5.2 settlement proxy).

        Does not move cash: unrealized MTM is reflected only through equity().
        """
        self._mark_price = float(price)

    def margin_ok(self, contracts: int, mark_price: float | None = None) -> bool:
        """Entry gate (spec §5.2): equity >= initial_margin * contracts.

        ``contracts`` is the WHOLE position the order would leave, held contracts
        included. Equity is cash + open MTM marked at ``mark_price`` (default: the last
        mark(), else the entry price), so an open loss shrinks what can be added and an
        open profit grows it. Flat, equity is cash.
        """
        if mark_price is None:
            mark_price = self._mark_price
        if mark_price is None:
            mark_price = self.position.avg_price
        return self.equity(mark_price) >= self.instrument.initial_margin * contracts

    def equity_after_fill(self, side: str, contracts: int, price: float, commission: float,
                          mark_price: float) -> float:
        """Equity just after a fill of ``contracts`` on ``side`` at ``price`` paying
        ``commission``, everything marked at ``mark_price`` (no state changes).

        Equity is linear in the position, so this is equity(mark) less the fee plus the
        fill's own P&L at the mark: a buy above the mark (slippage) costs, a sell below
        it costs. Covers opening, scaling in, reducing and flipping alike.
        """
        signed = contracts if side == "buy" else -contracts
        return (self.equity(mark_price) - commission
                + pnl_points(price, mark_price, signed, self.instrument))

    def margin_call_amount(self) -> float:
        """Variation-margin call on a maintenance breach (spec §5.2).

        Breach when equity < maintenance_margin * |qty|.
        On breach the call restores to INITIAL:
            initial_margin * |qty| - equity   (>= 0)
        Otherwise 0. Flat => 0. Uses the last mark() price.
        """
        call = self.margin_call(ts=0)
        return 0.0 if call is None else call.amount

    def margin_call(self, ts: int) -> MarginCall | None:
        """The margin call at the last mark() price, or None (spec §5.2).

        None when flat or when equity >= maintenance_margin * |qty| (maintenance is
        the floor; equal to it is not a breach). On a breach the call restores to
        INITIAL: amount = initial_margin * |qty| - equity. ``ts`` stamps the event.
        """
        qty = abs(self.position.qty)
        if qty == 0:
            return None
        mark = self._mark_price
        if mark is None:
            mark = self.position.avg_price
        eq = self.equity(mark_price=mark)
        maintenance = self.instrument.maintenance_margin * qty
        if eq >= maintenance:
            return None
        initial = self.instrument.initial_margin * qty
        return MarginCall(ts=ts, contracts=qty, equity=eq, maintenance=maintenance,
                          initial=initial, amount=max(0.0, initial - eq))

    def mark_open_at_end(self, ts: int) -> None:
        """Book a residual open position as a flagged open-at-end Trade (§5.9).

        Marks the open position to the last mark() price and appends a
        Trade with open_at_end=True (exit_ts/exit_price=None, pnl=unrealized).
        Its ``commission`` is entry fees only (the entry fees still pooled on the
        position; no exit fee has been paid), and its ``r_multiple`` is net of them.
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
        r_mult = self._calc_r_multiple(unrealized - self._open_commission, abs(self.position.qty))

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
                commission=self._open_commission,
            )
        )
