"""Order execution: directional slippage, gap-through stops, lifecycle (spec 5.4, 5.4a)."""

from __future__ import annotations

import copy
import math
import numbers
from dataclasses import dataclass, replace

from plutobt.config import BacktestConfig
from plutobt.instruments import Instrument, ticks_to_price
from plutobt.portfolio import Fill, Portfolio


@dataclass
class Order:
    side: str                    # "buy" | "sell"
    contracts: int
    kind: str                    # "market" | "limit"
    price: float | None = None   # trigger for stop/limit
    attached_stop: float | None = None
    status: str = "pending"      # "pending"|"filled"|"cancelled"|"rejected"
    margin_liquidation: bool = False   # the engine's forced exit on a margin call


class OrderError(ValueError):
    """A strategy asked for an order the engine cannot represent."""


class FrameError(ValueError):
    """A strategy's precompute left a bar frame the engine cannot run.

    Engine.run raises it. A problem already in the caller's own frame is a plain
    ValueError instead, since no strategy code caused it.
    """


def _ts_text(ts) -> str:
    """A bar's ts for a message, as an int when it is whole: a bar read back from a
    mixed-dtype frame carries its ts as a float (1700000180.0)."""
    if isinstance(ts, numbers.Real) and not isinstance(ts, numbers.Integral):
        if math.isfinite(ts) and float(ts).is_integer():
            return str(int(ts))
    return str(ts)


def _check_size(contracts, what: str) -> None:
    """The one rule for an order's size: an int (never a bool), 1 or more.

    Anything else raises OrderError naming ``what`` (the side, and where the order was
    caught), the value, its type and the fix. Nothing is coerced, so 4.0 is refused
    like 4.0000001.
    """
    if isinstance(contracts, bool):
        fix = "pass a number of contracts, not True or False"
    elif isinstance(contracts, numbers.Integral):
        if contracts >= 1:
            return
        fix = ("place no order when the size works out to 0" if contracts == 0
               else "a size is never negative; buy() and sell() pick the side")
    elif isinstance(contracts, numbers.Real):
        fix = "wrap the size in int(...)"
    else:
        fix = "pass the number of contracts as an int"
    raise OrderError(
        f"{what} must be a whole number of contracts (an int), 1 or more; "
        f"got {contracts!r} ({type(contracts).__name__}): {fix}"
    )


class Broker:
    def __init__(
        self,
        instrument: Instrument,
        config: BacktestConfig,
        portfolio: Portfolio,
    ) -> None:
        self.instrument = instrument
        self.config = config
        self.portfolio = portfolio
        self.pending: Order | None = None          # single queued order or None
        self.resting_stop: float | None = None     # GTC protective stop level

    def _slippage_ticks(self) -> float:
        st = self.config.slippage_ticks
        if st is None:
            st = self.instrument.slippage_ticks
        return st

    def _make_entry(
        self,
        side: str,
        contracts: int,
        stop: float | None,
        limit: float | None,
    ) -> Order:
        _check_size(contracts, f"{side} order size")
        if limit is not None:
            order = Order(
                side=side,
                contracts=contracts,
                kind="limit",
                price=limit,
                attached_stop=stop,
            )
        else:
            order = Order(
                side=side,
                contracts=contracts,
                kind="market",
                price=None,
                attached_stop=stop,
            )
        self._queue(order)
        return order

    def _queue(self, order: Order) -> None:
        """Make ``order`` the one pending order. An order it replaces is cancelled, so
        no Order object is left reading "pending" after it can no longer fill."""
        if self.pending is not None and self.pending is not order:
            self.pending.status = "cancelled"
        self.pending = order

    def buy(
        self,
        contracts: int,
        stop: float | None = None,
        limit: float | None = None,
    ) -> Order:
        return self._make_entry("buy", contracts, stop, limit)

    def sell(
        self,
        contracts: int,
        stop: float | None = None,
        limit: float | None = None,
    ) -> Order:
        return self._make_entry("sell", contracts, stop, limit)

    def close(self) -> Order | None:
        qty = self.portfolio.position.qty
        if qty == 0:
            return None
        side = "sell" if qty > 0 else "buy"
        order = Order(side=side, contracts=abs(qty), kind="market")
        self._queue(order)
        return order

    def liquidate(self) -> Order | None:
        """Queue the engine's forced exit on a margin call (spec §5.2).

        A market order for the whole position, replacing any pending order: it fills
        at the next bar's open and no break cancels it. Flat => None.
        """
        order = self.close()
        if order is not None:
            order.margin_liquidation = True
        return order

    def amend_stop(self, price: float) -> None:
        """Set the resting protective stop.

        Set while flat, it protects only an entry that fills on the NEXT bar, from that
        entry's own fill (checked on its entry bar, like an attached stop). If nothing
        fills on that bar, the flat book drops it, so a limit entry that fills later is
        unprotected. To protect a limit entry, attach the stop to the order
        (``buy(..., limit=..., stop=...)``); it arms whenever the limit fills."""
        self.resting_stop = price

    def cancel_pending_entry(self, keep_exits: bool = False) -> None:
        order = self.pending
        if order is None:
            return
        # Checked first: the trim below reads the size, and would repair a bad one. An
        # Order changed after it was placed, or queued straight into `pending`, never
        # met placement's check.
        _check_size(order.contracts, f"at a break, the pending {order.side} order's size")
        if order.margin_liquidation:
            # A forced liquidation is the engine's, not the strategy's: no break stops
            # it, and it fills at the next session's open.
            return
        if keep_exits and self._is_reducing(order):
            # An order that reduces the position is an EXIT: it survives the break
            # and fills at the next session's open, as a resting exit would. A flip
            # keeps only its closing contracts; the new position's half is an
            # entry, and entries do not cross a break.
            held = abs(self.portfolio.position.qty)
            if order.contracts > held:
                order.contracts = held
                order.attached_stop = None
            return
        order.status = "cancelled"
        self.pending = None
        # NOTE: resting_stop is deliberately left untouched (mandatory correction).

    # ------------------------------------------------------------------
    # Fill helpers
    # ------------------------------------------------------------------

    def _commission(self, contracts: int) -> float:
        return self.config.commission_per_side * contracts

    def _slip_amount(self) -> float:
        return ticks_to_price(self._slippage_ticks(), self.instrument)

    def _market_fill_price(self, side: str, open_px: float) -> float:
        slip = self._slip_amount()
        return open_px + slip if side == "buy" else open_px - slip

    def _emit_fill(self, ts: int, side: str, contracts: int, price: float,
                   margin_liquidation: bool = False) -> Fill:
        # The backstop every fill passes. An order is checked where it is first touched,
        # with the clearer message, but the resting stop and the margin trim size a fill
        # from the POSITION, which no order check sees.
        _check_size(contracts, f"cannot fill the {side} at ts {_ts_text(ts)}: its size")
        fill = Fill(
            ts=ts,
            side=side,
            contracts=contracts,
            price=price,
            commission=self._commission(contracts),
            margin_liquidation=margin_liquidation,
        )
        self.portfolio.apply_fill(fill)
        return fill

    def _position_side(self) -> str | None:
        qty = self.portfolio.position.qty
        if qty > 0:
            return "long"
        if qty < 0:
            return "short"
        return None

    def _open_through_stop(self, open_px: float) -> bool:
        """True when the bar opens at or beyond the held position's resting stop (a
        gap-through: the stop is triggered by the open itself)."""
        stop = self.resting_stop
        pos_side = self._position_side()
        if stop is None or pos_side is None:
            return False
        return open_px <= stop if pos_side == "long" else open_px >= stop

    @staticmethod
    def _marketable(order: Order, open_px: float) -> bool:
        """True for a limit already at or through the market at the open: a buy limit
        at or above the open, a sell limit at or below it."""
        if order.side == "buy":
            return open_px <= order.price
        return open_px >= order.price

    def _open_fill_price(self, order: Order, open_px: float) -> float:
        """An order filled at the open: the open plus slippage against the order, for a
        market order and a marketable limit alike, the limit capping the latter (a buy
        never pays above its limit, a sell never takes below it)."""
        price = self._market_fill_price(order.side, open_px)
        if order.kind == "limit":
            price = min(order.price, price) if order.side == "buy" else max(order.price, price)
        return price

    def _fire_stop(self, ts, at: float) -> Fill:
        """Fill the held position's resting stop, triggered at ``at`` (its own price, or
        the price it went live at when that was already beyond it), less slippage
        against the exit. Firing consumes the stop, so it fills at most once."""
        long = self._position_side() == "long"
        slip = self._slip_amount()
        fill = self._emit_fill(ts, "sell" if long else "buy",
                               abs(self.portfolio.position.qty),
                               at - slip if long else at + slip)
        self.resting_stop = None        # GTC stop consumed -> auto-cancel
        return fill

    def _after_stop(self, order: Order | None, stopped: Fill) -> Order | None:
        """The pending order once the stop has flattened the position it was sized
        against. A flip (same side as the stop's exit, more contracts than it closed)
        already has its closing half done, so it keeps just its new side: an entry from
        flat, margin-checked when it fills. Anything else (an exit, a scale-in, a forced
        liquidation) is stale and cancelled."""
        if order is None:
            return None
        if order.side == stopped.side and order.contracts > stopped.contracts:
            order.contracts -= stopped.contracts
            return order
        order.status = "cancelled"
        self.pending = None
        return None

    def _execute(self, order: Order, ts, price: float, mark: float) -> Fill | None:
        """Fill ``order`` at ``price`` if allow_short and margin (marked at ``mark``)
        admit it, else it is "rejected" (None). A filled order's attached stop becomes
        the resting stop, live from this fill."""
        order = self._admit(order, float(price), mark)
        if order is None:
            return None
        prior_side = self._position_side()
        fill = self._emit_fill(ts, order.side, order.contracts, price,
                               margin_liquidation=order.margin_liquidation)
        order.status = "filled"
        self.pending = None
        self._drop_stop_on_flip(prior_side)
        # An attached stop protects the position the fill leaves. A fill that flattens
        # leaves none, and its stop must not become the NEXT position's initial_stop
        # (that trade's R would read it).
        if order.attached_stop is not None and self.portfolio.position.qty != 0:
            self.resting_stop = order.attached_stop
            # §0.4: initial_stop is set on the position this fill opened or flipped; a
            # scale-in keeps the one it has.
            pos = self.portfolio.position
            if pos.initial_stop is None:
                pos.initial_stop = order.attached_stop
        return fill

    def _is_reducing(self, order: Order) -> bool:
        qty = self.portfolio.position.qty
        if qty > 0 and order.side == "sell":
            return True
        if qty < 0 and order.side == "buy":
            return True
        return False

    def _drop_stop_on_flip(self, prior_side: str | None) -> None:
        """A fill that flipped the position clears the old side's stop: a sell-stop
        under a long is no protective stop for the short that replaced it (it would
        fire at once as a buy-stop). An attached stop on the flip re-arms after this."""
        side = self._position_side()
        if prior_side is not None and side is not None and side != prior_side:
            self.resting_stop = None

    def _admit(self, order: Order, price: float, mark: float) -> Order | None:
        """The checks an order meets at the moment it fills at ``price``: allow_short,
        then margin.

        Returns the order (possibly trimmed to its closing contracts) or None when it
        is refused (status "rejected", pending cleared).
        """
        if not self.config.allow_short:
            # allow_short=False: nothing may leave a net short. A sell that only closes
            # a long goes through; a flip keeps just that closing half (its stop was for
            # the short); a short entry is refused.
            qty = self.portfolio.position.qty
            signed = order.contracts if order.side == "buy" else -order.contracts
            if qty + signed < 0:
                if qty > 0:
                    order.contracts = qty
                    order.attached_stop = None
                else:
                    return self._reject(order)
        held = abs(self.portfolio.position.qty)
        reducing = self._is_reducing(order)
        opening = order.contracts - held if reducing else order.contracts
        if opening > 0:
            # Initial margin covers the WHOLE position the fill leaves: a flip's new
            # side, or the held contracts plus a scale-in's. Equity is the account
            # just after this fill, marked at ``mark``: open P&L counts (a flip's new
            # side opens on what its closing half leaves, as close() then a new entry
            # would see it), and so do this fill's own commission and slippage. A pure
            # reduction opens nothing and is never gated.
            resulting = opening if reducing else held + opening
            after = self.portfolio.equity_after_fill(
                order.side, order.contracts, price, self._commission(order.contracts), mark)
            # Compared in cents: an open plus slippage that is not exactly representable
            # (GC 2000.3 + 0.2) would otherwise leave an exact-boundary account a
            # femto-dollar short and refuse it.
            required = self.instrument.initial_margin * resulting
            if round(after, 2) < round(required, 2):
                if not reducing:
                    return self._reject(order)
                # A flip whose new side fails margin keeps only its closing
                # contracts: the exit goes through, the reversal does not.
                order.contracts = held
                order.attached_stop = None
        return order

    def _reject(self, order: Order) -> None:
        order.status = "rejected"
        self.pending = None
        return None

    @staticmethod
    def _touch(cur: float, end: float, level: float, falling: bool) -> float | None:
        """Where a move from ``cur`` to ``end`` first reaches ``level``, for an order
        that triggers at or below its price (``falling``: a sell stop, a buy limit) or
        at or above it (a buy stop, a sell limit): ``cur`` itself when the price is
        already there, None when the move never gets there."""
        if falling:
            if cur <= level:
                return cur
            return level if end <= level else None
        if cur >= level:
            return cur
        return level if end >= level else None

    def _next_on_leg(self, cur: float, end: float, order: Order | None,
                     stop_first: bool = True) -> tuple[float, str] | None:
        """The first resting order the move from ``cur`` to ``end`` triggers, as (the
        price it triggers at, "stop" or "limit"); None if neither. At the same price
        ``stop_first`` picks which goes first."""
        hits = []
        side = self._position_side()
        if self.resting_stop is not None and side is not None:
            at = self._touch(cur, end, self.resting_stop, falling=side == "long")
            if at is not None:
                hits.append((abs(at - cur), 0 if stop_first else 1, at, "stop"))
        if order is not None:
            at = self._touch(cur, end, order.price, falling=order.side == "buy")
            if at is not None:
                hits.append((abs(at - cur), 1 if stop_first else 0, at, "limit"))
        if not hits:
            return None
        _, _, at, what = min(hits)
        return at, what

    def _walk(self, ts, order: Order | None, open_px: float, low: float, high: float,
              down_first: bool, stop_first: bool) -> list[Fill]:
        """One monotone intrabar ordering: open -> one extreme -> the other, each
        resting order triggering where the path first crosses its price."""
        fills: list[Fill] = []
        legs = [(open_px, low), (low, high)] if down_first else [(open_px, high), (high, low)]
        for cur, end in legs:
            while (hit := self._next_on_leg(cur, end, order, stop_first)) is not None:
                cur, what = hit
                if what == "stop":
                    stopped = self._fire_stop(ts, cur)
                    fills.append(stopped)
                    order = self._after_stop(order, stopped)
                else:
                    fill = self._execute(order, ts, cur, cur)
                    fills.extend([fill] if fill is not None else [])
                    order = None
        if self.portfolio.position.qty == 0:
            self.resting_stop = None      # no orphan stop on a flat position
        return fills

    def _intrabar_candidates(self, default_down_first: bool) -> list[tuple[bool, bool]]:
        """The orderings a bar allows, as (down_first, stop_first), the default first:
        both monotone paths, each with both tie orders at an equal trigger price."""
        d = default_down_first
        return [(d, True), (not d, True), (d, False), (not d, False)]

    def _open_candidates(self) -> list[bool]:
        """The orders a gap-through stop and a market or marketable order can take at the
        open, as stop_first, the default (the stop first) first."""
        return [True, False]

    def _open(self, ts, order: Order | None, open_px: float, at_open: bool,
              stop_first: bool) -> tuple[list[Fill], Order | None]:
        """The open: a gap-through stop and the market or marketable order, in the given
        order. Returns the fills and the order still pending after the open.

        Order first, the stop is not fired here: if it is still beyond the open after
        the order's fill, the walk fills it at the open, as it does any stop already
        beyond the price it went live at."""
        fills: list[Fill] = []
        if stop_first and self._open_through_stop(open_px):
            stopped = self._fire_stop(ts, open_px)
            fills.append(stopped)
            order = self._after_stop(order, stopped)
        if order is not None and at_open:
            fill = self._execute(order, ts, self._open_fill_price(order, open_px), open_px)
            fills.extend([fill] if fill is not None else [])
            order = None
        return fills, order

    def _path_candidates(self, order: Order | None, low: float,
                         high: float) -> list[tuple[bool, bool]]:
        """The intrabar orderings to try after the open, the default first. Only a
        pending limit the bar's range reaches makes the ordering matter: without one the
        held stop is the only order left and fills at the same price on either path, so
        the default is walked once."""
        side = self._position_side()
        down_first = (side == "long") if side is not None else (
            order is None or order.side == "buy")
        if order is not None and low <= order.price <= high:
            return self._intrabar_candidates(down_first)
        return [(down_first, True)]

    def _shadow(self, order: Order | None) -> tuple[Broker, Order | None]:
        """A throwaway copy of this broker, its portfolio and the pending order, to try
        an ordering on without touching the real book. Trades are not copied: closing
        only appends to them, and equity reads cash and the position."""
        pf = copy.copy(self.portfolio)
        pf.position = replace(self.portfolio.position)
        pf.trades = []
        shadow = copy.copy(self)
        shadow.portfolio = pf
        twin = None if order is None else replace(order)
        shadow.pending = twin
        return shadow, twin

    def fill_pending(self, next_bar) -> list[Fill]:
        """Fill against one bar: the pending order and the resting stop, walked along
        the bar's prices in time order.

        1. The open. If the open is at or beyond the held position's stop
           (gap-through), the stop fills at the open. The pending order fills at the
           open if it is a MARKET order or a marketable LIMIT (a buy limit at or above
           the open, a sell limit at or below it), both like a market order: the open
           plus slippage against the order, a marketable limit capped by its limit (a
           buy at min(limit, open + slip), a sell at max(limit, open - slip)). So a
           limit through the market is no free market order. When both act at the open,
           which traded first is a tie the bar does not resolve, so both orders are
           candidates in step 2: the stop first (its exit cancels a now stale order; a
           flip keeps its new side), or the order first (the stop then fills at the open
           too if it is still beyond the open for the position the order leaves).
        2. The path: worst case, per bar. A bar's OHLC does not say whether the high or
           the low came first, nor which of two orders at the same price traded first.
           So each bar is decided on its own: the engine tries the two monotone
           intrabar paths (open -> low -> high and open -> high -> low), each with both
           tie orders (stop before limit, limit before stop) and, when step 1 has a tie
           at the open, both orders there. Every candidate runs on a copy of the book,
           and the one that leaves the lowest equity marked at THIS bar's close is
           committed. This is the worst-case mode of Löw, Maier-Paape and Platen,
           "Correctness of Backtest Engines" (2015, arXiv:1509.08248). The choice is
           per bar, among those orderings only: it is not a worst case over the whole
           run (an ordering that is worse at this close can turn out better later), nor
           over all the ways a real bar could have moved (it can wiggle more than two
           monotone legs). A conservative convention, not a guarantee. On a tie the
           default ordering stands: the adverse extreme for the position held after the
           open first (down to the low for a long, up to the high for a short); flat,
           toward the pending limit (down for a buy limit, up for a sell limit), the
           adverse direction for the position it would open; at the same price the
           stop first; and at the open the stop first. Only a tie at the open, or a
           pending limit the bar's range reaches after the open, can make the ordering
           matter, so every other bar walks the default once.
           Along a path every resting order triggers in the order the path crosses its
           price: a resting LIMIT fills at its price with no slippage (it waits in the
           book and the market comes to it; one the open already trades through is
           marketable, above), the held STOP fills at its price less slippage against
           the exit. No order can trigger twice: a fill consumes the order, a stop
           fills once.

        A fill can arm a stop on the way: an order's attached stop (or one set with
        amend_stop while flat) is live from that fill (from the open for an order
        filled at the open) and is checked on the rest of the path. A stop already
        beyond the price it went live at fills there (the open's gap-through is the
        same rule), never at its better stop price. A stop that flattens the position
        leaves the pending order sized against a position that is gone: a flip keeps
        only its new side (an entry from flat), anything else is cancelled. Every
        fill meets allow_short and margin when it trades (marked at the open for an
        order filled there, at its own price for a resting limit); a limit the path
        never reaches stays pending, exactly as it was.
        """
        order = self.pending
        if order is not None:
            # Checked first, before the stop, the margin check or a trim reads the size.
            _check_size(order.contracts,
                        f"cannot fill the {order.side} order at ts {_ts_text(next_bar.ts)}: "
                        "its size")
            if order.kind not in ("market", "limit"):
                raise ValueError(f"unknown order kind: {order.kind!r}")
        ts = next_bar.ts
        open_px = float(next_bar.open)
        low, high = float(next_bar.low), float(next_bar.high)
        close_px = float(next_bar.close)
        at_open = order is not None and (order.kind == "market"
                                         or self._marketable(order, open_px))
        if at_open and self._open_through_stop(open_px):
            # A gap-through stop and an order filled at the open: which traded first is a
            # tie the bar does not resolve, so the open is part of the choice. Every
            # (open order, path) pair runs on a copy of the book.
            candidates = []
            for open_stop_first in self._open_candidates():
                shadow, twin = self._shadow(order)
                _, rest = shadow._open(ts, twin, open_px, at_open, open_stop_first)
                candidates += [(open_stop_first, *path)
                               for path in shadow._path_candidates(rest, low, high)]
            chosen = self._worst(ts, order, open_px, low, high, close_px, at_open,
                                 candidates)
            fills, order = self._open(ts, order, open_px, at_open, chosen[0])
            path = chosen[1:]
        else:
            # The open is settled (at most one order acts there); only the path is left.
            fills, order = self._open(ts, order, open_px, at_open, True)
            paths = self._path_candidates(order, low, high)
            path = paths[0] if len(paths) == 1 else self._worst(
                ts, order, open_px, low, high, close_px, False,
                [(None, *p) for p in paths])[1:]
        fills.extend(self._walk(ts, order, open_px, low, high, *path))
        return fills

    def _worst(self, ts, order: Order | None, open_px: float, low: float, high: float,
               close_px: float, at_open: bool, candidates: list[tuple]) -> tuple:
        """The candidate (open_stop_first, down_first, stop_first) that leaves the lowest
        equity marked at this bar's close, each run on a copy of the book. None for
        open_stop_first means the open is already done. Strictly lower only: on a tie
        the earlier (default) candidate stands."""
        if len(candidates) == 1:
            return candidates[0]
        best, worst = None, None
        for cand in candidates:
            open_stop_first, down_first, stop_first = cand
            shadow, twin = self._shadow(order)
            if open_stop_first is not None:
                _, twin = shadow._open(ts, twin, open_px, at_open, open_stop_first)
            shadow._walk(ts, twin, open_px, low, high, down_first, stop_first)
            eq = shadow.portfolio.equity(close_px)
            if worst is None or eq < worst - 1e-6:
                best, worst = cand, eq
        return best
