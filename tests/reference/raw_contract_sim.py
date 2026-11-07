"""An independent raw-contract simulator for the roll reconciliation proof (2.0 spec §8).

It imports nothing from plutobt. It holds the ACTUAL contracts at their raw prices,
works in whole ticks, and on each roll bar sells the old contract and buys the new one
(or the reverse for a short) at their raw opens, paying commission and slippage on both
legs as a separate roll cost, then re-places a resting stop on the new contract at the
same distance (old stop + the gap, the gap measured at the roll bar's open). The engine
instead trades one back-adjusted series and books the roll as a cost. If the adjustment
and the engine are right, the two agree exactly.

Rules (from the README fill bullets and the Broker docstrings), market orders and stops
only, no limits:

- An order decided on bar t is a market order filling at bar t+1's open, plus slippage
  against it (buy pays open + slip, sell gets open - slip). A later decision replaces a
  pending order. ``close`` is sized to the position when decided; flat, it is no order.
- An entry's attached stop arms from its fill (it is checked on the fill bar itself). A
  scale-in with a stop replaces the resting stop; one without keeps it. A fill that
  flips the position drops the old side's stop before the new one arms.
- A long's stop fires when the bar trades at or below it: at the open if the open is
  already through (gap-through), else at the stop, less slippage; a short's mirrors it.
  A stop fires once and leaves the position flat.
- When a market order and a gap-through stop both act at the open, which went first is
  a tie: both orders are tried and the one leaving the LOWER equity at this bar's close
  is kept, stop first unless the other is strictly lower. A stop that went first leaves
  a pending order on the same side as its exit, with more contracts, keeping only the
  extra contracts; any other pending order is cancelled.
- Commission is ``commission_per_side`` per contract on every fill. Roll legs move no
  slippage into prices: the roll cost is 2 x commission + 2 x slippage ticks x tick
  value, per contract.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field


@dataclass
class Order:
    side: str                  # "buy" | "sell"
    n: int
    stop: int | None = None    # in ticks, in the decision contract's raw terms
    stop_code: str | None = None


@dataclass
class State:
    qty: int = 0
    stop: int | None = None            # resting stop, ticks, active contract's terms
    flow: int = 0                      # -sum(sign * n * price) over fills, in ticks
    commission: float = 0.0
    roll_cost: float = 0.0
    pending: Order | None = None
    counters: dict = field(default_factory=dict)


class RawContractSim:
    def __init__(self, bars: dict[str, dict[int, tuple[int, int, int, int]]],
                 rolls: list[tuple[int, str, str]], *, tick_value: float,
                 commission_per_side: float, slippage_ticks: int):
        """``bars``: contract -> {ts: (open, high, low, close)} in ticks. ``rolls``:
        (roll_ts, from, to) in order. The first contract trades until the first roll,
        each next one from its roll_ts on."""
        self.bars, self.rolls = bars, rolls
        self.tv, self.comm, self.slip = tick_value, commission_per_side, slippage_ticks
        self.codes = [rolls[0][1]] + [r[2] for r in rolls] if rolls else list(bars)
        self.gap: dict[str, int] = {}         # to-contract -> gap at its roll (ticks)
        for ts, a, b in rolls:
            self.gap[b] = bars[b][ts][0] - bars[a][ts][0]

    # -- timeline ------------------------------------------------------------------
    def timeline(self) -> list[tuple[int, str]]:
        out = []
        for i, code in enumerate(self.codes):
            lo = self.rolls[i - 1][0] if i > 0 else None
            hi = self.rolls[i][0] if i < len(self.rolls) else None
            for ts in sorted(self.bars[code]):
                if (lo is None or ts >= lo) and (hi is None or ts < hi):
                    out.append((ts, code))
        return out

    def shift(self, level: int, from_code: str, to_code: str) -> int:
        """A level in ``from_code``'s raw terms, moved to ``to_code`` by the gaps."""
        i, j = self.codes.index(from_code), self.codes.index(to_code)
        for code in self.codes[i + 1:j + 1]:
            level += self.gap[code]
        return level

    # -- fills ---------------------------------------------------------------------
    def fill(self, s: State, side: str, n: int, price: int) -> None:
        sign = 1 if side == "buy" else -1
        before = s.qty
        s.flow -= sign * n * price
        s.qty += sign * n
        s.commission += self.comm * n
        if before != 0 and s.qty != 0 and (before > 0) != (s.qty > 0):
            s.stop = None                     # a flip drops the old side's stop
        if s.qty == 0:
            s.stop = None

    def gap_through(self, s: State, o: int) -> bool:
        if s.stop is None or s.qty == 0:
            return False
        return o <= s.stop if s.qty > 0 else o >= s.stop

    def fire_stop(self, s: State, at: int) -> tuple[str, int]:
        long = s.qty > 0
        side, n = ("sell" if long else "buy"), abs(s.qty)
        self.fill(s, side, n, at - self.slip if long else at + self.slip)
        s.stop = None
        return side, n

    def execute(self, s: State, order: Order, o: int, code: str) -> None:
        price = o + self.slip if order.side == "buy" else o - self.slip
        self.fill(s, order.side, order.n, price)
        if order.stop is not None and s.qty != 0:
            s.stop = self.shift(order.stop, order.stop_code, code)

    def open_step(self, s: State, o: int, code: str, stop_first: bool) -> None:
        order, s.pending = s.pending, None
        if stop_first and self.gap_through(s, o):
            side, n = self.fire_stop(s, o)
            if order is not None:
                if order.side == side and order.n > n:
                    order = Order(order.side, order.n - n, order.stop, order.stop_code)
                else:
                    order = None
        if order is not None:
            self.execute(s, order, o, code)

    def walk(self, s: State, o: int, h: int, lo: int) -> None:
        if s.stop is not None and s.qty != 0:
            if s.qty > 0 and lo <= s.stop:
                self.fire_stop(s, o if o <= s.stop else s.stop)
            elif s.qty < 0 and h >= s.stop:
                self.fire_stop(s, o if o >= s.stop else s.stop)
        if s.qty == 0:
            s.stop = None

    def equity(self, s: State, close: int) -> float:
        return (s.flow + s.qty * close) * self.tv - s.commission - s.roll_cost

    # -- run -----------------------------------------------------------------------
    def run(self, script: dict[int, tuple]) -> State:
        """``script``: decision ts -> ("buy"|"sell", n, stop_distance_ticks|None) or
        ("close",). Stops are set off the decision bar's close."""
        s = State(counters={"rolls_held": 0, "roll_bar_gap_stops": 0, "open_ties": 0})
        prev_code = None
        last_close = 0
        for ts, code in self.timeline():
            o, h, lo, c = self.bars[code][ts]
            if prev_code is not None and code != prev_code and s.qty != 0:
                old_open = self.bars[prev_code][ts][0]
                n = abs(s.qty)
                s.flow += s.qty * old_open - s.qty * o          # sell old, buy new
                s.roll_cost += (2 * self.comm + 2 * self.slip * self.tv) * n
                if s.stop is not None:
                    s.stop += self.gap[code]
                s.counters["rolls_held"] += 1
                if self.gap_through(s, o):
                    s.counters["roll_bar_gap_stops"] += 1
            if prev_code is not None:
                if s.pending is not None and self.gap_through(s, o):
                    s.counters["open_ties"] += 1
                    first, second = copy.deepcopy(s), copy.deepcopy(s)
                    for st, stop_first in ((first, True), (second, False)):
                        self.open_step(st, o, code, stop_first)
                        self.walk(st, o, h, lo)
                    worse = (second if self.equity(second, c)
                             < self.equity(first, c) - 1e-6 else first)
                    s.__dict__.update(worse.__dict__)
                else:
                    self.open_step(s, o, code, True)
                    self.walk(s, o, h, lo)
            prev_code, last_close = code, c
            action = script.get(ts)
            if action is not None:
                if action[0] == "close":
                    s.pending = (Order("sell" if s.qty > 0 else "buy", abs(s.qty))
                                 if s.qty != 0 else None)
                else:
                    side, n, dist = action
                    stop = None
                    if dist is not None:
                        stop = c - dist if side == "buy" else c + dist
                    s.pending = Order(side, n, stop, code)
        s.counters["final_qty"] = s.qty
        s.counters["last_close"] = last_close
        return s

    def price_pnl_ticks(self, s: State) -> int:
        return s.flow + s.qty * s.counters["last_close"]
