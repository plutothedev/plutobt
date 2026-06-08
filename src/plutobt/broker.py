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
    kind: str                    # "market" | "limit"
    price: float | None = None   # trigger for stop/limit
    attached_stop: float | None = None
    status: str = "pending"      # "pending"|"filled"|"cancelled"|"rejected"


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
        self.pending = order
        return order

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
        self.pending = order
        return order

    def amend_stop(self, price: float) -> None:
        self.resting_stop = price

    def cancel_pending_entry(self) -> None:
        if self.pending is not None:
            self.pending.status = "cancelled"
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

    def _emit_fill(self, ts: int, side: str, contracts: int, price: float) -> Fill:
        fill = Fill(
            ts=ts,
            side=side,
            contracts=contracts,
            price=price,
            commission=self._commission(contracts),
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

    def _stop_triggered(self, pos_side: str, stop: float, bar) -> bool:
        if pos_side == "long":
            return bar.low <= stop      # sell-stop protecting a long
        return bar.high >= stop         # buy-stop protecting a short

    def _stop_fill_price(self, pos_side: str, stop: float, bar) -> float:
        slip = self._slip_amount()
        if pos_side == "long":
            # exit is a SELL; worse = lower of (stop, open), slippage subtracts
            ref = min(stop, bar.open)
            return ref - slip
        # exit is a BUY; worse = higher of (stop, open), slippage adds
        ref = max(stop, bar.open)
        return ref + slip

    def _fire_resting_stop(self, bar) -> list[Fill]:
        fills: list[Fill] = []
        stop = self.resting_stop
        pos_side = self._position_side()
        if stop is None or pos_side is None:
            return fills
        if not self._stop_triggered(pos_side, stop, bar):
            return fills
        exit_side = "sell" if pos_side == "long" else "buy"
        contracts = abs(self.portfolio.position.qty)
        price = self._stop_fill_price(pos_side, stop, bar)
        fills.append(self._emit_fill(bar.ts, exit_side, contracts, price))
        self.resting_stop = None        # GTC stop consumed -> auto-cancel
        return fills

    def _limit_fill(self, order: Order, bar) -> float | None:
        limit = order.price
        if order.side == "buy":
            if bar.low <= limit:                 # traded through
                return min(limit, bar.open)      # better-at-open on gap-down
            return None
        # sell limit
        if bar.high >= limit:
            return max(limit, bar.open)          # better-at-open on gap-up
        return None

    def _is_reducing(self, order: Order) -> bool:
        qty = self.portfolio.position.qty
        if qty > 0 and order.side == "sell":
            return True
        if qty < 0 and order.side == "buy":
            return True
        return False

    def fill_pending(self, next_bar) -> list[Fill]:
        fills: list[Fill] = []
        stop_fills = self._fire_resting_stop(next_bar)    # adverse fills first
        fills.extend(stop_fills)
        order = self.pending
        # If the protective stop just flattened us, a same-bar exit order is stale.
        if stop_fills and self.portfolio.position.qty == 0:
            if order is not None:
                order.status = "cancelled"
                self.pending = None
            order = None
        if order is not None:
            if not self._is_reducing(order) and not self.portfolio.margin_ok(
                order.contracts
            ):
                order.status = "rejected"
                self.pending = None
                order = None
        if order is not None:
            if order.kind == "market":
                price = self._market_fill_price(order.side, next_bar.open)
                fills.append(
                    self._emit_fill(next_bar.ts, order.side, order.contracts, price)
                )
                order.status = "filled"
                self.pending = None
                if order.attached_stop is not None:
                    self.resting_stop = order.attached_stop
            elif order.kind == "limit":
                price = self._limit_fill(order, next_bar)
                if price is not None:
                    fills.append(
                        self._emit_fill(next_bar.ts, order.side, order.contracts, price)
                    )
                    order.status = "filled"
                    self.pending = None
                    if order.attached_stop is not None:
                        self.resting_stop = order.attached_stop
            else:
                raise ValueError(f"unknown order kind: {order.kind!r}")
        if self.portfolio.position.qty == 0:
            self.resting_stop = None      # no orphan stop on a flat position
        return fills
