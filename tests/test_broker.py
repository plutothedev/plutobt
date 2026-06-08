from collections import namedtuple
from dataclasses import fields, is_dataclass

from plutobt.broker import Broker, Order
from plutobt.config import BacktestConfig
from plutobt.instruments import get_instrument
from plutobt.portfolio import Fill, Portfolio

Bar = namedtuple("Bar", "ts open high low close")


def test_order_is_dataclass_with_exact_fields():
    assert is_dataclass(Order)
    names = {f.name for f in fields(Order)}
    assert names == {"side", "contracts", "kind", "price", "attached_stop", "status"}


def test_order_defaults():
    o = Order(side="buy", contracts=1, kind="market")
    assert o.side == "buy"
    assert o.contracts == 1
    assert o.kind == "market"
    assert o.price is None
    assert o.attached_stop is None
    assert o.status == "pending"


# ---------------------------------------------------------------------------
# M5.3 helpers + Broker construction tests
# ---------------------------------------------------------------------------

def make_broker(starting_cash=100_000.0, slippage_ticks=1.0, allow_short=True):
    inst = get_instrument("NQ")
    cfg = BacktestConfig(slippage_ticks=slippage_ticks, allow_short=allow_short)
    pf = Portfolio(starting_cash=starting_cash, instrument=inst)
    return Broker(inst, cfg, pf), inst, pf


def test_broker_starts_with_no_pending_and_no_resting_stop():
    broker, _, _ = make_broker()
    assert broker.pending is None
    assert broker.resting_stop is None


def test_buy_queues_single_pending_market_order():
    broker, _, _ = make_broker()
    order = broker.buy(contracts=2)
    assert order.side == "buy"
    assert order.contracts == 2
    assert order.kind == "market"
    assert order.status == "pending"
    assert broker.pending is order          # single slot, not a list
    assert not isinstance(broker.pending, list)


def test_sell_queues_single_pending_market_order():
    broker, _, _ = make_broker()
    order = broker.sell(contracts=1)
    assert order.side == "sell"
    assert broker.pending is order


def test_buy_with_attached_stop_records_attached_stop():
    broker, _, _ = make_broker()
    order = broker.buy(contracts=1, stop=14_900.0)
    assert order.attached_stop == 14_900.0
    # resting_stop is only armed when the entry fills, not at queue time
    assert broker.resting_stop is None


def test_buy_with_limit_makes_a_limit_order():
    broker, _, _ = make_broker()
    order = broker.buy(contracts=1, limit=15_000.0)
    assert order.kind == "limit"
    assert order.price == 15_000.0
