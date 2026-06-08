from dataclasses import fields, is_dataclass

from plutobt.broker import Order


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
