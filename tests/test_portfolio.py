import math

import pytest

from plutobt.portfolio import Position, Trade


def test_position_defaults_flat():
    p = Position()
    assert p.qty == 0
    assert p.avg_price is None
    assert p.initial_stop is None


def test_position_is_mutable_dataclass():
    p = Position(qty=2, avg_price=100.0, initial_stop=95.0)
    p.qty = -1
    p.avg_price = 101.5
    assert p.qty == -1
    assert p.avg_price == 101.5
    assert p.initial_stop == 95.0


def test_trade_value_type_fields():
    t = Trade(
        entry_ts=1000,
        exit_ts=2000,
        side="long",
        contracts=2,
        entry_price=100.0,
        exit_price=108.0,
        pnl=160.0,
    )
    assert t.entry_ts == 1000
    assert t.exit_ts == 2000
    assert t.side == "long"
    assert t.contracts == 2
    assert t.entry_price == 100.0
    assert t.exit_price == 108.0
    assert t.pnl == 160.0
    assert t.r_multiple is None
    assert t.open_at_end is False


def test_trade_open_at_end_flag():
    t = Trade(
        entry_ts=1000,
        exit_ts=None,
        side="short",
        contracts=1,
        entry_price=110.0,
        exit_price=None,
        pnl=-50.0,
        open_at_end=True,
    )
    assert t.exit_ts is None
    assert t.exit_price is None
    assert t.open_at_end is True
    assert math.isclose(t.pnl, -50.0)
