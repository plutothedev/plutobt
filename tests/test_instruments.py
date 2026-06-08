import dataclasses

import pytest

from plutobt.instruments import Instrument
from plutobt.instruments import REGISTRY, get_instrument


def test_instrument_is_frozen_dataclass_with_exact_fields():
    inst = Instrument(
        symbol="NQ",
        point_value=20.0,
        tick_size=0.25,
        tick_value=5.0,
        currency="USD",
        session_open_ct="08:30",
        session_close_ct="15:15",
        slippage_ticks=1.0,
        initial_margin=23000.0,
        maintenance_margin=20900.0,
        settlement="cash",
        price_format="decimal",
    )
    assert dataclasses.is_dataclass(inst)
    field_names = [f.name for f in dataclasses.fields(inst)]
    assert field_names == [
        "symbol",
        "point_value",
        "tick_size",
        "tick_value",
        "currency",
        "session_open_ct",
        "session_close_ct",
        "slippage_ticks",
        "initial_margin",
        "maintenance_margin",
        "settlement",
        "price_format",
    ]
    assert inst.symbol == "NQ"
    assert inst.tick_value == 5.0
    assert inst.session_open_ct == "08:30"
    assert inst.price_format == "decimal"
    # frozen => assignment raises
    with pytest.raises(dataclasses.FrozenInstanceError):
        inst.symbol = "ES"


def test_registry_has_all_nine_symbols():
    assert set(REGISTRY) == {"NQ", "MNQ", "ES", "MES", "YM", "RTY", "CL", "GC", "ZN"}


def test_get_instrument_returns_registry_object_by_symbol():
    nq = get_instrument("NQ")
    assert nq is REGISTRY["NQ"]
    assert nq.symbol == "NQ"


def test_get_instrument_unknown_symbol_raises_keyerror_listing_known():
    with pytest.raises(KeyError) as exc:
        get_instrument("SPY")
    # message lists the known symbols so the failure is actionable
    assert "SPY" in str(exc.value)
    assert "NQ" in str(exc.value)


@pytest.mark.parametrize(
    "symbol,point_value,tick_size,tick_value,settlement,price_format",
    [
        ("NQ", 20.0, 0.25, 5.0, "cash", "decimal"),
        ("MNQ", 2.0, 0.25, 0.5, "cash", "decimal"),
        ("ES", 50.0, 0.25, 12.5, "cash", "decimal"),
        ("MES", 5.0, 0.25, 1.25, "cash", "decimal"),
        ("YM", 5.0, 1.0, 5.0, "cash", "decimal"),
        ("RTY", 50.0, 0.10, 5.0, "cash", "decimal"),
        ("CL", 1000.0, 0.01, 10.0, "physical", "decimal"),
        ("GC", 100.0, 0.10, 10.0, "physical", "decimal"),
        ("ZN", 1000.0, 0.015625, 15.625, "physical", "thirty_seconds"),
    ],
)
def test_registry_specs_match_section_5_3(
    symbol, point_value, tick_size, tick_value, settlement, price_format
):
    inst = get_instrument(symbol)
    assert inst.symbol == symbol
    assert inst.point_value == point_value
    assert inst.tick_size == tick_size
    assert inst.tick_value == tick_value
    assert inst.settlement == settlement
    assert inst.price_format == price_format
    # tick_value is internally consistent: tick_size * point_value == tick_value
    assert inst.tick_value == pytest.approx(inst.tick_size * inst.point_value, rel=0, abs=1e-9)
    # currency + non-negative slippage/margins are populated
    assert inst.currency == "USD"
    assert inst.slippage_ticks >= 0
    assert inst.initial_margin > 0
    assert inst.maintenance_margin > 0
    assert inst.initial_margin >= inst.maintenance_margin


def test_index_futures_session_is_rth_0830_to_1515_ct():
    for symbol in ("NQ", "MNQ", "ES", "MES", "YM", "RTY"):
        inst = get_instrument(symbol)
        assert inst.session_open_ct == "08:30"
        assert inst.session_close_ct == "15:15"
