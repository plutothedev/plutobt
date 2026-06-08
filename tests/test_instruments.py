import dataclasses

import pytest

from plutobt.instruments import Instrument


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
