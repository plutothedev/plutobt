import dataclasses

import pytest

from plutobt.instruments import (
    REGISTRY,
    Instrument,
    get_instrument,
    parse_price,
    pnl_points,
    ticks_to_price,
)


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


def test_registry_has_all_nine_futures_symbols():
    # The 9 verified futures specs are still present and untouched. v0.5 ADDS equity
    # rows additively (CONTRACT §3.1), so this is a subset check, not exact equality.
    futures = {"NQ", "MNQ", "ES", "MES", "YM", "RTY", "CL", "GC", "ZN"}
    assert futures <= set(REGISTRY)


def test_get_instrument_returns_registry_object_by_symbol():
    nq = get_instrument("NQ")
    assert nq is REGISTRY["NQ"]
    assert nq.symbol == "NQ"


def test_get_instrument_unknown_symbol_raises_keyerror_listing_known():
    # "SPY" is now a real equity row (CONTRACT §3.1); use a genuinely unknown ticker.
    with pytest.raises(KeyError) as exc:
        get_instrument("TSLA")
    # message lists the known symbols so the failure is actionable
    assert "TSLA" in str(exc.value)
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


def test_parse_price_decimal_passthrough_float():
    nq = get_instrument("NQ")
    assert parse_price(20150.25, nq) == 20150.25


def test_parse_price_decimal_passthrough_string():
    nq = get_instrument("NQ")
    assert parse_price("20150.25", nq) == 20150.25


def test_parse_price_zn_32nds_half_tick():
    # spec §5.3 worked example: 110'085 = 110 + 8.5/32
    zn = get_instrument("ZN")
    assert parse_price("110'085", zn) == pytest.approx(110 + 8.5 / 32)


def test_parse_price_zn_32nds_whole_tick():
    # 110'080 = 110 + 8/32 (no half-32nd)
    zn = get_instrument("ZN")
    assert parse_price("110'080", zn) == pytest.approx(110 + 8.0 / 32)


def test_parse_price_zn_32nds_high_handle_and_31nds():
    # 131'315 = 131 + 31.5/32 (handle > 100, max 32nds region)
    zn = get_instrument("ZN")
    assert parse_price("131'315", zn) == pytest.approx(131 + 31.5 / 32)


def test_parse_price_zn_accepts_already_decimal_float():
    # a float on a 32nds instrument is treated as already-decimal points
    zn = get_instrument("ZN")
    assert parse_price(110.265625, zn) == pytest.approx(110.265625)


def test_parse_price_zn_rejects_malformed_quote():
    zn = get_instrument("ZN")
    with pytest.raises(ValueError):
        parse_price("110-085", zn)   # wrong separator
    with pytest.raises(ValueError):
        parse_price("110'85", zn)    # fractional part must be exactly 3 digits (CME ZN convention)


def test_ticks_to_price_nq_one_tick_is_quarter_point():
    nq = get_instrument("NQ")
    assert ticks_to_price(1, nq) == pytest.approx(0.25)
    assert ticks_to_price(4, nq) == pytest.approx(1.0)


def test_ticks_to_price_zn_one_tick_is_one_64th():
    # ZN tick = 1/64 = 0.015625 in decimal points
    zn = get_instrument("ZN")
    assert ticks_to_price(1, zn) == pytest.approx(1 / 64)
    assert ticks_to_price(2, zn) == pytest.approx(1 / 32)


def test_ticks_to_price_signed_and_fractional():
    cl = get_instrument("CL")          # tick 0.01
    assert ticks_to_price(-3, cl) == pytest.approx(-0.03)
    assert ticks_to_price(2.5, cl) == pytest.approx(0.025)


def test_pnl_points_nq_ten_point_long_one_contract():
    nq = get_instrument("NQ")
    # +10 points * $20/pt * 1 = $200
    assert pnl_points(20000.0, 20010.0, 1, nq) == pytest.approx(200.0)


def test_pnl_points_short_is_negative_of_long():
    nq = get_instrument("NQ")
    long_pnl = pnl_points(20000.0, 20010.0, 1, nq)
    short_pnl = pnl_points(20000.0, 20010.0, -1, nq)
    assert short_pnl == pytest.approx(-long_pnl)
    assert short_pnl == pytest.approx(-200.0)


def test_pnl_points_scales_with_contracts():
    es = get_instrument("ES")
    # +4 points * $50/pt * 3 = $600
    assert pnl_points(5000.0, 5004.0, 3, es) == pytest.approx(600.0)


# §10.1: per-symbol single-tick move -> exactly one tick_value of P&L, values
# taken verbatim from the §5.3 table.
@pytest.mark.parametrize(
    "symbol,expected_tick_value",
    [
        ("NQ", 5.0), ("MNQ", 0.5), ("ES", 12.5), ("MES", 1.25),
        ("YM", 5.0), ("RTY", 5.0), ("CL", 10.0), ("GC", 10.0), ("ZN", 15.625),
    ],
)
def test_section_10_1_one_tick_move_equals_tick_value(symbol, expected_tick_value):
    inst = get_instrument(symbol)
    one_tick_up = ticks_to_price(1, inst)          # decimal size of one tick
    pnl = pnl_points(0.0, one_tick_up, 1, inst)    # 1 contract, +1 tick
    assert pnl == pytest.approx(expected_tick_value)
    assert pnl == pytest.approx(inst.tick_value)


def test_section_10_1_nq_n_tick_move_dollars():
    # explicit N-tick check: 8 ticks * $5/tick = $40 on NQ
    nq = get_instrument("NQ")
    eight_ticks = ticks_to_price(8, nq)            # 2.0 points
    assert pnl_points(0.0, eight_ticks, 1, nq) == pytest.approx(40.0)


def test_section_10_1_zn_half_of_one_32nd_identity():
    # ½ of 1/32 = 1/64 = 0.015625 decimal points; * $1,000/pt = $15.625
    zn = get_instrument("ZN")
    assert zn.tick_size == pytest.approx(0.5 * (1 / 32))
    assert zn.tick_size == pytest.approx(1 / 64)
    assert zn.tick_size == pytest.approx(0.015625)
    one_tick = ticks_to_price(1, zn)
    assert pnl_points(0.0, one_tick, 1, zn) == pytest.approx(1000.0 / 64.0)
    assert pnl_points(0.0, one_tick, 1, zn) == pytest.approx(15.625)


def test_section_10_1_zn_32nds_parse_to_pnl_end_to_end():
    # buy 110'080, sell 110'085: +0.5/32 = 1 ZN tick = $15.625 on 1 contract
    zn = get_instrument("ZN")
    entry = parse_price("110'080", zn)
    exit_ = parse_price("110'085", zn)
    assert pnl_points(entry, exit_, 1, zn) == pytest.approx(15.625)


def test_every_registry_instrument_tick_value_matches_tick_math():
    # closes §10.1 across the whole registry: a +1 tick move on 1 contract
    # always equals the published tick_value, with no symbol left out.
    for symbol, inst in REGISTRY.items():
        one_tick = ticks_to_price(1, inst)
        assert pnl_points(0.0, one_tick, 1, inst) == pytest.approx(inst.tick_value), symbol


@pytest.mark.parametrize("symbol", ["AAPL", "MSFT", "SPY"])
def test_equity_rows_present_with_cash_equity_spec(symbol):
    inst = get_instrument(symbol)
    assert inst.symbol == symbol
    # CONTRACT §3.1 — cash-equity economics, NOT a futures perf-bond.
    assert inst.point_value == 1.0          # the multiplier IS point_value; 1:1 cash
    assert inst.tick_size == 0.01
    assert inst.tick_value == 0.01
    assert inst.currency == "USD"
    assert inst.settlement == "cash"
    assert inst.price_format == "decimal"
    # No performance bond for cash equity (documented limit, not an invariant).
    assert inst.initial_margin == 0.0
    assert inst.maintenance_margin == 0.0
    # Binding tick-math identity must hold for equity rows too.
    assert inst.tick_value == pytest.approx(inst.tick_size * inst.point_value, rel=0, abs=1e-9)


def test_equity_one_dollar_move_is_one_dollar_per_share():
    # point_value=1.0 => (exit-entry) * 1.0 * shares = exact cash equity.
    # A +1.00 move on 10 shares of AAPL is exactly $10 — no futures multiplier.
    aapl = get_instrument("AAPL")
    assert pnl_points(150.00, 151.00, 10, aapl) == pytest.approx(10.0)
    # One tick (0.01) on 1 share equals the published tick_value (0.01).
    one_tick = ticks_to_price(1, aapl)
    assert pnl_points(0.0, one_tick, 1, aapl) == pytest.approx(aapl.tick_value)
    assert pnl_points(0.0, one_tick, 1, aapl) == pytest.approx(0.01)
