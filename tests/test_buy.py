"""Task 6: quote, buy and assemble Databento bars (the money path). Never touches the
network: every client here is ``tests/fake_databento``."""
import math
import subprocess
import sys
import traceback
from datetime import date

import pandas as pd
import pytest

from plutobt import buy as B
from plutobt.config import BacktestConfig
from plutobt.contracts import calendar_roll_day, parse_contract
from plutobt.data import tag_sessions
from plutobt.engine import Engine
from plutobt.instruments import get_instrument
from plutobt.store import Store
from plutobt.strategies.ema_cross import EMACross
from tests import fake_databento as fake

KEY = "db-test-not-a-key"
H1 = "ohlcv-1h"


def client(market, key=KEY):
    return fake.Historical(key=key, market=market)


def nq_quote(tmp_path, market=None, start=date(2024, 2, 1), end=date(2024, 4, 1), bars="1h"):
    m = market or fake.nq_market()
    c = client(m)
    st = Store(tmp_path)
    return m, c, st, B.quote(c, "NQ", bars, start, end, st)


def _ts(d: date) -> int:
    return int(pd.Timestamp(d, tz="UTC").timestamp())


# -- constants -----------------------------------------------------------------

def test_constants():
    assert B.DATASET == "GLBX.MDP3"
    assert B.SCHEMAS == {"1m": "ohlcv-1m", "1h": "ohlcv-1h", "1d": "ohlcv-1d"}
    assert B.BUYABLE == frozenset({"NQ", "MNQ", "ES", "MES", "YM", "RTY", "CL", "GC", "ZN"})
    assert B.PAD_DAYS == 10
    assert issubclass(B.PriceChanged, B.BuyError)


# -- quote ---------------------------------------------------------------------

def test_quote_sums_cost_per_piece_and_charges_nothing(tmp_path):
    m, c, st, q = nq_quote(tmp_path)
    assert m.count("get_range") == 0
    assert q.root == "NQ" and q.bars == "1h"
    assert (q.start, q.end, q.data_ends) == (date(2024, 2, 1), date(2024, 4, 1), None)
    # Front H4 [2/1, 3/8) and M4 [3/8, 4/1): padded 10 days around the roll only;
    # the outer edges stop at the requested start and end.
    assert q.pieces == (
        B.Piece("NQH2024", "NQH4", date(2024, 2, 1), date(2024, 3, 18)),
        B.Piece("NQM2024", "NQM4", date(2024, 2, 27), date(2024, 4, 1)),
    )
    assert q.held == ()
    costs = [kw for n, kw in m.calls if n == "get_cost"]
    assert len(costs) == 2
    assert costs[0] == dict(dataset="GLBX.MDP3", symbols=["NQH4"], schema=H1,
                            start="2024-02-01", end="2024-03-18", stype_in="raw_symbol")
    assert q.price_usd == pytest.approx(0.01 * (46 + 34))
    # GLBX refuses continuous -> raw_symbol, so the front month is resolved in two free
    # steps: continuous -> instrument_id, then ONE call instrument_id -> raw_symbol.
    res = [kw for n, kw in m.calls if n == "resolve"]
    assert res == [dict(dataset="GLBX.MDP3", symbols=["NQ.v.0"], stype_in="continuous",
                        stype_out="instrument_id", start_date="2024-02-01",
                        end_date="2024-04-01"),
                   dict(dataset="GLBX.MDP3", symbols=["101", "102"],
                        stype_in="instrument_id", stype_out="raw_symbol",
                        start_date="2024-02-01", end_date="2024-04-01")]
    assert st.held(H1, "NQH2024") == []          # nothing written by a quote


def test_quote_refuses_unbuyable_root(tmp_path):
    with pytest.raises(B.BuyError, match="NQ, MNQ, ES"):
        B.quote(client(fake.nq_market()), "AAPL", "1h", date(2024, 2, 1), date(2024, 4, 1),
                Store(tmp_path))


def test_quote_refuses_unknown_bar_size_and_backwards_range(tmp_path):
    c = client(fake.nq_market())
    with pytest.raises(B.BuyError, match="1m, 1h, 1d"):
        B.quote(c, "NQ", "5m", date(2024, 2, 1), date(2024, 4, 1), Store(tmp_path))
    with pytest.raises(B.BuyError, match="before"):
        B.quote(c, "NQ", "1h", date(2024, 4, 1), date(2024, 4, 1), Store(tmp_path))


def test_quote_accepts_lower_case_root(tmp_path):
    q = B.quote(client(fake.nq_market()), "nq", "1h", date(2024, 2, 1), date(2024, 4, 1),
                Store(tmp_path))
    assert q.root == "NQ"


def test_quote_clamps_to_available_end(tmp_path):
    # Part of 3/20 is available: that day is excluded (floored to the UTC date).
    m = fake.nq_market(available_end="2024-03-20T13:00:00.000000000Z")
    _, _, _, q = nq_quote(tmp_path, market=m)
    assert q.end == date(2024, 3, 20)
    assert q.data_ends == date(2024, 3, 20)
    assert all(p.end <= date(2024, 3, 20) for p in q.pieces)
    assert q.pieces[-1] == B.Piece("NQM2024", "NQM4", date(2024, 2, 27), date(2024, 3, 20))
    res = [kw for n, kw in m.calls if n == "resolve"][0]
    assert res["end_date"] == "2024-03-20"


def test_quote_refuses_a_range_past_the_data(tmp_path):
    m = fake.nq_market(available_end="2024-01-10T00:00:00.000000000Z")
    with pytest.raises(B.BuyError, match="2024-01-10"):
        nq_quote(tmp_path, market=m)


def test_quote_window_never_runs_before_dataset_start(tmp_path):
    m = fake.nq_market(dataset_start="2024-01-25T00:00:00.000000000Z")
    _, _, _, q = nq_quote(tmp_path, market=m, start=date(2024, 1, 25))
    assert q.pieces[0].start == date(2024, 1, 25)
    with pytest.raises(B.BuyError, match="2024-01-25"):
        nq_quote(tmp_path, market=m, start=date(2024, 1, 20))


@pytest.mark.parametrize("bad", [math.nan, -1.0, "1.00", None, math.inf, True])
def test_quote_refuses_a_bad_price(tmp_path, bad):
    m = fake.nq_market(cost_fn=lambda kw: bad)
    with pytest.raises(B.BuyError, match="price"):
        nq_quote(tmp_path, market=m)


def test_quote_refuses_no_front_contract(tmp_path):
    m = fake.Market([])
    with pytest.raises(B.BuyError, match="no NQ"):
        nq_quote(tmp_path, market=m)


def test_quote_skips_contract_that_takes_over_after_end(tmp_path):
    # Databento calls M4 front from 3/8, but CME's calendar roll is 3/11, after the end.
    m = fake.nq_market(available_end="2024-03-09T00:00:00.000000000Z")
    _, _, _, q = nq_quote(tmp_path, market=m, end=date(2024, 3, 9))
    assert [p.contract for p in q.pieces] == ["NQH2024"]
    assert q.pieces[0].end == date(2024, 3, 9)


def test_second_fetch_is_free(tmp_path):
    m, c, st, q = nq_quote(tmp_path)
    B.buy(c, q, q.price_usd, st)
    before = m.count("get_cost")
    q2 = B.quote(c, "NQ", "1h", date(2024, 2, 1), date(2024, 4, 1), st)
    assert q2.pieces == () and q2.price_usd == 0.0
    assert m.count("get_cost") == before
    assert {p.contract for p in q2.held} == {"NQH2024", "NQM2024"}


def test_partly_held_request_quotes_only_the_missing_piece(tmp_path):
    m, c, st, q = nq_quote(tmp_path, end=date(2024, 3, 1))
    B.buy(c, q, q.price_usd, st)
    q2 = B.quote(c, "NQ", "1h", date(2024, 2, 1), date(2024, 4, 1), st)
    assert q2.held == (B.Piece("NQH2024", "NQH4", date(2024, 2, 1), date(2024, 3, 1)),)
    assert q2.pieces == (
        B.Piece("NQH2024", "NQH4", date(2024, 3, 1), date(2024, 3, 18)),
        B.Piece("NQM2024", "NQM4", date(2024, 2, 27), date(2024, 4, 1)),
    )


def test_quote_tells_decade_reused_raw_symbols_apart(tmp_path):
    # GLBX raw symbols carry a one-digit year: NQH4 is NQH2014 and NQH2024.
    m = fake.Market([
        fake.Listing("NQH4", date(2013, 12, 13), date(2014, 3, 14), 3500.0, 0.25, 1),
        fake.Listing("NQM4", date(2014, 3, 14), date(2014, 6, 13), 3600.0, 0.25, 2),
        fake.Listing("NQH4", date(2023, 12, 15), date(2024, 3, 8), 17000.0, 0.25, 3),
    ])
    c = client(m)
    st = Store(tmp_path)
    q = B.quote(c, "NQ", "1d", date(2014, 1, 2), date(2024, 3, 1), st)
    assert q.pieces == (
        B.Piece("NQH2014", "NQH4", date(2014, 1, 2), date(2014, 3, 24)),
        B.Piece("NQM2014", "NQM4", date(2014, 3, 4), date(2014, 6, 23)),
        B.Piece("NQH2024", "NQH4", date(2023, 12, 5), date(2024, 3, 1)),
    )
    B.buy(c, q, q.price_usd, st)
    assert st.held("ohlcv-1d", "NQH2014") == [(date(2014, 1, 2), date(2014, 3, 24))]
    assert st.held("ohlcv-1d", "NQH2024") == [(date(2023, 12, 5), date(2024, 3, 1))]
    old = st.frame("ohlcv-1d", "NQH2014", date(2014, 1, 2), date(2014, 3, 24))
    new = st.frame("ohlcv-1d", "NQH2024", date(2023, 12, 5), date(2024, 3, 1))
    assert old["close"].between(3400, 3600).all() and len(old)
    assert new["close"].between(16900, 17100).all() and len(new)


def test_quote_skips_a_first_contract_rolled_out_before_the_start(tmp_path):
    # Databento keeps H4 front until 3/13, past CME's calendar roll (3/11); a request
    # starting 3/12 never uses H4 under the calendar rule, so it is not bought.
    m = fake.Market([
        fake.Listing("NQH4", date(2023, 12, 15), date(2024, 3, 13), 17000.0, 0.25, 1),
        fake.Listing("NQM4", date(2024, 3, 13), date(2024, 6, 14), 17200.0, 0.25, 2),
    ])
    _, c, st, q = nq_quote(tmp_path, market=m, start=date(2024, 3, 12))
    assert q.pieces == (B.Piece("NQM2024", "NQM4", date(2024, 3, 12), date(2024, 4, 1)),)
    B.buy(c, q, q.price_usd, st)
    df, meta = B.assemble(st, q, session="rth")
    assert meta is None and df["ts"].iloc[0] >= _ts(date(2024, 3, 12))


# -- symbology: real GLBX answers ---------------------------------------------

# Verbatim from real Databento (live check 2026-11-17, NQ, 2024-02-01 to 2024-04-01).
REAL_CONT_TO_ID = {
    "result": {"NQ.v.0": [{"d0": "2024-02-01", "d1": "2024-03-13", "s": "750"},
                          {"d0": "2024-03-13", "d1": "2024-04-01", "s": "13743"}]},
    "symbols": ["NQ.v.0"], "stype_in": "continuous", "stype_out": "instrument_id",
    "start_date": "2024-02-01", "end_date": "2024-04-01", "partial": [], "not_found": [],
    "message": "OK", "status": 0}
REAL_ID_TO_RAW = {
    "result": {"13743": [{"d0": "2024-02-01", "d1": "2024-04-01", "s": "NQM4"}],
               "750": [{"d0": "2024-02-01", "d1": "2024-04-01", "s": "NQH4"}]},
    "symbols": ["13743", "750"], "stype_in": "instrument_id", "stype_out": "raw_symbol",
    "start_date": "2024-02-01", "end_date": "2024-04-01", "partial": [], "not_found": [],
    "message": "OK", "status": 0}


def real_nq_market(**kw):
    return fake.Market([
        fake.Listing("NQH4", date(2023, 12, 15), date(2024, 3, 13), 17000.0, 0.25, 750),
        fake.Listing("NQM4", date(2024, 3, 13), date(2024, 6, 14), 17200.0, 0.25, 13743),
    ], **kw)


def _resolve(c, symbols, stype_in, stype_out):
    return c.symbology.resolve(dataset="GLBX.MDP3", symbols=symbols, stype_in=stype_in,
                               stype_out=stype_out, start_date="2024-02-01",
                               end_date="2024-04-01")


def test_fake_refuses_continuous_to_raw_symbol_like_real_glbx():
    # The first 3.1 code asked for exactly this and real Databento answered 422.
    c = client(fake.nq_market())
    with pytest.raises(fake.BentoClientError) as ei:
        _resolve(c, ["NQ.v.0"], "continuous", "raw_symbol")
    text = str(ei.value)
    assert text.startswith("422 symbology_invalid_request")
    assert "`stype_in=continuous`, `stype_out=raw_symbol`" in text


def test_fake_answers_symbology_exactly_as_real_glbx():
    c = client(real_nq_market())
    assert _resolve(c, ["NQ.v.0"], "continuous", "instrument_id") == REAL_CONT_TO_ID
    assert _resolve(c, ["13743", "750"], "instrument_id", "raw_symbol") == REAL_ID_TO_RAW


def test_quote_on_real_shaped_answers_prices_the_roll(tmp_path):
    m = real_nq_market()
    _, c, st, q = nq_quote(tmp_path, market=m)
    # Databento's front is H4 until 3/13 (CME's calendar roll is 3/11): both contracts
    # are bought, padded 10 days around Databento's roll.
    assert q.pieces == (
        B.Piece("NQH2024", "NQH4", date(2024, 2, 1), date(2024, 3, 23)),
        B.Piece("NQM2024", "NQM4", date(2024, 3, 3), date(2024, 4, 1)),
    )
    ids = [kw for n, kw in m.calls if n == "resolve" and kw["stype_in"] == "instrument_id"]
    assert [kw["symbols"] for kw in ids] == [["750", "13743"]]


def test_quote_asks_for_every_front_id_in_one_call(tmp_path):
    m, _, _, q = nq_quote(tmp_path, start=date(2024, 2, 1), end=date(2024, 7, 1))
    assert [p.raw_symbol for p in q.pieces] == ["NQH4", "NQM4", "NQU4"]
    ids = [kw for n, kw in m.calls if n == "resolve" and kw["stype_in"] == "instrument_id"]
    assert len(ids) == 1
    assert ids[0]["symbols"] == ["101", "102", "103"]
    assert m.count("resolve") == 2


def _step2(c, change):
    real = c.symbology.resolve

    def resolve(**kw):
        answer = real(**kw)
        if kw["stype_in"] == "instrument_id":
            change(answer)
        return answer

    c.symbology.resolve = resolve


@pytest.mark.parametrize("how", ["missing", "late", "two"])
def test_quote_refuses_an_id_step_two_cannot_name(tmp_path, how):
    m = fake.nq_market()
    c = client(m)

    def change(answer):
        if how == "missing":
            answer["result"].pop("102")
        elif how == "late":               # an interval that starts after the id was front
            answer["result"]["102"] = [{"d0": "2024-03-20", "d1": "2024-04-01",
                                        "s": "NQM4"}]
        else:                              # two different symbols on the same day
            answer["result"]["102"].append({"d0": "2024-02-01", "d1": "2024-04-01",
                                            "s": "NQU4"})

    _step2(c, change)
    with pytest.raises(B.BuyError, match="102") as ei:
        B.quote(c, "NQ", "1h", date(2024, 2, 1), date(2024, 4, 1), Store(tmp_path))
    assert "nothing was charged" in str(ei.value).lower()
    assert m.count("get_cost") == 0 and m.count("get_range") == 0


def test_quote_accepts_a_repeated_identical_interval(tmp_path):
    m = fake.nq_market()
    c = client(m)
    _step2(c, lambda a: a["result"]["102"].append(dict(a["result"]["102"][0])))
    q = B.quote(c, "NQ", "1h", date(2024, 2, 1), date(2024, 4, 1), Store(tmp_path))
    assert [p.raw_symbol for p in q.pieces] == ["NQH4", "NQM4"]


def test_quote_refuses_an_id_step_two_lists_as_not_found(tmp_path):
    m = fake.nq_market()
    c = client(m)

    def change(answer):
        answer["not_found"] = ["102"]

    _step2(c, change)
    with pytest.raises(B.BuyError, match="Databento could not map NQ for part of that range; "
                                         "nothing was priced or bought."):
        B.quote(c, "NQ", "1h", date(2024, 2, 1), date(2024, 4, 1), Store(tmp_path))
    assert m.count("get_cost") == 0


def test_step_two_failure_is_redacted_and_charges_nothing(tmp_path):
    m = fake.nq_market()
    c = client(m)
    real = c.symbology.resolve

    def resolve(**kw):
        if kw["stype_in"] == "instrument_id":
            raise RuntimeError(f"401 Unauthorized for key {KEY}")
        return real(**kw)

    c.symbology.resolve = resolve
    with pytest.raises(B.BuyError) as ei:
        B.quote(c, "NQ", "1h", date(2024, 2, 1), date(2024, 4, 1), Store(tmp_path))
    text = str(ei.value)
    assert "Asking Databento which contract each front-month id is" in text
    assert "***" in text
    _no_key_anywhere(ei.value)
    assert m.count("get_cost") == 0 and m.count("get_range") == 0


def test_quote_accepts_ids_step_two_lists_as_partial(tmp_path):
    # Real GLBX lists an id in step 2's partial whenever its intervals do not cover the
    # whole window (the contract's own life is shorter than a long range): that is normal.
    m = fake.nq_market()
    c = client(m)
    _step2(c, lambda a: a.__setitem__("partial", ["101", "102"]))
    q = B.quote(c, "NQ", "1h", date(2024, 2, 1), date(2024, 4, 1), Store(tmp_path))
    assert [p.raw_symbol for p in q.pieces] == ["NQH4", "NQM4"]


def test_fake_recycles_ids_and_lists_them_partial_like_real_glbx():
    c = client(fake.nq_market())
    a = c.symbology.resolve(dataset="GLBX.MDP3", symbols=["101", "102", "103"],
                            stype_in="instrument_id", stype_out="raw_symbol",
                            start_date="2024-02-01", end_date="2024-07-01")
    assert a["result"]["101"] == [
        {"d0": "2024-02-01", "d1": "2024-04-07", "s": "NQH4"},
        {"d0": "2024-04-09", "d1": "2024-05-20", "s": "NQM4 C18000"},   # after a gap
        {"d0": "2024-05-20", "d1": "2024-07-01", "s": "CLN4-CLQ4"},
    ]
    assert a["result"]["103"] == [
        {"d0": "2024-02-01", "d1": "2024-04-12", "s": "ESM4 P4500"},
        {"d0": "2024-04-15", "d1": "2024-07-01", "s": "NQU4"},
    ]
    assert a["result"]["102"] == [{"d0": "2024-02-01", "d1": "2024-07-01", "s": "NQM4"}]
    assert a["partial"] == ["101", "103"] and a["not_found"] == []
    with pytest.raises(ValueError, match="overlap"):
        fake.Market([fake.Listing("NQH4", date(2023, 12, 15), date(2024, 3, 8), 1.0, 0.25,
                                  1, recycled=((date(2024, 3, 1), date(2024, 5, 1), "X"),))])


def _es_years_market() -> fake.Market:
    """ES quarterly fronts 2020-12 to 2025-03, ids recycled before and after their life."""
    listings = []
    prev = date(2020, 12, 10)
    months = [(y, mo) for y in range(2021, 2025) for mo in (3, 6, 9, 12)] + [(2025, 3)]
    for i, (y, mon) in enumerate(months):
        to = date(y, mon, 10)
        raw = f"ES{'HMUZ'[(mon // 3) - 1]}{y % 10}"
        recycled = ()
        if raw == "ESU1":            # later an option, a gap, then a same-root spread
            recycled = ((date(2021, 10, 20), date(2023, 1, 1), "ESH6 C4500"),
                        (date(2023, 1, 4), date(2026, 1, 1), "ESH6-ESM6"))
        if raw == "ESH4":            # earlier another root, a gap, then a spread
            recycled = ((date(2020, 6, 1), date(2022, 5, 1), "CLZ2"),
                        (date(2022, 5, 10), date(2023, 10, 11), "ESZ3-ESH4"))
        listings.append(fake.Listing(raw, prev, to, 4000.0 + 10 * i, 0.25, 500 + i,
                                     recycled=recycled))
        prev = to
    return fake.Market(listings)


def test_multi_year_quote_succeeds_when_step_two_lists_ids_as_partial(tmp_path):
    # The live check: ES 1m 2016..2026 was refused because step 2 listed ids in partial.
    m = _es_years_market()
    c = client(m)
    seen = []
    _step2(c, seen.append)
    q = B.quote(c, "ES", "1d", date(2021, 1, 4), date(2025, 1, 1), Store(tmp_path))
    assert len(seen) == 1 and len(seen[0]["partial"]) >= 17
    assert {"ESH6 C4500", "ESH6-ESM6", "CLZ2", "ESZ3-ESH4"} <= {
        iv["s"] for ivs in seen[0]["result"].values() for iv in ivs}
    raws = [p.raw_symbol for p in q.pieces]
    assert raws[:5] == ["ESH1", "ESM1", "ESU1", "ESZ1", "ESH2"]
    assert raws[-2:] == ["ESZ4", "ESH5"] and len(raws) == 17
    assert [p.contract for p in q.pieces][2] == "ESU2021"
    assert m.count("get_cost") == 17 and m.count("get_range") == 0


def test_decade_reuse_quote_succeeds_with_recycled_ids(tmp_path):
    # NQH4 is NQH2014 (id 1) and NQH2024 (id 3); both ids name unrelated instruments in
    # between, one of them an NQH4 option, so the ten-year step-2 answer lists all three
    # ids in partial.
    m = fake.Market([
        fake.Listing("NQH4", date(2013, 12, 13), date(2014, 3, 14), 3500.0, 0.25, 1,
                     recycled=((date(2014, 4, 20), date(2017, 1, 1), "ESZ6-ESH7"),
                               (date(2017, 1, 1), date(2024, 3, 1), "GCJ8"))),
        fake.Listing("NQM4", date(2014, 3, 14), date(2014, 6, 13), 3600.0, 0.25, 2,
                     recycled=((date(2014, 7, 13), date(2024, 3, 1), "ZNU5"),)),
        fake.Listing("NQH4", date(2023, 12, 15), date(2024, 3, 8), 17000.0, 0.25, 3,
                     recycled=((date(2014, 1, 2), date(2015, 6, 1), "NQH4 C4000"),
                               (date(2015, 6, 10), date(2023, 10, 16), "CLN9"))),
    ])
    c = client(m)
    seen = []
    _step2(c, seen.append)
    st = Store(tmp_path)
    q = B.quote(c, "NQ", "1d", date(2014, 1, 2), date(2024, 3, 1), st)
    assert sorted(seen[0]["partial"]) == ["1", "2", "3"]
    assert q.pieces == (
        B.Piece("NQH2014", "NQH4", date(2014, 1, 2), date(2014, 3, 24)),
        B.Piece("NQM2014", "NQM4", date(2014, 3, 4), date(2014, 6, 23)),
        B.Piece("NQH2024", "NQH4", date(2023, 12, 5), date(2024, 3, 1)),
    )
    B.buy(c, q, q.price_usd, st)
    old = st.frame("ohlcv-1d", "NQH2014", date(2014, 1, 2), date(2014, 3, 24))
    new = st.frame("ohlcv-1d", "NQH2024", date(2023, 12, 5), date(2024, 3, 1))
    assert old["close"].between(3400, 3600).all() and len(old)
    assert new["close"].between(16900, 17100).all() and len(new)


@pytest.mark.parametrize("intervals", [
    # the id names NQM4 for the start of the row, then a recycled option
    [("2024-02-01", "2024-03-20", "NQM4"), ("2024-03-20", "2024-04-01", "NQM4 C18000")],
    # the same symbol twice with a gap inside the row
    [("2024-02-01", "2024-03-15", "NQM4"), ("2024-03-18", "2024-04-01", "NQM4")],
    # covered only up to a day before the row ends
    [("2024-02-01", "2024-03-31", "NQM4")],
])
def test_quote_refuses_a_row_only_partly_covered_by_its_interval(tmp_path, intervals):
    # Id 102 is front [3/8, 4/1). An interval holding only part of that span is the
    # recycled-id case: never guessed.
    m = fake.nq_market()
    c = client(m)
    _step2(c, lambda a: a["result"].__setitem__(
        "102", [{"d0": d0, "d1": d1, "s": s} for d0, d1, s in intervals]))
    with pytest.raises(B.BuyError, match="102") as ei:
        B.quote(c, "NQ", "1h", date(2024, 2, 1), date(2024, 4, 1), Store(tmp_path))
    assert "nothing was charged" in str(ei.value).lower()
    assert m.count("get_cost") == 0 and m.count("get_range") == 0


@pytest.mark.parametrize("sym", ["ESM4", "CLM4", "NQM4 C18000", "ESH6 C4500", "NQM4-NQU4",
                                 "ESH6-ESM6", "NQ JUN24", "NQM24", "TT2:SA 06D 25M6"])
def test_quote_refuses_a_containing_interval_that_is_not_a_plain_outright_of_the_root(
        tmp_path, sym):
    m = fake.nq_market()
    c = client(m)
    _step2(c, lambda a: a["result"].__setitem__(
        "102", [{"d0": "2024-02-01", "d1": "2024-04-01", "s": sym}]))
    with pytest.raises(B.BuyError) as ei:
        B.quote(c, "NQ", "1h", date(2024, 2, 1), date(2024, 4, 1), Store(tmp_path))
    assert "nothing was charged" in str(ei.value).lower()
    assert m.count("get_cost") == 0 and m.count("get_range") == 0


# Verbatim step-2 intervals from real Databento (ES 2016-01-01..2026-01-01, live check
# 2026-11-17): CME recycled these ids across options, spreads and other roots.
REAL_ES_RECYCLED = {
    "1030": [{"d0": "2016-01-01", "d1": "2016-04-18", "s": "SIH6"},
             {"d0": "2016-04-18", "d1": "2016-06-22", "s": "FTLM601"},
             {"d0": "2016-06-22", "d1": "2017-05-14", "s": "TT2:SA 06D 25M6"},
             {"d0": "2017-05-14", "d1": "2020-06-07", "s": "CHIK0-WDCK0"},
             {"d0": "2020-06-07", "d1": "2021-10-03", "s": "ESU1"},
             {"d0": "2021-10-03", "d1": "2021-12-15", "s": "AVZX111"},
             {"d0": "2021-12-15", "d1": "2026-01-01", "s": "HRCG5"}],
    "42140878": [{"d0": "2023-08-17", "d1": "2026-01-01", "s": "ESH6"}],
    "736": [{"d0": "2016-01-01", "d1": "2016-10-17", "s": "GHYV6"},
            {"d0": "2016-10-17", "d1": "2016-12-04", "s": "WSZ6-WSX7"},
            {"d0": "2016-12-04", "d1": "2018-01-07", "s": "GUSH9"},
            {"d0": "2018-01-10", "d1": "2018-03-04", "s": "PDLG809-PTLG809"},
            {"d0": "2018-03-04", "d1": "2019-08-05", "s": "ESM9"},
            {"d0": "2019-08-05", "d1": "2020-01-28", "s": "RPU0-RPF0"},
            {"d0": "2020-01-28", "d1": "2020-03-17", "s": "PYPH004"},
            {"d0": "2020-03-17", "d1": "2026-01-01", "s": "AL6H6"}],
}


def _real_es_answer():
    return {"result": {k: [dict(iv) for iv in v] for k, v in REAL_ES_RECYCLED.items()},
            "partial": ["42140878", "736"], "not_found": []}


def test_real_recycled_intervals_map_each_front_row_to_its_outright():
    rows = [(date(2019, 3, 15), date(2019, 6, 14), "736"),
            (date(2021, 6, 18), date(2021, 9, 17), "1030"),
            (date(2025, 12, 19), date(2026, 1, 1), "42140878")]
    out = B._raw_rows(rows, _real_es_answer(), "ES", [KEY])
    assert [r["s"] for r in out] == ["ESM9", "ESU1", "ESH6"]
    assert out[0] == {"d0": "2019-03-15", "d1": "2019-06-14", "s": "ESM9"}


@pytest.mark.parametrize("row", [
    (date(2019, 7, 20), date(2019, 8, 20), "736"),      # runs from ESM9 into a spread
    (date(2018, 1, 5), date(2018, 1, 12), "736"),       # straddles the 1/7..1/10 gap
    (date(2021, 9, 20), date(2021, 12, 17), "1030"),    # runs from ESU1 into AVZX111
])
def test_real_recycled_intervals_refuse_a_row_no_interval_holds(row):
    with pytest.raises(B.BuyError, match=row[2]) as ei:
        B._raw_rows([row], _real_es_answer(), "ES", [KEY])
    assert "nothing was charged" in str(ei.value).lower()


def test_fake_market_refuses_a_reused_instrument_id():
    # CME never reuses an instrument id, even when GLBX reuses the raw symbol (NQH4).
    with pytest.raises(ValueError, match="instrument id"):
        fake.Market([
            fake.Listing("NQH4", date(2013, 12, 13), date(2014, 3, 14), 3500.0, 0.25, 7),
            fake.Listing("NQH4", date(2023, 12, 15), date(2024, 3, 8), 17000.0, 0.25, 7),
        ])


def test_outer_edges_buy_nothing_outside_the_request(tmp_path):
    m, c, st, q = nq_quote(tmp_path)
    B.buy(c, q, q.price_usd, st)
    assert q.pieces[0].start == q.start and q.pieces[-1].end == q.end
    for n, kw in m.calls:
        if n in ("get_range", "get_cost"):
            assert kw["start"] >= q.start.isoformat() and kw["end"] <= q.end.isoformat()
    for p in q.pieces:
        f = st.frame(H1, p.contract, p.start, p.end)
        assert f["ts"].min() >= _ts(q.start) and f["ts"].max() < _ts(q.end)


# -- buy -----------------------------------------------------------------------

def test_buy_saves_each_piece_to_the_store(tmp_path):
    m, c, st, q = nq_quote(tmp_path)
    B.buy(c, q, q.price_usd, st)
    calls = [kw for n, kw in m.calls if n == "get_range"]
    assert calls == [dict(dataset="GLBX.MDP3", symbols=[p.raw_symbol], schema=H1,
                          start=p.start.isoformat(), end=p.end.isoformat(),
                          stype_in="raw_symbol") for p in q.pieces]
    assert [kw for n, kw in m.calls if n == "to_df"] == [{"price_type": "float"}] * 2
    for p in q.pieces:
        assert st.held(H1, p.contract) == [(p.start, p.end)]
        f = st.frame(H1, p.contract, p.start, p.end)
        assert list(f.columns) == ["ts", "open", "high", "low", "close", "volume"]
        assert str(f["ts"].dtype) == "int64" and str(f["volume"].dtype) == "int64"
        want = fake.make_bars(m.listing(p.raw_symbol), H1, p.start.isoformat(),
                              p.end.isoformat())
        assert f["ts"].tolist() == (want.index.asi8 // 10**9).tolist()
        assert f["close"].tolist() == want["close"].tolist()


def test_buy_with_no_pieces_calls_nothing(tmp_path):
    m, c, st, q = nq_quote(tmp_path)
    B.buy(c, q, q.price_usd, st)
    q2 = B.quote(c, "NQ", "1h", date(2024, 2, 1), date(2024, 4, 1), st)
    n = len(m.calls)
    B.buy(c, q2, 0.0, st)
    assert len(m.calls) == n


def test_buying_the_same_quote_twice_buys_nothing_the_second_time(tmp_path):
    m, c, st, q = nq_quote(tmp_path)
    B.buy(c, q, q.price_usd, st)
    n_range, n_cost = m.count("get_range"), m.count("get_cost")
    with pytest.raises(B.BuyError) as ei:
        B.buy(c, q, q.price_usd, st)
    assert str(ei.value) == ("The data on this computer changed since the quote; quote "
                             "again. Nothing was bought.")
    assert (m.count("get_range"), m.count("get_cost")) == (n_range, n_cost)


def test_store_filled_by_another_run_while_waiting_buys_nothing(tmp_path):
    m, c, st, q = nq_quote(tmp_path)
    other = B.quote(c, "NQ", "1h", date(2024, 2, 1), date(2024, 3, 1), Store(tmp_path))
    B.buy(c, other, other.price_usd, Store(tmp_path))
    n = m.count("get_range")
    with pytest.raises(B.BuyError, match="changed since the quote"):
        B.buy(c, q, q.price_usd, st)
    assert m.count("get_range") == n


def test_bad_price_text_is_redacted(tmp_path):
    m = fake.nq_market(cost_fn=lambda kw: f"oops {KEY}")
    with pytest.raises(B.BuyError) as ei:
        nq_quote(tmp_path, market=m)
    assert "***" in str(ei.value)
    _no_key_anywhere(ei.value)


def test_price_changed_buys_nothing(tmp_path):
    m, c, st, q = nq_quote(tmp_path)
    m.rate_per_day = 0.02
    with pytest.raises(B.PriceChanged) as ei:
        B.buy(c, q, q.price_usd, st)
    assert ei.value.new_usd == pytest.approx(2 * q.price_usd)
    assert f"${2 * q.price_usd:.2f}" in str(ei.value)
    assert "Nothing was bought" in str(ei.value)
    assert m.count("get_range") == 0
    assert st.held(H1, "NQH2024") == [] and st.held(H1, "NQM2024") == []


def test_price_within_half_a_cent_still_buys(tmp_path):
    m, c, st, q = nq_quote(tmp_path)
    B.buy(c, q, q.price_usd + 0.004, st)
    assert m.count("get_range") == 2
    m2, c2, st2, q2 = nq_quote(tmp_path / "b")
    with pytest.raises(B.PriceChanged):
        B.buy(c2, q2, q2.price_usd + 0.006, st2)
    assert m2.count("get_range") == 0


def test_max_cost_checked_on_fresh_price(tmp_path):
    m, c, st, q = nq_quote(tmp_path)
    assert q.price_usd < 5
    m.rate_per_day = 0.1                        # fresh price 8.00
    # Refused by --max-cost first: the trader is never asked to approve a price that
    # --max-cost would refuse anyway.
    for approved in (q.price_usd, 8.0):
        with pytest.raises(B.BuyError,
                           match=r"Databento price \$8\.00 is above --max-cost \$5\.00\. "
                                 r"Nothing was bought\.") as ei:
            B.buy(c, q, approved, st, max_cost=5)
        assert not isinstance(ei.value, B.PriceChanged)
    assert m.count("get_range") == 0


def test_failure_midway_keeps_saved_pieces_no_retry(tmp_path):
    m, c, st, q = nq_quote(tmp_path)
    m.fail_get_range = {2: RuntimeError("gateway timeout")}
    with pytest.raises(B.BuyError) as ei:
        B.buy(c, q, q.price_usd, st)
    msg = str(ei.value)
    assert "NQH2024" in msg and "gateway timeout" in msg and "NQM2024" in msg
    assert m.count("get_range") == 2
    h = q.pieces[0]
    assert st.held(H1, "NQH2024") == [(h.start, h.end)]
    assert st.held(H1, "NQM2024") == []
    # The next quote asks only for what is missing.
    q2 = B.quote(c, "NQ", "1h", date(2024, 2, 1), date(2024, 4, 1), st)
    assert [p.contract for p in q2.pieces] == ["NQM2024"]


def test_failure_on_first_piece_says_nothing_was_saved(tmp_path):
    m, c, st, q = nq_quote(tmp_path)
    m.fail_get_range = {1: RuntimeError("boom")}
    with pytest.raises(B.BuyError, match="nothing was saved"):
        B.buy(c, q, q.price_usd, st)
    assert m.count("get_range") == 1


def test_spread_rows_dropped(tmp_path):
    m, c, st, q = nq_quote(tmp_path)
    m.spread_rows = True
    B.buy(c, q, q.price_usd, st)
    for p in q.pieces:
        f = st.frame(H1, p.contract, p.start, p.end)
        want = fake.make_bars(m.listing(p.raw_symbol), H1, p.start.isoformat(),
                              p.end.isoformat())
        assert len(f) == len(want)
        assert 3.333 not in set(f["open"])


def _no_key_anywhere(err):
    text = str(err) + "".join(traceback.format_exception(err))
    assert KEY not in text
    assert err.__cause__ is None and err.__context__ is None


def test_key_never_in_error_text(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABENTO_API_KEY", KEY)
    m = fake.nq_market()
    c = fake.Historical(market=m)               # key read from the environment
    st = Store(tmp_path)
    q = B.quote(c, "NQ", "1h", date(2024, 2, 1), date(2024, 4, 1), st)
    m.fail_get_range = {1: Exception(f"bad key {KEY}")}
    with pytest.raises(B.BuyError) as ei:
        B.buy(c, q, q.price_usd, st)
    assert "***" in str(ei.value)
    _no_key_anywhere(ei.value)


@pytest.mark.parametrize("where", ["fail_get_cost", "fail_resolve"])
def test_key_never_in_quote_error_text(tmp_path, where):
    m = fake.nq_market()
    setattr(m, where, Exception(f"401 Unauthorized for key {KEY}"))
    with pytest.raises(B.BuyError) as ei:
        B.quote(client(m), "NQ", "1h", date(2024, 2, 1), date(2024, 4, 1), Store(tmp_path))
    assert "***" in str(ei.value)
    _no_key_anywhere(ei.value)


def test_key_never_in_reprice_error_text(tmp_path):
    m, c, st, q = nq_quote(tmp_path)
    m.fail_get_cost = Exception(f"key={KEY} rejected")
    with pytest.raises(B.BuyError) as ei:
        B.buy(c, q, q.price_usd, st)
    assert "***" in str(ei.value)
    _no_key_anywhere(ei.value)
    assert m.count("get_range") == 0


def test_redact():
    assert B._redact(f"a {KEY} b {KEY}", KEY) == "a *** b ***"
    assert B._redact("nothing here", None) == "nothing here"
    assert B._redact("nothing here", "") == "nothing here"


# -- make_client ---------------------------------------------------------------

def test_make_client_without_databento(monkeypatch):
    monkeypatch.setitem(sys.modules, "databento", None)
    monkeypatch.setenv("DATABENTO_API_KEY", KEY)
    with pytest.raises(B.BuyError) as ei:
        B.make_client()
    text = str(ei.value)
    assert "not installed" in text
    assert 'pip install -e ".[databento]"' in text and "PlutoBT folder" in text
    assert "plutobt[databento]" not in text           # not on PyPI: that line 404s


def test_make_client_with_databento_installed_but_broken(monkeypatch):
    # Live check 2026-11-17: databento pulled pyarrow 26, which needs NumPy 2 and fails at
    # import under PlutoBT's numpy 1.26.4. The add-on IS installed; say so, with the reason.
    import builtins
    real_import = builtins.__import__

    def broken(name, *args, **kwargs):
        if name == "databento":
            raise ImportError("pyarrow requires NumPy 2.0 or newer, found 1.26.4"
                              f"\n  key {KEY}")
        return real_import(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, "databento", raising=False)
    monkeypatch.setenv("DATABENTO_API_KEY", KEY)
    monkeypatch.setattr(builtins, "__import__", broken)
    try:
        with pytest.raises(B.BuyError) as ei:
            B.make_client()
    finally:
        monkeypatch.setattr(builtins, "__import__", real_import)
    text = str(ei.value)
    assert "installed but failed to load" in text
    assert "not installed" not in text
    assert "pyarrow requires NumPy 2.0 or newer, found 1.26.4" in text
    assert 'pip install -e ".[databento]"' in text and "PlutoBT folder" in text
    assert "***" in text and text.isascii() and "\n" not in text
    _no_key_anywhere(ei.value)


@pytest.mark.parametrize("value", [None, "", "   "])
def test_make_client_without_key(monkeypatch, value):
    monkeypatch.setitem(sys.modules, "databento", fake)
    if value is None:
        monkeypatch.delenv("DATABENTO_API_KEY", raising=False)
    else:
        monkeypatch.setenv("DATABENTO_API_KEY", value)
    with pytest.raises(B.BuyError) as ei:
        B.make_client()
    assert str(ei.value) == ("Set DATABENTO_API_KEY to your Databento API key (it is read "
                             "from the environment only and never saved).")


def test_make_client_with_key(monkeypatch):
    monkeypatch.setitem(sys.modules, "databento", fake)
    monkeypatch.setenv("DATABENTO_API_KEY", KEY)
    c = B.make_client()
    assert isinstance(c, fake.Historical) and c.key == KEY


def test_make_client_error_hides_key(monkeypatch):
    class Boom:
        def __init__(self, key=None, **kw):
            raise RuntimeError(f"rejected {key}")

    mod = type(sys)("databento")
    mod.Historical = Boom
    monkeypatch.setitem(sys.modules, "databento", mod)
    monkeypatch.setenv("DATABENTO_API_KEY", KEY)
    with pytest.raises(B.BuyError) as ei:
        B.make_client()
    assert "***" in str(ei.value)
    _no_key_anywhere(ei.value)


# -- assemble ------------------------------------------------------------------

def bought(tmp_path, market=None, root="NQ", bars="1h", start=date(2024, 2, 1),
           end=date(2024, 4, 1)):
    m = market or (fake.cl_market() if root == "CL" else fake.nq_market())
    c = client(m)
    st = Store(tmp_path)
    q = B.quote(c, root, bars, start, end, st)
    B.buy(c, q, q.price_usd, st)
    return st, q


def test_assemble_two_contracts_builds_continuous_with_calendar_trigger(tmp_path):
    st, q = bought(tmp_path)
    df, meta = B.assemble(st, q, session="all")
    assert meta.roll_trigger == "calendar"
    assert [(r["from"], r["to"]) for r in meta.rolls] == [("NQH2024", "NQM2024")]
    assert df["ts"].iloc[0] >= _ts(q.start)
    assert df["ts"].iloc[-1] < _ts(q.end)
    assert meta.window == [int(df["ts"].iloc[0]), int(df["ts"].iloc[-1])]
    assert set(df["contract"]) == {"NQH2024", "NQM2024"}
    # The newest contract keeps its raw prices; the older one is shifted by the gap.
    assert (df.loc[df["contract"] == "NQM2024", "adj_offset"] == 0).all()


def test_assemble_cl_uses_volume_trigger(tmp_path):
    st, q = bought(tmp_path, root="CL", start=date(2024, 1, 2), end=date(2024, 2, 1))
    df, meta = B.assemble(st, q, session="rth")
    assert meta.roll_trigger == "volume"
    assert [(r["from"], r["to"]) for r in meta.rolls] == [("CLG2024", "CLH2024")]


def test_assemble_single_contract_has_no_meta(tmp_path):
    st, q = bought(tmp_path, start=date(2024, 2, 1), end=date(2024, 3, 1))
    assert [p.contract for p in q.pieces] == ["NQH2024"]
    df, meta = B.assemble(st, q, session="all")
    assert meta is None
    assert list(df.columns) == ["ts", "open", "high", "low", "close", "volume"]
    assert isinstance(df.index, pd.RangeIndex)
    assert df["ts"].iloc[0] >= _ts(q.start) and df["ts"].iloc[-1] < _ts(q.end)


def test_assemble_rth_filters_before_build_and_engine_accepts_meta(tmp_path):
    st, q = bought(tmp_path)
    df, meta = B.assemble(st, q, session="rth")
    assert "session" not in df.columns
    assert set(tag_sessions(df, get_instrument("NQ"))["session"]) == {"RTH"}
    (roll,) = meta.rolls
    roll_ts = roll["roll_ts"]
    assert roll_ts in set(df["ts"])
    assert pd.Timestamp(roll_ts, unit="s", tz="UTC").date() == calendar_roll_day(
        parse_contract("NQH2024"))
    cfg = BacktestConfig(commission_per_side=2.5, slippage_ticks=1.0)
    res = Engine(get_instrument("NQ"), cfg).run(df, EMACross(), roll_meta=meta)
    assert len(res.equity_curve) == len(df)


def test_assemble_daily_bars_are_never_session_filtered(tmp_path):
    st, q = bought(tmp_path, bars="1d")
    df, meta = B.assemble(st, q, session="rth")
    assert meta.roll_trigger == "calendar"
    # Daily bars open at 00:00 UTC (ETH by the clock); none is dropped.
    assert set(tag_sessions(df, get_instrument("NQ"))["session"]) == {"ETH"}
    assert len(df) > 30


def test_assemble_rth_drops_a_contract_that_rolls_in_after_the_end(tmp_path):
    # Data ends at the end date, 2024-03-11, which is H4's CME roll day. Regular hours
    # on 3/11 are past the end, so M4 never trades inside the range: build from H4 alone.
    m = fake.nq_market(available_end="2024-03-11T00:00:00.000000000Z")
    st, q = bought(tmp_path, market=m, end=date(2024, 3, 11))
    assert [p.contract for p in q.pieces] == ["NQH2024", "NQM2024"]
    df, meta = B.assemble(st, q, session="rth")
    assert meta is None and len(df)
    # The full day: trading day 3/11 opens Sunday 3/10 17:00 CT, inside the range.
    df_all, meta_all = B.assemble(st, q, session="all")
    assert [r["to"] for r in meta_all.rolls] == ["NQM2024"]
    assert df_all["ts"].iloc[-1] < _ts(q.end)


def test_assemble_volume_root_drops_a_last_contract_that_never_crossed(tmp_path):
    # Databento calls H4 front from Friday 1/19 and the data ends that day: H4's volume
    # beats G4's on 1/19 but there is no later shared day to roll at. G4 alone.
    m = fake.cl_market(available_end="2024-01-20T00:00:00.000000000Z")
    st, q = bought(tmp_path, market=m, root="CL", start=date(2024, 1, 2),
                   end=date(2024, 1, 20))
    assert [p.contract for p in q.pieces] == ["CLG2024", "CLH2024"]
    df, meta = B.assemble(st, q, session="rth")
    assert meta is None
    assert df["close"].between(71, 73).all()          # G4's prices, not H4's


def test_assemble_refuses_a_series_that_would_stop_early(tmp_path):
    # CLH4 never beats CLG4 by volume and G4's bars stop near 1/29: the series would end
    # weeks before the requested end. Refuse, keeping what was bought.
    m = fake.cl_market()
    m.listing("CLH4").volume = 10
    st, q = bought(tmp_path, market=m, root="CL", start=date(2024, 1, 2),
                   end=date(2024, 2, 15))
    with pytest.raises(B.BuyError, match="CLH2024 never took over from CLG2024") as ei:
        B.assemble(st, q, session="all")
    assert "kept" in str(ei.value)
    assert st.held(H1, "CLH2024") != []


def test_assemble_refuses_a_single_contract_that_does_not_cover_the_range(tmp_path):
    st, q = bought(tmp_path, start=date(2024, 2, 1), end=date(2024, 3, 1))
    # The same bought contract, asked for past what was bought.
    short = B.Quote(q.root, q.bars, q.start, date(2024, 3, 5), (), q.pieces, 0.0, None)
    with pytest.raises(B.BuyError, match="2024-03-01"):
        B.assemble(st, short, session="all")


def test_assemble_refuses_bad_session(tmp_path):
    st, q = bought(tmp_path)
    with pytest.raises(B.BuyError, match="session"):
        B.assemble(st, q, session="eth")


def test_assemble_refuses_data_not_held(tmp_path):
    _, _, st, q = nq_quote(tmp_path)
    with pytest.raises(B.BuyError, match="not fully held"):
        B.assemble(st, q, session="rth")


# -- import without databento --------------------------------------------------

def test_plutobt_imports_without_databento():
    import importlib.util

    mods = ["plutobt", "plutobt.buy", "plutobt.cli"]
    assert importlib.util.find_spec("plutobt.cli") is not None
    code = ("import sys; sys.modules['databento'] = None; "
            f"import {', '.join(mods)}; "
            "assert sys.modules['databento'] is None")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_failure_inside_get_range_still_warns_of_a_possible_bill(tmp_path):
    m, c, st, q = nq_quote(tmp_path)
    m.fail_get_range = {1: RuntimeError("connection reset")}
    with pytest.raises(B.BuyError) as ei:
        B.buy(c, q, q.price_usd, st)
    assert "Databento may have billed NQH2024" in str(ei.value)
    assert "billing page" in str(ei.value)


@pytest.mark.parametrize("where", ["partial", "not_found"])
def test_quote_refuses_a_partly_mapped_root_before_pricing(tmp_path, where):
    m = fake.nq_market()
    c = client(m)
    real = c.symbology.resolve

    def resolve(**kw):
        answer = real(**kw)
        answer[where] = ["NQ.v.0"]
        return answer

    c.symbology.resolve = resolve
    with pytest.raises(B.BuyError, match="Databento could not map NQ for part of that range; "
                                         "nothing was priced or bought."):
        B.quote(c, "NQ", "1h", date(2024, 2, 1), date(2024, 4, 1), Store(tmp_path))
    assert m.count("get_cost") == 0
