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
    # Front H4 [2/1, 3/8) and M4 [3/8, 4/1), each padded 10 days both ways.
    assert q.pieces == (
        B.Piece("NQH2024", "NQH4", date(2024, 1, 22), date(2024, 3, 18)),
        B.Piece("NQM2024", "NQM4", date(2024, 2, 27), date(2024, 4, 11)),
    )
    assert q.held == ()
    costs = [kw for n, kw in m.calls if n == "get_cost"]
    assert len(costs) == 2
    assert costs[0] == dict(dataset="GLBX.MDP3", symbols=["NQH4"], schema=H1,
                            start="2024-01-22", end="2024-03-18", stype_in="raw_symbol")
    assert q.price_usd == pytest.approx(0.01 * (56 + 44))
    res = [kw for n, kw in m.calls if n == "resolve"]
    assert res == [dict(dataset="GLBX.MDP3", symbols=["NQ.v.0"], stype_in="continuous",
                        stype_out="raw_symbol", start_date="2024-02-01",
                        end_date="2024-04-01")]
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
    _, _, _, q = nq_quote(tmp_path, market=m)
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
    assert q2.held == (B.Piece("NQH2024", "NQH4", date(2024, 1, 22), date(2024, 3, 11)),)
    assert q2.pieces == (
        B.Piece("NQH2024", "NQH4", date(2024, 3, 11), date(2024, 3, 18)),
        B.Piece("NQM2024", "NQM4", date(2024, 2, 27), date(2024, 4, 11)),
    )


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


def test_price_changed_buys_nothing(tmp_path):
    m, c, st, q = nq_quote(tmp_path)
    m.rate_per_day = 0.02
    with pytest.raises(B.PriceChanged) as ei:
        B.buy(c, q, q.price_usd, st)
    assert ei.value.new_usd == pytest.approx(2.0)
    assert "$2.00" in str(ei.value) and "Nothing was bought" in str(ei.value)
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
    m.rate_per_day = 0.06                       # fresh price 6.00
    with pytest.raises(B.PriceChanged) as ei:
        B.buy(c, q, q.price_usd, st, max_cost=5)
    # The trader says yes to the new price; --max-cost still refuses it.
    with pytest.raises(B.BuyError,
                       match=r"Databento price \$6\.00 is above --max-cost \$5\.00\. "
                             r"Nothing was bought\.") as ei2:
        B.buy(c, q, ei.value.new_usd, st, max_cost=5)
    assert not isinstance(ei2.value, B.PriceChanged)
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
    with pytest.raises(B.BuyError, match=r"pip install plutobt\[databento\]"):
        B.make_client()


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

    mods = ["plutobt", "plutobt.buy"]
    if importlib.util.find_spec("plutobt.cli") is not None:   # Task 7 adds the CLI
        mods.append("plutobt.cli")
    code = ("import sys; sys.modules['databento'] = None; "
            f"import {', '.join(mods)}; "
            "assert sys.modules['databento'] is None")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
