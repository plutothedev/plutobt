"""Worst-case intrabar path (broker.fill_pending).

Inside a bar the engine does not know whether the high or the low came first, nor, at an
equal price, whether the stop or the limit traded first. It evaluates every ordering the
bar allows (open -> low -> high, open -> high -> low, and both tie orders) and commits the
one that leaves the LOWEST account equity marked at the bar's close: the worst-case mode
of Löw, Maier-Paape and Platen (2015).
"""
import random
from collections import namedtuple

import pytest

from plutobt.broker import Broker
from plutobt.config import BacktestConfig
from plutobt.instruments import get_instrument
from plutobt.portfolio import Portfolio

Bar = namedtuple("Bar", "ts open high low close")
NQ = get_instrument("NQ")


def make(cls=Broker, slip=0.0, commission=0.0, cash=100_000.0):
    pf = Portfolio(starting_cash=cash, instrument=NQ)
    return cls(NQ, BacktestConfig(slippage_ticks=slip, commission_per_side=commission), pf), pf


def got(fills):
    return [(f.side, f.contracts, f.price) for f in fills]


def test_partial_cover_takes_the_path_that_retests_the_tightened_stop():
    # The reviewer's repro. Short 2 @18,000 stop 18,100; buy limit 1 @17,900 with a
    # tightened stop 17,950. Bar O18000 H18050 L17850 C17940. Up first (the short's
    # adverse side) covers 1 at 17,900 and never re-tests 17,950: 103,200. Down first
    # covers 1 at 17,900, then the rise to 18,050 crosses 17,950 and buys the other at
    # 17,950: 103,000. The bar allows both; the engine books the worse.
    broker, pf = make()
    broker.sell(2, stop=18_100.0)
    broker.fill_pending(Bar(1, 18_000.0, 18_000.0, 18_000.0, 18_000.0))
    broker.buy(1, limit=17_900.0, stop=17_950.0)
    fills = broker.fill_pending(Bar(2, 18_000.0, 18_050.0, 17_850.0, 17_940.0))
    assert got(fills) == [("buy", 1, 17_900.0), ("buy", 1, 17_950.0)]
    assert pf.position.qty == 0
    assert pf.equity(17_940.0) == pytest.approx(103_000.0)


def test_equal_price_stop_and_limit_take_the_worse_tie_order():
    # Long 1 @18,000.25 (1 tick slippage) stop 17,900; buy limit 1 @17,900. The bar
    # trades through 17,900 and closes there. Stop first: out at 17,899.75, then the
    # limit buys 1 at 17,900 (97,990 at the close). Limit first: long 2, then the stop
    # sells both at 17,899.75 (97,985). The engine books the worse, limit first.
    broker, pf = make(slip=1.0)
    broker.buy(1, stop=17_900.0)
    broker.fill_pending(Bar(1, 18_000.0, 18_000.0, 18_000.0, 18_000.0))
    assert pf.position.avg_price == 18_000.25
    add = broker.buy(1, limit=17_900.0)
    fills = broker.fill_pending(Bar(2, 18_000.0, 18_010.0, 17_850.0, 17_900.0))
    assert got(fills) == [("buy", 1, 17_900.0), ("sell", 2, 17_899.75)]
    assert add.status == "filled"
    assert pf.position.qty == 0
    assert pf.equity(17_900.0) == pytest.approx(97_985.0)


def test_a_path_that_does_not_matter_keeps_the_adverse_first_path():
    # Long 1 stop 17,900, sell limit 18,050 as a target. Either order fills the target
    # (the stop is never reached), so the default path stands and nothing changes.
    broker, pf = make()
    broker.buy(1, stop=17_900.0)
    broker.fill_pending(Bar(1, 18_000.0, 18_000.0, 18_000.0, 18_000.0))
    target = broker.sell(1, limit=18_050.0)
    fills = broker.fill_pending(Bar(2, 18_000.0, 18_060.0, 17_950.0, 18_010.0))
    assert got(fills) == [("sell", 1, 18_050.0)]
    assert target.status == "filled"
    assert pf.cash == pytest.approx(101_000.0)


def test_a_limit_filled_only_on_the_rejected_path_stays_pending():
    # Long 1 stop 17,900, sell limit 18,050 target; the bar reaches both. Down first:
    # stopped at 17,900 and the target is cancelled with the position it closed. Up
    # first: target at 18,050. The worse (stopped) path is booked, and the Order the
    # strategy holds reads what really happened to it on that path.
    broker, pf = make()
    broker.buy(1, stop=17_900.0)
    broker.fill_pending(Bar(1, 18_000.0, 18_000.0, 18_000.0, 18_000.0))
    target = broker.sell(1, limit=18_050.0)
    fills = broker.fill_pending(Bar(2, 18_000.0, 18_060.0, 17_890.0, 18_000.0))
    assert got(fills) == [("sell", 1, 17_900.0)]
    assert target.status == "cancelled"
    assert broker.pending is None
    assert pf.position.qty == 0
    assert pf.cash == pytest.approx(98_000.0)


# ---------------------------------------------------------------------------
# Property: the booked path is never kinder than any single ordering the bar allows.
# ---------------------------------------------------------------------------

def forced(down_first, stop_first):
    """A Broker that walks exactly one intrabar ordering (no worst-of choice)."""

    class Forced(Broker):
        def _intrabar_candidates(self, default_down_first):
            return [(down_first, stop_first)]

    return Forced


ORDERINGS = [(d, s) for d in (True, False) for s in (True, False)]


def scenario(r):
    q = lambda x: round(x * 4) / 4  # noqa: E731
    pos = r.choice([0, 1, 2, -1, -2])
    stop = None if pos == 0 or r.random() < 0.25 else \
        q(18_000 - (1 if pos > 0 else -1) * r.uniform(5, 120))
    amend = q(18_000 + r.uniform(-100, 100)) if pos == 0 and r.random() < 0.2 else None
    side = r.choice(["buy", "sell"])
    n = r.choice([1, 2, 3])
    kind = r.choice(["market", "limit", "limit", "limit", None])
    lim = q(18_000 + r.uniform(-150, 150))
    ast = q(lim + r.uniform(-120, 120)) if r.random() < 0.6 else None
    if r.random() < 0.15 and stop is not None:
        lim = stop                       # equal trigger prices: the tie case
    o = q(18_000 + r.uniform(-120, 120))
    c = q(18_000 + r.uniform(-120, 120))
    h = q(max(o, c) + r.uniform(0, 150))
    lo = q(min(o, c) - r.uniform(0, 150))
    slip = r.choice([0.0, 1.0, 2.0])
    cash = r.choice([1e7, 30_000.0])     # small accounts exercise margin rejections
    return pos, stop, amend, side, n, kind, lim, ast, Bar(2, o, h, lo, c), slip, cash


def play(cls, sc):
    pos, stop, amend, side, n, kind, lim, ast, bar, slip, cash = sc
    broker, pf = make(cls, slip=slip, commission=2.5, cash=cash)
    if pos:
        (broker.buy if pos > 0 else broker.sell)(abs(pos), stop=stop)
        broker.fill_pending(Bar(1, 18_000.0, 18_000.0, 18_000.0, 18_000.0))
    if amend is not None:
        broker.amend_stop(amend)
    place = broker.buy if side == "buy" else broker.sell
    if kind == "market":
        place(n, stop=ast)
    elif kind == "limit":
        place(n, stop=ast, limit=lim)
    broker.fill_pending(bar)
    return pf.equity(bar.close)


def test_property_booked_equity_is_the_worst_of_every_ordering():
    r = random.Random(20261102)
    branched = 0
    for _ in range(5_000):
        sc = scenario(r)
        eng = play(Broker, sc)
        alts = [play(forced(d, s), sc) for d, s in ORDERINGS]
        assert eng <= min(alts) + 1e-6, (sc, eng, alts)
        assert eng == pytest.approx(min(alts), abs=1e-6), (sc, eng, alts)
        branched += max(alts) - min(alts) > 1e-6
    # The scenarios must actually exercise path-dependent bars, or the test is vacuous.
    assert branched > 50, branched
