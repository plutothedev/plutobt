"""The intrabar path (broker.fill_pending): what fills first inside one bar.

At the open: the pending MARKET order, and a LIMIT already at or through the market
there (marketable), fill first, at the open, with slippage (a marketable limit capped by
its limit); by default a stop the open is already beyond fills at the open before them
(when both act there, the order between them is a tie the engine branches on). Then the
path: from the open to the held position's adverse extreme (a long's low, a short's
high), then to the other extreme. Each resting order (a limit, the held stop, a stop
armed by a fill on the way) triggers where the path first crosses its price.
That is the DEFAULT ordering; where the bar allows a worse one by that bar's close (the
other extreme first, the limit first at an equal price, or the order first at the open)
the engine books that instead, bar by bar: see test_worst_case_path.py.
"""
from collections import namedtuple

import pandas as pd
import pytest

from plutobt.broker import Broker
from plutobt.config import BacktestConfig
from plutobt.engine import Engine
from plutobt.instruments import get_instrument
from plutobt.portfolio import Portfolio
from plutobt.strategy import Strategy

Bar = namedtuple("Bar", "ts open high low close")
NQ = get_instrument("NQ")


def make_broker(slippage_ticks=0.0, cash=100_000.0):
    cfg = BacktestConfig(slippage_ticks=slippage_ticks)
    pf = Portfolio(starting_cash=cash, instrument=NQ)
    return Broker(NQ, cfg, pf), pf


def long_1(broker, stop):
    broker.buy(1, stop=stop)
    broker.fill_pending(Bar(1, 18_000.0, 18_010.0, 17_990.0, 18_000.0))


def short_1(broker, stop):
    broker.sell(1, stop=stop)
    broker.fill_pending(Bar(1, 18_000.0, 18_010.0, 17_990.0, 18_000.0))


def got(fills):
    return [(f.side, f.contracts, f.price) for f in fills]


# ---------------------------------------------------------------------------
# 1. A resting limit and the held stop trigger in the order the path crosses them.
# ---------------------------------------------------------------------------

def test_scale_in_limit_above_the_stop_fills_before_the_stop():
    # The auditor's repro. Long 1 @18,000 stop 17,900; buy limit 1 @18,000. The bar
    # opens 18,040 and trades down to 17,890: the path down passes 18,000 before
    # 17,900, so the add fills, then the stop sells both.
    broker, pf = make_broker()
    long_1(broker, stop=17_900.0)
    add = broker.buy(1, limit=18_000.0)
    fills = broker.fill_pending(Bar(2, 18_040.0, 18_045.0, 17_890.0, 17_895.0))
    assert got(fills) == [("buy", 1, 18_000.0), ("sell", 2, 17_900.0)]
    assert add.status == "filled"
    assert pf.position.qty == 0
    assert pf.cash == pytest.approx(100_000.0 - 2 * 100 * 20)


def test_scale_in_limit_below_the_stop_is_cancelled_by_the_stop():  # positive control
    # The limit (17,850) sits below the stop (17,900): the path down reaches the stop
    # first, the long is gone, and the add (sized for it) is cancelled.
    broker, pf = make_broker()
    long_1(broker, stop=17_900.0)
    add = broker.buy(1, limit=17_850.0)
    fills = broker.fill_pending(Bar(2, 18_040.0, 18_045.0, 17_800.0, 17_895.0))
    assert got(fills) == [("sell", 1, 17_900.0)]
    assert add.status == "cancelled"
    assert broker.pending is None
    assert pf.position.qty == 0


def test_limit_and_stop_at_the_same_price_the_stop_goes_first():
    # A tie is broken adverse first: the stop fills, the add is stale.
    broker, pf = make_broker()
    long_1(broker, stop=17_900.0)
    add = broker.buy(1, limit=17_900.0)
    fills = broker.fill_pending(Bar(2, 18_040.0, 18_045.0, 17_850.0, 17_895.0))
    assert got(fills) == [("sell", 1, 17_900.0)]
    assert add.status == "cancelled"


def test_short_scale_in_limit_below_the_stop_fills_before_the_stop():
    # The mirror: short 1 @18,000 stop 18,100; sell limit 1 @18,000; the bar opens
    # 17,960 and trades up to 18,110.
    broker, pf = make_broker()
    short_1(broker, stop=18_100.0)
    broker.sell(1, limit=18_000.0)
    fills = broker.fill_pending(Bar(2, 17_960.0, 18_110.0, 17_955.0, 18_105.0))
    assert got(fills) == [("sell", 1, 18_000.0), ("buy", 2, 18_100.0)]
    assert pf.position.qty == 0


def test_scale_in_with_its_own_stop_arms_it_from_the_fill():
    # The add's attached stop (17,950) replaces the 17,900 stop when the add fills at
    # 18,000; the rest of the path down (to 17,940) crosses it.
    broker, pf = make_broker(slippage_ticks=1.0)
    long_1(broker, stop=17_900.0)
    broker.buy(1, limit=18_000.0, stop=17_950.0)
    fills = broker.fill_pending(Bar(2, 18_040.0, 18_045.0, 17_940.0, 17_945.0))
    assert got(fills) == [("buy", 1, 18_000.0), ("sell", 2, 17_949.75)]


def test_scale_in_limit_reached_stop_not_reached_keeps_both_lots():  # positive control
    broker, pf = make_broker()
    long_1(broker, stop=17_900.0)
    broker.buy(1, limit=18_000.0)
    fills = broker.fill_pending(Bar(2, 18_040.0, 18_045.0, 17_950.0, 17_995.0))
    assert got(fills) == [("buy", 1, 18_000.0)]
    assert pf.position.qty == 2
    assert broker.resting_stop == 17_900.0


def test_long_with_a_target_goes_down_first_so_the_stop_wins_a_straddle():
    # Adverse first for a long: the path goes to the low before the high, so a bar
    # that straddles both the stop and the target books the stop; the target exit is
    # then stale and cancelled.
    broker, pf = make_broker()
    long_1(broker, stop=17_900.0)
    target = broker.sell(1, limit=18_100.0)
    fills = broker.fill_pending(Bar(2, 18_000.0, 18_120.0, 17_880.0, 18_000.0))
    assert got(fills) == [("sell", 1, 17_900.0)]
    assert target.status == "cancelled"
    assert pf.position.qty == 0


def test_short_with_a_target_goes_up_first_so_the_stop_wins_a_straddle():
    broker, pf = make_broker()
    short_1(broker, stop=18_100.0)
    target = broker.buy(1, limit=17_900.0)
    fills = broker.fill_pending(Bar(2, 18_000.0, 18_120.0, 17_880.0, 18_000.0))
    assert got(fills) == [("buy", 1, 18_100.0)]
    assert target.status == "cancelled"


def test_limit_flip_after_the_stop_opens_the_new_side_on_the_way_back_up():
    # Long 1 stop 17,900; sell 2 limit 18,050. Down to 17,880 first (the stop sells 1),
    # then up to 18,060: the flip, re-sized to its new side, sells 1 at 18,050, and its
    # attached stop (18,055), live from that fill, is crossed on the way up.
    broker, pf = make_broker()
    long_1(broker, stop=17_900.0)
    flip = broker.sell(2, limit=18_050.0, stop=18_055.0)
    fills = broker.fill_pending(Bar(2, 18_000.0, 18_060.0, 17_880.0, 18_000.0))
    assert got(fills) == [("sell", 1, 17_900.0), ("sell", 1, 18_050.0),
                          ("buy", 1, 18_055.0)]
    assert flip.status == "filled"
    assert pf.position.qty == 0


def test_flat_buy_limit_entry_then_its_stop_on_the_same_leg():
    # Flat: the path goes toward the pending limit first (down for a buy), which is
    # also the new long's adverse direction.
    broker, pf = make_broker()
    broker.buy(1, limit=17_950.0, stop=17_900.0)
    fills = broker.fill_pending(Bar(1, 18_000.0, 18_100.0, 17_850.0, 18_050.0))
    assert got(fills) == [("buy", 1, 17_950.0), ("sell", 1, 17_900.0)]


def test_flat_sell_limit_entry_then_its_stop_on_the_same_leg():
    broker, pf = make_broker()
    broker.sell(1, limit=18_050.0, stop=18_100.0)
    fills = broker.fill_pending(Bar(1, 18_000.0, 18_150.0, 17_900.0, 17_950.0))
    assert got(fills) == [("sell", 1, 18_050.0), ("buy", 1, 18_100.0)]


class _Script(Strategy):
    def __init__(self, plan):
        self.plan = plan

    def on_bar(self, ctx):
        k = (int(ctx.bar.ts) - 1_700_000_000) // 300
        for step in self.plan.get(k, []):
            step(ctx.broker)


def _frame(rows):
    return pd.DataFrame([dict(ts=1_700_000_000 + 300 * k, open=o, high=h, low=lo,
                              close=c, volume=1)
                         for k, (o, h, lo, c) in enumerate(rows)])


def test_engine_scale_in_before_the_stop_ends_at_96000():
    # The auditor's repro through Engine.run, with no costs.
    df = _frame([(18_000, 18_010, 17_990, 18_000), (18_000, 18_050, 17_990, 18_040),
                 (18_040, 18_045, 17_890, 17_895), (17_895, 17_900, 17_880, 17_890)])
    plan = {0: [lambda b: b.buy(1, stop=17_900.0)], 1: [lambda b: b.buy(1, limit=18_000.0)]}
    cfg = BacktestConfig(commission_per_side=0.0, slippage_ticks=0.0, check_causality=False)
    r = Engine(NQ, cfg).run(df, _Script(plan))
    assert r.equity_curve.iloc[-1] == pytest.approx(96_000.0)
    assert [t.contracts for t in r.trades] == [2]


# ---------------------------------------------------------------------------
# 2 + 3. A limit that is marketable at the open fills like a market order there.
# ---------------------------------------------------------------------------

def test_marketable_buy_limit_pays_the_same_slippage_as_a_market_order():
    # buy(limit=99999) is no free market order: open 18,000 + 4 ticks.
    for limit in (None, 99_999.0):
        broker, pf = make_broker(slippage_ticks=4.0)
        broker.buy(1, limit=limit)
        fills = broker.fill_pending(Bar(1, 18_000.0, 18_010.0, 17_990.0, 18_000.0))
        assert got(fills) == [("buy", 1, 18_001.0)], limit


def test_marketable_sell_limit_pays_the_same_slippage_as_a_market_order():
    for limit in (None, 1.0):
        broker, pf = make_broker(slippage_ticks=4.0)
        broker.sell(1, limit=limit)
        fills = broker.fill_pending(Bar(1, 18_000.0, 18_010.0, 17_990.0, 18_000.0))
        assert got(fills) == [("sell", 1, 17_999.0)], limit


def test_marketable_limit_slippage_is_capped_by_the_limit():
    broker, pf = make_broker(slippage_ticks=4.0)
    broker.buy(1, limit=18_000.5)
    fills = broker.fill_pending(Bar(1, 18_000.0, 18_010.0, 17_990.0, 18_000.0))
    assert got(fills) == [("buy", 1, 18_000.5)]
    broker, pf = make_broker(slippage_ticks=4.0)
    broker.sell(1, limit=17_999.5)
    fills = broker.fill_pending(Bar(1, 18_000.0, 18_010.0, 17_990.0, 18_000.0))
    assert got(fills) == [("sell", 1, 17_999.5)]


def test_limit_at_the_open_fills_at_the_open():
    broker, pf = make_broker(slippage_ticks=4.0)
    broker.buy(1, limit=18_000.0)
    fills = broker.fill_pending(Bar(1, 18_000.0, 18_010.0, 17_990.0, 18_000.0))
    assert got(fills) == [("buy", 1, 18_000.0)]


def test_resting_limit_reached_inside_the_bar_has_no_slippage():  # positive control
    broker, pf = make_broker(slippage_ticks=4.0)
    broker.buy(1, limit=17_995.0)
    fills = broker.fill_pending(Bar(1, 18_000.0, 18_010.0, 17_990.0, 18_000.0))
    assert got(fills) == [("buy", 1, 17_995.0)]


def test_marketable_target_fills_at_the_open_before_the_held_stop():
    # The auditor's case B. Long 1 stop 17,900, sell limit 18,050 pending; the bar opens
    # 18,100, through the limit, then trades down to 17,890. The limit fills at the
    # open first, so the stop has nothing left to protect.
    broker, pf = make_broker()
    long_1(broker, stop=17_900.0)
    target = broker.sell(1, limit=18_050.0)
    fills = broker.fill_pending(Bar(2, 18_100.0, 18_110.0, 17_890.0, 17_895.0))
    assert got(fills) == [("sell", 1, 18_100.0)]
    assert target.status == "filled"
    assert pf.cash == pytest.approx(102_000.0)


def test_marketable_limit_flip_fills_at_the_open_and_the_old_stop_never_fires():
    broker, pf = make_broker()
    long_1(broker, stop=17_950.0)
    broker.sell(2, limit=17_970.0)
    fills = broker.fill_pending(Bar(2, 17_980.0, 17_990.0, 17_900.0, 17_920.0))
    assert got(fills) == [("sell", 2, 17_980.0)]
    assert pf.position.qty == -1
    assert broker.resting_stop is None


def test_marketable_limit_after_a_gap_through_stop_is_resized_like_a_market_order():
    # The open (17,940) is already through the 17,950 stop: the stop fills at the
    # open first, then the flip, re-sized to its new side, sells 1 at the open.
    broker, pf = make_broker()
    long_1(broker, stop=17_950.0)
    flip = broker.sell(2, limit=17_930.0)
    fills = broker.fill_pending(Bar(2, 17_940.0, 17_960.0, 17_900.0, 17_920.0))
    assert got(fills) == [("sell", 1, 17_940.0), ("sell", 1, 17_940.0)]
    assert (flip.status, flip.contracts) == ("filled", 1)
    assert pf.position.qty == -1


def test_marketable_limit_attached_stop_is_live_from_the_open():
    broker, pf = make_broker(slippage_ticks=1.0)
    broker.buy(1, limit=18_100.0, stop=17_900.0)
    fills = broker.fill_pending(Bar(1, 18_000.0, 18_010.0, 17_850.0, 17_990.0))
    assert got(fills) == [("buy", 1, 18_000.25), ("sell", 1, 17_899.75)]
