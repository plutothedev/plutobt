from collections import namedtuple
from dataclasses import fields, is_dataclass

import pytest

from plutobt.broker import Broker, Order
from plutobt.config import BacktestConfig
from plutobt.instruments import get_instrument
from plutobt.portfolio import Portfolio

Bar = namedtuple("Bar", "ts open high low close")


def test_order_is_dataclass_with_exact_fields():
    assert is_dataclass(Order)
    names = {f.name for f in fields(Order)}
    assert names == {"side", "contracts", "kind", "price", "attached_stop", "status",
                     "margin_liquidation"}


def test_order_defaults():
    o = Order(side="buy", contracts=1, kind="market")
    assert o.side == "buy"
    assert o.contracts == 1
    assert o.kind == "market"
    assert o.price is None
    assert o.attached_stop is None
    assert o.status == "pending"
    assert o.margin_liquidation is False


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


# ---------------------------------------------------------------------------
# M5.4 — fill_pending market fills with directional slippage (§10.14)
# ---------------------------------------------------------------------------

def test_market_buy_fills_above_open_by_one_tick():  # §10.14
    broker, inst, pf = make_broker(slippage_ticks=1.0)
    broker.buy(contracts=1)
    bar = Bar(ts=1000, open=15_000.0, high=15_010.0, low=14_990.0, close=15_005.0)
    fills = broker.fill_pending(bar)
    assert len(fills) == 1
    f = fills[0]
    assert f.side == "buy"
    assert f.price == 15_000.25          # 15000 + 1 tick (0.25) up
    assert broker.pending is None
    assert pf.position.qty == 1


def test_market_sell_fills_below_open_by_one_tick():  # §10.14
    broker, inst, pf = make_broker(slippage_ticks=1.0)
    broker.sell(contracts=1)
    bar = Bar(ts=1000, open=15_000.0, high=15_010.0, low=14_990.0, close=15_005.0)
    fills = broker.fill_pending(bar)
    assert fills[0].side == "sell"
    assert fills[0].price == 14_999.75   # 15000 - 1 tick (0.25) down
    assert pf.position.qty == -1


def test_slippage_scales_with_ticks_directionally():  # §10.14
    broker, inst, pf = make_broker(slippage_ticks=4.0)
    broker.buy(contracts=1)
    bar = Bar(ts=1000, open=15_000.0, high=15_020.0, low=14_990.0, close=15_010.0)
    fills = broker.fill_pending(bar)
    assert fills[0].price == 15_001.0    # 15000 + 4 ticks * 0.25

    broker2, _, pf2 = make_broker(slippage_ticks=4.0)
    broker2.sell(contracts=1)
    fills2 = broker2.fill_pending(bar)
    assert fills2[0].price == 14_999.0   # 15000 - 4 ticks * 0.25


def test_fill_pending_returns_empty_when_nothing_pending():
    broker, _, _ = make_broker()
    bar = Bar(ts=1000, open=15_000.0, high=15_010.0, low=14_990.0, close=15_005.0)
    assert broker.fill_pending(bar) == []


def test_instrument_default_slippage_used_when_config_slippage_is_none():
    inst = get_instrument("NQ")           # instrument.slippage_ticks == 1.0
    cfg = BacktestConfig()                # slippage_ticks=None
    pf = Portfolio(starting_cash=100_000.0, instrument=inst)
    broker = Broker(inst, cfg, pf)
    broker.buy(contracts=1)
    bar = Bar(ts=1000, open=15_000.0, high=15_010.0, low=14_990.0, close=15_005.0)
    fills = broker.fill_pending(bar)
    assert fills[0].price == 15_000.25    # NQ tick 0.25, 1 tick slippage from instrument default


# ---------------------------------------------------------------------------
# M5.5 — Attached-stop arms resting_stop on entry fill; amend_stop
# ---------------------------------------------------------------------------

def test_attached_stop_arms_resting_stop_on_fill():
    broker, _, _ = make_broker()
    broker.buy(contracts=1, stop=14_900.0)
    bar = Bar(ts=1000, open=15_000.0, high=15_010.0, low=14_990.0, close=15_005.0)
    broker.fill_pending(bar)
    assert broker.resting_stop == 14_900.0


def test_amend_stop_updates_resting_stop():
    broker, _, _ = make_broker()
    broker.buy(contracts=1, stop=14_900.0)
    bar = Bar(ts=1000, open=15_000.0, high=15_010.0, low=14_990.0, close=15_005.0)
    broker.fill_pending(bar)
    broker.amend_stop(14_950.0)
    assert broker.resting_stop == 14_950.0


def test_entry_without_attached_stop_leaves_resting_stop_none():
    broker, _, _ = make_broker()
    broker.buy(contracts=1)
    bar = Bar(ts=1000, open=15_000.0, high=15_010.0, low=14_990.0, close=15_005.0)
    broker.fill_pending(bar)
    assert broker.resting_stop is None


# ---------------------------------------------------------------------------
# M5.6 — Resting stop fires with gap-through at worse-of(stop,open)−slippage (§10.3)
# ---------------------------------------------------------------------------

def open_long(broker, entry_open=15_000.0, stop=14_900.0):
    broker.buy(contracts=1, stop=stop)
    entry_bar = Bar(ts=1000, open=entry_open, high=entry_open + 10,
                    low=entry_open - 10, close=entry_open + 5)
    broker.fill_pending(entry_bar)


def test_long_stop_normal_touch_fills_at_stop_minus_slippage():  # §10.3 baseline
    broker, inst, pf = make_broker(slippage_ticks=1.0)
    open_long(broker, entry_open=15_000.0, stop=14_900.0)
    # next bar dips to 14,895 — stop touched but no gap; ref = worse(stop, open)
    stop_bar = Bar(ts=2000, open=14_920.0, high=14_925.0, low=14_895.0, close=14_910.0)
    fills = broker.fill_pending(stop_bar)
    assert len(fills) == 1
    f = fills[0]
    assert f.side == "sell"
    # worse of (stop=14900, open=14920) for a sell-stop is the LOWER -> 14900; - 1 tick
    assert f.price == 14_899.75
    assert pf.position.qty == 0
    assert broker.resting_stop is None


def test_long_stop_gap_through_fills_at_open_not_stop():  # §10.3 gap-through
    broker, inst, pf = make_broker(slippage_ticks=1.0)
    open_long(broker, entry_open=15_000.0, stop=14_900.0)
    # next bar GAPS DOWN: opens 14,850 (below the 14,900 stop)
    gap_bar = Bar(ts=2000, open=14_850.0, high=14_870.0, low=14_840.0, close=14_860.0)
    fills = broker.fill_pending(gap_bar)
    assert len(fills) == 1
    f = fills[0]
    assert f.side == "sell"
    # worse of (stop=14900, open=14850) is 14850 (the open), then - 1 tick slippage
    assert f.price == 14_849.75          # fills AT the gap open, not the 14,900 stop
    assert pf.position.qty == 0
    assert broker.resting_stop is None


def test_long_stop_not_touched_does_not_fire():
    broker, inst, pf = make_broker(slippage_ticks=1.0)
    open_long(broker, entry_open=15_000.0, stop=14_900.0)
    quiet_bar = Bar(ts=2000, open=15_010.0, high=15_030.0, low=14_950.0, close=15_020.0)
    fills = broker.fill_pending(quiet_bar)
    assert fills == []
    assert pf.position.qty == 1
    assert broker.resting_stop == 14_900.0


def test_short_stop_gap_through_fills_at_open_above_stop():  # §10.3 symmetric short
    broker, inst, pf = make_broker(slippage_ticks=1.0)
    broker.sell(contracts=1, stop=15_100.0)
    entry_bar = Bar(ts=1000, open=15_000.0, high=15_010.0, low=14_990.0, close=15_005.0)
    broker.fill_pending(entry_bar)            # short 1 @ ~14,999.75
    # next bar GAPS UP: opens 15,150 (above the 15,100 buy-stop)
    gap_bar = Bar(ts=2000, open=15_150.0, high=15_170.0, low=15_140.0, close=15_160.0)
    fills = broker.fill_pending(gap_bar)
    f = fills[0]
    assert f.side == "buy"
    # worse of (stop=15100, open=15150) for a buy-stop is the HIGHER -> 15150; + 1 tick
    assert f.price == 15_150.25
    assert pf.position.qty == 0


# ---------------------------------------------------------------------------
# M5.7 — Limit fills at limit-or-better-at-open with NO slippage
# ---------------------------------------------------------------------------

def test_buy_limit_fills_at_limit_when_bar_trades_through():
    broker, inst, pf = make_broker(slippage_ticks=1.0)
    broker.buy(contracts=1, limit=14_950.0)
    bar = Bar(ts=1000, open=15_000.0, high=15_010.0, low=14_940.0, close=14_960.0)
    fills = broker.fill_pending(bar)
    assert len(fills) == 1
    assert fills[0].price == 14_950.0     # exactly the limit, NO slippage
    assert pf.position.qty == 1


def test_buy_limit_fills_better_at_open_on_favorable_gap():
    broker, inst, pf = make_broker(slippage_ticks=1.0)
    broker.buy(contracts=1, limit=14_950.0)
    # bar opens BELOW the limit -> price improves to the open
    bar = Bar(ts=1000, open=14_900.0, high=14_960.0, low=14_890.0, close=14_955.0)
    fills = broker.fill_pending(bar)
    assert fills[0].price == 14_900.0     # better-at-open, NO slippage
    assert pf.position.qty == 1


def test_buy_limit_does_not_fill_when_price_never_reaches():
    broker, inst, pf = make_broker(slippage_ticks=1.0)
    broker.buy(contracts=1, limit=14_950.0)
    bar = Bar(ts=1000, open=15_000.0, high=15_020.0, low=14_980.0, close=15_010.0)
    fills = broker.fill_pending(bar)
    assert fills == []
    assert pf.position.qty == 0
    assert broker.pending is not None     # rests for the next bar


def test_sell_limit_fills_at_limit_or_better_at_open():
    broker, inst, pf = make_broker(slippage_ticks=1.0)
    broker.sell(contracts=1, limit=15_050.0)
    bar = Bar(ts=1000, open=15_000.0, high=15_060.0, low=14_990.0, close=15_040.0)
    fills = broker.fill_pending(bar)
    assert fills[0].price == 15_050.0     # exactly the limit
    assert pf.position.qty == -1

    broker2, _, pf2 = make_broker(slippage_ticks=1.0)
    broker2.sell(contracts=1, limit=15_050.0)
    gap = Bar(ts=1000, open=15_100.0, high=15_120.0, low=15_080.0, close=15_110.0)
    fills2 = broker2.fill_pending(gap)
    assert fills2[0].price == 15_100.0    # better-at-open on favorable gap-up


# ---------------------------------------------------------------------------
# M5.8 — close() queues a flattening market order
# ---------------------------------------------------------------------------

def test_close_long_queues_opposite_market_order():
    broker, inst, pf = make_broker()
    open_long(broker, entry_open=15_000.0, stop=14_900.0)
    order = broker.close()
    assert order is not None
    assert order.side == "sell"
    assert order.contracts == 1
    assert order.kind == "market"
    assert broker.pending is order


def test_close_short_queues_buy():
    broker, inst, pf = make_broker()
    broker.sell(contracts=2, stop=15_200.0)
    entry = Bar(ts=1000, open=15_000.0, high=15_010.0, low=14_990.0, close=15_005.0)
    broker.fill_pending(entry)
    order = broker.close()
    assert order.side == "buy"
    assert order.contracts == 2


def test_close_when_flat_returns_none_and_queues_nothing():
    broker, _, _ = make_broker()
    assert broker.close() is None
    assert broker.pending is None


def test_close_fill_flattens_and_clears_resting_stop():
    broker, inst, pf = make_broker(slippage_ticks=1.0)
    open_long(broker, entry_open=15_000.0, stop=14_900.0)
    broker.close()
    exit_bar = Bar(ts=2000, open=15_050.0, high=15_060.0, low=15_040.0, close=15_055.0)
    fills = broker.fill_pending(exit_bar)
    assert fills[-1].side == "sell"
    assert pf.position.qty == 0
    assert broker.resting_stop is None      # auto-cancel once flat


# ---------------------------------------------------------------------------
# M5.9 — Intrabar stop-first tie-break for a bracket bar (§10.9)
# ---------------------------------------------------------------------------

def test_bracket_same_bar_books_the_stop_first():  # §10.9
    broker, inst, pf = make_broker(slippage_ticks=1.0)
    # long 1 @ ~15,000.25, protective stop 14,900
    open_long(broker, entry_open=15_000.0, stop=14_900.0)
    # strategy also has a profit-target SELL LIMIT at 15,100 pending for next bar
    broker.sell(contracts=1, limit=15_100.0)
    # next bar straddles BOTH: low 14,880 (hits stop) AND high 15,120 (hits target)
    straddle = Bar(ts=2000, open=14_950.0, high=15_120.0, low=14_880.0, close=15_000.0)
    fills = broker.fill_pending(straddle)
    # stop-first: the FIRST fill is the protective sell stop (the loss)
    assert fills[0].side == "sell"
    # worse of (stop=14900, open=14950) -> 14900; - 1 tick = 14,899.75 (the loss exit)
    assert fills[0].price == 14_899.75
    # position is flat after the stop; the target limit must NOT also open a new short
    assert pf.position.qty == 0


def test_target_only_fills_when_stop_not_hit():
    broker, inst, pf = make_broker(slippage_ticks=1.0)
    open_long(broker, entry_open=15_000.0, stop=14_900.0)
    broker.sell(contracts=1, limit=15_100.0)
    # bar reaches the target but never the stop
    bar = Bar(ts=2000, open=15_050.0, high=15_120.0, low=15_010.0, close=15_090.0)
    fills = broker.fill_pending(bar)
    assert len(fills) == 1
    assert fills[0].side == "sell"
    assert fills[0].price == 15_100.0     # target hit, no slippage
    assert pf.position.qty == 0
    assert broker.resting_stop is None    # closing the long auto-cancels the stop


# ---------------------------------------------------------------------------
# M5.10 — Margin gate rejects an entry that fails margin_ok
# ---------------------------------------------------------------------------

def test_entry_rejected_when_margin_not_ok():  # §5.2 margin gate
    # NQ initial margin is far above $500, so a single contract fails the gate
    broker, inst, pf = make_broker(starting_cash=500.0, slippage_ticks=1.0)
    assert pf.margin_ok(1) is False        # precondition
    order = broker.buy(contracts=1)
    bar = Bar(ts=1000, open=15_000.0, high=15_010.0, low=14_990.0, close=15_005.0)
    fills = broker.fill_pending(bar)
    assert fills == []
    assert order.status == "rejected"
    assert broker.pending is None
    assert pf.position.qty == 0
    assert broker.resting_stop is None


def test_entry_allowed_when_margin_ok():
    broker, inst, pf = make_broker(starting_cash=100_000.0)
    assert pf.margin_ok(1) is True
    broker.buy(contracts=1)
    bar = Bar(ts=1000, open=15_000.0, high=15_010.0, low=14_990.0, close=15_005.0)
    fills = broker.fill_pending(bar)
    assert len(fills) == 1
    assert pf.position.qty == 1


def test_exit_not_gated_by_margin():
    # open with enough cash, then drop scenario: a close() must still fill
    broker, inst, pf = make_broker(starting_cash=100_000.0, slippage_ticks=1.0)
    open_long(broker, entry_open=15_000.0, stop=14_900.0)
    order = broker.close()
    bar = Bar(ts=2000, open=15_050.0, high=15_060.0, low=15_040.0, close=15_055.0)
    fills = broker.fill_pending(bar)
    assert order.status == "filled"
    assert len(fills) == 1
    assert pf.position.qty == 0


# ---------------------------------------------------------------------------
# Scale-in margin: an order that adds contracts is accepted only if equity (cash +
# open MTM, marked at the fill bar's open) covers INITIAL margin on the whole
# resulting position, not just the new contracts against cash. NQ IM = 23,000.
# ---------------------------------------------------------------------------

def _flat_bar(ts, px):
    return Bar(ts=ts, open=px, high=px, low=px, close=px)


def test_five_from_flat_still_refused():  # unchanged from-flat behaviour
    broker, inst, pf = make_broker(slippage_ticks=0.0)
    order = broker.buy(contracts=5)               # 115,000 > 100,000
    assert broker.fill_pending(_flat_bar(1000, 100.0)) == []
    assert order.status == "rejected"
    assert pf.position.qty == 0


def test_four_from_flat_still_accepted():  # positive control
    broker, inst, pf = make_broker(slippage_ticks=0.0)
    order = broker.buy(contracts=4)               # 92,000 <= 100,000
    broker.fill_pending(_flat_bar(1000, 100.0))
    assert order.status == "filled"
    assert pf.position.qty == 4


def test_scale_in_counts_the_contracts_already_held():
    # Long 4 (92,000 of IM); 4 more would need 184,000 on 100,000 of equity.
    broker, inst, pf = make_broker(slippage_ticks=0.0)
    broker.buy(contracts=4)
    broker.fill_pending(_flat_bar(1000, 100.0))
    order = broker.buy(contracts=4)
    assert broker.fill_pending(_flat_bar(2000, 100.0)) == []
    assert order.status == "rejected"
    assert pf.position.qty == 4


def test_scale_in_counts_the_open_loss():
    # Long 1 from 100; at the next open (-3,800) it is down 78,000: equity 22,000, and
    # 4 contracts need 92,000. Cash alone (100,000) would have passed 3 new ones.
    broker, inst, pf = make_broker(slippage_ticks=0.0)
    broker.buy(contracts=1)
    broker.fill_pending(_flat_bar(1000, 100.0))
    order = broker.buy(contracts=3)
    assert broker.fill_pending(_flat_bar(2000, -3_800.0)) == []
    assert order.status == "rejected"
    assert pf.position.qty == 1


def test_scale_in_counts_the_open_profit():
    # Cash 30,000, long 1 from 100; at 2,100 it is up 40,000: equity 70,000 covers
    # 3 contracts (69,000), though cash alone covers no 2 new ones (46,000).
    broker, inst, pf = make_broker(starting_cash=30_000.0, slippage_ticks=0.0)
    broker.buy(contracts=1)
    broker.fill_pending(_flat_bar(1000, 100.0))
    order = broker.buy(contracts=2)
    broker.fill_pending(_flat_bar(2000, 2_100.0))
    assert order.status == "filled"
    assert pf.position.qty == 3


def test_scale_in_within_margin_is_accepted():  # positive control
    broker, inst, pf = make_broker(slippage_ticks=0.0)
    broker.buy(contracts=1)
    broker.fill_pending(_flat_bar(1000, 100.0))
    order = broker.buy(contracts=3)               # 4 x 23,000 = 92,000 <= 100,000
    broker.fill_pending(_flat_bar(2000, 100.0))
    assert order.status == "filled"
    assert pf.position.qty == 4


def test_short_scale_in_counts_the_contracts_already_held():
    broker, inst, pf = make_broker(slippage_ticks=0.0)
    broker.sell(contracts=3)
    broker.fill_pending(_flat_bar(1000, 100.0))
    order = broker.sell(contracts=2)              # 5 x 23,000 > 100,000
    assert broker.fill_pending(_flat_bar(2000, 100.0)) == []
    assert order.status == "rejected"
    assert pf.position.qty == -3


def test_reducing_and_closing_are_never_blocked_by_margin():
    # Long 4 from 100, then the market collapses: equity is far below any requirement.
    broker, inst, pf = make_broker(slippage_ticks=0.0)
    broker.buy(contracts=4)
    broker.fill_pending(_flat_bar(1000, 100.0))
    partial = broker.sell(contracts=1)
    broker.fill_pending(_flat_bar(2000, -1_000.0))
    assert partial.status == "filled"
    assert pf.position.qty == 3
    exit_ = broker.close()
    broker.fill_pending(_flat_bar(3000, -1_000.0))
    assert exit_.status == "filled"
    assert pf.position.qty == 0


# ---------------------------------------------------------------------------
# M5.11 — cancel_pending_entry() cancels only the entry, never resting_stop
# ---------------------------------------------------------------------------

def test_cancel_pending_entry_cancels_order_only():
    broker, _, _ = make_broker()
    order = broker.buy(contracts=1)
    broker.cancel_pending_entry()
    assert order.status == "cancelled"
    assert broker.pending is None


def test_cancel_pending_entry_does_not_touch_resting_stop():
    broker, _, _ = make_broker()
    open_long(broker, entry_open=15_000.0, stop=14_900.0)   # resting_stop = 14,900
    broker.buy(contracts=1)                                  # a new pending entry
    broker.cancel_pending_entry()
    assert broker.pending is None
    assert broker.resting_stop == 14_900.0                  # untouched


def test_cancel_pending_entry_noop_when_nothing_pending():
    broker, _, _ = make_broker()
    broker.cancel_pending_entry()        # must not raise
    assert broker.pending is None


def test_cancelled_entry_does_not_fill_next_bar():
    broker, inst, pf = make_broker()
    broker.buy(contracts=1)
    broker.cancel_pending_entry()
    bar = Bar(ts=1000, open=15_000.0, high=15_010.0, low=14_990.0, close=15_005.0)
    assert broker.fill_pending(bar) == []
    assert pf.position.qty == 0


# ---------------------------------------------------------------------------
# M5.12 — Broker public API surface guard
# ---------------------------------------------------------------------------

def test_broker_public_api_surface():
    broker, _, _ = make_broker()
    for name in (
        "buy", "sell", "close", "amend_stop",
        "cancel_pending_entry", "fill_pending",
    ):
        assert callable(getattr(broker, name))
    assert hasattr(broker, "pending")
    assert hasattr(broker, "resting_stop")


# ---------------------------------------------------------------------------
# Entry-bar stops: an attached stop is live from the entry fill, so the entry bar's
# own range is checked against it (adverse extreme first, §5.4).
# ---------------------------------------------------------------------------

def test_entry_bar_pierce_fires_attached_stop_on_the_entry_bar():
    broker, inst, pf = make_broker(slippage_ticks=1.0)
    broker.buy(contracts=1, stop=14_900.0)
    # opens 15,000 (entry), trades down to 14,850 through the 14,900 stop
    bar = Bar(ts=1000, open=15_000.0, high=15_010.0, low=14_850.0, close=14_990.0)
    fills = broker.fill_pending(bar)
    assert [(f.side, f.price) for f in fills] == [("buy", 15_000.25), ("sell", 14_899.75)]
    assert pf.position.qty == 0
    assert broker.resting_stop is None
    assert broker.pending is None
    tr = pf.trades[0]
    assert tr.entry_ts == tr.exit_ts == 1000
    # R reads the attached stop: (14,899.75 - 15,000.25) / (15,000.25 - 14,900)
    assert tr.r_multiple == pytest.approx(-100.5 / 100.25)
    # consumed: nothing fires on the next bar, even one that trades through again
    again = Bar(ts=2000, open=14_950.0, high=14_960.0, low=14_800.0, close=14_850.0)
    assert broker.fill_pending(again) == []


def test_entry_bar_gap_through_fills_stop_at_the_open():
    # The entry bar opens already beyond the stop: the stop fills at the open less
    # slippage, never at the better stop price.
    broker, inst, pf = make_broker(slippage_ticks=1.0)
    broker.buy(contracts=1, stop=14_900.0)
    bar = Bar(ts=1000, open=14_850.0, high=14_870.0, low=14_840.0, close=14_860.0)
    fills = broker.fill_pending(bar)
    assert [(f.side, f.price) for f in fills] == [("buy", 14_850.25), ("sell", 14_849.75)]
    assert pf.position.qty == 0


def test_short_entry_bar_pierce_fires_attached_stop():
    broker, inst, pf = make_broker(slippage_ticks=1.0)
    broker.sell(contracts=2, stop=15_100.0)
    bar = Bar(ts=1000, open=15_000.0, high=15_120.0, low=14_990.0, close=15_010.0)
    fills = broker.fill_pending(bar)
    assert [(f.side, f.contracts, f.price) for f in fills] == [
        ("sell", 2, 14_999.75), ("buy", 2, 15_100.25)]
    assert pf.position.qty == 0


def test_entry_bar_stop_not_touched_stays_resting():  # positive control
    broker, inst, pf = make_broker(slippage_ticks=1.0)
    broker.buy(contracts=1, stop=14_900.0)
    bar = Bar(ts=1000, open=15_000.0, high=15_010.0, low=14_900.25, close=15_005.0)
    fills = broker.fill_pending(bar)
    assert [(f.side, f.price) for f in fills] == [("buy", 15_000.25)]
    assert pf.position.qty == 1
    assert broker.resting_stop == 14_900.0
    assert pf.position.initial_stop == 14_900.0


def test_limit_entry_bar_pierce_fires_attached_stop():
    # A buy limit fills on the way down, so a low beyond the stop came after the fill.
    broker, inst, pf = make_broker(slippage_ticks=1.0)
    broker.buy(contracts=1, limit=14_950.0, stop=14_900.0)
    bar = Bar(ts=1000, open=15_000.0, high=15_010.0, low=14_880.0, close=14_920.0)
    fills = broker.fill_pending(bar)
    assert [(f.side, f.price) for f in fills] == [("buy", 14_950.0), ("sell", 14_899.75)]
    assert pf.position.qty == 0


def test_limit_entry_stop_already_beyond_the_fill_exits_at_the_fill_not_the_open():
    # A stop above a buy limit's fill is live (and beyond) the moment the limit fills,
    # with the market at 14,950: it exits there less slippage, never at the 14,960
    # stop, a price the market had already left when the stop went live.
    broker, inst, pf = make_broker(slippage_ticks=1.0)
    broker.buy(contracts=1, limit=14_950.0, stop=14_960.0)
    bar = Bar(ts=1000, open=15_000.0, high=15_010.0, low=14_940.0, close=14_945.0)
    fills = broker.fill_pending(bar)
    assert [(f.side, f.price) for f in fills] == [("buy", 14_950.0), ("sell", 14_949.75)]
    assert pf.position.qty == 0


def test_flip_bar_checks_the_new_sides_stop_once():
    # Long 1 with stop 14,900 (not touched); sell 2 flips short with stop 15,100,
    # and the flip bar's high (15,150) takes it out. One fill per order, no more.
    broker, inst, pf = make_broker(slippage_ticks=1.0)
    open_long(broker, entry_open=15_000.0, stop=14_900.0)
    broker.sell(contracts=2, stop=15_100.0)
    bar = Bar(ts=2000, open=15_000.0, high=15_150.0, low=14_950.0, close=15_050.0)
    fills = broker.fill_pending(bar)
    assert [(f.side, f.contracts, f.price) for f in fills] == [
        ("sell", 2, 14_999.75), ("buy", 1, 15_100.25)]
    assert pf.position.qty == 0
    assert broker.resting_stop is None
    assert [t.side for t in pf.trades] == ["long", "short"]


def test_flip_bar_new_stop_not_touched_stays_resting():
    broker, inst, pf = make_broker(slippage_ticks=1.0)
    open_long(broker, entry_open=15_000.0, stop=14_900.0)
    broker.sell(contracts=2, stop=15_100.0)
    bar = Bar(ts=2000, open=15_000.0, high=15_050.0, low=14_950.0, close=15_020.0)
    fills = broker.fill_pending(bar)
    assert len(fills) == 1
    assert pf.position.qty == -1
    assert broker.resting_stop == 15_100.0


def test_old_stop_hit_first_resizes_a_pending_flip_to_its_new_side():
    # Adverse first: the held long's stop fires at the top of the bar and flattens it.
    # The pending sell(2) was sized as a flip (close 1 + open 1); its closing half is
    # done, so it opens just the new short 1 at the open. That short's stop (15,100) is
    # then live and the bar's high (15,150) takes it out: one fill per order.
    broker, inst, pf = make_broker(slippage_ticks=1.0)
    open_long(broker, entry_open=15_000.0, stop=14_900.0)
    order = broker.sell(contracts=2, stop=15_100.0)
    bar = Bar(ts=2000, open=14_950.0, high=15_150.0, low=14_850.0, close=15_000.0)
    fills = broker.fill_pending(bar)
    assert [(f.side, f.contracts, f.price) for f in fills] == [
        ("sell", 1, 14_899.75), ("sell", 1, 14_949.75), ("buy", 1, 15_100.25)]
    assert (order.status, order.contracts) == ("filled", 1)
    assert pf.position.qty == 0
    assert broker.resting_stop is None
    assert [t.side for t in pf.trades] == ["long", "short"]


def test_old_stop_hit_first_resized_flip_keeps_its_new_short():
    # Long 2 stopped out; the pending sell(5) opens short 3 with its own stop resting.
    broker, inst, pf = make_broker(slippage_ticks=0.0)
    broker.buy(contracts=2, stop=14_900.0)
    broker.fill_pending(Bar(ts=1000, open=15_000.0, high=15_010.0, low=14_990.0,
                            close=15_000.0))
    order = broker.sell(contracts=5, stop=15_100.0)
    bar = Bar(ts=2000, open=14_950.0, high=15_000.0, low=14_850.0, close=14_900.0)
    fills = broker.fill_pending(bar)
    assert [(f.side, f.contracts, f.price) for f in fills] == [
        ("sell", 2, 14_900.0), ("sell", 3, 14_950.0)]
    assert (order.status, order.contracts) == ("filled", 3)
    assert pf.position.qty == -3
    assert broker.resting_stop == 15_100.0
    assert pf.position.initial_stop == 15_100.0


def test_old_stop_hit_first_cancels_a_pending_pure_exit():  # positive control
    # A pending close() is stale once the stop has flattened the position.
    broker, inst, pf = make_broker(slippage_ticks=0.0)
    open_long(broker, entry_open=15_000.0, stop=14_900.0)
    order = broker.close()
    bar = Bar(ts=2000, open=14_950.0, high=14_960.0, low=14_850.0, close=14_900.0)
    fills = broker.fill_pending(bar)
    assert [(f.side, f.contracts) for f in fills] == [("sell", 1)]
    assert order.status == "cancelled"
    assert pf.position.qty == 0


# ---------------------------------------------------------------------------
# A flip with no new stop: the old position's stop is for the other side, so it is
# cleared, never left resting on the new position.
# ---------------------------------------------------------------------------

def test_market_flip_without_a_new_stop_clears_the_old_stop():
    broker, inst, pf = make_broker(slippage_ticks=0.0)
    broker.buy(contracts=1, stop=95.0)
    broker.fill_pending(Bar(1, 100.0, 101.0, 99.0, 100.0))         # long 1, sell-stop 95
    broker.sell(contracts=2)
    broker.fill_pending(Bar(2, 100.0, 101.0, 99.5, 100.0))         # flip to short 1
    assert pf.position.qty == -1
    assert broker.resting_stop is None
    assert pf.position.initial_stop is None
    # As a buy-stop on the short, the stale 95 would fire on any bar above 95.
    assert broker.fill_pending(Bar(3, 100.0, 100.5, 99.75, 100.25)) == []
    assert pf.position.qty == -1


def test_limit_flip_without_a_new_stop_clears_the_old_stop():
    broker, inst, pf = make_broker(slippage_ticks=0.0)
    broker.buy(contracts=1, stop=95.0)
    broker.fill_pending(Bar(1, 100.0, 101.0, 99.0, 100.0))
    broker.sell(contracts=2, limit=101.0)
    broker.fill_pending(Bar(2, 100.0, 102.0, 99.5, 101.0))         # limit fills at 101
    assert pf.position.qty == -1
    assert broker.resting_stop is None
    assert broker.fill_pending(Bar(3, 100.0, 100.5, 99.75, 100.25)) == []
    assert pf.position.qty == -1


def test_scale_in_without_a_new_stop_keeps_the_stop():  # positive control
    broker, inst, pf = make_broker(slippage_ticks=0.0)
    broker.buy(contracts=1, stop=95.0)
    broker.fill_pending(Bar(1, 100.0, 101.0, 99.0, 100.0))
    broker.buy(contracts=1)
    broker.fill_pending(Bar(2, 100.0, 101.0, 99.0, 100.0))
    assert pf.position.qty == 2
    assert broker.resting_stop == 95.0


def test_partial_close_keeps_the_stop():  # positive control
    broker, inst, pf = make_broker(slippage_ticks=0.0)
    broker.buy(contracts=2, stop=95.0)
    broker.fill_pending(Bar(1, 100.0, 101.0, 99.0, 100.0))
    broker.sell(contracts=1)
    broker.fill_pending(Bar(2, 100.0, 101.0, 99.0, 100.0))
    assert pf.position.qty == 1
    assert broker.resting_stop == 95.0


# ---------------------------------------------------------------------------
# allow_short=False: an order that would leave a net short is refused; a sell that only
# closes a long goes through, and a flip keeps just that closing half.
# ---------------------------------------------------------------------------

def test_no_shorts_refuses_a_sell_from_flat():
    broker, inst, pf = make_broker(slippage_ticks=0.0, allow_short=False)
    order = broker.sell(contracts=1, stop=110.0)
    assert broker.fill_pending(Bar(1, 100.0, 101.0, 99.0, 100.0)) == []
    assert order.status == "rejected"
    assert broker.pending is None
    assert pf.position.qty == 0
    assert broker.resting_stop is None


def test_shorts_allowed_by_default_sell_from_flat_fills():  # positive control
    broker, inst, pf = make_broker(slippage_ticks=0.0)
    order = broker.sell(contracts=1)
    broker.fill_pending(Bar(1, 100.0, 101.0, 99.0, 100.0))
    assert order.status == "filled"
    assert pf.position.qty == -1


def test_no_shorts_still_closes_a_long():  # positive control
    broker, inst, pf = make_broker(slippage_ticks=0.0, allow_short=False)
    broker.buy(contracts=2, stop=90.0)
    broker.fill_pending(Bar(1, 100.0, 101.0, 99.0, 100.0))
    partial = broker.sell(contracts=1)
    broker.fill_pending(Bar(2, 100.0, 101.0, 99.0, 100.0))
    assert partial.status == "filled" and pf.position.qty == 1
    exit_ = broker.close()
    broker.fill_pending(Bar(3, 100.0, 101.0, 99.0, 100.0))
    assert exit_.status == "filled" and pf.position.qty == 0


def test_no_shorts_trims_a_flip_to_its_closing_half():
    broker, inst, pf = make_broker(slippage_ticks=0.0, allow_short=False)
    broker.buy(contracts=1, stop=90.0)
    broker.fill_pending(Bar(1, 100.0, 101.0, 99.0, 100.0))
    order = broker.sell(contracts=3, stop=110.0)
    fills = broker.fill_pending(Bar(2, 100.0, 101.0, 99.0, 100.0))
    assert [(f.side, f.contracts) for f in fills] == [("sell", 1)]
    assert (order.status, order.contracts, order.attached_stop) == ("filled", 1, None)
    assert pf.position.qty == 0
    assert broker.resting_stop is None


def test_no_shorts_refuses_a_flip_resized_by_a_stop_out():
    # The long's stop closes it; what is left of the pending sell(2) is a short entry.
    broker, inst, pf = make_broker(slippage_ticks=0.0, allow_short=False)
    broker.buy(contracts=1, stop=95.0)
    broker.fill_pending(Bar(1, 100.0, 101.0, 99.0, 100.0))
    order = broker.sell(contracts=2)
    fills = broker.fill_pending(Bar(2, 96.0, 96.0, 94.0, 94.0))
    assert [(f.side, f.contracts, f.price) for f in fills] == [("sell", 1, 95.0)]
    assert order.status == "rejected"
    assert pf.position.qty == 0
