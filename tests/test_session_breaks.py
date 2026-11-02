"""break_on="session" (wave 1): only a trading-session change is a break;
pending exits survive it; orders are whole contracts."""
import numpy as np
import pandas as pd
import pytest

from plutobt.broker import Broker, FrameError, Order, OrderError
from plutobt.config import BacktestConfig
from plutobt.engine import Engine
from plutobt.instruments import get_instrument
from plutobt.portfolio import Fill, Portfolio
from plutobt.strategy import Strategy

NQ = get_instrument("NQ")
T0 = 1_700_000_000


class Scripted(Strategy):
    """Acts on bar indices: {index: "buy" | "close" | (side, contracts)}."""

    def __init__(self, plan):
        self.plan = plan
        self.i = -1

    def precompute(self, df):
        return df

    def on_bar(self, ctx):
        self.i += 1
        act = self.plan.get(self.i)
        if act == "buy":
            ctx.broker.buy(contracts=1)
        elif act == "close":
            ctx.broker.close()
        elif isinstance(act, tuple):
            side, contracts = act
            getattr(ctx.broker, side)(contracts=contracts)


def _frame(offsets, sessions=None):
    ts = [T0 + o for o in offsets]
    px = [100.0 + i for i in range(len(ts))]
    df = pd.DataFrame({"ts": pd.Series(ts, dtype="int64"), "open": px, "high": [p + 1 for p in px],
                       "low": [p - 1 for p in px], "close": px,
                       "volume": pd.Series([10] * len(ts), dtype="int64")})
    if sessions is not None:
        df["session"] = pd.Series(sessions, dtype="int64")
    return df


def _run(df, plan, **cfg):
    return Engine(NQ, BacktestConfig(periods_per_year=252.0, **cfg)).run(df, Scripted(plan))


BREAK = [0, 60, 120, 180, 240]   # with sessions [1, 1, 1, 2, 2]: a break after bar 2


def test_a_quiet_stretch_inside_a_session_is_not_a_break():
    df = _frame([0, 60, 120, 300, 360], sessions=[1] * 5)   # nobody traded 180-240
    kept = _run(df, {2: "buy"}, break_on="session").trades
    assert [t.entry_ts for t in kept] == [T0 + 300]
    assert _run(df, {2: "buy"}).trades == []                 # the default gap rule drops it


def test_a_session_change_cancels_a_pending_entry():
    # CHARACTERIZATION PIN (passes before this task: the old session rule already
    # cancels entries at a session change; mutation-proved in Step 6).
    df = _frame(BREAK, sessions=[1, 1, 1, 2, 2])
    assert _run(df, {2: "buy"}, break_on="session").trades == []


# The same break, as each mode sees it: session mode reads the session change after
# bar 2; gap mode (the default) reads the 480 s gap after bar 2 (the bars are 60 s).
MODES = {
    "session": (lambda: _frame(BREAK, sessions=[1, 1, 1, 2, 2]), {"break_on": "session"}),
    "gap": (lambda: _frame([0, 60, 120, 600, 660]), {}),
}
AFTER_BREAK = {"session": T0 + 180, "gap": T0 + 600}   # the next session's first bar


@pytest.mark.parametrize("mode", MODES)
def test_a_pending_close_survives_a_break_in_either_mode(mode):
    # gap mode used to cancel the exit too (fixed 2026-11-02), so a close decided on a
    # session's last bar was dropped and the position was held overnight and beyond.
    make, cfg = MODES[mode]
    (t,) = _run(make(), {0: "buy", 2: "close"}, **cfg).trades
    assert t.exit_ts == AFTER_BREAK[mode] and not t.open_at_end


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("entry,exit_", [("buy", ("sell", 1)), (("sell", 1), ("buy", 1))])
def test_a_plain_exit_order_survives_a_break(mode, entry, exit_):
    # sell(1) exiting a long and buy(1) covering a short are exits too, not
    # only close() (money re-audit, 2026-09-26).
    make, cfg = MODES[mode]
    (t,) = _run(make(), {0: entry, 2: exit_}, **cfg).trades
    assert t.exit_ts == AFTER_BREAK[mode] and not t.open_at_end


@pytest.mark.parametrize("mode", MODES)
def test_a_pending_entry_is_cancelled_at_a_break_in_either_mode(mode):
    make, cfg = MODES[mode]
    assert _run(make(), {2: "buy"}, **cfg).trades == []
    # carry_overnight=True carries it to the next session's open instead.
    (t,) = _run(make(), {2: "buy"}, carry_overnight=True, **cfg).trades
    assert t.entry_ts == AFTER_BREAK[mode]


@pytest.mark.parametrize("mode", MODES)
def test_a_flip_at_a_break_keeps_only_its_closing_contracts(mode):
    flip = {0: "buy", 2: ("sell", 2)}
    same = _run(_frame(BREAK, sessions=[1] * 5), flip, break_on="session").trades
    assert [t.side for t in same] == ["long", "short"]          # no break: a real flip
    make, cfg = MODES[mode]
    (t,) = _run(make(), flip, **cfg).trades
    assert t.side == "long" and t.exit_ts == AFTER_BREAK[mode] and not t.open_at_end
    # flat after the break, no short opened across it


def test_a_flip_whose_new_side_fails_margin_keeps_only_its_exit():
    # PRE-EXISTING (money re-review, 2026-09-27): _is_reducing checks the side
    # only, so a flip skipped the margin check and long 1 then sell(50) opened a
    # 49-contract short on $100k. The exit part always goes through; the new
    # position's part must pass margin like any entry.
    flip = {0: "buy", 2: ("sell", 50)}
    (t,) = _run(_frame(BREAK, sessions=[1] * 5), flip, break_on="session").trades
    assert t.side == "long" and not t.open_at_end


def test_session_mode_needs_a_session_column():
    with pytest.raises(ValueError, match="session"):
        _run(_frame([0, 60, 120]), {}, break_on="session")


def test_an_unknown_break_mode_is_refused():
    with pytest.raises(ValueError, match="break_on"):
        _run(_frame([0, 60, 120], sessions=[1, 1, 1]), {}, break_on="tick")


@pytest.mark.parametrize("bad", [0, -1, 1.5, True, "1"])
def test_orders_are_whole_contracts(bad):
    broker = Broker(NQ, BacktestConfig(), Portfolio(starting_cash=100_000.0, instrument=NQ))
    with pytest.raises(OrderError):
        broker.buy(contracts=bad)


def test_numpy_integers_are_whole_contracts():
    # CHARACTERIZATION PIN (passes before this task; mutation-proved in Step 6).
    broker = Broker(NQ, BacktestConfig(), Portfolio(starting_cash=100_000.0, instrument=NQ))
    assert broker.buy(contracts=np.int64(2)).contracts == 2


# ---------------------------------------------------------------------------
# Approved additions (Task 8 review): every fill is checked, the portfolio refuses a
# fill of fewer than 1 contract, and the commission must be a finite number.
# ---------------------------------------------------------------------------


class ResizedAfterPlacing(Strategy):
    """Buys 1, then places sell(1) to exit and changes that Order's size before it
    fills (sizing code run after placing: an Order is a plain dataclass)."""

    def __init__(self, resize):
        self.resize = resize
        self.i = -1
        self.broker = None

    def on_bar(self, ctx):
        self.i += 1
        self.broker = ctx.broker
        if self.i == 0:
            ctx.broker.buy(contracts=1)
        elif self.i == 2:
            ctx.broker.sell(contracts=1).contracts = self.resize


@pytest.mark.parametrize("resize", [int(0.5), 0.5, -1], ids=["int(0.5)", "0.5", "-1"])
def test_an_order_changed_after_placing_is_refused_at_the_fill(resize):
    # A fill reads order.contracts when it FILLS, after the placement check. Unchecked,
    # int(0.5) divided the exit fee by zero (ZeroDivisionError), 0.5 booked half a
    # contract, and -1 bought a second contract instead of selling the first.
    strategy = ResizedAfterPlacing(resize)
    engine = Engine(NQ, BacktestConfig(periods_per_year=252.0, commission_per_side=2.5))
    with pytest.raises(OrderError, match="cannot fill the sell order"):
        engine.run(_frame(BREAK), strategy)
    book = strategy.broker.portfolio
    assert book.trades == [] and book.position.qty == 1         # nothing booked...
    assert book.cash == pytest.approx(100_000.0 - 2.5)          # ...past the entry's fee


@pytest.mark.parametrize("contracts", [0, 0.5, True])
def test_an_order_queued_straight_into_pending_is_refused_at_the_fill(contracts):
    broker = Broker(NQ, BacktestConfig(), Portfolio(starting_cash=100_000.0, instrument=NQ))
    broker.pending = Order(side="buy", contracts=contracts, kind="market")
    with pytest.raises(OrderError, match="cannot fill the buy order"):
        broker.fill_pending(_frame([0]).iloc[0])
    assert broker.portfolio.trades == [] and broker.portfolio.position.qty == 0


def test_a_numpy_integer_order_fills():
    # CHARACTERIZATION PIN (passes before the fill-time check; mutation-proved: the size
    # rule reading isinstance(contracts, int) goes red).
    (t,) = _run(_frame(BREAK), {0: ("buy", np.int64(2)), 2: "close"}).trades
    assert t.contracts == 2 and not t.open_at_end


@pytest.mark.parametrize("contracts", [0, -1])
def test_the_portfolio_refuses_a_fill_of_fewer_than_one_contract(contracts):
    # The last line of defence behind the broker. Unchecked, 0 contracts against a held
    # position divided the exit fee by zero, and a sell of -1 bought a second contract.
    p = Portfolio(starting_cash=100_000.0, instrument=NQ)
    p.apply_fill(Fill(ts=1, side="buy", contracts=1, price=100.0, commission=2.5))
    with pytest.raises(ValueError, match="1 or more contracts"):
        p.apply_fill(Fill(ts=2, side="sell", contracts=contracts, price=101.0, commission=0.0))
    assert p.trades == [] and p.position.qty == 1
    assert p.cash == pytest.approx(100_000.0 - 2.5)


@pytest.mark.parametrize("fee", [float("nan"), float("inf"), float("-inf")])
def test_a_non_finite_commission_is_refused(fee):
    # TOML can hold nan and inf. Unchecked, a NaN fee ran without an error and reported
    # Sharpe 0.0 and expectancy 0.0, and a -inf fee a 100% win rate.
    with pytest.raises(ValueError, match="commission_per_side"):
        _run(_frame(BREAK), {0: "buy", 2: "close"}, commission_per_side=fee)


def test_a_negative_commission_is_a_rebate_and_runs():
    # CHARACTERIZATION PIN (passes before the finiteness check; mutation-proved: a check
    # that also refuses a negative fee goes red).
    (t,) = _run(_frame(BREAK), {0: "buy", 2: "close"}, commission_per_side=-0.5).trades
    assert t.commission == pytest.approx(-1.0)                   # both sides rebated
    assert t.net_pnl == pytest.approx(t.pnl + 1.0)


# ---------------------------------------------------------------------------
# Task 9 review (2026-09-28): a flip's margin sees its closing loss, an exit leaves no
# stale stop, session IDs are checked, and an order's size is checked first and named.
# ---------------------------------------------------------------------------


class Plan(Strategy):
    """Acts on bar indices: {index: callable(broker)}, keeping what each call returned."""

    def __init__(self, plan):
        self.plan, self.i, self.orders = plan, -1, {}

    def on_bar(self, ctx):
        self.i += 1
        act = self.plan.get(self.i)
        if act is not None:
            self.orders[self.i] = act(ctx.broker)


def _closes(closes, sessions=None):
    """A bar a minute at these closes (open = close, high and low a point away)."""
    n = len(closes)
    df = pd.DataFrame({"ts": pd.Series([T0 + 60 * i for i in range(n)], dtype="int64"),
                       "open": closes, "high": [c + 1 for c in closes],
                       "low": [c - 1 for c in closes], "close": closes,
                       "volume": pd.Series([1] * n, dtype="int64")})
    df["session"] = pd.Series(sessions or [1] * n, dtype="int64")
    return df


def _run_plan(df, plan, **cfg):
    """Session mode, no slippage; NQ's initial margin is $23,000 a contract on $100,000.
    Returns (trades, {bar index: what the plan's call returned there})."""
    strategy = Plan(plan)
    cfg = {"periods_per_year": 252.0, "slippage_ticks": 0.0, "break_on": "session", **cfg}
    return Engine(NQ, BacktestConfig(**cfg)).run(df, strategy).trades, strategy.orders


def _seen(trades):
    return [(t.side, t.contracts, t.exit_ts, t.open_at_end) for t in trades]


def test_a_flip_sizes_its_new_side_from_the_equity_its_closing_half_leaves():
    # Long 1 NQ from 3,000 is down $40,000 at the flip's open (1,000), leaving $60,000,
    # and 4 new contracts need $92,000. Checked against the cash from before that loss,
    # sell(5) opened short 4, which close() then sell(4) refuses: now both end flat.
    df = _closes([3000.0, 3000.0, 1000.0, 1000.0, 1000.0, 1000.0])
    flip, _ = _run_plan(df, {0: lambda b: b.buy(1), 2: lambda b: b.sell(5)})
    apart, _ = _run_plan(df, {0: lambda b: b.buy(1), 2: lambda b: b.close(),
                              3: lambda b: b.sell(4)})
    assert _seen(flip) == _seen(apart) == [("long", 1, T0 + 180, False)]


def test_a_flip_opens_exactly_what_margin_allows():
    # CHARACTERIZATION PIN (review candidate). 4 new contracts need $92,000 (fits
    # $100,000); 5 need $115,000 (does not). Mutation-proved: opening = the whole order,
    # and opening one short of order.contracts - held, go red.
    df = _closes([100.0] * 5)
    fits, _ = _run_plan(df, {0: lambda b: b.buy(1), 2: lambda b: b.sell(5)})
    assert [(t.side, t.contracts) for t in fits] == [("long", 1), ("short", 4)]
    over, _ = _run_plan(df, {0: lambda b: b.buy(1), 2: lambda b: b.sell(6)})
    assert [(t.side, t.contracts, t.open_at_end) for t in over] == [("long", 1, False)]


def test_a_partial_exit_stays_partial_on_an_account_below_initial_margin():
    # CHARACTERIZATION PIN (review candidate). Long 3 from 5,000, the market falls to
    # 1,000: selling 1 realizes -$80,000, so cash ($20,000) is below one contract's
    # initial margin. The next sell(1) still exits exactly one contract; it opens
    # nothing, so margin does not apply to it. Mutation-proved: opening = the whole order.
    # on_margin_call="record": the account is far below maintenance, and the default
    # forced liquidation would close all three before the partial exits are tested.
    df = _closes([5000.0, 5000.0, 1000.0, 1000.0, 1000.0, 1000.0])
    trades, _ = _run_plan(df, {0: lambda b: b.buy(3), 2: lambda b: b.sell(1),
                               3: lambda b: b.sell(1)}, on_margin_call="record")
    closed = [t.contracts for t in trades if not t.open_at_end]
    held = [t.contracts for t in trades if t.open_at_end]
    assert closed == [1, 1] and held == [1]


def test_a_flip_trimmed_at_a_break_leaves_no_stop_behind():
    # CHARACTERIZATION PIN (review candidate). sell(2, stop=...) while long 1 is trimmed
    # to an exit at the break; its stop was for the short that never opened, so the next
    # position's R uses that position's own stop, and the Order reads as the exit it
    # became. Mutation-proved: the trim keeping the Order's stop goes red.
    df = _closes([100.0] * 7 + [110.0] * 2, sessions=[1, 1, 1, 2, 2, 2, 2, 2, 2])
    trades, orders = _run_plan(df, {0: lambda b: b.buy(1, stop=90.0),
                                    2: lambda b: b.sell(2, stop=50.0),
                                    4: lambda b: b.buy(1, stop=95.0), 6: lambda b: b.close()})
    first, second = trades
    assert first.side == "long" and not first.open_at_end
    assert second.pnl == 200.0 and second.r_multiple == 200.0 / ((100.0 - 95.0) * 20.0)
    assert (orders[2].contracts, orders[2].attached_stop, orders[2].status) == (1, None, "filled")


def test_a_flip_trimmed_by_margin_leaves_no_stop_behind():
    # CHARACTERIZATION PIN (review candidate): the same for a flip whose new side fails
    # margin. Mutation-proved: the trim keeping the Order's stop goes red.
    df = _closes([100.0] * 7 + [110.0] * 2)
    trades, orders = _run_plan(df, {0: lambda b: b.buy(1, stop=90.0),
                                    2: lambda b: b.sell(50, stop=150.0),
                                    4: lambda b: b.buy(1, stop=95.0), 6: lambda b: b.close()})
    first, second = trades
    assert first.side == "long" and not first.open_at_end
    assert second.pnl == 200.0 and second.r_multiple == 200.0 / ((100.0 - 95.0) * 20.0)
    assert (orders[2].contracts, orders[2].attached_stop, orders[2].status) == (1, None, "filled")


@pytest.mark.parametrize("exit_", [lambda b: b.sell(1, stop=50.0),
                                   lambda b: b.sell(1, limit=100.0, stop=50.0)],
                         ids=["market", "limit"])
def test_an_exit_that_flattens_leaves_no_stop_for_the_next_trade(exit_):
    # PRE-EXISTING (review, 2026-09-28): an exit order's stop= was written onto the flat
    # position its fill left, so the NEXT trade's R used it: 0.2 for a trade with no
    # stop at all, and a stop of 50 in place of the next trade's own 95.
    df = _closes([100.0] * 7 + [110.0] * 2)
    plain, _ = _run_plan(df, {0: lambda b: b.buy(1, stop=90.0), 2: exit_,
                              4: lambda b: b.buy(1), 6: lambda b: b.close()})
    assert plain[1].r_multiple is None
    own, _ = _run_plan(df, {0: lambda b: b.buy(1, stop=90.0), 2: exit_,
                            4: lambda b: b.buy(1, stop=95.0), 6: lambda b: b.close()})
    assert own[1].r_multiple == 200.0 / ((100.0 - 95.0) * 20.0)


_SIZE_RULE = "must be a whole number of contracts (an int), 1 or more; got "


@pytest.mark.parametrize(("size", "tail"), [
    (4.0, "4.0 (float): wrap the size in int(...)"),
    (True, "True (bool): pass a number of contracts, not True or False"),
    (0, "0 (int): place no order when the size works out to 0"),
    (-2, "-2 (int): a size is never negative; buy() and sell() pick the side"),
], ids=["float", "bool", "zero", "negative"])
def test_a_bad_size_names_the_side_the_value_its_type_and_the_fix(size, tail):
    # Refused, never coerced, so 4.0 and 4.0000001 behave the same (review, 2026-09-28).
    # At the fill the ts reads as an int, though the bar carries it as a float.
    broker = Broker(NQ, BacktestConfig(), Portfolio(starting_cash=100_000.0, instrument=NQ))
    with pytest.raises(OrderError) as placed:
        broker.buy(contracts=size)
    assert str(placed.value) == "buy order size " + _SIZE_RULE + tail
    broker.pending = Order(side="sell", contracts=size, kind="market")
    with pytest.raises(OrderError) as filled:
        broker.fill_pending(_frame([0]).iloc[0])
    assert str(filled.value) == f"cannot fill the sell order at ts {T0}: its size " + (
        _SIZE_RULE + tail)


def test_a_bad_size_on_an_exit_is_refused_before_a_break_trims_it():
    # Unchecked, the break's trim read 1.5 and cut it to the 1 contract held: a bad size
    # silently repaired (review, 2026-09-28).
    engine = Engine(NQ, BacktestConfig(break_on="session"))
    with pytest.raises(OrderError, match=r"at a break, the pending sell order's size .*1\.5"):
        engine.run(_frame(BREAK, sessions=[1, 1, 1, 2, 2]), ResizedAfterPlacing(1.5))


@pytest.mark.parametrize("size", [None, 60.5], ids=["None", "60.5"])
def test_a_bad_size_is_refused_before_the_margin_check_reads_it(size):
    # Unchecked, None raised a bare TypeError in the margin check, and 60.5 (a flip too
    # big for margin) was cut to the 1 contract held: a bad size silently repaired.
    with pytest.raises(OrderError, match="cannot fill the sell order"):
        Engine(NQ, BacktestConfig()).run(_frame(BREAK), ResizedAfterPlacing(size))


def test_a_resting_stop_survives_a_session_break_and_fills_at_a_gapped_open():
    # CHARACTERIZATION PIN (review, 2026-09-28): a break cancels a pending entry only; a
    # resting protective stop stays, and a gap through it fills at the open, not at the
    # stop. Mutation-proved: clearing the resting stop at a session break goes red.
    df = _closes([100.0, 100.0, 100.0, 90.0, 90.0], sessions=[1, 1, 1, 2, 2])
    (t,), _ = _run_plan(df, {0: lambda b: b.buy(1, stop=95.0)})
    assert (t.exit_ts, t.exit_price, t.open_at_end) == (T0 + 180, 90.0, False)


@pytest.mark.parametrize("labels", [
    ["ETH", "ETH", "ETH", "ETH", "ETH", "RTH", "RTH"],
    ["ETH", "RTH", "RTH", "ETH", "ETH", "RTH", "RTH"],
], ids=["weekend", "two-days"])
def test_session_mode_refuses_rth_eth_labels(labels):
    # plutobt's own tag_sessions writes RTH/ETH; the first list is what it writes from
    # Friday 15:55 CT to Monday 08:31 CT, and it never decreases as text. Read as
    # session IDs, a Friday 15:59 CT entry filled at Sunday's open (review, 2026-09-28).
    df = _frame([60 * i for i in range(7)])
    df["session"] = pd.Series(labels)
    with pytest.raises(ValueError) as e:
        _run(df, {1: "buy"}, break_on="session")
    assert "never decreases (e.g. the CME trading day as an int YYYYMMDD), not RTH/ETH" in (
        str(e.value))
    assert f"row 0 (ts {T0}) holds 'ETH' (str), not a number" in str(e.value)


def test_session_mode_refuses_a_session_id_that_decreases():
    df = _frame(BREAK, sessions=[20260925, 20260925, 20260924, 20260924, 20260924])
    with pytest.raises(ValueError) as e:
        _run(df, {1: "buy"}, break_on="session")
    assert "never decreases" in str(e.value)
    assert f"row 2 (ts {T0 + 120}) holds 20260924, below the 20260925 before it" in str(e.value)


@pytest.mark.parametrize("session", [
    pd.Series([1.0, 1.0, float("nan"), 1.0, 1.0]),
    pd.Series([1, 1, None, 1, 1], dtype="Int64"),
    pd.Series([1, 1, None, 1, 1], dtype=object),
], ids=["nan", "pd.NA", "None"])
def test_session_mode_refuses_a_bar_with_no_session_id(session):
    # Unchecked, NaN silently cancelled every entry, pd.NA crashed, and None never broke.
    df = _frame(BREAK)
    df["session"] = session
    with pytest.raises(ValueError, match=rf"row 2 \(ts {T0 + 120}\) has no session ID"):
        _run(df, {1: "buy"}, break_on="session")


class Precompute(Strategy):
    """A strategy whose precompute is `fn`; with none, the engine must never call it."""

    def __init__(self, fn=None):
        self.fn = fn

    def precompute(self, df):
        assert self.fn is not None, "the engine ran the strategy's precompute"
        return self.fn(df)


@pytest.mark.parametrize("cfg", [{"break_on": "tick"},
                                 {"commission_per_side": float("nan")},
                                 {"slippage_ticks": float("inf")},
                                 {"periods_per_year": float("nan")}],
                         ids=["break_on", "commission", "slippage", "periods"])
def test_a_bad_config_is_refused_before_the_strategy_runs(cfg):
    with pytest.raises(ValueError, match=next(iter(cfg))):
        Engine(NQ, BacktestConfig(**cfg)).run(_frame(BREAK, sessions=[1] * 5), Precompute())


def test_a_bad_session_column_in_the_callers_frame_is_a_plain_value_error():
    df = _frame(BREAK)
    df["session"] = pd.Series([1.0, 1.0, float("nan"), 1.0, 1.0])
    with pytest.raises(ValueError, match="has no session ID") as e:
        Engine(NQ, BacktestConfig(break_on="session")).run(df, Precompute())
    assert not isinstance(e.value, FrameError)       # the caller's frame, before any strategy


@pytest.mark.parametrize("breaks_it", [
    lambda df: df.drop(columns=["session"]),
    lambda df: df.assign(session=df["session"].where(df.index != 2)),
    lambda df: df.assign(session=df["session"].to_numpy()[::-1]),
], ids=["removed", "emptied", "reversed"])
def test_a_precompute_that_breaks_the_session_column_raises_frame_error(breaks_it):
    # A FrameError names the strategy's precompute, so PlutoDesk can answer it with a 422.
    df = _frame(BREAK, sessions=[1, 1, 1, 2, 2])
    with pytest.raises(FrameError,
                       match="the strategy's precompute removed or broke the session column"):
        Engine(NQ, BacktestConfig(break_on="session")).run(df, Precompute(breaks_it))


@pytest.mark.parametrize("name", ["slippage_ticks", "periods_per_year"])
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_a_non_finite_slippage_or_periods_per_year_is_refused(name, bad):
    # TOML can hold nan and inf (review, 2026-09-28); None is each one's default.
    with pytest.raises(ValueError, match=name):
        Engine(NQ, BacktestConfig(**{name: bad})).run(_frame(BREAK), Scripted({}))


def test_slippage_and_periods_per_year_may_be_left_to_their_defaults():
    # CHARACTERIZATION PIN: None is the instrument's slippage (NQ: 1 tick) and a
    # periods_per_year derived later. Mutation-proved: the finiteness check without its
    # None guard goes red.
    cfg = BacktestConfig(slippage_ticks=None, periods_per_year=None)
    (t,) = Engine(NQ, cfg).run(_frame(BREAK), Scripted({0: "buy", 2: "close"})).trades
    assert t.entry_price == 101.0 + 0.25


# ---------------------------------------------------------------------------
# Task 9 re-review (2026-09-28): every fill checks its size, the strategy gets a copy
# of the position, the flip margin reads the fill bar's open, bars stay in time order.
# ---------------------------------------------------------------------------


def _bars(rows, sessions=None):
    """A bar a minute from (open, close) pairs, high and low a point outside both."""
    n = len(rows)
    df = pd.DataFrame({"ts": pd.Series([T0 + 60 * i for i in range(n)], dtype="int64"),
                       "open": [float(o) for o, _ in rows],
                       "high": [max(o, c) + 1.0 for o, c in rows],
                       "low": [min(o, c) - 1.0 for o, c in rows],
                       "close": [float(c) for _, c in rows],
                       "volume": pd.Series([1] * n, dtype="int64")})
    df["session"] = pd.Series(sessions or [1] * n, dtype="int64")
    return df


@pytest.mark.parametrize(("closes", "then", "qty"), [
    ([100.0, 100.0, 100.0, 80.0, 80.0], None, 0.5),
    ([100.0, 100.0, 100.0, 80.0, 80.0], None, 2.5),
    ([100.0] * 5, lambda b: b.sell(60), 2.5),
], ids=["stop-0.5", "stop-2.5", "margin-trim-2.5"])
def test_a_fill_sized_from_the_position_is_checked_too(closes, then, qty):
    # Re-review candidate: the resting stop fills abs(position.qty) and the margin trim
    # fills the qty held, sizes no order check sees. A strategy writing the book's qty
    # booked 2.5 contracts, or hit the portfolio's bare ValueError on 0.5 (a 500 in
    # PlutoDesk). The book stays reachable as ctx.broker.portfolio, so it writes there.
    def act(broker):
        broker.portfolio.position.qty = qty
        return then(broker) if then is not None else None

    with pytest.raises(OrderError, match=rf"cannot fill the sell at ts {T0 + 180}: its size"):
        _run_plan(_closes(closes), {0: lambda b: b.buy(1, stop=95.0), 2: act})


def test_a_flips_margin_is_marked_at_the_fill_bars_open_not_its_close():
    # CHARACTERIZATION PIN (re-review candidate): every other test bar has open == close,
    # so a check marked at the fill bar's CLOSE (look-ahead) passed them all. The flip
    # fills at bar 3's open (1,000: long 1 from 3,000 is down $40,000, leaving $60,000,
    # and 4 new contracts need $92,000); bar 3 then closes back at 3,000, which the fill
    # could not have known. Mutation-proved: the check marked at next_bar.close goes red.
    df = _bars([(3000, 3000), (3000, 3000), (3000, 3000), (1000, 3000), (3000, 3000)])
    trades, _ = _run_plan(df, {0: lambda b: b.buy(1), 2: lambda b: b.sell(5)})
    assert [(t.side, t.contracts, t.open_at_end) for t in trades] == [("long", 1, False)]


class CtxPlan(Strategy):
    """Acts on bar indices: {index: callable(ctx)}."""

    def __init__(self, plan):
        self.plan, self.i = plan, -1

    def on_bar(self, ctx):
        self.i += 1
        act = self.plan.get(self.i)
        if act is not None:
            act(ctx)


def test_writing_ctx_position_cannot_erase_an_open_loss():
    # PRE-EXISTING since v0.1 (re-review, 2026-09-28): ctx.position WAS the book's own
    # Position, so ctx.position.qty = 0 (a strategy's idea of an exit) erased an open
    # -$400 loss: no trade, and equity back at $100,000. It is a copy for the bar now.
    def zero_it(ctx):
        ctx.position.qty = 0

    cfg = BacktestConfig(periods_per_year=252.0, slippage_ticks=0.0)
    result = Engine(NQ, cfg).run(_closes([100.0, 100.0, 90.0, 80.0, 80.0, 80.0]),
                                 CtxPlan({0: lambda c: c.broker.buy(1), 3: zero_it}))
    (t,) = result.trades
    assert t.open_at_end and t.pnl == -400.0
    assert result.equity_curve.iloc[-1] == 100_000.0 - 400.0


class TrailByWriting(Strategy):
    """Buys 1 with a stop at 90, then 'trails' by writing ctx.position.initial_stop each
    bar (the wrong API; ctx.broker.amend_stop is the right one) and closes on bar 3."""

    def __init__(self):
        self.i, self.stops = -1, []

    def on_bar(self, ctx):
        self.i += 1
        if self.i == 0:
            ctx.broker.buy(1, stop=90.0)
        elif ctx.position.qty > 0:
            ctx.position.initial_stop = float(ctx.bar.close) - 0.25
            self.stops.append(ctx.broker.resting_stop)
            if self.i == 3:
                ctx.broker.close()


def test_writing_ctx_position_initial_stop_moves_neither_the_stop_nor_r():
    # PRE-EXISTING since v0.1 (re-review, 2026-09-28): the write rewrote the trade's R
    # denominator (0.05 read as 2.0) while the real stop never moved.
    strategy = TrailByWriting()
    cfg = BacktestConfig(periods_per_year=252.0, slippage_ticks=0.0)
    (t,) = Engine(NQ, cfg).run(_closes([100.0, 100.0, 100.25, 100.5, 100.5, 100.5]),
                               strategy).trades
    assert t.r_multiple == t.pnl / ((100.0 - 90.0) * 20.0)       # 0.05, on the 90 stop
    assert strategy.stops == [90.0, 90.0, 90.0]                   # the real stop stayed


# How a frame's bar timestamps go wrong, and the problem its refusal names. A NaN makes
# the whole ts column float, so the stamps beside it print as floats.
_BAD_TS = {
    "reversed": (lambda df: df.iloc[[4, 3, 2, 1, 0]],
                 f"row 1 (ts {T0 + 180}) is not after row 0 (ts {T0 + 240})"),
    "shuffled": (lambda df: df.iloc[[2, 0, 3, 1, 4]],
                 f"row 1 (ts {T0}) is not after row 0 (ts {T0 + 120})"),
    "repeated": (lambda df: df.iloc[[0, 1, 1, 3, 4]],
                 f"row 2 (ts {T0 + 60}) is not after row 1 (ts {T0 + 60})"),
    "nan": (lambda df: df.assign(ts=df["ts"].where(df.index != 2)),
            f"row 2 (ts nan) is not after row 1 (ts {T0 + 60.0})"),
    "nan-first": (lambda df: df.assign(ts=df["ts"].where(df.index != 0)),
                  f"row 1 (ts {T0 + 60.0}) is not after row 0 (ts nan)"),
    "missing": (lambda df: df.drop(columns=["ts"]), "the frame has no 'ts' column"),
}
_MODES = {"one-session": ([1] * 5, "session"), "gap": (None, "gap")}


@pytest.mark.parametrize("mode", list(_MODES))
@pytest.mark.parametrize("bad", list(_BAD_TS))
def test_bars_out_of_time_order_in_the_callers_frame_are_refused(bad, mode):
    # Nothing checked the time order: a reversed or shuffled frame (one session, or any
    # frame in gap mode) ran time backwards without a word, a NaN ts crashed on a cryptic
    # int(nan), and a frame with no ts on a KeyError deep in the run (re-review, 2026-09-28).
    (breaks, problem), (sessions, break_on) = _BAD_TS[bad], _MODES[mode]
    df = breaks(_frame(BREAK, sessions=sessions))
    with pytest.raises(ValueError) as e:
        Engine(NQ, BacktestConfig(break_on=break_on)).run(df, Precompute())
    assert not isinstance(e.value, FrameError)                    # the caller's frame
    assert "must strictly increase" in str(e.value)
    assert problem in str(e.value)


@pytest.mark.parametrize("mode", list(_MODES))
@pytest.mark.parametrize("bad", list(_BAD_TS))
def test_a_precompute_that_reorders_the_bars_raises_frame_error(bad, mode):
    # The same frames, made by the strategy's precompute. Unnamed, a precompute that drops
    # ts raised a KeyError: a 500 in PlutoDesk, not its 422 for a FrameError.
    (breaks, problem), (sessions, break_on) = _BAD_TS[bad], _MODES[mode]
    with pytest.raises(FrameError) as e:
        Engine(NQ, BacktestConfig(break_on=break_on)).run(_frame(BREAK, sessions=sessions),
                                                          Precompute(breaks))
    assert (f"the strategy's precompute removed or broke the bar timestamps: {problem}."
            in str(e.value))


@pytest.mark.parametrize("precomputed", [False, True], ids=["callers-frame", "precompute"])
def test_time_order_is_checked_before_the_session_ids(precomputed):
    # Reversing a two-session frame also makes its session IDs decrease; the time order
    # is the cause, so it is what the refusal names.
    df = _frame(BREAK, sessions=[1, 1, 1, 2, 2])
    reverse = Precompute(lambda d: d.iloc[::-1])
    frame, strategy = (df, reverse) if precomputed else (df.iloc[::-1], Precompute())
    with pytest.raises(ValueError, match="bar timestamps .* must strictly increase"):
        Engine(NQ, BacktestConfig(break_on="session")).run(frame, strategy)
