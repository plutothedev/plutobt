"""BacktestResult.margin_refusals: the orders the margin check refused outright or cut
down to their closing half, each counted once. Counted where the check runs, so a
thin run can say why it is thin instead of reading as "the signal never fired".

Self-contained on purpose: PlutoDesk vendors this file whole, and its vendored subset
has no plutobt.data, no golden files and a different conftest."""
import pandas as pd

from plutobt.config import BacktestConfig
from plutobt.engine import BacktestResult, Engine
from plutobt.instruments import get_instrument
from plutobt.strategy import Strategy

NQ = get_instrument("NQ")          # initial margin $23,000: 5 contracts fail on $100k
T0 = 1_700_000_000
EVEN = [0, 60, 120, 180, 240]      # one bar a minute: no gap, so no break in gap mode


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
    df = pd.DataFrame({"ts": pd.Series(ts, dtype="int64"), "open": px,
                       "high": [p + 1 for p in px], "low": [p - 1 for p in px], "close": px,
                       "volume": pd.Series([10] * len(ts), dtype="int64")})
    if sessions is not None:
        df["session"] = pd.Series(sessions, dtype="int64")
    return df


def _run(df, strategy, **cfg):
    return Engine(NQ, BacktestConfig(periods_per_year=252.0, **cfg)).run(df, strategy)


def test_an_entry_the_margin_check_refuses_is_counted():
    one = _run(_frame(EVEN), Scripted({0: ("buy", 5)}))
    assert one.margin_refusals == 1 and one.trades == []
    two = _run(_frame(EVEN), Scripted({0: ("buy", 5), 2: ("buy", 5)}))
    assert two.margin_refusals == 2


def test_a_flip_cut_to_its_closing_half_is_counted_and_its_exit_still_fills():
    result = _run(_frame(EVEN), Scripted({0: "buy", 2: ("sell", 50)}))
    assert result.margin_refusals == 1
    (trade,) = result.trades                   # the long closed; no short opened
    assert trade.side == "long" and trade.exit_ts == T0 + 180 and not trade.open_at_end


class FlipLimitBetweenStopAndHigh(Strategy):
    """Long 1 (filled at 101), then a stop at 101.5 under it and a sell limit for 50 at
    102.5 above it. Bar 2 (open 102, low 101, high 103) reaches both after the open, so
    the worst-case search tries both orderings on throwaway copies of the book, and the
    margin check refuses the flip on every one of them."""

    def __init__(self):
        self.i = -1
        self.order = None

    def precompute(self, df):
        return df

    def on_bar(self, ctx):
        self.i += 1
        if self.i == 0:
            ctx.broker.buy(contracts=1)
        elif self.i == 1:
            ctx.broker.amend_stop(101.5)
            self.order = ctx.broker.sell(contracts=50, limit=102.5)


def test_orderings_tried_and_thrown_away_are_not_counted():
    # Up first, the limit flips the long and margin cuts it to its closing contract; down
    # first, the stop flattens the long and the flip, now a 49-contract entry, is refused.
    # Either way one order met one refusal on the committed path. Every ordering the
    # search tried ran on a copy holding its own copy of the order, so counting the
    # copies would read 3.
    strategy = FlipLimitBetweenStopAndHigh()
    result = _run(_frame(EVEN), strategy)
    assert strategy.order.status != "pending"
    assert result.margin_refusals == 1


def test_a_session_break_trim_is_not_a_margin_refusal():
    flip = {0: "buy", 2: ("sell", 50)}
    no_break = _run(_frame(EVEN, sessions=[1] * 5), Scripted(flip), break_on="session")
    assert no_break.margin_refusals == 1       # the margin check cut the flip
    at_break = _run(_frame(EVEN, sessions=[1, 1, 1, 2, 2]), Scripted(flip),
                    break_on="session")
    assert at_break.margin_refusals == 0       # the break trimmed it first


class WritesRejected(Strategy):
    def precompute(self, df):
        return df

    def on_bar(self, ctx):
        if ctx.position.qty == 0 and ctx.broker.pending is None:
            order = ctx.broker.buy(contracts=1)
            order.status = "rejected"          # the strategy's own word, not the check's


def test_a_status_the_strategy_writes_is_not_a_margin_refusal():
    result = _run(_frame(EVEN), WritesRejected())
    assert result.margin_refusals == 0 and len(result.trades) == 1


def test_a_run_without_refusals_reports_zero():
    assert _run(_frame(EVEN), Scripted({0: "buy", 2: "close"})).margin_refusals == 0
    bare = BacktestResult(equity_curve=pd.Series(dtype="float64"), trades=[], metrics={},
                          manifest={})
    assert bare.margin_refusals == 0


class RefusedEveryBar(Strategy):
    """An entry the margin check refuses on every bar, and no reference kept to it: a
    dropped Order's id is free for the next one, so only holding the Order keeps the
    count exact."""

    def precompute(self, df):
        return df

    def on_bar(self, ctx):
        ctx.broker.buy(contracts=5)


def test_orders_nobody_keeps_are_each_counted():
    # CHARACTERIZATION PIN (passes on 0fee11a): it goes red if the broker keeps only the
    # ids and drops the Orders (CPython reuses a freed id: 26 of 199 counted, review probe).
    n = 200
    result = _run(_frame([60 * i for i in range(n)]), RefusedEveryBar())
    assert result.margin_refusals == n - 1     # the last bar's order meets no fill


class StopThenFlip(Strategy):
    """Long 1; on bar 2 a flip too big for margin, with the stop moved up first."""

    def __init__(self, stop):
        self.i, self.stop = -1, stop

    def precompute(self, df):
        return df

    def on_bar(self, ctx):
        self.i += 1
        if self.i == 0:
            ctx.broker.buy(contracts=1)
        elif self.i == 2:
            if self.stop is not None:
                ctx.broker.amend_stop(self.stop)
            ctx.broker.sell(contracts=50)


def test_a_flip_a_stop_made_stale_is_not_a_margin_refusal():
    # The bar walks in time order: the market flip fills at the open (103), before the
    # bar's low (102) reaches the 102.5 stop, so the margin check cuts the flip either
    # way and the stop never fires on a position that is already flat.
    assert _run(_frame(EVEN), StopThenFlip(None)).margin_refusals == 1    # cut
    assert _run(_frame(EVEN), StopThenFlip(102.5)).margin_refusals == 1   # cut at the open
