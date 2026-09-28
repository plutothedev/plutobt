"""break_on="session" (wave 1): only a trading-session change is a break;
pending exits survive it; orders are whole contracts."""
import numpy as np
import pandas as pd
import pytest

from plutobt.broker import Broker, Order, OrderError
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


def test_a_pending_close_survives_a_session_break_in_session_mode():
    df = _frame(BREAK, sessions=[1, 1, 1, 2, 2])
    (t,) = _run(df, {0: "buy", 2: "close"}, break_on="session").trades
    assert t.exit_ts == T0 + 180 and not t.open_at_end


@pytest.mark.parametrize("entry,exit_", [("buy", ("sell", 1)), (("sell", 1), ("buy", 1))])
def test_a_plain_exit_order_survives_a_session_break(entry, exit_):
    # sell(1) exiting a long and buy(1) covering a short are exits too, not
    # only close() (money re-audit, 2026-09-26).
    df = _frame(BREAK, sessions=[1, 1, 1, 2, 2])
    (t,) = _run(df, {0: entry, 2: exit_}, break_on="session").trades
    assert t.exit_ts == T0 + 180 and not t.open_at_end


def test_a_flip_at_a_session_break_keeps_only_its_closing_contracts():
    flip = {0: "buy", 2: ("sell", 2)}
    same = _run(_frame(BREAK, sessions=[1] * 5), flip, break_on="session").trades
    assert [t.side for t in same] == ["long", "short"]          # no break: a real flip
    (t,) = _run(_frame(BREAK, sessions=[1, 1, 1, 2, 2]), flip, break_on="session").trades
    assert t.side == "long" and t.exit_ts == T0 + 180 and not t.open_at_end   # flat, no short


def test_a_flip_whose_new_side_fails_margin_keeps_only_its_exit():
    # PRE-EXISTING (money re-review, 2026-09-27): _is_reducing checks the side
    # only, so a flip skipped the margin check and long 1 then sell(50) opened a
    # 49-contract short on $100k. The exit part always goes through; the new
    # position's part must pass margin like any entry.
    flip = {0: "buy", 2: ("sell", 50)}
    (t,) = _run(_frame(BREAK, sessions=[1] * 5), flip, break_on="session").trades
    assert t.side == "long" and not t.open_at_end


def test_known_default_gap_mode_drops_a_pending_exit_at_a_gap():
    # CHARACTERIZATION PIN (passes before this task; mutation-proved in Step 6).
    # KNOWN, reported to pluto 2026-09-26, deliberately unchanged here: the
    # golden run and the plutobt video numbers depend on the default mode.
    df = _frame([0, 60, 120, 600, 660])
    (t,) = _run(df, {0: "buy", 2: "close"}).trades
    assert t.open_at_end


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


@pytest.mark.parametrize("contracts", [0, 0.5])
def test_an_order_queued_straight_into_pending_is_refused_at_the_fill(contracts):
    broker = Broker(NQ, BacktestConfig(), Portfolio(starting_cash=100_000.0, instrument=NQ))
    broker.pending = Order(side="buy", contracts=contracts, kind="market")
    with pytest.raises(OrderError, match="cannot fill the buy order"):
        broker.fill_pending(_frame([0]).iloc[0])
    assert broker.portfolio.trades == [] and broker.portfolio.position.qty == 0


def test_a_numpy_integer_order_fills():
    # CHARACTERIZATION PIN (passes before the fill-time check; mutation-proved: that
    # check reading isinstance(contracts, int) goes red).
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
