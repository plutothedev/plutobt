import pandas as pd
import pytest

from plutobt.strategy import Strategy, Context
from plutobt.instruments import get_instrument
from plutobt.portfolio import Position
from plutobt.broker import Broker
from plutobt.config import BacktestConfig
from plutobt.portfolio import Portfolio


def test_strategy_base_precompute_is_identity_passthrough():
    # Base precompute returns the frame unchanged (subclasses override).
    df = pd.DataFrame({"ts": [1, 2], "open": [1.0, 2.0], "high": [1.0, 2.0],
                       "low": [1.0, 2.0], "close": [1.0, 2.0], "volume": [1, 1]})
    out = Strategy().precompute(df)
    pd.testing.assert_frame_equal(out, df)


def test_strategy_base_on_bar_is_noop():
    # Base on_bar does nothing and returns None.
    nq = get_instrument("NQ")
    pf = Portfolio(starting_cash=100_000.0, instrument=nq)
    broker = Broker(nq, BacktestConfig(), pf)
    ctx = Context(bar=object(), position=Position(), broker=broker,
                  instrument=nq, tick=nq.tick_size, equity=100_000.0, state={})
    assert Strategy().on_bar(ctx) is None


def test_context_holds_read_only_bar_and_typed_position():
    nq = get_instrument("NQ")
    pf = Portfolio(starting_cash=100_000.0, instrument=nq)
    broker = Broker(nq, BacktestConfig(), pf)
    Row = __import__("collections").namedtuple("Row", ["ts", "close", "long"])
    bar = Row(ts=10, close=20000.0, long=True)
    ctx = Context(bar=bar, position=Position(qty=0), broker=broker,
                  instrument=nq, tick=0.25, equity=100_000.0, state={})
    assert ctx.bar.long is True
    assert ctx.bar.close == 20000.0
    assert ctx.position.qty == 0
    assert ctx.tick == 0.25
    assert ctx.instrument.symbol == "NQ"
    # state is a mutable dict persisted by the engine across bars
    ctx.state["hh"] = 20000.0
    assert ctx.state["hh"] == 20000.0
