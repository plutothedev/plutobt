"""The look-ahead check refuses a precompute that changes a class attribute (final audit B).
Until 3.0.3 it compared an attribute before and after by its pickle, and a function or a
class pickles as its name only: a precompute that replaced a method, or a class nested in
the strategy, with a new one under the same name looked unchanged, and the run traded on
the look-ahead. A helper swapped inside a list the class holds was missed another way:
the new helper does not pickle under the old name, so the list was compared by identity,
and it is the same list.

The strategies here sit at module level on purpose: there their methods and nested classes
pickle by name, which is the shape of the gap. (A class defined inside a test function does
not pickle at all, so the check compares it by identity and always saw these swaps.)"""
import functools
import pickle

import numpy as np
import pandas as pd
import pytest

from plutobt.config import BacktestConfig
from plutobt.engine import Engine
from plutobt.instruments import get_instrument
from plutobt.lookahead import LookaheadError
from plutobt.strategy import Strategy

NQ = get_instrument("NQ")


def _bars(n=60):
    rng = np.random.default_rng(1)
    c = 20000.0 + np.cumsum(rng.normal(0, 5.0, size=n))
    return pd.DataFrame({"ts": np.arange(0, 300 * n, 300, dtype="int64"),
                         "open": c, "high": c + 2.0, "low": c - 2.0, "close": c,
                         "volume": np.ones(n, dtype="int64")})


def _eng(**cfg):
    return Engine(NQ, BacktestConfig(commission_per_side=0.0, slippage_ticks=0.0, **cfg))


def helper_a(x):
    return x


class SwapsOnBar(Strategy):
    """Replaces on_bar, on its first precompute (the traded call), with a wrapper of the
    same name that carries the whole frame's lowest close."""

    def precompute(self, df):
        cls = type(self)
        if not getattr(cls.on_bar, "__wrapped__", None):
            top = float(df["close"].min())
            inner = cls.on_bar

            @functools.wraps(inner)
            def on_bar(self, ctx):
                ctx.state["top"] = top
                return inner(self, ctx)
            cls.on_bar = on_bar
        return df

    def on_bar(self, ctx):
        if ctx.bar.close <= ctx.state.get("top", -1.0) and ctx.position.qty == 0:
            ctx.broker.buy(1)


class SwapsNested(Strategy):
    """Replaces its nested Params class with a new one under the same qualified name,
    carrying the whole frame's lowest close."""

    class Params:
        top = None

    def precompute(self, df):
        cls = type(self)
        if cls.Params.top is None:
            new = type("Params", (), {"top": float(df["close"].min())})
            new.__qualname__ = "SwapsNested.Params"
            new.__module__ = __name__
            cls.Params = new
        return df

    def on_bar(self, ctx):
        if ctx.bar.close <= (self.Params.top or -1.0) and ctx.position.qty == 0:
            ctx.broker.buy(1)


class SwapsInList(Strategy):
    """Swaps the helper inside the list it holds, in place, for one that carries the whole
    frame's lowest close."""

    steps = [helper_a]

    def precompute(self, df):
        cls = type(self)
        if cls.steps[0] is helper_a:
            top = float(df["close"].min())
            cls.steps[0] = lambda x: top
        return df

    def on_bar(self, ctx):
        if ctx.bar.close <= self.steps[0](-1.0) and ctx.position.qty == 0:
            ctx.broker.buy(1)


class ResetsSame(Strategy):
    """Sets a class attribute to the value it holds, and a helper to the same object."""

    fast = 20
    helper = staticmethod(helper_a)

    def precompute(self, df):
        type(self).fast = 20
        type(self).helper = type(self).__dict__["helper"]
        return df

    def on_bar(self, ctx):
        pass


class EditsArray(Strategy):
    """Edits a class-level numpy array in place with a value of the whole frame."""

    table = np.zeros(3)

    def precompute(self, df):
        type(self).table[0] = float(df["close"].min())
        return df

    def on_bar(self, ctx):
        pass


def test_a_same_named_on_bar_wrapper_is_refused():
    assert pickle.dumps(SwapsOnBar.on_bar)            # pickles by name: the 3.0.2 gap
    with pytest.raises(LookaheadError, match=r"class attribute SwapsOnBar\.on_bar"):
        _eng().run(_bars(), SwapsOnBar())
    assert not hasattr(SwapsOnBar.on_bar, "__wrapped__")    # put back


def test_a_nested_class_replaced_under_its_own_name_is_refused():
    assert pickle.dumps(SwapsNested.Params)
    with pytest.raises(LookaheadError, match=r"SwapsNested\.Params"):
        _eng().run(_bars(), SwapsNested())
    assert SwapsNested.Params.top is None                       # put back


def test_a_helper_swapped_inside_a_list_in_place_is_refused():
    with pytest.raises(LookaheadError, match=r"SwapsInList\.steps"):
        _eng().run(_bars(), SwapsInList())
    assert SwapsInList.steps == [helper_a]                     # restored in place


def test_setting_the_same_function_or_an_equal_value_again_still_runs():
    result = _eng().run(_bars(), ResetsSame())
    assert result is not None
    assert ResetsSame.fast == 20


def test_a_numpy_array_edited_in_place_is_still_refused_and_restored():
    with pytest.raises(LookaheadError, match=r"EditsArray\.table"):
        _eng().run(_bars(), EditsArray())
    assert EditsArray.table.tolist() == [0.0, 0.0, 0.0]
