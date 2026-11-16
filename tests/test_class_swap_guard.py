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
import types

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


def test_a_helper_swapped_for_a_lambda_inside_a_list_runs_as_on_3_0_2():
    # After the swap the list has no ordinary pickle (a lambda does not pickle), so the
    # check compares it by identity, as 3.0.2 did, and it is the same list. Seeing it took a
    # rule that also refused harmless objects that open a connection on first read (the
    # fifth review); it is a documented limit. A same-named swap that still pickles is
    # refused (test_class_attribute_matrix: S_bound_method_in_list_swap).
    assert _eng().run(_bars(), SwapsInList()) is not None


def test_setting_the_same_function_or_an_equal_value_again_still_runs():
    result = _eng().run(_bars(), ResetsSame())
    assert result is not None
    assert ResetsSame.fast == 20


def test_a_numpy_array_edited_in_place_is_still_refused_and_restored():
    with pytest.raises(LookaheadError, match=r"EditsArray\.table"):
        _eng().run(_bars(), EditsArray())
    assert EditsArray.table.tolist() == [0.0, 0.0, 0.0]


# ---- the whole-branch review's findings (3.0.3)


class HoldsZoneInfo(Strategy):
    """A session time zone on the class, read by precompute: ran on 3.0.2."""

    from zoneinfo import ZoneInfo as _ZoneInfo
    tz = _ZoneInfo("America/New_York")
    opened = pd.Timestamp("2024-01-02 09:30", tz=_ZoneInfo("America/New_York"))

    def precompute(self, df):
        assert str(self.tz) == "America/New_York" and self.opened.hour == 9
        return df

    def on_bar(self, ctx):
        pass


def test_a_zoneinfo_class_attribute_still_runs():
    # ZoneInfo reduces through a bound builtin method made anew on every access, so an id
    # for it differed between two picklings of the same untouched object (review C1).
    assert _eng().run(_bars(), HoldsZoneInfo()) is not None


class CachedSessionOpen(Strategy):
    """A cached_property made from the parameters only: never warned."""

    tzname = "America/New_York"

    @functools.cached_property
    def session_open(self):
        from zoneinfo import ZoneInfo
        return pd.Timestamp("2024-01-02 09:30", tz=ZoneInfo(self.tzname))

    def precompute(self, df):
        assert self.session_open.hour == 9
        return df

    def on_bar(self, ctx):
        pass


def test_a_parameter_only_cached_property_holding_a_zoneinfo_is_not_warned():
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert _eng().run(_bars(), CachedSessionOpen()) is not None


class EditsUnpicklable(Strategy):
    """A dict holding a lambda (no ordinary pickle) edited in place with a value of the
    whole frame."""

    memo = {"pick": lambda x: x, "top": None}

    def precompute(self, df):
        if type(self).memo["top"] is None:
            type(self).memo["top"] = float(df["close"].min())
        return df

    def on_bar(self, ctx):
        if ctx.bar.close <= (self.memo["top"] or -1.0) and ctx.position.qty == 0:
            ctx.broker.buy(1)


def test_an_attribute_with_no_pickle_edited_in_place_runs_as_on_3_0_2():
    # A dict holding a lambda has no ordinary pickle, so the check compares it by identity,
    # as 3.0.2 did: an edit in place is not seen (a documented limit). Seeing it took a
    # second pickle that broke other cases (the fifth review).
    assert _eng().run(_bars(), EditsUnpicklable()) is not None


class SwapsCachedMethod(Strategy):
    """Replaces an lru_cache-wrapped method with a new one under the same name, carrying
    the whole frame's lowest close."""

    @functools.cache  # noqa: B019 - the swap of a cached method is what is tested
    def level(self):
        return -1.0

    def precompute(self, df):
        cls = type(self)
        if cls.level(self) == -1.0:
            top = float(df["close"].min())

            def level(self):
                return top
            level.__qualname__ = "SwapsCachedMethod.level"
            cls.level = functools.lru_cache(maxsize=None)(level)
        return df

    def on_bar(self, ctx):
        if ctx.bar.close <= self.level() and ctx.position.qty == 0:
            ctx.broker.buy(1)


def test_an_lru_cached_method_replaced_under_its_own_name_is_refused():
    # An lru_cache wrapper pickles as its name too (review M1, re-graded: the README says a
    # replaced method is refused).
    with pytest.raises(LookaheadError, match=r"SwapsCachedMethod\.level"):
        _eng().run(_bars(), SwapsCachedMethod())


# ---- the fix-pass re-review's findings (3.0.3)


class Packed:
    """Pickles (its state is a list), but cannot be rebuilt from that pickle: it has no
    __setstate__ to take a list."""

    def __init__(self):
        self.v = 1

    def __getstate__(self):
        return [self.v]


def _packed():
    return Packed()


class RestoreCannotRebuild(Strategy):
    """Edits in place an attribute whose old value cannot be rebuilt, and another that can."""

    a = {"p": _packed(), "t": None}
    b = []

    def precompute(self, df):
        type(self).a["t"] = float(df["close"].min())
        type(self).b.append(1.0)
        return df

    def on_bar(self, ctx):
        pass


def test_an_attribute_that_cannot_be_rebuilt_is_still_a_refusal_and_the_rest_are_put_back():
    # Rebuilding its old value raised, which turned the refusal into a crash and left the
    # attributes after it edited (the fix-pass re-review).
    with pytest.raises(LookaheadError, match=r"RestoreCannotRebuild\.a"):
        _eng().run(_bars(), RestoreCannotRebuild())
    assert RestoreCannotRebuild.b == []


class _Unpicklable:
    """An object with no pickle at all, whose bound method a strategy keeps."""

    def __reduce__(self):
        raise TypeError("no pickle")

    def first(self, x):
        return x

    def second(self, x):
        return x


_HOLDER = _Unpicklable()
_OTHER = _Unpicklable()


class SwapsBoundMethodInList(Strategy):
    """Swaps, in place, a bound method of an unpicklable object in the list it holds for
    the same-named method of another such object."""

    steps = [_HOLDER.first, 0.0]

    def precompute(self, df):
        cls = type(self)
        if cls.steps[0].__self__ is _HOLDER:
            cls.steps[0] = _OTHER.first
        return df

    def on_bar(self, ctx):
        pass


def test_a_bound_method_of_an_unpicklable_object_swapped_in_a_list_runs_as_on_3_0_2():
    # The list has no ordinary pickle (its method's object has none), so it is compared by
    # identity, as 3.0.2 did (a documented limit).
    assert _eng().run(_bars(), SwapsBoundMethodInList()) is not None


# ---- fix pass 2's re-review: state changed through a bound method the class holds


class _Model:
    def __init__(self):
        self.mean = None

    def fit(self, df):
        self.mean = float(df["close"].mean())

    def predict(self):
        return self.mean


_MODEL = _Model()


class FitsThroughBoundMethod(Strategy):
    """Holds a model's bound methods; precompute fits the model on the whole frame."""

    fit = _MODEL.fit
    predict = _MODEL.predict

    def precompute(self, df):
        if self.predict() is None:
            self.fit(df)
        return df

    def on_bar(self, ctx):
        if ctx.bar.close <= self.predict() and ctx.position.qty == 0:
            ctx.broker.buy(1)


_SEEN: list = []
_LIST_MODEL = _Model()


class AppendsThroughBoundBuiltin(Strategy):
    """Holds a list's bound append (the list itself is no class attribute); precompute
    appends a value of the whole frame."""

    push = _SEEN.append

    def precompute(self, df):
        if not _SEEN:
            self.push(float(df["close"].min()))
        return df

    def on_bar(self, ctx):
        pass


class FitsThroughMethodInList(Strategy):
    """Holds a model's bound method inside a list (the model itself is no class
    attribute); precompute fits through it."""

    steps = [_LIST_MODEL.fit]

    def precompute(self, df):
        if _LIST_MODEL.mean is None:
            self.steps[0](df)
        return df

    def on_bar(self, ctx):
        pass


@pytest.mark.parametrize("strategy", [FitsThroughBoundMethod, AppendsThroughBoundBuiltin,
                                      FitsThroughMethodInList])
def test_state_changed_through_a_bound_method_the_class_holds_is_refused(strategy):
    # Refused on 3.0.2 (the bound method pickles with its object's state). Keying a bound
    # method by its object's identity alone hid the change (fix pass 2's re-review, C1).
    with pytest.raises(LookaheadError, match="changed class attribute"):
        _eng().run(_bars(), strategy())


class RestoreCannotRebuildAlone(Strategy):
    """As RestoreCannotRebuild, run by one test only (a class left unrestored refuses
    every later run)."""

    a = {"p": _packed(), "t": None}

    def precompute(self, df):
        type(self).a["t"] = float(df["close"].min())
        return df

    def on_bar(self, ctx):
        pass


def test_the_refusal_names_an_attribute_it_could_not_put_back():
    # It stays as precompute left it; the words say so instead of "restored" (fix pass 2's
    # re-review).
    with pytest.raises(LookaheadError) as refused:
        _eng().run(_bars(), RestoreCannotRebuildAlone())
    words = str(refused.value)
    assert "could not be put back: RestoreCannotRebuildAlone.a" in words
    assert "restart" in words


# ---- fix pass 3's re-review: depth, a swapped bound method, a rerun after a leak


class _Stage:
    def __init__(self, nxt=None):
        self.seen = 0.0
        self.out = nxt.update if nxt is not None else None

    def update(self, x):
        self.seen = x


def _chain(depth):
    stage = None
    for _ in range(depth):
        stage = _Stage(stage)
    return stage


class DeepPipeline(Strategy):
    """A pipeline of stages, each holding the next one's bound update, 22 deep."""

    head = _chain(22)

    def __init__(self):
        self.mine = _chain(22)

    def precompute(self, df):
        return df

    def on_bar(self, ctx):
        pass


def test_a_deep_chain_of_bound_methods_is_checked_in_linear_time():
    # Testing each owner by pickling it again doubled the work at every level: 3.0.2 checked
    # this in 0.03 s, the third fix pass in about 30 s (fix pass 3's re-review, C1).
    import time

    started = time.perf_counter()
    assert _eng().run(_bars(), DeepPipeline()) is not None
    assert time.perf_counter() - started < 5.0


class _Fitter:
    def __init__(self):
        self.mean = None

    def fit(self, df):
        self.mean = None


def _leaky_fit(self, df):
    self.mean = float(df["close"].mean())


_FITTER = _Fitter()


class SwapsBoundMethod(Strategy):
    """Swaps the bound fit it holds for a same-named bound method of the same object that
    fits on the whole frame."""

    fit = _FITTER.fit

    def precompute(self, df):
        cls = type(self)
        if cls.fit.__func__ is _Fitter.fit:
            leaky = functools.wraps(_Fitter.fit)(_leaky_fit)
            cls.fit = types.MethodType(leaky, _FITTER)
        return df

    def on_bar(self, ctx):
        pass


def test_a_bound_method_swapped_for_a_same_named_one_is_refused():
    # Written as getattr(its object, its name), which function it was dropped out of the
    # comparison (fix pass 3's re-review, I1).
    with pytest.raises(LookaheadError, match=r"SwapsBoundMethod\.fit"):
        _eng().run(_bars(), SwapsBoundMethod())
    assert SwapsBoundMethod.fit.__func__ is _Fitter.fit


class FitsOnceAndCannotBeRestored(Strategy):
    """Fits once into an attribute whose old value cannot be rebuilt."""

    a = {"p": _packed(), "t": None}

    def precompute(self, df):
        if type(self).a["t"] is None:
            type(self).a["t"] = float(df["close"].mean())
        return df

    def on_bar(self, ctx):
        pass


def test_a_rerun_after_an_edit_that_could_not_be_put_back_is_refused():
    # The second run starts from the leaked value; it ran cleanly (fix pass 3's re-review, I3).
    with pytest.raises(LookaheadError, match="could not be put back"):
        _eng().run(_bars(), FitsOnceAndCannotBeRestored())
    with pytest.raises(LookaheadError, match="FitsOnceAndCannotBeRestored.a"):
        _eng().run(_bars(), FitsOnceAndCannotBeRestored())


# ---- fix pass 5's re-review: a restore that writes the swap back


class _Predictor:
    def __init__(self):
        self.coef = 0.0

    def predict(self, x):
        return x * self.coef


_PREDICTOR = _Predictor()
_FITTED: dict = {}


class RebindsTheMethodItHolds(Strategy):
    """Rebinds _Predictor.predict to a same-named wrapper carrying the whole frame's mean,
    and stores the new bound method in the list it holds: unpickling the old list looks the
    method up by name, so it gets the new one."""

    attr = [_PREDICTOR.predict]

    def precompute(self, df):
        if "mean" not in _FITTED:
            _FITTED["mean"] = float(df["close"].mean())

            @functools.wraps(_Predictor.predict)
            def predict(self_, x):
                return x - _FITTED["mean"]
            _Predictor.predict = predict
            type(self).attr[0] = _PREDICTOR.predict
        return df

    def on_bar(self, ctx):
        pass


def test_a_swap_that_also_rebinds_the_method_on_its_class_is_refused_but_not_undone():
    # A documented limit: the refusal stands, but rebuilding the list looks the method up by
    # name and finds the rebound one, so the swap stays (fix pass 5's re-review, I2). 3.0.2
    # ran this case. Checking each restore by its bytes refused good restores (fix pass 6's
    # re-review), so the limit is stated instead.
    with pytest.raises(LookaheadError, match="RebindsTheMethodItHolds.attr"):
        _eng().run(_bars(), RebindsTheMethodItHolds())
