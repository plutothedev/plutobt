"""State the look-ahead check's snapshot cannot see (final audit B, 2026-11-02).

Class attributes are shared by every copy the check runs, so a fit-once value stored on
the class escaped the snapshot: refused, and restored. An attribute precompute sets on
self from the full frame (a quantile threshold) is a look-ahead the frame check cannot
see if on_bar reads it: warned, naming it. A precompute that fails on a short prefix
says the check was running and how to get past it.
"""
import pathlib
import warnings

import numpy as np
import pandas as pd
import pytest

from plutobt.broker import FrameError
from plutobt.config import BacktestConfig
from plutobt.data import load_bars
from plutobt.engine import Engine
from plutobt.instruments import get_instrument
from plutobt.lookahead import LookaheadError, LookaheadWarning
from plutobt.strategies.ema_cross import EMACross
from plutobt.strategy import Strategy

NQ = get_instrument("NQ")
ROOT = pathlib.Path(__file__).resolve().parents[1]


def _bars(n=60):
    rng = np.random.default_rng(1)
    c = 20000.0 + np.cumsum(rng.normal(0, 5.0, size=n))
    return pd.DataFrame({"ts": np.arange(0, 300 * n, 300, dtype="int64"),
                         "open": c, "high": c + 2.0, "low": c - 2.0, "close": c,
                         "volume": np.ones(n, dtype="int64")})


def _eng(**cfg):
    return Engine(NQ, BacktestConfig(commission_per_side=0.0, slippage_ticks=0.0, **cfg))


def _run_recording(strategy, df=None, **cfg):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = _eng(**cfg).run(_bars() if df is None else df, strategy)
    return result, [w for w in caught if issubclass(w.category, LookaheadWarning)]


# ---------------------------------------------------------------------------
# 5. Class attributes changed by precompute: refused, restored.
# ---------------------------------------------------------------------------

def _class_fit_once():
    class ClassFitOnce(Strategy):
        mu = None
        sd = None

        def precompute(self, d):
            if ClassFitOnce.mu is None:
                ClassFitOnce.mu, ClassFitOnce.sd = d.close.mean(), d.close.std()
            d["z"] = (d.close - ClassFitOnce.mu) / ClassFitOnce.sd
            return d
    return ClassFitOnce


def test_a_fit_once_class_attribute_is_refused_naming_it():
    # The auditor's repro: the traded call fitted mu on the full frame, every prefix
    # then reused it, and the check passed.
    cls = _class_fit_once()
    with pytest.raises(LookaheadError, match=r"class attribute ClassFitOnce\.mu"):
        _eng().run(_bars(), cls())


def test_the_class_is_restored_so_a_second_run_is_refused_too():
    # Left set, the second run would find mu fitted and pass on the old full-sample fit.
    cls = _class_fit_once()
    for _ in range(2):
        with pytest.raises(LookaheadError, match="class attribute"):
            _eng().run(_bars(), cls())
        assert cls.mu is None and cls.sd is None


def test_an_in_place_mutated_class_cache_is_refused_and_restored():
    class Cached(Strategy):
        cache = {}

        def precompute(self, d):
            self.cache.setdefault("hi", d.high.max())
            return d.assign(x=d.high / self.cache["hi"])

    with pytest.raises(LookaheadError, match=r"class attribute Cached\.cache"):
        _eng().run(_bars(), Cached())
    assert Cached.cache == {}


def test_a_class_attribute_added_by_precompute_is_refused_and_removed():
    class Adds(Strategy):
        def precompute(self, d):
            type(self).hi = d.high.max()
            return d.assign(x=d.high / type(self).hi)

    with pytest.raises(LookaheadError, match=r"class attribute Adds\.hi"):
        _eng().run(_bars(), Adds())
    assert "hi" not in vars(Adds)


def test_a_parent_class_attribute_is_watched_too():
    class Base(Strategy):
        level = None

    class Child(Base):
        def precompute(self, d):
            if Base.level is None:
                Base.level = d.close.median()
            return d.assign(x=(d.close > Base.level).astype(float))

    with pytest.raises(LookaheadError, match=r"class attribute Base\.level"):
        _eng().run(_bars(), Child())
    assert Base.level is None


def test_a_class_attribute_set_only_by_a_prefix_call_is_refused_and_removed():
    # The traded (full-frame) call leaves the class alone; a prefix call writes it, and
    # the prefixes also leak. Whichever is found, the class is put back.
    class Later(Strategy):
        def precompute(self, d):
            if len(d) < 30:
                type(self).short = True
            return d.assign(x=d.close.shift(-1).ffill())

    with pytest.raises(LookaheadError, match=r"class attribute Later\.short"):
        _eng().run(_bars(), Later())
    assert "short" not in vars(Later)


def test_class_attributes_read_but_not_written_run():  # positive control
    result, caught = _run_recording(EMACross(), load_bars(
        str(ROOT / "data" / "sample" / "nq_5m.json")))
    assert len(result.trades) > 0 and caught == []


def test_class_state_is_not_checked_with_the_check_off():
    cls = _class_fit_once()
    _eng(check_causality=False).run(_bars(), cls())
    assert cls.mu is not None


def test_a_write_to_the_strategy_base_class_is_refused_and_removed():
    # Strategy itself is shared by every copy too: it used to be left out of the scan.
    class OnBase(Strategy):
        def precompute(self, d):
            Strategy.cache = float(d.close.mean())
            return d.assign(x=d.close - Strategy.cache)

    try:
        with pytest.raises(LookaheadError, match=r"class attribute Strategy\.cache"):
            _eng().run(_bars(), OnBase())
        assert "cache" not in vars(Strategy)
    finally:
        if "cache" in vars(Strategy):
            del Strategy.cache


SHARED = [1, 2]


def test_a_shared_list_edited_in_place_is_restored_in_place():
    # The class attribute is an alias of a module list: restoring it by assignment
    # would leave the module's list grown. A list, dict or set is put back in place.
    class Hist(Strategy):
        hist = SHARED

        def precompute(self, d):
            self.hist.append(len(d))
            return d.assign(x=d.close.diff())

    with pytest.raises(LookaheadError, match=r"class attribute Hist\.hist"):
        _eng().run(_bars(), Hist())
    assert Hist.hist is SHARED
    assert SHARED == [1, 2]


# ---------------------------------------------------------------------------
# 6. Instance attributes precompute sets: warned, naming each, at the run call.
# ---------------------------------------------------------------------------

class StoredThreshold(Strategy):
    def precompute(self, d):
        r = d.close.pct_change()
        d["r"] = r
        self.thr = r.quantile(0.9)      # full-sample quantile kept on self for on_bar
        return d

    def on_bar(self, ctx):
        if ctx.bar.r > self.thr and ctx.position.qty == 0:
            ctx.broker.buy(1)


def test_an_attribute_set_by_precompute_is_warned_naming_it():
    _, caught = _run_recording(StoredThreshold())
    (w,) = caught
    text = str(w.message)
    assert "precompute set self.thr from the full frame" in text
    assert "if on_bar uses it, that is look-ahead" in text
    assert "compute per-bar values as columns instead" in text
    # at the user's Engine.run line, not inside the engine
    assert pathlib.Path(w.filename).resolve() == pathlib.Path(__file__).resolve()


def test_an_attribute_precompute_changes_is_warned():
    class Changes(Strategy):
        def __init__(self):
            self.level = 0.0

        def precompute(self, d):
            self.level = float(d.close.max())
            return d

    _, caught = _run_recording(Changes())
    assert ["self.level" in str(w.message) for w in caught] == [True]


def test_each_attribute_gets_its_own_warning():
    class Two(Strategy):
        def precompute(self, d):
            self.a, self.b = d.close.mean(), d.close.std()
            return d

    _, caught = _run_recording(Two())
    assert sorted(str(w.message).split()[2] for w in caught) == ["self.a", "self.b"]


def test_attributes_set_before_the_run_are_not_warned():  # positive control
    class Params(Strategy):
        def __init__(self):
            self.spans = [20, 50]
            self.table = pd.Series([1.0, 2.0, 3.0])
            self.weights = np.ones(5)
            self.opts = {"stop": 20, "levels": np.arange(3)}
            self.rng_seed = 7

        def precompute(self, d):
            return d.assign(x=d.close.rolling(self.spans[0]).mean())

    result, caught = _run_recording(Params())
    assert len(result.equity_curve) == 60 and caught == []


def test_the_warning_does_not_refuse_the_run():
    result, _ = _run_recording(StoredThreshold())
    assert len(result.equity_curve) == 60


# ---------------------------------------------------------------------------
# 9. A precompute that fails on a short prefix says the check was running.
# ---------------------------------------------------------------------------

class NeedsTwenty(Strategy):
    def precompute(self, d):
        if len(d) < 20:
            raise ZeroDivisionError("window of 20 bars, got fewer")
        return d.assign(x=d.close.rolling(20).mean())


def test_a_precompute_that_fails_on_a_short_prefix_names_the_prefix_and_the_check():
    with pytest.raises(FrameError) as e:
        _eng().run(_bars(), NeedsTwenty())
    text = str(e.value)
    assert "ZeroDivisionError: window of 20 bars, got fewer" in text
    assert "df[:1]" in text and "1 bar" in text
    assert "look-ahead check" in text
    assert "check_causality = false" in text and "warm-up" in text
    assert isinstance(e.value.__cause__, ZeroDivisionError)


def test_the_same_precompute_runs_with_the_check_off():  # control
    assert len(_eng(check_causality=False).run(_bars(), NeedsTwenty()).equity_curve) == 60


def test_a_precompute_that_fails_on_the_full_frame_is_not_wrapped():
    class Broken(Strategy):
        def precompute(self, d):
            raise KeyError("no such column")

    with pytest.raises(KeyError, match="no such column"):
        _eng().run(_bars(), Broken())
