"""Price-level guard on back-adjusted series (spec §7.4)."""
import warnings

import numpy as np
import pandas as pd
import pytest

from plutobt.config import BacktestConfig
from plutobt.data import load_bars
from plutobt.engine import Engine
from plutobt.instruments import get_instrument
from plutobt.lookahead import PriceLevelWarning
from plutobt.strategy import Strategy
from tests.test_engine_roll import series


def make(name, fn):
    class S(Strategy):
        def precompute(self, df):
            df[name] = fn(df)
            return df
    return S()


def ema(df):
    return df["close"].ewm(span=5, adjust=False).mean()


def rsi(df):
    d = df["close"].diff()
    up = d.clip(lower=0).ewm(alpha=1 / 5, adjust=False).mean()
    down = (-d.clip(upper=0)).ewm(alpha=1 / 5, adjust=False).mean()
    return 100 - 100 / (1 + up / down.replace(0, np.nan))


def atr(df):
    return (df["high"] - df["low"]).rolling(3).mean()


def above(df):
    return (df["close"] > 18030.0).astype(int)


def pct(df):
    return df["close"].pct_change()


def logret(df):
    return np.log(df["close"]).diff()


def guard_warnings(strategy, df=None, meta=None, **cfg):
    if df is None:
        df, meta = series()
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        Engine(get_instrument("NQ"), BacktestConfig(**cfg)).run(df, strategy,
                                                                 roll_meta=meta)
    return [w for w in seen if issubclass(w.category, PriceLevelWarning)]


@pytest.mark.parametrize("fn", [ema, rsi, atr], ids=["ema", "rsi", "atr"])
def test_level_free_indicators_pass(fn):
    assert guard_warnings(make("sig", fn)) == []


@pytest.mark.parametrize("fn", [above, pct, logret], ids=["level", "pct", "log"])
def test_level_dependent_columns_warn(fn):
    (w,) = guard_warnings(make("sig", fn))
    msg = str(w.message)
    assert "'sig'" in msg and "back-adjusted" in msg and "raw_close" in msg


def test_guard_off_with_causality_check_off():
    assert guard_warnings(make("sig", above), check_causality=False) == []


def test_no_guard_without_contract_column():
    bars = load_bars("data/sample/nq_5m.json").iloc[:120].reset_index(drop=True)
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        Engine(get_instrument("NQ"), BacktestConfig()).run(bars, make("sig", above))
    assert not [w for w in seen if issubclass(w.category, PriceLevelWarning)]


def test_warning_names_the_callers_line():
    (w,) = guard_warnings(make("sig", above))
    assert w.filename.endswith("test_price_level_guard.py")


def test_pandas_frame_untouched():
    df, meta = series()
    before = df.copy()
    guard_warnings(make("sig", ema), df=df, meta=meta)
    pd.testing.assert_frame_equal(df, before)
