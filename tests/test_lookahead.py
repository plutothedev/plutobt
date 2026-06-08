import numpy as np
import pandas as pd
import pytest

from plutobt.lookahead import assert_causal


def _frame(n=60):
    rng = np.random.default_rng(0)
    close = 20000.0 + np.cumsum(rng.normal(0, 5.0, size=n))
    return pd.DataFrame({
        "ts": np.arange(n, dtype="int64"),
        "open": close, "high": close + 1.0, "low": close - 1.0,
        "close": close, "volume": np.ones(n, dtype="int64"),
    })


def _noncausal_ema_with_global_mean(df):
    df = df.copy()
    df["ema_f"] = df["close"].ewm(span=10, adjust=False, min_periods=10).mean()
    df["long"] = df["ema_f"] > df["close"].mean()  # mean of <=t prefix is causal-per-prefix? -> NO
    return df


def _causal_ema_fixed(df):
    df = df.copy()
    df["ema_f"] = df["close"].ewm(span=10, adjust=False, min_periods=10).mean()
    df["ema_s"] = df["close"].ewm(span=20, adjust=False, min_periods=20).mean()
    df["long"] = df["ema_f"] > df["ema_s"]
    return df


def _leaky(df):
    df = df.copy()
    df["ema_f"] = df["close"].ewm(span=10, adjust=False, min_periods=10).mean()
    df["peek"] = df["close"].shift(-1)  # FUTURE bar -> leak
    return df


def test_assert_causal_passes_on_causal_precompute():
    # Pure ewm(adjust=False) recursion is causal; full-sample grid.
    assert assert_causal(_causal_ema_fixed, _frame(), sample=None) is None


def test_assert_causal_peek_positive_control_raises():
    # Injected leak (.shift(-1)) must be caught: detector works.
    with pytest.raises(AssertionError, match="non-causal"):
        assert_causal(_leaky, _frame())


def test_assert_causal_full_sample_normalization_is_a_leak():
    # df.close.mean() over the WHOLE series is non-causal -> caught.
    with pytest.raises(AssertionError, match="non-causal"):
        assert_causal(_noncausal_ema_with_global_mean, _frame())


def test_assert_causal_sampled_grid_runs_cheaply():
    # sample=8 evaluates only 8 truncation points but still passes a causal fn.
    assert_causal(_causal_ema_fixed, _frame(120), sample=8)
