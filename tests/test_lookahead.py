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


# ---------------------------------------------------------------------------
# Film-ready audit (2026-11-02): base-column rewrites, row changes, sparse leaks.
# ---------------------------------------------------------------------------

def _close_from_next_bar(df):
    # A "smoothed close" that is really the next bar's close: a leak in a BASE column.
    df = df.copy()
    df["close"] = df["close"].shift(-1).fillna(df["close"])
    return df


def _causal_dropna(df):
    df = df.copy()
    df["sma"] = df["close"].rolling(5).mean()
    return df.dropna().reset_index(drop=True)


def _sparse_leak(df):
    # Causal everywhere except row 37, which reads the next bar's close.
    df = df.copy()
    df["x"] = df["close"].ewm(span=10, adjust=False).mean()
    if len(df) > 38:
        df.loc[37, "x"] = df.loc[38, "close"]
    return df


def test_assert_causal_catches_a_base_column_rewritten_with_future_data():
    # Unchecked, only columns the input lacked were compared: this passed (probe p04).
    with pytest.raises(AssertionError, match="non-causal column 'close'"):
        assert_causal(_close_from_next_bar, _frame())


def test_assert_causal_passes_an_honest_base_column_rewrite_control():
    # Control for the check above: a same-bar (causal) rewrite is not a leak.
    def same_bar(df):
        df = df.copy()
        df["close"] = (df["high"] + df["low"]) / 2.0
        return df

    assert assert_causal(same_bar, _frame()) is None


@pytest.mark.parametrize("fn", [_causal_dropna, lambda df: df.iloc[::-1].copy(),
                                lambda df: df.iloc[1:].copy()],
                         ids=["dropna", "reversed", "first-row-dropped"])
def test_assert_causal_refuses_a_precompute_that_drops_or_reorders_rows(fn):
    # Unchecked, dropna crashed on an IndexError (probe p12); a reorder compared the
    # wrong rows. Both are refused by name: one row per input bar, in order.
    with pytest.raises(ValueError, match="one row per input bar, in order"):
        assert_causal(fn, _frame())


def test_assert_causal_compares_non_numeric_columns():
    # A text signal column used to crash the float compare; an honest one passes and a
    # leaky one is caught.
    def honest(df):
        df = df.copy()
        df["side"] = np.where(df["close"] > df["close"].shift(1), "up", "down")
        return df

    def leaky(df):
        df = df.copy()
        df["side"] = np.where(df["close"].shift(-1) > df["close"], "up", "down")
        return df

    assert assert_causal(honest, _frame()) is None
    with pytest.raises(AssertionError, match="non-causal column 'side'"):
        assert_causal(leaky, _frame())


def test_assert_causal_sampled_mode_can_miss_a_sparse_leak():
    # Documented limit: a sampled grid only truncates at its sample points, so a leak on
    # one row between them goes unseen. The full grid (sample=None) catches it.
    assert assert_causal(_sparse_leak, _frame(), sample=5) is None
    with pytest.raises(AssertionError, match="non-causal column 'x'"):
        assert_causal(_sparse_leak, _frame(), sample=None)
    assert "can miss" in assert_causal.__doc__.lower()


def test_lookahead_error_is_an_assertion_and_a_frame_error():
    from plutobt.broker import FrameError
    from plutobt.lookahead import LookaheadError

    with pytest.raises(LookaheadError) as e:
        assert_causal(_leaky, _frame())
    assert isinstance(e.value, AssertionError) and isinstance(e.value, FrameError)


# ---------------------------------------------------------------------------
# Review fix (2026-11-02): the grid covers every prefix df[:1] .. df[:n-1].
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fn", [
    lambda d: d.assign(x=d["close"].pct_change().bfill()),
    lambda d: d.assign(x=d["close"].diff().bfill()),
    lambda d: d.assign(x=np.r_[d["close"].to_numpy()[1:2], d["close"].to_numpy()[1:]]
                       if len(d) > 1 else d["close"].to_numpy()),
], ids=["pct_change-bfill", "diff-bfill", "row0-is-close1"])
def test_assert_causal_catches_a_leak_into_row_zero(fn):
    # The grid started at the 2-bar prefix, so row 0 was never computed alone: a
    # backfilled first return (row 0 = close1/close0 - 1) passed and on_bar(0) saw it.
    with pytest.raises(AssertionError, match="non-causal column 'x' at t=0"):
        assert_causal(fn, _frame())


def test_assert_causal_row_zero_control_passes():
    # Control: the same return without the backfill is NaN on row 0 in every prefix.
    assert assert_causal(lambda d: d.assign(x=d["close"].pct_change()), _frame()) is None


def _prefix_lengths(n, sample):
    calls = []

    def record(d):
        calls.append(len(d))
        return d.assign(x=d["close"].cumsum())

    assert_causal(record, _frame(n), sample=sample)
    return calls


def test_full_grid_computes_every_proper_prefix_once():
    # The full frame once, then df[:1] .. df[:n-1]; df[:n] again could never mismatch.
    assert _prefix_lengths(10, None) == [10] + list(range(1, 10))


def test_sampled_grid_spans_the_first_and_last_proper_prefix():
    calls = _prefix_lengths(60, 5)
    assert calls[0] == 60 and calls[1] == 1 and calls[-1] == 59 and len(calls) == 6
