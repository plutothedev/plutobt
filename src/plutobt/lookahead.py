"""Expanding-window (prefix-consistency) causality detector (§5.1, §10.2).

For each sampled bar t from 0 to n-2, ``precompute(df.iloc[:t+1])`` MUST equal
``precompute(df).iloc[:t+1]`` on every column, the base columns included. A
mismatch means a value at or before t depended on data at index > t (a lookahead
leak). This is the Lopez de Prado / Freqtrade truncation detector; the injected
``.shift(-1)`` "peek" test is the positive control. ``Engine.run`` runs it on every
strategy by default (``check_causality``).
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from plutobt.broker import FrameError


class LookaheadError(FrameError, AssertionError):
    """A precompute column at bar t changed when the bars after t were removed.

    A FrameError (so a ValueError: the strategy's precompute made a frame the engine
    must not run) and an AssertionError (what ``assert_causal`` always raised).
    """


# Float noise, not a leak: two numbers match within RTOL relative or ATOL absolute.
# pandas centres rolling moments (skew, kurt) on the whole input, so their last digits
# move by about 1e-11 relative when later bars are removed; a real leak moves a value
# by far more (one part per million is caught 1000 times over).
RTOL = 1e-9
ATOL = 1e-10

_ROW_RULE = ("precompute must return one row per input bar, in order (no dropped, "
             "added or reordered rows); mark warm-up bars with NaN instead of dropping them")


def _values(s: pd.Series) -> tuple[bool, np.ndarray]:
    """(numeric?, values): float64 with NaN for numbers and bools, objects otherwise."""
    if pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
        return True, s.to_numpy(dtype="float64", na_value=np.nan)
    return False, s.to_numpy(dtype=object)


def _mismatch_at(a: tuple[bool, np.ndarray], b: tuple[bool, np.ndarray]) -> int | None:
    """Position of the first value that differs between two ``_values`` of one length,
    NaN matching NaN; numbers within RTOL/ATOL, anything else by equality. None if none."""
    (num_a, x), (num_b, y) = a, b
    if num_a and num_b:
        same = np.isclose(x, y, rtol=RTOL, atol=ATOL, equal_nan=True)
    else:
        na_x, na_y = pd.isna(x), pd.isna(y)
        same = np.array([(p and q) or (not p and not q and bool(u == v))
                         for u, v, p, q in zip(x, y, na_x, na_y, strict=True)], dtype=bool)
    bad = np.flatnonzero(~same)
    return int(bad[0]) if bad.size else None


def _first_mismatch(a: pd.Series, b: pd.Series) -> int | None:
    """``_mismatch_at`` for two Series of one length."""
    return _mismatch_at(_values(a), _values(b))


def assert_causal(
    precompute_fn: Callable[[pd.DataFrame], pd.DataFrame],
    df: pd.DataFrame,
    sample: int | None = None,
) -> None:
    """Raise LookaheadError if ``precompute_fn`` reads bars after the one it fills.

    Every column of the output is checked, the base ones (ts, open, high, low, close,
    volume, session) too, so a precompute that rewrites ``close`` with the next bar's
    close is caught like a new column would be. At each truncation point t the whole
    prefix result is compared with the full result's first t+1 rows. The truncation
    points run from t=0 (the 1-bar prefix, which catches a leak into row 0 such as a
    backfilled first return) to t=n-2; the n-bar prefix is the full frame itself.
    Numbers match within float noise, rtol=1e-9 and atol=1e-10 (RTOL, ATOL): pandas'
    rolling skew and kurt move their last digits (about 1e-11 relative) when later bars
    are removed, which is not a leak.

    ``sample=None`` (or >= n-1) truncates at every bar: exhaustive, at a cost that grows
    with the square of the bar count. ``sample=k`` truncates at k evenly spaced bars
    from t=0 to t=n-2 only, and CAN MISS a sparse leak: one that touches a few rows (say, the
    last bar of each session) between two truncation points is never seen.

    A precompute that drops, adds or reorders rows is refused with a FrameError (a
    ValueError) before any compare, since its rows no longer line up with the bars.
    """
    full = precompute_fn(df.copy()).reset_index(drop=True)
    n = len(df)
    _check_rows(full, df, "precompute(df)")
    if n < 2:
        return None

    if sample is None or sample >= n - 1:
        grid = range(0, n - 1)
    else:
        # Evenly spaced truncation points across [0, n-2]: prefixes df[:1] .. df[:n-1].
        grid = np.unique(np.linspace(0, n - 2, num=sample, dtype="int64"))

    expected = {col: _values(full[col]) for col in full.columns}
    for t in grid:
        t = int(t)
        prefix = precompute_fn(df.iloc[: t + 1].copy()).reset_index(drop=True)
        _check_rows(prefix, df.iloc[: t + 1], f"precompute(df[:{t + 1}])")
        for col, (numeric, values) in expected.items():
            if col not in prefix.columns:
                raise LookaheadError(
                    f"non-causal column {col!r} at t={t}: it is missing from "
                    f"precompute(df[:{t + 1}]), so it depends on how many bars follow"
                )
            bad = _mismatch_at(_values(prefix[col]), (numeric, values[: t + 1]))
            if bad is not None:
                raise LookaheadError(
                    f"non-causal column {col!r} at t={bad}: "
                    f"prefix={prefix[col].iloc[bad]!r} != full={full[col].iloc[bad]!r} "
                    f"(the value changed when the bars after t={t} were removed)"
                )
    return None


def _check_rows(out: pd.DataFrame, given: pd.DataFrame, call: str) -> None:
    """Refuse a precompute result whose rows are not the given bars, in order."""
    if len(out) != len(given):
        raise FrameError(f"{call} returned {len(out)} rows for {len(given)} input bars: "
                         f"{_ROW_RULE}")
    if "ts" in out.columns and "ts" in given.columns:
        a, b = out["ts"].to_numpy(), given["ts"].to_numpy()
        if a.dtype == b.dtype and np.array_equal(a, b):
            return
        bad = _first_mismatch(out["ts"].reset_index(drop=True),
                              given["ts"].reset_index(drop=True))
        if bad is not None:
            raise FrameError(f"{call} row {bad} has ts {out['ts'].iloc[bad]!r}, the input "
                             f"bar there has ts {given['ts'].iloc[bad]!r}: {_ROW_RULE}")
