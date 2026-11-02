"""Expanding-window (prefix-consistency) causality detector (§5.1, §10.2).

For each sampled bar t from 0 to n-2, ``precompute(df.iloc[:t+1])`` MUST equal
``precompute(df).iloc[:t+1]`` on every column, the base columns included. A
mismatch means a value at or before t depended on data at index > t (a lookahead
leak). This is the Lopez de Prado / Freqtrade truncation detector; the injected
``.shift(-1)`` "peek" test is the positive control. ``Engine.run`` runs it on every
strategy by default (``check_causality``).
"""
from __future__ import annotations

import warnings
from collections.abc import Callable

import numpy as np
import pandas as pd

from plutobt.broker import FrameError


class LookaheadError(FrameError, AssertionError):
    """A precompute column at bar t changed when the bars after t were removed.

    A FrameError (so a ValueError: the strategy's precompute made a frame the engine
    must not run) and an AssertionError (what ``assert_causal`` always raised).
    """


# How two numbers of a float column match (the full frame's value b, a prefix's a):
#     |a - b| <= RTOL * |b| + SCALE_TOL * scale
# where scale is that column's largest finite |value| in the full frame. Why both terms:
# - Honest float noise exists. pandas centres rolling skew and kurt on the whole input,
#   so their last digits move when later bars are removed: 5e-8 relative on the NQ
#   sample's rolling(20).kurt, 4e-7 on 1,500 one-minute-like bars. RTOL covers it.
# - That noise is absolute, not relative to each value: where a kurtosis crosses zero
#   its relative change is unbounded, so the column's own scale sets a floor. A floor
#   fixed in absolute units would not do: realized variance of 1e-5 returns sits near
#   1e-10, and an atol of 1e-10 let a whole-sample mean subtracted from it pass, while
#   a price column in the tens of thousands would need a far larger one.
# A leak moves a value by far more than a part per million of its column (a shift, a
# backfill, a whole-sample mean), so it is caught; a change smaller than that is out of
# reach by design. Integer, bool, text, categorical and datetime columns (by the full
# frame's dtype) match exactly: they carry no float noise.
RTOL = 1e-6
SCALE_TOL = 1e-6

# sample="auto": every prefix up to AUTO_FULL_MAX bars; above, AUTO_SPACED evenly
# spaced prefixes plus the first and last AUTO_EDGE (leaks cluster at the edges: a
# backfilled first bar, a whole-sample statistic that only moves near the end).
AUTO_FULL_MAX = 2000
AUTO_SPACED = 250
AUTO_EDGE = 20

_ROW_RULE = ("precompute must return one row per input bar, in order (no dropped, "
             "added or reordered rows); mark warm-up bars with NaN instead of dropping them")


_FLOAT, _EXACT, _OBJECT = "float", "exact", "object"


def _values(s: pd.Series) -> tuple[str, np.ndarray]:
    """(kind, values): float64 with NaN for a float column; the raw int or bool array for
    an integer or bool column without missing values (compared exactly); objects for
    everything else (text, categorical, datetime, nullable ints with NA, arrays)."""
    dtype = s.dtype
    if pd.api.types.is_bool_dtype(dtype) or pd.api.types.is_integer_dtype(dtype):
        if s.hasnans:
            return _OBJECT, s.to_numpy(dtype=object)
        return _EXACT, s.to_numpy()
    if pd.api.types.is_float_dtype(dtype):
        return _FLOAT, s.to_numpy(dtype="float64", na_value=np.nan)
    return _OBJECT, s.to_numpy(dtype=object)


_SEQUENCES = (np.ndarray, list, tuple, pd.Series, pd.DataFrame)
_FLOATS = (float, np.floating)


def _finite_max(x: np.ndarray) -> float:
    x = np.abs(x[np.isfinite(x)])
    return float(x.max()) if x.size else 0.0


def _scale(kind: str, values: np.ndarray) -> float:
    """A column's scale: its largest finite |value| (floats, float cells and arrays of
    numbers in an object column); 0 for a column compared exactly."""
    if kind == _FLOAT:
        return _finite_max(values)
    if kind == _EXACT:
        return 0.0
    scale = 0.0
    for cell in values:
        if isinstance(cell, _FLOATS) or isinstance(cell, _SEQUENCES):
            try:
                scale = max(scale, _finite_max(np.asarray(cell, dtype="float64").ravel()))
            except (TypeError, ValueError):
                pass
    return scale


def _same_object(u, v, col, rtol: float, atol: float) -> bool:
    """Whether two cells of an object column match: float numbers and arrays of floats
    by the column's tolerance, integer arrays exactly, anything else by ==. A cell whose
    == is neither true nor false is refused with a FrameError naming the column."""
    if isinstance(u, _SEQUENCES) or isinstance(v, _SEQUENCES):
        try:
            x, y = np.asarray(u), np.asarray(v)
            if x.shape != y.shape:
                return False
            if x.dtype.kind in "iub" and y.dtype.kind in "iub":
                return bool(np.array_equal(x, y))
            x, y = x.astype("float64"), y.astype("float64")
            return bool(np.isclose(x, y, rtol=rtol, atol=atol, equal_nan=True).all())
        except (TypeError, ValueError):
            x, y = np.asarray(u, dtype=object), np.asarray(v, dtype=object)
            return x.shape == y.shape and all(
                _same_object(p, q, col, rtol, atol)
                for p, q in zip(x.ravel(), y.ravel(), strict=True))
    if isinstance(u, _FLOATS) and isinstance(v, _FLOATS):
        return bool(np.isclose(u, v, rtol=rtol, atol=atol, equal_nan=True))
    try:
        return bool(u == v)
    except (TypeError, ValueError) as e:
        raise FrameError(
            f"column {col!r} holds {type(u).__name__} values that cannot be compared "
            f"(== gave neither true nor false: {e}), so the causality check cannot test "
            "it; store numbers, text or arrays of numbers per bar"
        ) from e


def _mismatch_at(a: tuple[str, np.ndarray], b: tuple[str, np.ndarray], col=None, *,
                 scale: float | None = None, exact: bool = False) -> int | None:
    """Position of the first value of ``a`` (got) that does not match ``b`` (expected,
    the full frame's), two ``_values`` of one length; None if none. NaN matches NaN.
    The expected column's kind decides the rule: a float column within RTOL and
    SCALE_TOL * scale (``scale`` defaults to the expected values' own), an integer or
    bool column exactly, an object column cell by cell (_same_object). ``exact=True``
    compares floats exactly too."""
    (kind_a, x), (kind_b, y) = a, b
    if exact:
        rtol = atol = 0.0
    else:
        rtol = RTOL
        atol = SCALE_TOL * (_scale(kind_b, y) if scale is None else scale)
    if kind_a == _EXACT and kind_b == _EXACT:
        same = x == y
    elif _OBJECT not in (kind_a, kind_b):
        xf, yf = x.astype("float64"), y.astype("float64")
        if kind_b == _EXACT:
            same = (xf == yf) | (np.isnan(xf) & np.isnan(yf))
        else:
            same = np.isclose(xf, yf, rtol=rtol, atol=atol, equal_nan=True)
    else:
        na_x, na_y = pd.isna(x), pd.isna(y)
        same = np.array([(p and q) or (not p and not q
                                       and _same_object(u, v, col, rtol, atol))
                         for u, v, p, q in zip(x, y, na_x, na_y, strict=True)], dtype=bool)
    bad = np.flatnonzero(~np.asarray(same, dtype=bool))
    return int(bad[0]) if bad.size else None


def _first_mismatch(a: pd.Series, b: pd.Series) -> int | None:
    """Position of the first value of ``a`` that is not exactly ``b``'s (NaN matching
    NaN), for columns that must come back unchanged (ts, the base price columns)."""
    return _mismatch_at(_values(a), _values(b), a.name, exact=True)


def _grid(n: int, sample: int | str | None) -> np.ndarray | range:
    """The truncation points t (prefix df[:t+1]) for ``n`` bars, from 0 to n-2."""
    if sample == "auto":
        if n <= AUTO_FULL_MAX:
            return range(0, n - 1)
        spaced = np.linspace(0, n - 2, num=AUTO_SPACED, dtype="int64")
        edges = np.r_[np.arange(min(AUTO_EDGE, n - 1)),
                      np.arange(max(0, n - 1 - AUTO_EDGE), n - 1)]
        return np.unique(np.r_[spaced, edges])
    if sample is None or sample >= n - 1:
        return range(0, n - 1)
    # Evenly spaced truncation points across [0, n-2]: prefixes df[:1] .. df[:n-1].
    return np.unique(np.linspace(0, n - 2, num=sample, dtype="int64"))


def assert_causal(
    precompute_fn: Callable[[pd.DataFrame], pd.DataFrame],
    df: pd.DataFrame,
    sample: int | str | None = None,
    *,
    full: pd.DataFrame | None = None,
    stacklevel: int = 2,
) -> None:
    """Raise LookaheadError if ``precompute_fn`` reads bars after the one it fills.

    Every column of the output is checked, the base ones (ts, open, high, low, close,
    volume, session) too, so a precompute that rewrites ``close`` with the next bar's
    close is caught like a new column would be. At each truncation point t the whole
    prefix result is compared with the full result's first t+1 rows. The truncation
    points run from t=0 (the 1-bar prefix, which catches a leak into row 0 such as a
    backfilled first return) to t=n-2; the n-bar prefix is the full frame itself.
    Float columns match within float noise, per column: a prefix value a matches the
    full value b when |a - b| <= RTOL * |b| + SCALE_TOL * scale, RTOL = SCALE_TOL = 1e-6
    and scale the column's largest finite |value| in the full frame. pandas' rolling
    skew and kurt move their last digits when later bars are removed (they centre on the
    whole input), which is not a leak; scaling the floor to the column keeps a leak in
    a tiny feature (realized variance near 1e-10) as visible as one in a price. So a
    change smaller than about a part per million of the column is out of reach, by
    design. Integer, bool, text, categorical and datetime columns (by the full frame's
    dtype) match exactly. pandas' rolling skew and kurt of raw PRICE LEVELS over short
    windows (kurt under about 50 bars, skew under about 10) on a series that wanders far
    from its mean can move by far more than that, up to about 1% of the column, between
    a prefix and the full frame: such a column is refused although honest. Compute them
    on returns, where pandas' result does not move at all.

    ``sample=None`` (or >= n-1) truncates at every bar: exhaustive, at a cost that grows
    with the square of the bar count. ``sample=k`` truncates at k evenly spaced bars
    from t=0 to t=n-2 only. ``sample="auto"`` (Engine.run's default) is the full grid
    up to 2,000 bars and, above that, 250 evenly spaced prefixes plus the first 20 and
    last 20, with a UserWarning that the check was sampled. A sampled grid CAN MISS a
    sparse leak: one that touches a few rows (say, the last bar of each session)
    between two truncation points is never seen.

    A precompute that drops, adds or reorders rows is refused with a FrameError (a
    ValueError) before any compare, since its rows no longer line up with the bars.

    ``full`` is the frame the prefixes are compared with: by default
    ``precompute_fn(df)``; Engine.run passes the frame it trades, so what is checked is
    exactly what is traded. ``precompute_fn`` must start from the same state on every
    call: Engine.run gives each call its own deepcopy of the strategy as it was passed
    in, so nothing one call saves on the strategy reaches the next.
    """
    if full is None:
        full = precompute_fn(df.copy())
    full = full.reset_index(drop=True)
    n = len(df)
    _check_rows(full, df, "precompute(df)")
    if n < 2:
        return None

    grid = _grid(n, sample)
    if sample == "auto" and len(grid) < n - 1:   # an explicit k was asked for
        warnings.warn(f"causality check sampled: {len(grid)} of {n - 1} prefixes on {n} "
                      "bars (set causality_sample = None to check every prefix)",
                      UserWarning, stacklevel=stacklevel)

    expected = {}
    for col in full.columns:
        kind, values = _values(full[col])
        expected[col] = (kind, values, _scale(kind, values))
    for t in grid:
        t = int(t)
        prefix = precompute_fn(df.iloc[: t + 1].copy()).reset_index(drop=True)
        _check_rows(prefix, df.iloc[: t + 1], f"precompute(df[:{t + 1}])")
        for col, (kind, values, scale) in expected.items():
            if col not in prefix.columns:
                raise LookaheadError(
                    f"non-causal column {col!r} at t={t}: it is missing from "
                    f"precompute(df[:{t + 1}]), so it depends on how many bars follow"
                )
            bad = _mismatch_at(_values(prefix[col]), (kind, values[: t + 1]), col,
                               scale=scale)
            if bad is not None:
                raise LookaheadError(
                    f"non-causal column {col!r} at t={bad}: "
                    f"prefix={prefix[col].iloc[bad]!r} != full={full[col].iloc[bad]!r} "
                    f"(the value changed when the bars after t={t} were removed)"
                )
    return None


def first_difference(got: pd.DataFrame, expected: pd.DataFrame) -> tuple | None:
    """(column, row) of the first value of ``expected`` that ``got`` does not match, by
    the same rule as the causality check (a missing column is row None); None if all
    match. Both frames must have the same number of rows."""
    got, expected = got.reset_index(drop=True), expected.reset_index(drop=True)
    for col in expected.columns:
        if col not in got.columns:
            return col, None
        bad = _mismatch_at(_values(got[col]), _values(expected[col]), col)
        if bad is not None:
            return col, bad
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
