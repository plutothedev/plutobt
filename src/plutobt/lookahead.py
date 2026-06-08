"""Expanding-window (prefix-consistency) causality detector (§5.1, §10.2).

For each sampled bar t, ``precompute(df.iloc[:t+1]).iloc[t]`` MUST equal
``precompute(df).iloc[t]`` on every derived column. A mismatch means the
value at t depended on data at index > t (a lookahead leak). This is the
Lopez de Prado / Freqtrade truncation detector; the injected ``.shift(-1)``
"peek" test is the positive control.
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd


def assert_causal(
    precompute_fn: Callable[[pd.DataFrame], pd.DataFrame],
    df: pd.DataFrame,
    sample: int | None = None,
) -> None:
    full = precompute_fn(df.copy()).reset_index(drop=True)
    n = len(full)
    derived = [c for c in full.columns if c not in df.columns]
    if not derived:
        return None

    if sample is None or sample >= n:
        grid = range(1, n)
    else:
        # Evenly spaced truncation points across [1, n-1].
        grid = np.unique(np.linspace(1, n - 1, num=sample, dtype="int64"))

    for t in grid:
        prefix = precompute_fn(df.iloc[: t + 1].copy()).reset_index(drop=True)
        for col in derived:
            a = prefix[col].iloc[t]
            b = full[col].iloc[t]
            a_nan = isinstance(a, float) and np.isnan(a)
            b_nan = isinstance(b, float) and np.isnan(b)
            if a_nan and b_nan:
                continue
            if a_nan != b_nan or not np.allclose(
                np.asarray(a, dtype="float64"),
                np.asarray(b, dtype="float64"),
                rtol=1e-12, atol=1e-12,
            ):
                raise AssertionError(
                    f"non-causal column {col!r} at t={t}: "
                    f"prefix={a!r} != full={b!r}"
                )
    return None
