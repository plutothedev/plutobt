"""The Deflated Sharpe Ratio: how good the best of N tries has to look before it is more
than luck (3.0 spec §4; Bailey and Lopez de Prado, "The Deflated Sharpe Ratio", 2014).

Every Sharpe here is PER BAR (not annualized), the convention of ``metrics.psr``.

    expected_max_sharpe(N, V)  the luck bar: the Sharpe the best of N zero-edge tries
                               reaches by luck alone, given the spread V of their Sharpes
    deflated_sharpe(r, N, V)   the probability the best try truly beats that luck bar
                               (``psr`` with the luck bar as its benchmark)
    null_var_sr(T)             the spread zero-edge tries show over T returns, for when
                               the real spread is unknown

Counting near-identical settings as separate tries overstates N, so the result errs
strict. The paper's estimate of an effective N is out of scope.

The maths lives here and ``plutobt.deflate`` holds the command line (it re-exports these
names). The package imports this module, never that one, so ``python -m plutobt.deflate``
runs a module nothing has imported yet.
"""
from __future__ import annotations

import math
from statistics import NormalDist

from plutobt.metrics import psr

EULER_GAMMA = 0.5772156649015329
SURVIVES = 0.95
MAX_TRIALS = 10 ** 15      # past this, 1 - 1/(N e) rounds to 1.0 and the formula has no value
_Z = NormalDist().inv_cdf


def expected_max_sharpe(n_trials: int, var_sr: float) -> float:
    """The luck bar: ``sqrt(var_sr) * ((1 - g) Z(1 - 1/N) + g Z(1 - 1/(N e)))``, the
    expected maximum of N zero-edge per-bar Sharpes (paper Eq. 1, mean 0). One try has
    nothing to deflate: 0."""
    if isinstance(n_trials, bool) or int(n_trials) != n_trials or n_trials < 1:
        raise ValueError(f"n_trials must be a whole number of at least 1, got {n_trials!r}")
    if n_trials > MAX_TRIALS:
        raise ValueError(
            f"n_trials = {n_trials:,} is too large to compute a luck bar for (the limit is "
            f"{MAX_TRIALS:,})")
    if not math.isfinite(var_sr) or var_sr < 0:
        raise ValueError(f"var_sr must be a finite variance of 0 or more, got {var_sr}")
    n = int(n_trials)
    if n == 1:
        return 0.0
    bracket = ((1 - EULER_GAMMA) * _Z(1 - 1 / n)
               + EULER_GAMMA * _Z(1 - 1 / (n * math.e)))
    return math.sqrt(var_sr) * bracket


def null_var_sr(n_returns: int) -> float:
    """``1 / (T - 1)``: the variance of a per-bar Sharpe estimate over T returns when the
    true Sharpe is zero."""
    if n_returns < 2:
        raise ValueError(f"a Sharpe needs at least 2 returns, got {n_returns}")
    return 1.0 / (n_returns - 1)


def deflated_sharpe(returns, n_trials: int, var_sr: float) -> float:
    """PSR of ``returns`` against the luck bar of ``n_trials`` tries with Sharpe
    variance ``var_sr``. NaN when the returns have no volatility."""
    return psr(returns, sr0=expected_max_sharpe(n_trials, var_sr))


def verdict(dsr: float, n_trials: int) -> str:
    """The one-line verdict the report and the command print."""
    if not math.isfinite(dsr):
        return "not computable (no return volatility)"
    if dsr >= SURVIVES:
        return "survives deflation"
    return f"does not survive deflation (N = {n_trials:,} tries)"
