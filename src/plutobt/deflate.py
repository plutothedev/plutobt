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
"""
from __future__ import annotations

import math
from statistics import NormalDist

from plutobt.metrics import psr

EULER_GAMMA = 0.5772156649015329
SURVIVES = 0.95
_Z = NormalDist().inv_cdf


def expected_max_sharpe(n_trials: int, var_sr: float) -> float:
    """The luck bar: ``sqrt(var_sr) * ((1 - g) Z(1 - 1/N) + g Z(1 - 1/(N e)))``, the
    expected maximum of N zero-edge per-bar Sharpes (paper Eq. 1, mean 0). One try has
    nothing to deflate: 0."""
    if int(n_trials) != n_trials or n_trials < 1:
        raise ValueError(f"n_trials must be a whole number of at least 1, got {n_trials}")
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


# ---------------------------------------------------------------------------
# python -m plutobt.deflate: the no-code deflator (spec §4.1)
# ---------------------------------------------------------------------------

_HELP = """\
Deflate a Sharpe ratio for the number of versions you tried before you kept the best one.

TradingView users: the Strategy Tester computes its Sharpe from MONTHLY returns, after
taking off a risk-free rate (2% a year by default, the risk_free_rate argument of
strategy()), so deflating it against zero here is slightly strict. TradingView does not
document whether its figure is annualized. If yours is annualized, pass it as it is with
--periods-per-year 12 --returns <months tested>. If it is a plain monthly figure,
multiply it by 3.46 (the square root of 12) first.

example:
  python -m plutobt.deflate --sharpe 2.1 --periods-per-year 12 --returns 60 --trials 40
"""

_LABEL = "IN-SAMPLE, single pre-specified strategy; not predictive of live results."


class _Refused(ValueError):
    pass


def _parser():
    import argparse

    p = argparse.ArgumentParser(
        prog="python -m plutobt.deflate", description=_HELP,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sharpe", type=float, required=True,
                   help="the best Sharpe you kept, ANNUALIZED")
    p.add_argument("--periods-per-year", type=float, required=True,
                   help="return periods a year behind that Sharpe (12 monthly, 252 daily)")
    p.add_argument("--returns", type=int, required=True,
                   help="how many returns stand behind it (months tested, days tested)")
    p.add_argument("--trials", type=int, required=True,
                   help="how many versions you tried, the kept one included")
    p.add_argument("--skew", type=float, default=None,
                   help="skew of the returns (give with --kurtosis)")
    p.add_argument("--kurtosis", type=float, default=None,
                   help="Pearson kurtosis of the returns, normal = 3 (give with --skew)")
    return p


def _check(a) -> None:
    if not math.isfinite(a.sharpe):
        raise _Refused(f"--sharpe must be a finite number, got {a.sharpe}")
    if not (math.isfinite(a.periods_per_year) and a.periods_per_year > 0):
        raise _Refused(f"--periods-per-year must be above 0, got {a.periods_per_year}")
    if a.returns < 2:
        raise _Refused(f"--returns must be at least 2 (a Sharpe needs 2 returns), "
                       f"got {a.returns}")
    if a.trials < 1:
        raise _Refused(f"--trials must be at least 1 (the version you kept), got {a.trials}")
    if (a.skew is None) != (a.kurtosis is None):
        raise _Refused("give both --skew and --kurtosis, or neither (then normal returns "
                       "are assumed)")
    if a.skew is not None:
        if not (math.isfinite(a.skew) and math.isfinite(a.kurtosis)):
            raise _Refused("--skew and --kurtosis must be finite numbers")
        if a.kurtosis < 1 + a.skew ** 2:
            raise _Refused(
                f"skew {a.skew:g} with kurtosis {a.kurtosis:g} is impossible: Pearson "
                f"kurtosis is always at least 1 + skew squared ({1 + a.skew ** 2:g}). "
                "Kurtosis here is Pearson (normal = 3), not excess (normal = 0)")


def _report(a) -> str:
    from plutobt.metrics import psr_from_moments

    root = math.sqrt(a.periods_per_year)
    sr = a.sharpe / root
    normal = a.skew is None
    g3, g4 = (0.0, 3.0) if normal else (a.skew, a.kurtosis)
    var_sr = null_var_sr(a.returns)
    bar = expected_max_sharpe(a.trials, var_sr)
    dsr = psr_from_moments(sr, a.returns, g3, g4, bar)
    tries = "1 try" if a.trials == 1 else f"{a.trials:,} tries"
    lines = [f"PlutoBT deflate: best Sharpe {a.sharpe:.2f} (annualized, "
             f"{a.periods_per_year:g} periods a year) over {a.returns:,} returns, "
             f"best of {tries}"]
    if a.trials == 1:
        lines.append("Luck bar: one try, nothing to deflate (this is the plain PSR "
                     "against zero)")
    else:
        lines.append(f"Luck bar: the best of {a.trials:,} zero-edge tries would show an "
                     f"annualized Sharpe of about {bar * root:.2f} by luck alone")
    lines.append(f"Deflated Sharpe: {dsr:.3f}")
    lines.append(f"Verdict: {verdict(dsr, a.trials)}")
    if a.trials > 1:
        lines.append(f"Assumed: the spread of the {tries} is what zero-edge tries show "
                     f"over {a.returns:,} returns (their real spread is unknown here)")
    if normal:
        lines.append("Assumed: normal returns (skew 0, kurtosis 3); pass --skew and "
                     "--kurtosis if you know them")
    else:
        lines.append(f"Using: skew {g3:g}, kurtosis {g4:g}")
    if a.returns < 30:
        lines.append("Warning: under 30 returns, every Sharpe statistic here is unreliable")
    lines.append(_LABEL)
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    """Print the deflated verdict; 0 when it ran, 2 when the input was refused."""
    import sys

    a = _parser().parse_args(argv)
    try:
        _check(a)
    except _Refused as refused:
        print(f"deflate: {refused}", file=sys.stderr)
        return 2
    sys.stdout.write(_report(a))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
