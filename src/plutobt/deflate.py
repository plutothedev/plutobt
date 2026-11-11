"""``python -m plutobt.deflate``: deflate a Sharpe you already have, no code needed (3.0
spec §4.1).

The maths is in ``plutobt.deflation``; its names are re-exported here so
``from plutobt.deflate import deflated_sharpe`` keeps working. Nothing in the package
imports this module, which is what lets ``python -m plutobt.deflate`` run without Python's
"found in sys.modules" warning.
"""
from __future__ import annotations

import math

from plutobt.deflation import (
    EULER_GAMMA,
    MAX_TRIALS,
    SURVIVES,
    deflated_sharpe,
    expected_max_sharpe,
    null_var_sr,
    verdict,
)

__all__ = ["EULER_GAMMA", "SURVIVES", "deflated_sharpe", "expected_max_sharpe", "main",
           "null_var_sr", "verdict"]

_MAX_SHARPE = 1e6          # per period; far past any real Sharpe, far short of overflow

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

_LABEL = "IN-SAMPLE, best of several tries; not predictive of live results."
_LABEL_ONE = "IN-SAMPLE, a single try; not predictive of live results."


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
    if a.trials > MAX_TRIALS:
        raise _Refused(f"--trials {a.trials:,} is too large to compute a luck bar for "
                       f"(the limit is {MAX_TRIALS:,})")
    if abs(a.sharpe) / math.sqrt(a.periods_per_year) > _MAX_SHARPE:
        raise _Refused(f"--sharpe {a.sharpe:g} with --periods-per-year "
                       f"{a.periods_per_year:g} is a Sharpe of over {_MAX_SHARPE:g} a "
                       "period, which is not a Sharpe ratio")
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
    if math.isfinite(dsr):
        lines.append(f"Verdict: {verdict(dsr, a.trials)}")
    else:
        lines.append("Verdict: not computable: this skew and kurtosis leave a Sharpe of "
                     f"{sr:.3g} a period no standard error (1 - skew x SR + (kurtosis - 1)"
                     "/4 x SR squared must be above 0)")
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
    lines.append(_LABEL_ONE if a.trials == 1 else _LABEL)
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
