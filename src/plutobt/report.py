"""Human-readable IN-SAMPLE metrics report for a BacktestResult."""
from __future__ import annotations

import math

from plutobt.engine import BacktestResult
from plutobt.metrics import (
    MIN_TRADE_SAMPLE,
    TRADING_DAYS_PER_YEAR,
    max_drawdown_recovered,
    trading_days_covered,
)

_LABEL = "IN-SAMPLE, single pre-specified strategy; not predictive of live results."


def _num(x: float, spec: str, *, nan: str = "n/a", inf: str | None = None) -> str:
    """Format a metric, naming why a non-finite value has no number (never a bare
    nan/inf on screen)."""
    if math.isnan(x):
        return nan
    if math.isinf(x) and inf is not None:
        return inf
    return format(x, spec)


def _fixed_sig(x: float, sig: int) -> str:
    """``x`` in fixed point with ``sig`` significant digits, trailing zeros kept
    (0.0125479, never 1.25479e-02: it is read off the screen)."""
    if x == 0.0:
        return format(x, f".{sig - 1}f")
    return format(x, f".{max(0, sig - 1 - math.floor(math.log10(abs(x))))}f")


def _sharpe_texts(sharpe_raw: float, factor: float, sharpe: float) -> tuple[str, str]:
    """The per-bar Sharpe and the annualization factor as printed, with digits enough
    (at least 6 significant, and 3 decimals of the factor) that the printed per-bar
    times the printed factor rounds to the printed Sharpe (3 decimals): a viewer can
    redo the multiplication from the screen. sharpe is sharpe_raw x factor, so a few
    more digits always get there."""
    want = format(sharpe, ".3f")
    for extra in range(10):
        bar = _fixed_sig(sharpe_raw, 6 + extra)
        fac = format(factor, f".{3 + extra}f")
        if format(float(bar) * float(fac), ".3f") == want:
            break
    return bar, fac


def _signed(x: float) -> str:
    """A CI bound with its sign and digits enough not to round to zero: 0.0004 prints
    +0.0004, never 0.000, so a bound that looks like zero always shows its side."""
    for decimals in range(3, 10):
        text = f"{x:+.{decimals}f}"
        if x == 0.0 or float(text) != 0.0:
            return text
    return f"{x:+.3e}"


def _days(x: float) -> str:
    """A day count to at most one decimal, no trailing zeros (20, 201.6)."""
    return f"{x:.1f}".rstrip("0").rstrip(".")


def format_report(result: BacktestResult, symbol: str) -> str:
    m = result.metrics
    man = result.manifest
    # Direct indexing (not .get) so a missing required key raises KeyError, never blanks.
    sharpe = m["sharpe"]
    sharpe_raw = m["sharpe_raw"]
    sharpe_se = m["sharpe_se"]
    ci_lo, ci_hi = m["sharpe_ci"]
    psr = m["psr"]
    sortino = m["sortino"]
    calmar = m["calmar"]
    pf = m["profit_factor"]
    expectancy = m["expectancy"]
    win_rate = m["win_rate"]
    dd_depth = m["max_drawdown_depth"]
    dd_dur = m["max_drawdown_duration"]
    ppy = m["periods_per_year"]
    t_obs = m["T"]
    # Closed trades: the count every trade statistic (win rate / PF / expectancy) and the
    # low-sample warning are computed over, so the Trades line can never disagree with them.
    n_trades = m["n_trades"]
    # A position still open at the last bar is marked to the last close: it is in equity
    # and Sharpe, but not in the trade stats, so it is disclosed on its own line.
    n_open = sum(1 for t in result.trades if t.open_at_end)
    # avg_r may be None (§0.2); handle gracefully for display.
    avg_r_val = m.get("avg_r")
    avg_r_str = f"{avg_r_val:.3f}" if avg_r_val is not None else "n/a"

    # The annualization factor: sharpe = sharpe_raw * sqrt(periods_per_year), and the SE
    # and CI carry the same factor (compute_metrics), so print it rather than hide it.
    factor = math.sqrt(ppy)
    if math.isnan(sharpe_raw) or math.isnan(sharpe):
        bar_str, factor_str = "", f"{factor:.3f}"
    else:
        bar_str, factor_str = _sharpe_texts(sharpe_raw, factor, sharpe)
    no_trades = "n/a (no trades)"
    no_wins_or_losses = "n/a (no winning or losing trades)"
    pf_nan = no_trades if n_trades == 0 else no_wins_or_losses
    # Every closed trade a scratch: no winner and no loser, so profit factor is 0/0 and
    # a 0.000 win rate would read as "lost them all".
    all_scratch = n_trades > 0 and math.isnan(pf)
    win_rate_str = (no_wins_or_losses if all_scratch
                    else _num(win_rate, ".3f", nan=no_trades))
    ci_known = not (math.isnan(ci_lo) or math.isnan(ci_hi))
    ci_str = f"[{_signed(ci_lo)}, {_signed(ci_hi)}]" if ci_known else "n/a"
    # compute_metrics leaves the Sharpe, its SE/CI and PSR NaN with no returns or zero
    # return volatility (a flat or constant-growth curve), and Sortino/Calmar NaN for
    # the 0/0 of a flat curve: say why, not "nan" (and never a 0.000 Sharpe).
    no_vol = "n/a (no returns)" if t_obs == 0 else "n/a (zero return volatility)"
    # Calmar divides an ANNUALIZED return (mean bar return x periods_per_year) by the
    # max drawdown. From under a year of bars that is an extrapolation (20 trading days
    # is x12.6), so a finite Calmar is printed only once the run covers a year; the
    # value stays in result.metrics["calmar"].
    days = trading_days_covered(len(result.equity_curve), ppy)
    if math.isfinite(calmar) and days < TRADING_DAYS_PER_YEAR:
        calmar_str = f"n/a (needs >= 1 year; this run covers {_days(days)} trading days)"
    else:
        calmar_str = _num(calmar, ".3f", nan=no_vol, inf="inf (no drawdown)")
    # The duration is the longest time below a prior peak, at any depth (not the
    # deepest drawdown's period). It may still be open at the last bar: it is then a
    # lower bound, not a finished period.
    dd_open = ("   (not recovered by the last bar)"
               if not max_drawdown_recovered(result.equity_curve) else "")
    expectancy_str = (f"{expectancy:.3f}   ($ per closed trade, net of costs)"
                      if math.isfinite(expectancy)
                      else _num(expectancy, ".3f", nan=no_trades))

    lines = [
        f"=== PlutoBT report: {symbol} ===",
        _LABEL,
        f"Observations T = {t_obs}   (periods_per_year = {ppy})",
        f"Annualization = sqrt(periods_per_year) = sqrt({ppy:g}) = {factor_str}",
        f"Sharpe/bar    = {bar_str}   (raw, per bar)" if bar_str
        else f"Sharpe/bar    = {no_vol}",
        f"Sharpe        = {sharpe:.3f}   (annualized: Sharpe/bar x {factor_str})"
        if bar_str else f"Sharpe        = {no_vol}",
        f"Sharpe SE     = {_num(sharpe_se, '.3f', nan=no_vol)}   (iid, skew/kurtosis-"
        f"adjusted; no autocorrelation adjustment)   95% CI {ci_str}",
    ]
    # Segment 8: an interval that straddles zero means the Sharpe cannot be told apart
    # from no edge at all. Say so on its own line, next to the interval. Decided on the
    # unrounded bounds; _signed prints them so the screen never contradicts it.
    if ci_known and ci_lo <= 0.0 <= ci_hi:
        lines.append("⚠ Sharpe CI includes 0: not statistically distinguishable from zero")
    lines += [
        f"PSR           = {_num(psr, '.3f', nan=no_vol if math.isnan(sharpe) else 'n/a')}"
        "   (Probabilistic Sharpe Ratio vs 0: 1 minus the one-sided p-value; "
        "skew/kurtosis-adjusted, Bailey & Lopez de Prado)",
        f"Sortino       = {_num(sortino, '.3f', nan=no_vol, inf='inf (no downside)')}",
        f"Calmar        = {calmar_str}",
        f"Profit factor = {_num(pf, '.3f', nan=pf_nan, inf='inf (no losing trades)')}",
        f"Expectancy    = {expectancy_str}",
        f"Win rate      = {win_rate_str}",
        f"Avg R         = {avg_r_str}",
        f"Max DD depth  = {dd_depth:.6f}",
        f"Max DD duration = {dd_dur} bars   (longest time below a prior peak){dd_open}",
        f"Trades        = {n_trades}   (closed)",
        f"Open at end   = {n_open}"
        + (" (marked to last close, excluded from trade stats)" if n_open else ""),
        f"data_sha256   = {man['data_sha256']}   (of the loaded bars as CSV, not the file)",
        f"engine version = {man['version']}",
    ]

    # §5.7 low-sample warning: every trade-based stat (profit factor, expectancy, win
    # rate, Avg R) is unreliable below MIN_TRADE_SAMPLE closed trades (the same
    # n_trades as the Trades line).
    if n_trades < MIN_TRADE_SAMPLE:
        lines.append(
            f"⚠ insufficient sample: n_trades={n_trades} < {MIN_TRADE_SAMPLE}; "
            f"profit factor, expectancy, win rate and Avg R unreliable"
        )

    return "\n".join(lines)
