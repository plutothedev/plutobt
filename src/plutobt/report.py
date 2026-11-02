"""Human-readable IN-SAMPLE metrics report for a BacktestResult."""
from __future__ import annotations

import math

from plutobt.engine import BacktestResult
from plutobt.metrics import MIN_TRADE_SAMPLE

_LABEL = "IN-SAMPLE, single pre-specified strategy; not predictive of live results."


def _num(x: float, spec: str, *, nan: str = "n/a", inf: str | None = None) -> str:
    """Format a metric, naming why a non-finite value has no number (never a bare
    nan/inf on screen)."""
    if math.isnan(x):
        return nan
    if math.isinf(x) and inf is not None:
        return inf
    return format(x, spec)


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
    no_trades = "n/a (no trades)"
    pf_nan = no_trades if n_trades == 0 else "n/a (no winning or losing trades)"
    ci_known = not (math.isnan(ci_lo) or math.isnan(ci_hi))
    ci_str = f"[{ci_lo:.3f}, {ci_hi:.3f}]" if ci_known else "n/a"
    # compute_metrics leaves SE/CI NaN with no returns or zero return volatility (a flat
    # curve), and Sortino/Calmar NaN for the 0/0 of a flat curve: say why, not "nan".
    no_vol = "n/a (no returns)" if t_obs == 0 else "n/a (zero return volatility)"

    lines = [
        f"=== PlutoBT report: {symbol} ===",
        _LABEL,
        f"Observations T = {t_obs}   (periods_per_year = {ppy})",
        f"Annualization = sqrt(periods_per_year) = sqrt({ppy:g}) = {factor:.3f}",
        f"Sharpe/bar    = {sharpe_raw:.5f}   (raw, per bar)",
        f"Sharpe        = {sharpe:.3f}   (annualized: Sharpe/bar x {factor:.3f})",
        f"Sharpe SE     = {_num(sharpe_se, '.3f', nan=no_vol)}   (iid, skew/kurtosis-"
        f"adjusted; no autocorrelation adjustment)   95% CI {ci_str}",
    ]
    # Segment 8: an interval that straddles zero means the Sharpe cannot be told apart
    # from no edge at all. Say so on its own line, next to the interval.
    if ci_known and ci_lo <= 0.0 <= ci_hi:
        lines.append("⚠ Sharpe CI includes 0: not statistically distinguishable from zero")
    lines += [
        f"PSR           = {_num(psr, '.3f')}   (P[true Sharpe > 0], skew/kurtosis-"
        "adjusted, Bailey & Lopez de Prado)",
        f"Sortino       = {_num(sortino, '.3f', nan=no_vol, inf='inf (no downside)')}",
        f"Calmar        = {_num(calmar, '.3f', nan=no_vol, inf='inf (no drawdown)')}",
        f"Profit factor = {_num(pf, '.3f', nan=pf_nan, inf='inf (no losing trades)')}",
        f"Expectancy    = {_num(expectancy, '.3f', nan=no_trades)}",
        f"Win rate      = {_num(win_rate, '.3f', nan=no_trades)}",
        f"Avg R         = {avg_r_str}",
        f"Max DD depth  = {dd_depth:.6f}",
        f"Max DD duration = {dd_dur} bars",
        f"Trades        = {n_trades}   (closed)",
        f"Open at end   = {n_open}"
        + (" (marked to last close, excluded from trade stats)" if n_open else ""),
        f"data_sha256   = {man['data_sha256']}",
        f"engine version = {man['version']}",
    ]

    # §5.7 low-sample warning: trade-based stats (PF/expectancy) are unreliable
    # below MIN_TRADE_SAMPLE closed trades (the same n_trades as the Trades line).
    if n_trades < MIN_TRADE_SAMPLE:
        lines.append(
            f"⚠ insufficient sample: n_trades={n_trades} < {MIN_TRADE_SAMPLE}; "
            f"PF/expectancy unreliable"
        )

    return "\n".join(lines)
