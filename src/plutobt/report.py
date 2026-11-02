"""Human-readable IN-SAMPLE metrics report for a BacktestResult."""
from __future__ import annotations

from plutobt.engine import BacktestResult
from plutobt.metrics import MIN_TRADE_SAMPLE

_LABEL = "IN-SAMPLE, single pre-specified strategy; not predictive of live results."


def format_report(result: BacktestResult, symbol: str) -> str:
    m = result.metrics
    man = result.manifest
    # Direct indexing (not .get) so a missing required key raises KeyError, never blanks.
    sharpe = m["sharpe"]
    ci_lo, ci_hi = m["sharpe_ci"]
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

    lines = [
        f"=== PlutoBT report: {symbol} ===",
        _LABEL,
        f"Observations T = {t_obs}   (periods_per_year = {ppy})",
        f"Sharpe        = {sharpe:.3f}   95% CI [{ci_lo:.3f}, {ci_hi:.3f}]",
        f"Sortino       = {sortino:.3f}",
        f"Calmar        = {calmar:.3f}",
        f"Profit factor = {pf:.3f}",
        f"Expectancy    = {expectancy:.3f}",
        f"Win rate      = {win_rate:.3f}",
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
