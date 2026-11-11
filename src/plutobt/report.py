"""Human-readable IN-SAMPLE metrics report for a BacktestResult."""
from __future__ import annotations

import math

from plutobt.deflation import LOW_ACTIVE, MIN_ACTIVE
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
        lines.append("WARNING: Sharpe CI includes 0: not statistically distinguishable from zero")
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
    ]
    if "rolls_taken" in m:
        lines += [
            f"Rolls taken   = {m['rolls_taken']}   (contract rolls a position was held "
            "through)",
            f"Roll cost     = {m['roll_cost_total']:.2f}   ($, commission and slippage on "
            "both legs, inside net P&L)",
            f"Roll yield    = {m['roll_yield']:.2f}   ($)",
            "Roll yield: the jump an unadjusted front-month chart would show at each "
            "roll, which is carry, not P&L.",
        ]
    lines += [
        f"data_sha256   = {man['data_sha256']}   (of the loaded bars as CSV, not the file)",
        f"engine version = {man['version']}",
    ]

    # §5.7 low-sample warning: every trade-based stat (profit factor, expectancy, win
    # rate, Avg R) is unreliable below MIN_TRADE_SAMPLE closed trades (the same
    # n_trades as the Trades line).
    if n_trades < MIN_TRADE_SAMPLE:
        lines.append(
            f"WARNING: insufficient sample: n_trades={n_trades} < {MIN_TRADE_SAMPLE}; "
            f"profit factor, expectancy, win rate and Avg R unreliable"
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 3.0: the study report
# ---------------------------------------------------------------------------

_STUDY_LABEL = "IN-SAMPLE, best of a grid search; not predictive of live results."
_OOS_LABEL = "OUT-OF-SAMPLE (rerun flat in each test window)."


def _params(p: dict) -> str:
    return ", ".join(f"{k}={v}" for k, v in p.items())


def _rows(r: range) -> str:
    return f"rows {r.start}-{r.stop - 1}"


def format_study(result, symbol: str) -> str:
    """Render a ``StudyResult`` (``plutobt.study.run_study``): every setting, the
    Deflated Sharpe against every recorded try, PBO, then the out-of-sample walk-forward
    and purged CV. Sharpes print annualized (per-bar x sqrt(periods_per_year))."""
    from plutobt import __version__

    root = math.sqrt(result.periods_per_year)
    grid = result.grid
    names = [c for c in grid.columns if c not in ("sharpe", "sharpe_ann", "trades",
                                                   "net_pnl")]
    lines = [
        f"=== PlutoBT study: {result.strategy.split(':')[-1]} on {symbol}, "
        f"{len(grid)} settings ===",
        _STUDY_LABEL,
        f"Settings   (Sharpe annualized: per bar x sqrt({result.periods_per_year:g}) = "
        f"x {root:.3f})",
    ]
    head = [*names, "Sharpe", "Trades", "Net P&L"]
    body = []
    best_key = tuple(result.best.get(n) for n in names)
    for i in range(len(grid)):
        # column by column: iterrows would turn every int setting into a float
        values = [grid[n].iloc[i] for n in names]
        values = [v.item() if hasattr(v, "item") else v for v in values]
        cells = [str(v) for v in values]
        cells += [_num(float(grid["sharpe_ann"].iloc[i]), ".3f", nan="flat"),
                  str(int(grid["trades"].iloc[i])), f"{grid['net_pnl'].iloc[i]:,.2f}"]
        mark = "   <- best" if tuple(values) == best_key else ""
        body.append((cells, mark))
    widths = [max(len(h), *(len(c[i]) for c, _ in body)) for i, h in enumerate(head)]
    lines.append(("  " + "  ".join(h.ljust(w) for h, w in zip(head, widths, strict=True)))
                 .rstrip())
    for cells, mark in body:
        row = "  " + "  ".join(c.ljust(w) for c, w in zip(cells, widths, strict=True))
        lines.append(row.rstrip() + mark)
    if result.tie:
        tied = int((grid["sharpe"] == grid["sharpe"].max()).sum())
        lines.append(f"  tie: {tied} settings share the best Sharpe; the first in grid "
                     "order is used")

    d = result.deflation
    if result.ledger_path is None:
        count = f"N = {d['n_trials']:,} (ledger OFF: N counts this study only, earlier " \
                "tries unknown)"
    elif result.n_earlier == 0 and result.ledger_new:
        count = f"N = {d['n_trials']:,} (new ledger)"
    else:
        # a rerun of settings the ledger already holds adds no try: say so, or "0 earlier"
        # reads as if the earlier run had not been seen
        again = f"{result.n_repeat:,} of them already in the ledger, " if result.n_repeat \
            else ""
        count = f"N = {d['n_trials']:,} ({result.n_this:,} this study, {again}" \
                f"{result.n_earlier:,} earlier)"
    lines += ["Deflated Sharpe", f"  {count}"]
    if result.ledger_path is not None:
        # the default path is relative to where you run: a study run from another folder
        # starts a separate count, and this line (the full path) is how you notice
        lines.append(f"  Ledger: {result.ledger_path}")
    if result.n_other_bars:
        lines.append(f"  The ledger also holds {result.n_other_bars:,} tries of this "
                     "strategy on other bars: N does not count them")
    lines += [
        f"  Luck bar = {d['luck_bar_ann']:.3f}   (annualized: what the best of N zero-edge "
        "tries shows by luck alone)",
        f"  Deflated Sharpe = {_num(d['dsr'], '.3f')}   (probability the best setting "
        "truly beats the luck bar)",
        f"  Verdict: {d['verdict']}",
    ]
    if MIN_ACTIVE <= d["active_bars"] < LOW_ACTIVE:
        lines.append(
            f"  WARNING: only {d['active_bars']} bars with a non-zero return: under "
            f"{LOW_ACTIVE}, zero-edge strategies pass more often than the 5% this bar "
            "promises")

    p = result.pbo
    if p is None:
        lines += ["Probability of Backtest Overfitting", f"  not computed: {result.pbo_note}"]
    else:
        lines += [
            "Probability of Backtest Overfitting",
            f"  PBO = {p.pbo:.3f}   (share of {p.n_splits:,} splits whose in-sample best fell "
            "to the bottom half out-of-sample; above 0.05 is a warning)",
            f"  Degradation slope = {_num(p.degradation_slope, '.3f')}   (out-of-sample on "
            "in-sample Sharpe of the winner, across splits)",
            f"  Out-of-sample loss = {p.prob_oos_loss:.3f}   (share of splits where the "
            "winner lost money out-of-sample)",
        ]
    if p is not None and p.dropped_rows:
        lines.append(f"  {p.dropped_rows} oldest rows dropped so the blocks are equal")
    if p is not None and p.collapsed:
        lines.append(f"  {p.collapsed} setting(s) had returns identical to another and "
                     "count once here")
    if p is not None and p.pbo > 0.05 and p.prob_oos_loss < 0.05:
        lines.append("  the in-sample winner rarely lost money out-of-sample: a PBO this "
                     "high then says the settings are too alike to rank, not that they fail")

    wf = result.walk_forward
    kind = "anchored" if len(wf.windows) > 1 and wf.windows[1]["train"].start == 0 \
        else "rolling"
    lines += [_OOS_LABEL, f"Walk-forward   ({kind}, {len(wf.windows)} windows)"]
    for k, w in enumerate(wf.windows, 1):
        lines.append(
            f"  window {k}: train {_rows(w['train'])}, test {_rows(w['test'])}: "
            f"{_params(w['params'])}   in-sample {w['is_sharpe'] * root:.3f}, "
            f"out-of-sample {_num(w['oos_sharpe'] * root, '.3f', nan='flat')}"
            + ("   (ended open: one exit charged)" if w["open_at_end"] else ""))
    lines += [
        "  Each window restarts flat from 100,000: its sizing and margin checks do not "
        "follow the stitched curve",
        f"  Out-of-sample Sharpe = {_num(wf.sharpe_ann, '.3f', nan='flat')}   "
        f"SE {_num(wf.sharpe_se, '.3f')}   PSR {_num(wf.psr, '.3f')}",
        f"  In-sample Sharpe of the picks (mean) = {wf.is_sharpe_mean * root:.3f}",
    ]
    if wf.open_windows:
        lines.append(f"  {wf.open_windows} window(s) ended with a position open: each was "
                     "marked at its last close and charged one exit")
    if wf.margin_calls or wf.margin_refusals:
        lines.append(f"  WARNING: {wf.margin_calls} margin call(s) and "
                     f"{wf.margin_refusals} refused or cut order(s) inside the walk-forward "
                     "reruns: those windows did not trade as the settings table shows")
    if wf.below_margin_windows:
        lines.append(f"  WARNING: in {wf.below_margin_windows} window(s) the stitched curve "
                     "stood below the initial margin of the contracts held (its low: "
                     f"{wf.equity.min():,.0f}): one account could not have carried them")
    if wf.winner_changed:
        lines.append("  WARNING: winner changed between windows: the pick is not stable")

    cv = result.cv
    lines.append(f"Purged CV   ({len(cv.folds)} folds, embargo {cv.embargo} bars)")
    for k, f in enumerate(cv.folds, 1):
        lines.append(
            f"  fold {k}: test {_rows(f['test'])}: {_params(f['params'])}   out-of-sample "
            f"{_num(f['oos_sharpe'] * root, '.3f', nan='flat')}"
            + ("   (not real-time order: trained on later data)" if f["not_real_time"]
               else ""))
    lines.append(f"  Spread of fold Sharpes = {_num(cv.spread * root, '.3f')}")
    if cv.margin_calls or cv.margin_refusals:
        lines.append(f"  WARNING: {cv.margin_calls} margin call(s) and "
                     f"{cv.margin_refusals} refused or cut order(s) inside the fold reruns")
    if cv.capped:
        lines.append("  WARNING: embargo capped to keep 30 training rows: some trades are "
                     "longer, so these folds still carry part of the leak (walk-forward "
                     "does not)")
    lines += [
        f"data_sha256   = {result.data_sha256}",
        f"engine version = {__version__}",
    ]
    return "\n".join(lines) + "\n"
