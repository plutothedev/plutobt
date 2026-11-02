"""PlutoBT metrics: bar-based and trade-based statistics (§5.7, §5.8).

v1 Sharpe scope note: this module reports the PER-BAR Sharpe ratio,
annualized by sqrt(periods_per_year). It does NOT emit a separate
daily-resampled Sharpe (explicit omission, not silent): intraday 5m
returns are typically negatively autocorrelated, which biases the
naive intraday Sharpe downward/conservative. A daily-resampled Sharpe
is named as future work.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from plutobt.engine import BacktestResult
    from plutobt.portfolio import Trade

IN_SAMPLE_LABEL = (
    "IN-SAMPLE, single pre-specified strategy; not predictive of live results."
)

# Min-sample threshold for trade-based stats (§5.7). compute_metrics does NOT
# flag this; format_report (M9) imports this constant and surfaces the
# "insufficient sample" warning from the n_trades output key below the threshold.
MIN_TRADE_SAMPLE = 30

# A trade whose net P&L is within a millionth of a dollar of zero is a scratch: that is
# float noise from price differences, never money; it is neither a win nor a loss.
# Public, with trade_net_pnl, so every per-trade statistic (PlutoDesk's included) shares
# this one tolerance.
NET_ZERO_TOL_USD = 1e-6


# Zero return volatility, for every ratio over the std: a std at or below
# VOL_REL_TOL * max(1, mean |r|) is float noise, not volatility. A curve compounding a
# constant 0.1% a bar leaves a std of about 1e-16 after pct_change, which an exact
# std > 0 check divided by into an annualized Sharpe of about 1e15.
VOL_REL_TOL = 1e-12


def has_volatility(returns) -> bool:
    """True when ``returns`` has a spread a ratio can divide by: at least one return
    and a population std above ``VOL_REL_TOL * max(1, mean |r|)``."""
    r = np.asarray(returns, dtype=float)
    if r.size == 0:
        return False
    return float(r.std(ddof=0)) > VOL_REL_TOL * max(1.0, float(np.abs(r).mean()))


def sharpe(returns: pd.Series, periods_per_year: float) -> float:
    """Annualized Sharpe from per-bar returns (risk-free already removed upstream).

    Uses population std (ddof=0). Annualization factor = sqrt(periods_per_year),
    which is printed by ``compute_metrics``. NaN with no returns or zero return
    volatility (``has_volatility``): 0/0 has no number, and 0.0 would read as a
    measured "no edge".
    """
    r = np.asarray(returns, dtype=float)
    if not has_volatility(r):
        return float("nan")
    sr_bar = r.mean() / r.std(ddof=0)
    return float(sr_bar * math.sqrt(periods_per_year))


def _skew_kurtosis(r: np.ndarray) -> tuple[float, float]:
    """Population (biased) skew g3 and Pearson (non-excess) kurtosis g4 of ``r``; the
    moments both the Sharpe SE and PSR use. ``r`` must have a nonzero spread."""
    d = r - r.mean()
    m2 = (d**2).mean()
    return float((d**3).mean() / m2**1.5), float((d**4).mean() / m2**2)


def sharpe_se(sr: float, n: int, *, skew: float = 0.0, kurtosis: float = 3.0) -> float:
    """Standard error of a per-bar Sharpe ratio ``sr`` over ``n`` iid returns.

    Mertens (2002) / Opdyke (2007), for iid returns that need not be normal:
    SE^2 = (1 - g3*sr + (g4 - 1)/4 * sr^2) / n, with population skew g3 and Pearson
    (non-excess) kurtosis g4: the same variance term ``psr`` divides by. The defaults
    (g3 = 0, g4 = 3, normal returns) reduce it to Lo (2002), sqrt((1 + 0.5*sr^2)/n).
    No autocorrelation adjustment (Lo's eta(q) is not applied). NaN when the variance
    term is not positive (unreachable from real sample moments; see ``psr``).
    """
    variance = 1.0 - skew * sr + ((kurtosis - 1.0) / 4.0) * sr**2
    if variance <= 0.0:
        return float("nan")
    return math.sqrt(variance / n)


def sharpe_ci(sr: float, n: int, *, skew: float = 0.0,
              kurtosis: float = 3.0) -> tuple[float, float]:
    """95% confidence interval for the Sharpe ratio: sr +/- 1.96 * SE (``sharpe_se``,
    same moments)."""
    se = sharpe_se(sr, n, skew=skew, kurtosis=kurtosis)
    return (sr - 1.96 * se, sr + 1.96 * se)


def _norm_cdf(x: float) -> float:
    """Standard normal CDF via the error function (stdlib only, no scipy)."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def psr(
    returns: pd.Series,
    sr0: float = 0.0,
    *,
    skew: float | None = None,
    kurtosis: float | None = None,
) -> float:
    """Probabilistic Sharpe Ratio vs benchmark ``sr0`` (Bailey & Lopez de Prado).

    Conventions (stated per §9 item 10): per-bar (non-annualized) SR_hat;
    sqrt(n-1) scaling; non-excess (Pearson) kurtosis g4; population (biased)
    skew g3 and kurtosis g4. Returns NaN with zero return volatility
    (``has_volatility``) or fewer than 2 returns.

    ``skew`` / ``kurtosis`` override the sample g3 / g4 (testing hook for the
    variance-adjustment guard; the kurtosis-skew inequality makes a non-positive
    adjustment unreachable from real sample moments).

    MinTRL (Minimum Track Record Length) is OUT of v1; PSR stays in.
    """
    r = np.asarray(returns, dtype=float)
    n = r.size
    if n < 2 or not has_volatility(r):
        return float("nan")
    sr_hat = r.mean() / r.std(ddof=0)
    g3, g4 = _skew_kurtosis(r)                  # population skew, Pearson kurtosis
    g3 = g3 if skew is None else skew
    g4 = g4 if kurtosis is None else kurtosis
    # The variance-adjustment term can go non-positive for extreme
    # skew/kurtosis; sqrt of it would raise. Guard with NaN instead.
    variance_adj = 1.0 - g3 * sr_hat + ((g4 - 1.0) / 4.0) * sr_hat**2
    if variance_adj <= 0.0:
        return float("nan")
    denom = math.sqrt(variance_adj)
    z = (sr_hat - sr0) * math.sqrt(n - 1) / denom
    return _norm_cdf(z)


def sortino(returns: pd.Series, mar: float, periods_per_year: float) -> float:
    """Sortino ratio with FULL-N semideviation denominator (Sortino & Price 1994).

    DR = sqrt((1/N) * sum(min(R_t - MAR, 0)^2)) -- positive excursions zeroed
    but ALL N periods in the denominator, NOT divided by the downside count.
    MAR is at the same frequency as ``returns``; DR is annualized by
    sqrt(periods_per_year). Returns +inf when there is no downside and a positive
    mean excess; NaN for 0/0 (a flat curve: no downside and no excess) and for no
    returns at all, the same convention as ``profit_factor``.
    """
    r = np.asarray(returns, dtype=float)
    n = r.size
    if n == 0:
        return float("nan")
    excess = r - mar
    downside = np.minimum(excess, 0.0)
    dr = math.sqrt((downside**2).sum() / n)
    if dr == 0.0:
        return float("inf") if excess.mean() > 0.0 else float("nan")
    return float((excess.mean() / dr) * math.sqrt(periods_per_year))


def max_drawdown(equity: pd.Series) -> tuple[float, int]:
    """Max drawdown depth (signed fraction, <= 0) and duration in bars.

    Depth = min over t of equity_t / running_peak_t - 1.
    Duration = duration of the deepest drawdown episode (peak-to-recovery);
    if it never recovers, runs to the last bar. Positional (integer)
    bar distance is used, independent of the index dtype.
    """
    depth, duration, _ = _deepest_drawdown(equity)
    return (depth, duration)


def max_drawdown_recovered(equity: pd.Series) -> bool:
    """Whether the deepest drawdown (the one ``max_drawdown`` measures) regained its
    peak by the last bar. False means its duration runs to the last bar and is a lower
    bound, not a finished episode. True with no drawdown at all."""
    return _deepest_drawdown(equity)[2]


def _deepest_drawdown(equity: pd.Series) -> tuple[float, int, bool]:
    """(depth, duration in bars, recovered) of the deepest drawdown episode; the one
    computation behind ``max_drawdown`` and ``max_drawdown_recovered``."""
    eq = np.asarray(equity, dtype=float)
    n = eq.size
    if n == 0:
        return (0.0, 0, True)
    running_max = np.maximum.accumulate(eq)
    dd = eq / running_max - 1.0
    depth = float(dd.min())
    if depth == 0.0:
        return (0.0, 0, True)
    trough_pos = int(dd.argmin())
    # Peak position: last index at/before the trough where equity == running_max.
    peak_val = running_max[trough_pos]
    peak_pos = trough_pos
    while peak_pos > 0 and eq[peak_pos] < peak_val:
        peak_pos -= 1
    # Recovery: first bar after the trough whose equity regains the peak.
    recovery_pos = n - 1
    recovered = False
    for j in range(trough_pos + 1, n):
        if eq[j] >= peak_val:
            recovery_pos = j
            recovered = True
            break
    duration = recovery_pos - peak_pos
    return (depth, int(duration), recovered)


def trailing_hwm_drawdown(equity: pd.Series) -> float:
    """Worst drawdown from the trailing high-water mark, as a positive fraction.

    Ratchets up on the running equity peak (incl. unrealized) and never falls;
    answers the prop-firm (Apex/Topstep) audience (§5.7). Returns 0.0 if the
    curve never dips below its running peak.
    """
    eq = np.asarray(equity, dtype=float)
    if eq.size == 0:
        return 0.0
    running_max = np.maximum.accumulate(eq)
    dd = eq / running_max - 1.0
    return float(-dd.min())


# The trading-day basis of periods_per_year (bars per day x 252, see compute_metrics).
TRADING_DAYS_PER_YEAR = 252


def trading_days_covered(n_bars: int, periods_per_year: float) -> float:
    """How many trading days ``n_bars`` bars span, on the periods_per_year basis
    (bars per day x 252): ``n_bars * 252 / periods_per_year``. The 20-day NQ sample
    (1,620 bars at 20412 a year) is 20.0. A run under 252 of them is under a year, so
    an annualized return (and Calmar) from it is an extrapolation."""
    return n_bars * TRADING_DAYS_PER_YEAR / periods_per_year


def calmar(annual_return: float, max_dd: float) -> float:
    """Calmar/MAR ratio = AnnualizedReturn / |MaxDD|; +inf when MaxDD is 0 and the
    return is positive, NaN for 0/0 (a flat curve), as ``profit_factor``'s 0/0."""
    if max_dd == 0.0:
        return float("inf") if annual_return > 0.0 else float("nan")
    return float(annual_return / abs(max_dd))


def _closed_trades(trades: list[Trade]) -> list[Trade]:
    """Closed (entry-to-flat) trades; open-at-end marks are excluded from
    win-rate / PF / expectancy per §5.7 (still counted in equity/Sharpe)."""
    return [t for t in trades if not t.open_at_end]


def trade_net_pnl(t: Trade) -> float:
    """A closed trade's net P&L (price P&L minus its fees), with a float-noise scratch
    snapped to 0.0. When a price move exactly pays the fees, float noise in the price
    difference leaves a few 1e-12 of either sign on the trade (more on a large position),
    and a bare ``> 0`` or ``< 0`` would book that noise as a win or a loss; within
    ``NET_ZERO_TOL_USD`` (a millionth of a dollar) of zero the trade is a scratch instead,
    neither. The band is far below any real amount, so it never hides real money: a trade
    netting +$0.003 still counts. Every trade statistic here reads this, and any per-trade
    statistic built elsewhere should too, so they all share one tolerance.
    """
    net = t.net_pnl
    return 0.0 if abs(net) < NET_ZERO_TOL_USD else net


def profit_factor(trades: list[Trade]) -> float:
    """Gross profit / gross loss over closed trades, each trade NET of its fees.

    Gross profit and gross loss here are sums of the trades' NET P&L (a scratch, within
    ``NET_ZERO_TOL_USD`` of zero, is in neither), and 0/0 (no winner and no loser)
    returns NaN, not +inf.
    inf-guard: returns +inf when there are profits but no losers; returns NaN
    when there are no closed trades at all. Open-at-end trades are excluded.
    Min-sample flagging (n < MIN_TRADE_SAMPLE) is surfaced by ``format_report``
    (M9) from the n_trades output key -- not here, and not in ``compute_metrics``.
    """
    closed = _closed_trades(trades)
    if not closed:
        return float("nan")
    gross_profit = sum(trade_net_pnl(t) for t in closed if trade_net_pnl(t) > 0.0)
    gross_loss = -sum(trade_net_pnl(t) for t in closed if trade_net_pnl(t) < 0.0)
    if gross_loss == 0.0:
        return float("inf") if gross_profit > 0.0 else float("nan")
    return float(gross_profit / gross_loss)


def expectancy(trades: list[Trade]) -> float:
    """Per-trade expectancy = WinRate*AvgWin - LossRate*AvgLoss (§5.7).

    Computed over closed trades (open-at-end excluded), each trade NET of its fees; a
    scratch (within ``NET_ZERO_TOL_USD`` of zero) counts in n but is neither a win nor
    a loss. AvgLoss is a positive magnitude. Returns NaN with no closed trades.
    """
    closed = _closed_trades(trades)
    n = len(closed)
    if n == 0:
        return float("nan")
    wins = [trade_net_pnl(t) for t in closed if trade_net_pnl(t) > 0.0]
    losses = [trade_net_pnl(t) for t in closed if trade_net_pnl(t) < 0.0]
    win_rate_val = len(wins) / n
    loss_rate = len(losses) / n
    avg_win = (sum(wins) / len(wins)) if wins else 0.0
    avg_loss = (-sum(losses) / len(losses)) if losses else 0.0  # positive magnitude
    return float(win_rate_val * avg_win - loss_rate * avg_loss)


def avg_r(trades: list[Trade]) -> float | None:
    """Average Van Tharp R-multiple over closed trades that HAVE an initial stop.

    Each R is net of its trade's fees, against a price-risk 1R with no fees in it
    (``Portfolio._calc_r_multiple``), so a stop-out exactly at the stop is below -1R.
    Stop-less trades carry r_multiple is None and are excluded (count printed by
    ``compute_metrics``). Returns None when no closed trade has an R-multiple.
    """
    rs = [t.r_multiple for t in _closed_trades(trades) if t.r_multiple is not None]
    if not rs:
        return None
    return float(sum(rs) / len(rs))


def outlier_sensitivity(trades: list[Trade], equity: pd.Series) -> dict:
    """Recompute headline trade stats with the single best/worst trade removed,
    and report the share of total P&L from the top-1 / top-5 trades (§5.7).

    ``collapses_without_top1`` flags when total P&L would go non-positive once
    the single best trade is removed (operationalizes "succeeding on outliers").
    Every P&L here is NET of fees: trades are ranked, summed and shared by their net
    P&L, with a scratch (within ``NET_ZERO_TOL_USD`` of zero) as 0.0.

    v1 scope note: Sharpe-recomputed-ex-outlier is OUT of v1 — only profit
    factor + expectancy are recomputed ex-best/ex-worst here.
    """
    closed = _closed_trades(trades)
    pnls = [trade_net_pnl(t) for t in closed]
    total = sum(pnls)
    ordered = sorted(pnls, reverse=True)
    top1 = ordered[0] if ordered else 0.0
    top5 = sum(ordered[:5])
    pct_top1 = (top1 / total) if total != 0.0 else float("nan")
    pct_top5 = (top5 / total) if total != 0.0 else float("nan")

    best = max(closed, key=lambda t: trade_net_pnl(t)) if closed else None
    worst = min(closed, key=lambda t: trade_net_pnl(t)) if closed else None
    ex_best = [t for t in closed if t is not best]
    ex_worst = [t for t in closed if t is not worst]
    total_ex_best = sum(trade_net_pnl(t) for t in ex_best)

    return {
        "pct_pnl_top1": pct_top1,
        "pct_pnl_top5": pct_top5,
        "profit_factor_ex_best": profit_factor(ex_best),
        "profit_factor_ex_worst": profit_factor(ex_worst),
        "expectancy_ex_best": expectancy(ex_best),
        "expectancy_ex_worst": expectancy(ex_worst),
        "total_pnl": total,
        "total_pnl_ex_best": total_ex_best,
        "collapses_without_top1": bool(total > 0.0 and total_ex_best <= 0.0),
    }


def cost_sensitivity(run_fn, slippage_ticks_list: list[float]) -> pd.DataFrame:
    """Run ``run_fn(slippage_ticks)`` across a list of slippage values and tabulate
    its returned metric dict (§5.7 cost-sensitivity sweep).

    Returns a DataFrame indexed by slippage_ticks (caller order preserved), one
    column per metric key. §10.15 asserts metrics degrade monotonically with
    slippage; this function only tabulates -- monotonicity is a property of the
    engine under test, verified by the caller's assertions.
    """
    rows = {}
    for s in slippage_ticks_list:
        rows[s] = run_fn(s)
    df = pd.DataFrame.from_dict(rows, orient="index")
    df = df.reindex(slippage_ticks_list)  # preserve caller order
    df.index.name = "slippage_ticks"
    return df


def sharpe_stats(returns: pd.Series, periods_per_year: float) -> dict:
    """The Sharpe, per bar and annualized, with its standard error and 95% CI, exactly
    as ``compute_metrics`` reports them (it calls this): keys ``sharpe_raw``,
    ``sharpe``, ``sharpe_se`` and ``sharpe_ci``. Public so a script that builds its
    own return series (the cold-open demo) prints the engine's error bar, not its own.
    """
    returns = pd.Series(returns, dtype="float64")
    t_count = int(returns.size)
    # Sharpe: sharpe_raw = per-bar, sharpe = annualized (§0.2)
    sr_raw = sharpe(returns, 1.0)
    sr_ann = sharpe(returns, periods_per_year)
    # SE and CI are reported in the SAME annualized units as `sharpe`, because
    # every reader pairs the adjacent keys (format_report prints them in one
    # sentence; the chat model quotes them together). They used to be left on
    # the raw per-bar Sharpe, which produced "Sharpe 0.640 (95% CI [-0.05,
    # 0.13])" - a point estimate outside its own interval - live on pluto's
    # screen. `sharpe` is exactly sqrt(periods_per_year) times `sharpe_raw`,
    # and a constant times an estimator has that constant times its standard
    # error, so the same sqrt(periods_per_year) factor scales sharpe_se and
    # sharpe_ci. The SE is the iid non-normal one (Mertens 2002 / Opdyke 2007),
    # with the sample skew and kurtosis psr uses, so the CI and PSR tell the same
    # story: Lo's normal-only SE printed "CI includes 0" beside "PSR = 0.986" on a
    # right-skewed series (review, 2026-11-02). They can still differ in a sliver,
    # since PSR scales by sqrt(T-1) and the SE by sqrt(T). Serial correlation is
    # not adjusted for (Lo's eta(q) is not applied). The raw-unit statistics stay
    # available through the pinned pure functions and sharpe_raw.
    scale = math.sqrt(periods_per_year)
    # Zero return volatility (has_volatility): the Sharpe is 0/0 (NaN), and the SE at
    # sr = 0 is a fixed sqrt(1/T) that would print a fake interval around it
    # ([-15.58, 15.58] at T = 323, ppy = 20412). NaN, like the T = 0 case.
    r_arr = np.asarray(returns, dtype=float)
    has_vol = has_volatility(r_arr)
    if has_vol:
        g3, g4 = _skew_kurtosis(r_arr)
        se = sharpe_se(sr_raw, t_count, skew=g3, kurtosis=g4) * scale
        raw_lo, raw_hi = sharpe_ci(sr_raw, t_count, skew=g3, kurtosis=g4)
        ci = (raw_lo * scale, raw_hi * scale)
    else:
        se = float("nan")
        ci = (float("nan"), float("nan"))
    return {"sharpe_raw": sr_raw, "sharpe": sr_ann, "sharpe_se": se, "sharpe_ci": ci}


def compute_metrics(result: BacktestResult, periods_per_year: float | None) -> dict:
    """Assemble the full metrics bundle for a BacktestResult, labeled IN-SAMPLE.

    §0.2 output keys (SINGLE SOURCE OF TRUTH — M9's format_report reads ONLY these):
    label, sharpe, sharpe_raw, sharpe_se, sharpe_ci (tuple; se and ci are
    ANNUALIZED, the same units as sharpe), psr, sortino,
    calmar, annual_return, max_drawdown_depth, max_drawdown_duration,
    trailing_hwm_drawdown, win_rate, profit_factor, expectancy, avg_r,
    T, periods_per_year, n_trades, exposure.

    The trade-stat keys (win_rate, profit_factor, expectancy, avg_r) are NET of fees,
    over closed trades; a scratch (net within ``NET_ZERO_TOL_USD`` of zero) counts in
    n_trades but is neither a win nor a loss.

    ``label`` is always the FIRST key (IN-SAMPLE stamp).

    periods_per_year derivation: if the argument is None, fall back to the value
    stored in result.manifest["config"]["periods_per_year"]. For 5m RTH index data
    this is 81 bars/day * 252 = 20412.

    §0.7 annual_return = mean(equity.pct_change().dropna()) * periods_per_year
    (simple returns, consistent with the Sharpe return series).

    §0.10 scope: Minimum Track Record Length (MinTRL) is OUT of v1; PSR stays in.
    Sharpe-recomputed-ex-outlier is OUT of v1; outlier_sensitivity recomputes
    profit factor + expectancy only.

    v1 reports the PER-BAR Sharpe (annualized via sqrt(periods_per_year));
    no daily-resampled Sharpe is emitted (explicit omission, see module docstring).
    """
    equity = result.equity_curve
    trades = result.trades

    if periods_per_year is None:
        periods_per_year = result.manifest.get("config", {}).get("periods_per_year")
    if periods_per_year is None:
        raise ValueError(
            "periods_per_year not provided and not present in result.manifest"
        )

    returns = equity.pct_change().dropna()
    t_count = int(returns.size)

    stats = sharpe_stats(returns, periods_per_year)
    sr_raw, sr_ann = stats["sharpe_raw"], stats["sharpe"]
    se, ci = stats["sharpe_se"], stats["sharpe_ci"]
    psr_val = psr(returns, sr0=0.0) if t_count > 1 else float("nan")

    # Sortino: annualized (§0.2 key "sortino")
    sortino_ann = sortino(returns, mar=0.0, periods_per_year=periods_per_year)

    # Drawdown
    dd_depth, dd_duration = max_drawdown(equity)
    trailing_dd = trailing_hwm_drawdown(equity)

    # §0.7: annual_return = mean(returns) * periods_per_year
    annual_ret = float(returns.mean() * periods_per_year) if t_count > 0 else 0.0
    calmar_ratio = calmar(annual_ret, dd_depth)

    # Trade stats, NET of fees (closed trades only for win-rate/PF/expectancy/avg_r)
    closed = _closed_trades(trades)
    n_closed = len(closed)
    wins = [t for t in closed if trade_net_pnl(t) > 0.0]
    # NaN with no closed trades (§0.3 said 0.0): 0 of 0 is no measurement, and it
    # matches profit_factor / expectancy, which are NaN there too.
    win_rate_val = (len(wins) / n_closed) if n_closed > 0 else float("nan")
    pf = profit_factor(trades)
    exp = expectancy(trades)
    a_r = avg_r(trades)

    # Exposure: fraction of equity bars inside any trade's [entry_ts, exit_ts]
    # interval (open-at-end trades carry exit_ts to the last bar).
    if t_count > 0 and trades:
        ts_arr = equity.index.to_numpy()
        total_bars = len(ts_arr)
        in_market = np.zeros(total_bars, dtype=bool)
        for t in trades:  # O(n_trades * T); acceptable at v1 scale
            entry = t.entry_ts
            exit_ = t.exit_ts if t.exit_ts is not None else ts_arr[-1]
            in_market |= (ts_arr >= entry) & (ts_arr <= exit_)
        exposure_val = float(in_market.sum() / total_bars)
    else:
        exposure_val = 0.0

    # Assemble — FIRST key must be "label" (§0.2)
    metrics: dict = {}
    metrics["label"] = IN_SAMPLE_LABEL
    metrics["sharpe"] = sr_ann
    metrics["sharpe_raw"] = sr_raw
    metrics["sharpe_se"] = se
    metrics["sharpe_ci"] = ci
    metrics["psr"] = psr_val
    metrics["sortino"] = sortino_ann
    metrics["calmar"] = calmar_ratio
    metrics["annual_return"] = annual_ret
    metrics["max_drawdown_depth"] = dd_depth
    metrics["max_drawdown_duration"] = dd_duration
    metrics["trailing_hwm_drawdown"] = trailing_dd
    metrics["win_rate"] = win_rate_val
    metrics["profit_factor"] = pf
    metrics["expectancy"] = exp
    metrics["avg_r"] = a_r
    metrics["T"] = t_count
    metrics["periods_per_year"] = float(periods_per_year)
    metrics["n_trades"] = n_closed
    metrics["exposure"] = exposure_val
    return metrics
