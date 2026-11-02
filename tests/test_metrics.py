"""Tests for plutobt.metrics (M7 tasks, §10.7, §10.15)."""

import math
import pathlib

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from plutobt.metrics import (
    IN_SAMPLE_LABEL,
    avg_r,
    calmar,
    compute_metrics,
    cost_sensitivity,
    expectancy,
    max_drawdown,
    outlier_sensitivity,
    profit_factor,
    psr,
    sharpe,
    sharpe_ci,
    sharpe_se,
    sortino,
    trailing_hwm_drawdown,
)
from plutobt.portfolio import Trade

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]

# Canonical reference returns series reused across §10.7 tasks.
# mean = 0.01, population std (ddof=0) = 0.025298221281347035, sharpe = 0.39528470752104744
REF_RETURNS = [0.02, -0.01, 0.03, -0.02, 0.04, -0.01, 0.02, -0.03, 0.01, 0.05]

# §0.2 EXACT required key set (single source of truth, M9 reads only these).
_REQUIRED_KEYS = [
    "label", "sharpe", "sharpe_raw", "sharpe_se", "sharpe_ci",
    "psr", "sortino", "calmar", "annual_return",
    "max_drawdown_depth", "max_drawdown_duration", "trailing_hwm_drawdown",
    "win_rate", "profit_factor", "expectancy", "avg_r",
    "T", "periods_per_year", "n_trades", "exposure",
]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _trade(pnl: float, r_multiple=None, open_at_end: bool = False) -> Trade:
    """Minimal Trade for trade-stat tests; only fields the metric reads matter.
    Trade.side is "long"/"short" per §0.5.
    """
    return Trade(
        entry_ts=0, exit_ts=1, side="long", contracts=1,
        entry_price=100.0, exit_price=100.0 + pnl, pnl=pnl,
        r_multiple=r_multiple, open_at_end=open_at_end,
    )


def _result_with(equity: pd.Series, trades, manifest=None):
    """Build a minimal BacktestResult-like object for the assembler."""
    from plutobt.engine import BacktestResult  # imported lazily to avoid cycles
    return BacktestResult(
        equity_curve=equity,
        trades=trades,
        metrics={},
        manifest=manifest if manifest is not None else {"periods_per_year": None},
    )


# ---------------------------------------------------------------------------
# M7.1 — sharpe (per-bar, risk-free, factor printed)
# ---------------------------------------------------------------------------

def test_sharpe_per_bar_matches_hand_calc():
    r = pd.Series(REF_RETURNS)
    # periods_per_year=1.0 => raw per-bar Sharpe (factor sqrt(1)=1)
    got = sharpe(r, periods_per_year=1.0)
    assert_allclose(got, 0.39528470752104744, rtol=0, atol=1e-12)


def test_sharpe_annualizes_by_sqrt_periods_per_year():
    r = pd.Series(REF_RETURNS)
    ppy = 20412.0  # 81 bars/day * 252
    got = sharpe(r, periods_per_year=ppy)
    expected = 0.39528470752104744 * math.sqrt(ppy)
    assert_allclose(got, expected, rtol=1e-12, atol=0)


def test_sharpe_zero_std_is_nan_not_zero():
    # Was 0.0, pinned here. Zero volatility makes the Sharpe 0/0: there is no number,
    # and 0.000 on screen read as a measured "no edge" beside "SE n/a" (review,
    # 2026-11-02). NaN, which the report prints as "n/a (zero return volatility)".
    r = pd.Series([0.0, 0.0, 0.0, 0.0])
    assert math.isnan(sharpe(r, periods_per_year=1.0))
    assert math.isnan(sharpe(pd.Series([], dtype=float), periods_per_year=1.0))


def _constant_growth_returns(rate=0.001, n=50):
    """Bar returns of a curve compounding a constant ``rate`` per bar, via pct_change
    as compute_metrics takes them: float noise leaves a std of about 1e-16."""
    eq = pd.Series(100_000.0 * np.cumprod(np.r_[1.0, np.full(n, 1.0 + rate)]))
    return eq, eq.pct_change().dropna()


def test_a_constant_return_curve_has_zero_volatility():
    # Review finding (2026-11-02): the check was exact std > 0, so a constant
    # 0.1%/bar curve printed an annualized Sharpe of about 1e15 from float noise.
    eq, r = _constant_growth_returns()
    assert r.std(ddof=0) > 0.0                    # positive control: exact check misses
    assert math.isnan(sharpe(r, periods_per_year=20412.0))
    assert math.isnan(psr(r, sr0=0.0))
    out = compute_metrics(_result_with(eq, []), periods_per_year=20412.0)
    assert math.isnan(out["sharpe"]) and math.isnan(out["sharpe_raw"])
    assert math.isnan(out["sharpe_se"]) and all(map(math.isnan, out["sharpe_ci"]))
    assert math.isnan(out["psr"])
    assert out["sortino"] == math.inf             # no downside, positive mean: honest


def test_volatility_just_above_the_tolerance_still_counts():
    # The tolerance is relative, std <= 1e-12 * max(1, mean |r|): it only swallows
    # float noise. A real 1e-9 wobble on a 0.1%/bar mean is volatility.
    r = pd.Series([0.001 + 1e-9, 0.001 - 1e-9] * 25)
    assert sharpe(r, periods_per_year=1.0) == pytest.approx(1e6, rel=1e-6)
    assert not math.isnan(psr(r, sr0=0.0))


# ---------------------------------------------------------------------------
# M7.2 — sharpe_se and sharpe_ci (Lo 2002)
# ---------------------------------------------------------------------------

def test_sharpe_se_lo_2002():
    # SE(SR) = sqrt((1 + 0.5*SR^2)/n)
    sr = 0.39528470752104744
    n = 10
    got = sharpe_se(sr, n)
    expected = math.sqrt((1.0 + 0.5 * sr**2) / n)  # 0.32834813841409244
    assert_allclose(got, 0.32834813841409244, rtol=0, atol=1e-12)
    assert_allclose(got, expected, rtol=0, atol=1e-15)


def test_sharpe_ci_is_sr_plus_minus_196_se():
    sr = 0.39528470752104744
    n = 10
    lo, hi = sharpe_ci(sr, n)
    se = sharpe_se(sr, n)
    assert_allclose(lo, sr - 1.96 * se, rtol=0, atol=1e-15)
    assert_allclose(hi, sr + 1.96 * se, rtol=0, atol=1e-15)
    # This CI straddles 0 -> not significant.
    assert lo < 0.0 < hi


def _skew_kurt(r):
    """Population skew g3 and Pearson (non-excess) kurtosis g4, by hand."""
    r = np.asarray(r, dtype=float)
    d = r - r.mean()
    m2 = (d**2).mean()
    return (d**3).mean() / m2**1.5, (d**4).mean() / m2**2


def test_sharpe_se_adjusts_for_skew_and_kurtosis():
    # Mertens (2002) / Opdyke (2007), iid non-normal returns:
    # SE^2 = (1 - g3*SR + (g4 - 1)/4 * SR^2) / n, per-bar SR, Pearson kurtosis g4:
    # the same variance term PSR divides by.
    sr, n = 0.39528470752104744, 10
    g3, g4 = _skew_kurt(REF_RETURNS)
    got = sharpe_se(sr, n, skew=g3, kurtosis=g4)
    expected = math.sqrt((1.0 - g3 * sr + (g4 - 1.0) / 4.0 * sr**2) / n)
    assert_allclose(got, expected, rtol=0, atol=1e-15)
    assert got < sharpe_se(sr, n) - 0.004    # thin tails (g4 = 1.77 < 3) shrink it
    # Normal moments (g3 = 0, g4 = 3) reduce it to Lo's iid-normal formula exactly.
    assert sharpe_se(sr, n, skew=0.0, kurtosis=3.0) == sharpe_se(sr, n)


def test_sharpe_ci_carries_the_moments():
    sr, n = 0.39528470752104744, 10
    g3, g4 = _skew_kurt(REF_RETURNS)
    lo, hi = sharpe_ci(sr, n, skew=g3, kurtosis=g4)
    se = sharpe_se(sr, n, skew=g3, kurtosis=g4)
    assert_allclose(lo, sr - 1.96 * se, rtol=0, atol=1e-15)
    assert_allclose(hi, sr + 1.96 * se, rtol=0, atol=1e-15)


def test_sharpe_se_is_nan_when_the_variance_term_is_not_positive():
    # Same guard as psr: 1 - 5*0.395 + 0 < 0 has no square root, so no SE.
    assert math.isnan(sharpe_se(0.39528470752104744, 10, skew=5.0, kurtosis=1.0))
    lo, hi = sharpe_ci(0.39528470752104744, 10, skew=5.0, kurtosis=1.0)
    assert math.isnan(lo) and math.isnan(hi)


def _lottery(n, k):
    """n per-bar returns: k evenly spaced +2% bars, the rest -0.08%. Strongly
    right-skewed (g3 about 3.4 at k = 14 of 200)."""
    r = np.full(n, -0.0008)
    r[np.linspace(0, n - 1, k).astype(int)] = 0.02
    return r


def _metrics_of_returns(r, ppy=20412.0):
    eq = pd.Series(100_000.0 * np.concatenate([[1.0], np.cumprod(1.0 + r)]))
    return compute_metrics(_result_with(eq, []), periods_per_year=ppy)


def test_skewed_series_ci_and_psr_tell_the_same_story():
    # Review finding (2026-11-02): the CI used Lo's iid-normal SE while PSR adjusts for
    # skew and kurtosis, so on a right-skewed series the report printed "CI includes 0"
    # next to "PSR = 0.986". Positive control: Lo's interval really did include 0 here.
    r = _lottery(200, 14)
    out = _metrics_of_returns(r)
    sr = out["sharpe_raw"]
    assert sr - 1.96 * sharpe_se(sr, 200) < 0.0          # the old interval: includes 0
    lo, hi = out["sharpe_ci"]
    assert lo > 0.0 and out["psr"] > 0.975              # now both say "Sharpe > 0"
    # The SE is the non-normal one, annualized like the Sharpe.
    g3, g4 = _skew_kurt(r)
    raw = math.sqrt((1.0 - g3 * sr + (g4 - 1.0) / 4.0 * sr**2) / 200)
    assert_allclose(out["sharpe_se"], raw * math.sqrt(20412.0), rtol=1e-6, atol=0)


def test_ci_excludes_zero_exactly_when_psr_exceeds_975_on_skewed_series():
    # Agreement on both sides of the line: k = 13 (PSR 0.970, CI includes 0) through
    # k = 19 (PSR 0.9999, CI excludes 0).
    for k in range(10, 20):
        out = _metrics_of_returns(_lottery(200, k))
        assert (out["sharpe_ci"][0] > 0.0) == (out["psr"] > 0.975), k


def test_ci_and_psr_disagree_only_in_the_t_versus_t_minus_1_sliver():
    # The CI excludes 0 when z_T = SR*sqrt(T)/sqrt(V) > 1.96; PSR > 0.975 when
    # z_(T-1) = SR*sqrt(T-1)/sqrt(V) > 1.959964 (Bailey & Lopez de Prado scale PSR by
    # sqrt(T-1)). Same V, so they can only disagree when z_(T-1) sits in
    # (1.96*sqrt((T-1)/T), 1.959964]: a sliver under 0.2% wide at T = 300.
    rng = np.random.default_rng(20261102)
    disagreements = 0
    for _ in range(400):
        T = int(rng.integers(30, 300))
        r = rng.lognormal(0.0, 1.0, T) * 1e-3 - rng.uniform(1.0e-3, 2.2e-3)
        out = _metrics_of_returns(r)
        sr = out["sharpe_raw"]
        g3, g4 = _skew_kurt(r)
        v = 1.0 - g3 * sr + (g4 - 1.0) / 4.0 * sr**2
        z_prev = sr * math.sqrt(T - 1) / math.sqrt(v)
        if (out["sharpe_ci"][0] > 0.0) != (out["psr"] > 0.975):
            disagreements += 1
            assert 1.96 * math.sqrt((T - 1) / T) - 1e-9 < z_prev <= 1.959964 + 1e-6, T
    assert disagreements <= 4


# ---------------------------------------------------------------------------
# M7.3 — psr (Probabilistic Sharpe Ratio, PSR(0))
# ---------------------------------------------------------------------------

def test_psr_zero_reference():
    # PSR(0) = Phi( SR_hat*sqrt(n-1) / sqrt(1 - g3*SR + ((g4-1)/4)*SR^2) )
    # For REF_RETURNS: SR=0.39528470752104744, g3=-0.03705794133009825,
    # g4=1.7675781249999996, n=10 -> z=1.1602447548720627, PSR=0.877025414964864
    r = pd.Series(REF_RETURNS)
    got = psr(r, sr0=0.0)
    assert_allclose(got, 0.877025414964864, rtol=0, atol=1e-9)


def test_psr_against_sr0_benchmark_lowers_probability():
    r = pd.Series(REF_RETURNS)
    # Comparing against a higher benchmark Sharpe must reduce the probability.
    assert psr(r, sr0=0.30) < psr(r, sr0=0.0)


def test_psr_zero_std_returns_nan():
    r = pd.Series([0.0, 0.0, 0.0, 0.0])
    assert math.isnan(psr(r, sr0=0.0))


def test_psr_nonpositive_variance_adj_returns_nan_not_raise():
    # The variance-adjustment term 1 - g3*sr + ((g4-1)/4)*sr^2 can in principle
    # go <= 0 for extreme skew/kurtosis (sqrt would raise). The kurtosis-skew
    # inequality makes this unreachable from sample moments, so we drive it via
    # the skew/kurtosis override hook: sr_hat~0.395, g3=5, g4=1 ->
    # variance_adj = 1 - 5*0.395 + 0 = -0.976 < 0.
    r = pd.Series(REF_RETURNS)
    got = psr(r, sr0=0.0, skew=5.0, kurtosis=1.0)
    assert math.isnan(got)  # NaN, not a ValueError from math.sqrt of a negative


# ---------------------------------------------------------------------------
# M7.4 — sortino (full-N downside denominator, MAR) — §10.7 divide-by-N
# ---------------------------------------------------------------------------

def test_sortino_full_n_denominator():
    # DR = sqrt((1/N) * sum(min(R-MAR,0)^2)); MAR=0, N=10.
    # sum of squared downside = 0.0015 -> DR = 0.012247448713915891
    # sortino = mean/DR = 0.01/0.012247... = 0.816496580927726 (per-bar)
    r = pd.Series(REF_RETURNS)
    got = sortino(r, mar=0.0, periods_per_year=1.0)
    assert_allclose(got, 0.816496580927726, rtol=0, atol=1e-12)


def test_sortino_rejects_divide_by_downside_count():
    # The WRONG denominator divides by the count of losers (4), giving
    # DR_wrong = sqrt(0.0015/4) = 0.019364916731037084 -> sortino_wrong ~ 0.5164.
    # Assert the correct full-N value differs from the divide-by-count value.
    r = pd.Series(REF_RETURNS)
    correct = sortino(r, mar=0.0, periods_per_year=1.0)
    wrong = 0.01 / math.sqrt(0.0015 / 4)
    assert not np.isclose(correct, wrong)
    assert correct > wrong  # full-N (larger N) => smaller DR => larger ratio


def test_sortino_annualizes_consistently():
    r = pd.Series(REF_RETURNS)
    ppy = 20412.0
    got = sortino(r, mar=0.0, periods_per_year=ppy)
    expected = 0.816496580927726 * math.sqrt(ppy)
    assert_allclose(got, expected, rtol=1e-12, atol=0)


def test_sortino_no_downside_returns_inf():
    r = pd.Series([0.01, 0.02, 0.03])  # all >= MAR
    assert math.isinf(sortino(r, mar=0.0, periods_per_year=1.0))


def test_sortino_flat_returns_is_nan_not_inf():
    # 0/0: no excess return over no downside. A zero-variance curve has no defined
    # Sortino, the same convention as profit_factor's 0/0 (NaN), never +inf.
    r = pd.Series([0.0, 0.0, 0.0])
    assert math.isnan(sortino(r, mar=0.0, periods_per_year=20412.0))


def test_sortino_no_returns_is_nan_not_zero():
    # No data must never read as a measured 0.0.
    assert math.isnan(sortino(pd.Series([], dtype=float), mar=0.0, periods_per_year=1.0))


# ---------------------------------------------------------------------------
# M7.5 — max_drawdown (depth + duration)
# ---------------------------------------------------------------------------

def test_max_drawdown_depth_and_duration():
    # peak 120 @ idx3, trough 90 @ idx4, recovery 130 @ idx6.
    # depth = (90-120)/120 = -0.25 ; duration = 6 - 3 = 3 bars (peak->recovery)
    eq = pd.Series([100, 110, 105, 120, 90, 95, 130, 125],
                   index=[0, 1, 2, 3, 4, 5, 6, 7], dtype=float)
    depth, duration = max_drawdown(eq)
    assert_allclose(depth, -0.25, rtol=0, atol=1e-12)
    assert duration == 3


def test_max_drawdown_unrecovered_runs_to_end():
    # Falls and never recovers: duration runs from the peak to the last bar.
    eq = pd.Series([100, 120, 110, 90, 80], index=[0, 1, 2, 3, 4], dtype=float)
    depth, duration = max_drawdown(eq)
    assert_allclose(depth, (80 - 120) / 120, rtol=0, atol=1e-12)
    assert duration == 3  # idx4 - idx1


def test_max_drawdown_duration_is_the_longest_time_under_water_at_any_depth():
    # A deep, quick dip (0 -> 2: 2 bars) and a shallow, slow one (2 -> 7: 5 bars). Depth
    # is the deep one's; duration is the longest stretch below a prior peak, the
    # shallow one's, regardless of depth.
    eq = pd.Series([100, 50, 100, 99, 99, 99, 99, 100], dtype=float)
    depth, duration = max_drawdown(eq)
    assert_allclose(depth, -0.5, rtol=0, atol=1e-12)
    assert duration == 5


def test_max_drawdown_duration_counts_an_open_stretch_to_the_last_bar():
    # The deepest dip is still open at the end but short (5 -> 6: 1 bar); the longest
    # stretch under water (0 -> 5: 5 bars) recovered.
    eq = pd.Series([100, 99, 99, 99, 99, 100, 50], dtype=float)
    depth, duration = max_drawdown(eq)
    assert_allclose(depth, -0.5, rtol=0, atol=1e-12)
    assert duration == 5
    # An open stretch longer than every recovered one is the duration, to the last bar.
    eq = pd.Series([100, 90, 100, 99, 99, 99, 99], dtype=float)
    assert max_drawdown(eq)[1] == 4


def test_max_drawdown_monotonic_up_is_zero():
    eq = pd.Series([100, 101, 102, 103], dtype=float)
    depth, duration = max_drawdown(eq)
    assert depth == 0.0
    assert duration == 0


# ---------------------------------------------------------------------------
# M7.6 — trailing_hwm_drawdown and calmar
# ---------------------------------------------------------------------------

def test_trailing_hwm_drawdown_is_worst_dip_from_running_peak():
    # Same curve as max_drawdown; worst trailing dip from running HWM = 0.25 (positive).
    eq = pd.Series([100, 110, 105, 120, 90, 95, 130, 125],
                   index=[0, 1, 2, 3, 4, 5, 6, 7], dtype=float)
    got = trailing_hwm_drawdown(eq)
    assert_allclose(got, 0.25, rtol=0, atol=1e-12)


def test_trailing_hwm_drawdown_never_negative():
    eq = pd.Series([100, 101, 102], dtype=float)
    assert trailing_hwm_drawdown(eq) == 0.0


def test_calmar_annual_return_over_abs_max_dd():
    # 0.20 / |-0.25| = 0.8
    assert_allclose(calmar(0.20, -0.25), 0.8, rtol=0, atol=1e-12)


def test_calmar_zero_drawdown_returns_inf():
    assert math.isinf(calmar(0.20, 0.0))


def test_calmar_zero_return_over_zero_drawdown_is_nan_not_inf():
    # 0/0 on a flat curve: undefined, NaN (as profit_factor's 0/0), never +inf.
    assert math.isnan(calmar(0.0, 0.0))


# ---------------------------------------------------------------------------
# M7.7 — profit_factor (inf-guard + open-at-end exclusion)
# ---------------------------------------------------------------------------

def test_profit_factor_hand_calc():
    # gross_profit = 600, gross_loss = 150 -> PF = 4.0
    trades = [_trade(300.0), _trade(200.0), _trade(100.0),
              _trade(-100.0), _trade(-50.0)]
    assert_allclose(profit_factor(trades), 4.0, rtol=0, atol=1e-12)


def test_profit_factor_no_losers_is_inf():
    trades = [_trade(100.0), _trade(50.0)]
    assert math.isinf(profit_factor(trades))


def test_profit_factor_excludes_open_at_end_trade():
    # Open-at-end trade is excluded from PF (§5.7) even if it would change the ratio.
    trades = [_trade(300.0), _trade(-100.0), _trade(-999.0, open_at_end=True)]
    assert_allclose(profit_factor(trades), 3.0, rtol=0, atol=1e-12)


def test_profit_factor_empty_is_nan():
    assert math.isnan(profit_factor([]))


# ---------------------------------------------------------------------------
# M7.8 — expectancy and avg_r (Van Tharp R; N/A excluded)
# ---------------------------------------------------------------------------

def test_expectancy_hand_calc():
    # win_rate=0.6, avg_win=200, loss_rate=0.4, avg_loss=75
    # expectancy = 0.6*200 - 0.4*75 = 120 - 30 = 90
    trades = [_trade(300.0), _trade(200.0), _trade(100.0),
              _trade(-100.0), _trade(-50.0)]
    assert_allclose(expectancy(trades), 90.0, rtol=0, atol=1e-12)


def test_avg_r_excludes_none_r_multiple():
    # R-multiples present: [2.0, 1.5, None, -1.0] -> avg over the 3 non-None = 0.8333...
    trades = [_trade(0.0, r_multiple=2.0), _trade(0.0, r_multiple=1.5),
              _trade(0.0, r_multiple=None), _trade(0.0, r_multiple=-1.0)]
    assert_allclose(avg_r(trades), 5.0 / 6.0, rtol=0, atol=1e-12)


def test_avg_r_all_stopless_is_none():
    trades = [_trade(0.0, r_multiple=None), _trade(0.0, r_multiple=None)]
    assert avg_r(trades) is None


def test_expectancy_empty_is_nan():
    assert math.isnan(expectancy([]))


# ---------------------------------------------------------------------------
# M7.9 — outlier_sensitivity
# ---------------------------------------------------------------------------

def test_outlier_sensitivity_top_trade_share_and_recompute():
    # PnLs: one dominant winner (1000) + small ones.
    trades = [_trade(1000.0), _trade(50.0), _trade(40.0),
              _trade(-30.0), _trade(-60.0)]
    eq = pd.Series([100.0, 1100.0, 1150.0, 1190.0, 1160.0, 1100.0])
    out = outlier_sensitivity(trades, eq)
    total = 1000 + 50 + 40 - 30 - 60  # = 1000
    assert_allclose(out["pct_pnl_top1"], 1000.0 / total, rtol=0, atol=1e-12)
    assert_allclose(out["pct_pnl_top5"], 1.0, rtol=0, atol=1e-12)  # all 5 trades
    # Profit factor with the single best winner removed must drop.
    assert out["profit_factor_ex_best"] < profit_factor(trades)
    # With the single worst loser removed, profit factor must rise.
    assert out["profit_factor_ex_worst"] > profit_factor(trades)
    # Expectancy is recomputed ex-best AND ex-worst (symmetry); removing the
    # best winner lowers expectancy, removing the worst loser raises it.
    assert out["expectancy_ex_best"] < expectancy(trades)
    assert out["expectancy_ex_worst"] > expectancy(trades)
    # Collapse flag: total P&L is dominated by one trade.
    assert out["collapses_without_top1"] is True


def test_outlier_sensitivity_balanced_does_not_collapse():
    trades = [_trade(100.0), _trade(90.0), _trade(110.0),
              _trade(-40.0), _trade(-50.0)]
    eq = pd.Series([100.0, 200.0, 290.0, 400.0, 360.0, 310.0])
    out = outlier_sensitivity(trades, eq)
    assert out["collapses_without_top1"] is False


# ---------------------------------------------------------------------------
# M7.10 — cost_sensitivity (sweep) — §10.15 monotonic
# ---------------------------------------------------------------------------

def test_cost_sensitivity_sweep_is_monotonic_non_increasing():
    # A run_fn whose metrics degrade as slippage rises (the realistic direction).
    def run_fn(slippage_ticks: float) -> dict:
        return {
            "sharpe": 1.6 - 0.4 * slippage_ticks,
            "profit_factor": 2.4 - 0.3 * slippage_ticks,
            "expectancy": 150.0 - 25.0 * slippage_ticks,
        }

    df = cost_sensitivity(run_fn, [1.0, 2.0, 4.0])
    assert list(df.index) == [1.0, 2.0, 4.0]
    assert list(df.columns) == ["sharpe", "profit_factor", "expectancy"]
    # §10.15: every metric column is monotonically non-increasing with slippage.
    for col in df.columns:
        diffs = df[col].diff().dropna()
        assert (diffs <= 1e-12).all(), f"{col} not monotonic non-increasing"


def test_cost_sensitivity_preserves_slippage_order_as_index():
    def run_fn(slippage_ticks: float) -> dict:
        return {"sharpe": -slippage_ticks}

    df = cost_sensitivity(run_fn, [4.0, 1.0, 2.0])
    assert list(df.index) == [4.0, 1.0, 2.0]  # caller order preserved, not sorted


# ---------------------------------------------------------------------------
# M7.11 — compute_metrics (assembler, §0.2 key set, IN-SAMPLE label)
# ---------------------------------------------------------------------------

def test_compute_metrics_first_key_is_label_and_is_in_sample():
    eq = pd.Series([100.0, 110.0, 105.0, 120.0, 90.0, 95.0, 130.0, 125.0])
    trades = [_trade(300.0), _trade(-100.0)]
    out = compute_metrics(_result_with(eq, trades), periods_per_year=20412.0)
    # First key MUST be "label" and carry the IN-SAMPLE stamp.
    assert next(iter(out)) == "label"
    assert out["label"] == IN_SAMPLE_LABEL


def test_compute_metrics_includes_exactly_required_keys():
    """§0.2: compute_metrics must include EXACTLY the required keys — no extras."""
    eq = pd.Series([100.0, 110.0, 105.0, 120.0, 90.0, 95.0, 130.0, 125.0])
    out = compute_metrics(_result_with(eq, [_trade(10.0)]), periods_per_year=20412.0)
    assert set(out) == set(_REQUIRED_KEYS), (
        f"Key mismatch.\n"
        f"  Missing: {set(_REQUIRED_KEYS) - set(out)}\n"
        f"  Extra:   {set(out) - set(_REQUIRED_KEYS)}"
    )
    # T = number of bar returns = len(equity) - 1
    assert out["T"] == 7
    assert_allclose(out["periods_per_year"], 20412.0, rtol=0, atol=0)


def test_compute_metrics_sharpe_raw_vs_annualized():
    """sharpe = annualized, sharpe_raw = per-bar (§0.2)."""
    eq_vals = [100.0] + list(np.cumprod(1.0 + np.array(REF_RETURNS)) * 100.0)
    eq = pd.Series(eq_vals)
    out = compute_metrics(_result_with(eq, []), periods_per_year=20412.0)
    rets = pd.Series(eq_vals).pct_change().dropna()
    assert_allclose(out["sharpe_raw"], sharpe(rets, 1.0), rtol=1e-12, atol=0)
    assert_allclose(out["sharpe"], sharpe(rets, 20412.0), rtol=1e-12, atol=0)


def test_compute_metrics_ci_brackets_the_sharpe_it_sits_beside():
    """The bundle's own consistency: sharpe_ci and sharpe_se must be in the SAME
    units as `sharpe` (annualized), because every reader pairs the adjacent keys.

    Found live (2026-08-26): the assembly annualized `sharpe` but left se/ci on
    the raw per-bar Sharpe, so the chat and format_report both printed
    "Sharpe 0.640 (95% CI [-0.05, 0.13])" - a point estimate outside its own
    interval, which no reader can use and some will trust."""
    eq_vals = [100.0] + list(np.cumprod(1.0 + np.array(REF_RETURNS)) * 100.0)
    eq = pd.Series(eq_vals)
    out = compute_metrics(_result_with(eq, []), periods_per_year=20412.0)
    lo, hi = out["sharpe_ci"]
    assert lo <= out["sharpe"] <= hi
    # The interval IS sharpe +/- 1.96 se, in the same annualized units.
    assert_allclose(lo, out["sharpe"] - 1.96 * out["sharpe_se"], rtol=1e-9, atol=0)
    assert_allclose(hi, out["sharpe"] + 1.96 * out["sharpe_se"], rtol=1e-9, atol=0)
    # And both scale from the raw skew/kurtosis-adjusted SE by sqrt(ppy), so the pure
    # functions (pinned above) remain the single source of the statistics.
    rets = eq.pct_change().dropna()
    g3, g4 = _skew_kurt(rets)
    raw_se = sharpe_se(sharpe(rets, 1.0), int(rets.size), skew=g3, kurtosis=g4)
    assert_allclose(out["sharpe_se"], raw_se * math.sqrt(20412.0), rtol=1e-9, atol=0)


def test_compute_metrics_empty_returns_gives_nan_se_and_ci_not_zero():
    """A one-point equity curve has zero bar returns (t_count == 0). sharpe_se and
    sharpe_ci must stay NaN, not collapse to a falsely precise 0.0 / (0.0, 0.0):
    "no data" must never read as "certainty of zero risk" to a downstream reader."""
    eq = pd.Series([100.0])
    out = compute_metrics(_result_with(eq, []), periods_per_year=20412.0)
    assert out["T"] == 0
    assert math.isnan(out["sharpe_se"])
    lo, hi = out["sharpe_ci"]
    assert math.isnan(lo)
    assert math.isnan(hi)


def test_compute_metrics_sharpe_ci_is_tuple():
    """§0.2: sharpe_ci must be a (low, high) tuple."""
    eq = pd.Series([100.0, 110.0, 105.0, 120.0])
    out = compute_metrics(_result_with(eq, []), periods_per_year=1.0)
    assert isinstance(out["sharpe_ci"], tuple)
    assert len(out["sharpe_ci"]) == 2
    lo, hi = out["sharpe_ci"]
    assert lo < hi


def test_compute_metrics_annual_return_per_s07():
    """§0.7: annual_return = mean(returns) * periods_per_year."""
    eq = pd.Series([100.0, 110.0, 105.0, 120.0, 90.0, 95.0, 130.0, 125.0])
    ppy = 20412.0
    out = compute_metrics(_result_with(eq, []), periods_per_year=ppy)
    rets = eq.pct_change().dropna()
    expected_ann_ret = float(rets.mean() * ppy)
    assert_allclose(out["annual_return"], expected_ann_ret, rtol=1e-12, atol=0)


def test_compute_metrics_win_rate_per_s03():
    """§0.3: win_rate = closed_wins / n_closed; 0.0 when no closed trades."""
    eq = pd.Series([100.0, 110.0, 120.0, 130.0])
    # 3 closed trades: 2 wins, 1 loss -> win_rate = 2/3
    trades = [_trade(10.0), _trade(20.0), _trade(-5.0)]
    out = compute_metrics(_result_with(eq, trades), periods_per_year=1.0)
    assert_allclose(out["win_rate"], 2.0 / 3.0, rtol=0, atol=1e-12)


def test_compute_metrics_win_rate_nan_when_no_closed_trades():
    """win_rate is NaN when n_closed == 0 (was 0.0 under §0.3). 0 wins of 0 trades is
    no measurement, and 0.000 on screen reads as "lost every trade"; NaN matches
    profit_factor and expectancy, which were already NaN with no closed trades."""
    eq = pd.Series([100.0, 110.0])
    out = compute_metrics(_result_with(eq, []), periods_per_year=1.0)
    assert math.isnan(out["win_rate"])


def test_compute_metrics_flat_curve_has_no_defined_ratios():
    """A flat, zero-trade curve: every ratio is 0/0, so NaN, consistently. The Sharpe
    SE/CI are NaN too: Lo's SE at zero volatility is a fake +/-1.96*sqrt(ppy/T)
    interval around a Sharpe that is itself 0/0."""
    eq = pd.Series([100_000.0] * 50)
    out = compute_metrics(_result_with(eq, []), periods_per_year=20412.0)
    assert out["T"] == 49
    for key in ("sharpe", "sharpe_raw", "psr", "sortino", "calmar", "profit_factor",
                "expectancy", "win_rate", "sharpe_se"):
        assert math.isnan(out[key]), key
    lo, hi = out["sharpe_ci"]
    assert math.isnan(lo) and math.isnan(hi)


def test_compute_metrics_derives_periods_per_year_from_manifest():
    """periods_per_year=None => derive from manifest's config (single source)."""
    eq = pd.Series([100.0, 101.0, 102.0])
    res = _result_with(eq, [], manifest={"config": {"periods_per_year": 20412.0}})
    out = compute_metrics(res, periods_per_year=None)
    assert_allclose(out["periods_per_year"], 20412.0, rtol=0, atol=0)


def test_compute_metrics_n_trades_is_closed_count():
    """n_trades counts closed trades only (open-at-end excluded)."""
    eq = pd.Series([100.0, 110.0, 120.0])
    closed = [_trade(10.0), _trade(-5.0)]
    open_end = [_trade(20.0, open_at_end=True)]
    out = compute_metrics(_result_with(eq, closed + open_end), periods_per_year=1.0)
    assert out["n_trades"] == 2  # only closed trades


def test_compute_metrics_exposure_fraction():
    """exposure = fraction of equity bars inside any [entry_ts, exit_ts] interval."""
    # 10-bar equity curve, ts index 0..9.
    eq = pd.Series([100.0 + i for i in range(10)], index=list(range(10)), dtype=float)
    # Trade A spans ts 1..3 (bars 1,2,3 = 3 bars); Trade B spans ts 6..7 (bars 6,7 = 2).
    # Disjoint intervals -> 5 of 10 bars in market -> exposure = 0.5.
    trade_a = Trade(entry_ts=1, exit_ts=3, side="long", contracts=1,
                    entry_price=100.0, exit_price=103.0, pnl=3.0)
    trade_b = Trade(entry_ts=6, exit_ts=7, side="long", contracts=1,
                    entry_price=106.0, exit_price=107.0, pnl=1.0)
    out = compute_metrics(_result_with(eq, [trade_a, trade_b]), periods_per_year=1.0)
    assert_allclose(out["exposure"], 0.5, rtol=0, atol=1e-12)


# ---------------------------------------------------------------------------
# M7.12 — public API gate + lint clean
# ---------------------------------------------------------------------------

def test_metrics_module_exposes_full_public_api():
    import plutobt.metrics as m

    for name in (
        "sharpe", "sharpe_se", "sharpe_ci", "psr", "sortino",
        "max_drawdown", "trailing_hwm_drawdown", "calmar",
        "profit_factor", "expectancy", "avg_r",
        "outlier_sensitivity", "cost_sensitivity", "compute_metrics",
    ):
        assert hasattr(m, name), name


# ---------------------------------------------------------------------------
# §10.15 — cost-sweep monotonicity backed by the REAL engine
# ---------------------------------------------------------------------------

def test_real_engine_metrics_degrade_monotonically_with_slippage():
    """§10.15: run the REAL engine at slippage_ticks in {1,2,4} on the sample
    bars; assert sharpe and profit_factor are non-increasing as slippage rises.

    This proves the ENGINE degrades under cost pressure, not just a synthetic
    run_fn (the existing test_cost_sensitivity_sweep_is_monotonic_non_increasing
    only validates the tabulator utility with a hand-crafted function).
    """
    from plutobt.config import BacktestConfig
    from plutobt.data import load_bars
    from plutobt.engine import Engine
    from plutobt.instruments import get_instrument
    from tests.conftest import GoldenStrategy

    sample = _REPO_ROOT / "data" / "sample" / "nq_5m.json"
    df = load_bars(str(sample))
    nq = get_instrument("NQ")
    ppy = 20412.0
    slippages = [1.0, 2.0, 4.0]

    def run_fn(slippage_ticks: float) -> dict:
        cfg = BacktestConfig(
            commission_per_side=2.5,
            slippage_ticks=slippage_ticks,
            periods_per_year=ppy,
            carry_overnight=False,
        )
        result = Engine(nq, cfg).run(df, GoldenStrategy())
        m = compute_metrics(result, periods_per_year=ppy)
        pf = m["profit_factor"]
        exp = m["expectancy"]
        return {
            "sharpe": m["sharpe"],
            # profit_factor / expectancy may be NaN (no trades); coerce to a
            # numeric sentinel that won't falsely break monotonicity.
            "profit_factor": pf if not math.isnan(pf) else float("-inf"),
            "expectancy": exp if not math.isnan(exp) else float("-inf"),
        }

    tab = cost_sensitivity(run_fn, slippages)
    assert list(tab.index) == slippages

    # Assert non-increasing (with small float tolerance) for each metric.
    for col in ["sharpe", "profit_factor", "expectancy"]:
        vals = tab[col].tolist()
        for lo, hi in zip(vals, vals[1:], strict=False):
            assert lo >= hi - 1e-10, (
                f"{col} not non-increasing with slippage: "
                f"{slippages[vals.index(lo)]}→{slippages[vals.index(hi)]}: "
                f"{lo} > {hi}"
            )
