"""Tests for plutobt.metrics (M7 tasks, §10.7, §10.15)."""

import math

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from plutobt.metrics import sharpe

# Canonical reference returns series reused across §10.7 tasks.
# mean = 0.01, population std (ddof=0) = 0.025298221281347035, sharpe = 0.39528470752104744
REF_RETURNS = [0.02, -0.01, 0.03, -0.02, 0.04, -0.01, 0.02, -0.03, 0.01, 0.05]


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


def test_sharpe_zero_std_returns_zero():
    r = pd.Series([0.0, 0.0, 0.0, 0.0])
    assert sharpe(r, periods_per_year=1.0) == 0.0


# ---------------------------------------------------------------------------
# M7.2 — sharpe_se and sharpe_ci (Lo 2002)
# ---------------------------------------------------------------------------
from plutobt.metrics import sharpe_ci, sharpe_se


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


# ---------------------------------------------------------------------------
# M7.3 — psr (Probabilistic Sharpe Ratio, PSR(0))
# ---------------------------------------------------------------------------
from plutobt.metrics import psr


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


# ---------------------------------------------------------------------------
# M7.4 — sortino (full-N downside denominator, MAR) — §10.7 divide-by-N
# ---------------------------------------------------------------------------
from plutobt.metrics import sortino


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


# ---------------------------------------------------------------------------
# M7.5 — max_drawdown (depth + duration)
# ---------------------------------------------------------------------------
from plutobt.metrics import max_drawdown


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


def test_max_drawdown_monotonic_up_is_zero():
    eq = pd.Series([100, 101, 102, 103], dtype=float)
    depth, duration = max_drawdown(eq)
    assert depth == 0.0
    assert duration == 0


# ---------------------------------------------------------------------------
# M7.6 — trailing_hwm_drawdown and calmar
# ---------------------------------------------------------------------------
from plutobt.metrics import calmar, trailing_hwm_drawdown


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


# ---------------------------------------------------------------------------
# M7.7 — profit_factor (inf-guard + open-at-end exclusion)
# ---------------------------------------------------------------------------
from plutobt.portfolio import Trade

from plutobt.metrics import profit_factor


def _trade(pnl: float, r_multiple=None, open_at_end: bool = False) -> Trade:
    # Minimal Trade for trade-stat tests; only fields the metric reads matter.
    # Trade.side is "long"/"short" per §0.5.
    return Trade(
        entry_ts=0, exit_ts=1, side="long", contracts=1,
        entry_price=100.0, exit_price=100.0 + pnl, pnl=pnl,
        r_multiple=r_multiple, open_at_end=open_at_end,
    )


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
from plutobt.metrics import avg_r, expectancy


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
from plutobt.metrics import outlier_sensitivity


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
    # Collapse flag: total P&L is dominated by one trade.
    assert out["collapses_without_top1"] is True


def test_outlier_sensitivity_balanced_does_not_collapse():
    trades = [_trade(100.0), _trade(90.0), _trade(110.0),
              _trade(-40.0), _trade(-50.0)]
    eq = pd.Series([100.0, 200.0, 290.0, 400.0, 360.0, 310.0])
    out = outlier_sensitivity(trades, eq)
    assert out["collapses_without_top1"] is False
