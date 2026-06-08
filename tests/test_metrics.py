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
