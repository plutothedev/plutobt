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
