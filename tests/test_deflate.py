"""Deflated Sharpe and the luck bar (3.0 spec §4). Numbers from the paper are the ones
the fact-check confirmed (Bailey and Lopez de Prado 2014, pp. 9-10)."""
import math

import numpy as np
import pandas as pd
import pytest

from plutobt.deflate import (
    EULER_GAMMA,
    SURVIVES,
    deflated_sharpe,
    expected_max_sharpe,
    null_var_sr,
    verdict,
)
from plutobt.metrics import psr, psr_from_moments

# The paper's worked example: annualized SR 2.5, daily (250 a year), T = 1250, skew -3,
# kurtosis 10, N = 100, V = 1/2 annualized (1/500 per period).
SR = 2.5 / math.sqrt(250)
T, SKEW, KURT, V = 1250, -3.0, 10.0, 0.5 / 250


def test_constants():
    assert EULER_GAMMA == pytest.approx(0.5772156649015329, abs=1e-15)
    assert SURVIVES == 0.95


def test_luck_bar_matches_the_video_card():
    assert expected_max_sharpe(100, 1.0) == pytest.approx(2.5306, abs=1e-4)
    assert expected_max_sharpe(1000, 1.0) == pytest.approx(3.2551, abs=1e-4)


def test_one_trial_has_no_luck_bar():
    assert expected_max_sharpe(1, 0.7) == 0.0


def test_luck_bar_scales_with_spread():
    assert expected_max_sharpe(100, 4.0) == pytest.approx(2 * expected_max_sharpe(100, 1.0))


def test_paper_worked_example():
    sr0 = expected_max_sharpe(100, V)
    assert sr0 == pytest.approx(0.1132, abs=1e-4)
    assert psr_from_moments(SR, T, SKEW, KURT, sr0) == pytest.approx(0.9004, abs=1e-4)
    # "with N = 46 trials, DSR would be 0.9505"; normal returns reach it at N = 88
    assert psr_from_moments(SR, T, SKEW, KURT, expected_max_sharpe(46, V)) == \
        pytest.approx(0.9505, abs=1e-4)
    assert psr_from_moments(SR, T, 0.0, 3.0, expected_max_sharpe(88, V)) == \
        pytest.approx(0.9505, abs=1e-4)


def _max_sharpe_sims(n, t, sims, seed):
    rng = np.random.default_rng(seed)
    out = np.empty(sims)
    for s in range(sims):
        r = rng.standard_normal((t, n))
        out[s] = (r.mean(axis=0) / r.std(axis=0)).max()
    return out


@pytest.mark.parametrize("n", [10, 100, 1000])
def test_luck_bar_matches_brute_force(n):
    t = 500
    sims = {10: 2000, 100: 600, 1000: 150}[n]
    simulated = _max_sharpe_sims(n, t, sims, seed=n).mean()
    formula = expected_max_sharpe(n, null_var_sr(t))
    if n >= 100:
        assert formula == pytest.approx(simulated, rel=0.03)
    else:
        # The paper's Eq. 1 is a large-N approximation: at N = 10 it runs about 2.3% above
        # the true expected maximum (1.5746 against 1.5393 for unit variance), the strict
        # side, so a small study's luck bar is a little high, never low.
        assert simulated <= formula <= simulated * 1.05


def test_deflated_equals_psr_against_the_luck_bar():
    r = pd.Series(np.random.default_rng(1).normal(0.05, 1.0, 800))
    assert deflated_sharpe(r, 40, 0.002) == psr(r, sr0=expected_max_sharpe(40, 0.002))


def test_psr_from_moments_matches_psr():
    from plutobt.metrics import _skew_kurtosis

    for seed in range(5):
        r = np.random.default_rng(seed).normal(0.02, 1.0, 300)
        g3, g4 = _skew_kurtosis(r)
        sr = r.mean() / r.std(ddof=0)
        assert psr_from_moments(sr, r.size, g3, g4, 0.03) == psr(pd.Series(r), sr0=0.03)


def test_psr_from_moments_nan_cases():
    assert math.isnan(psr_from_moments(0.1, 1, 0.0, 3.0))
    assert math.isnan(psr_from_moments(1.0, 100, 5.0, 3.0))   # non-positive variance term


def test_null_var_sr():
    assert null_var_sr(101) == pytest.approx(0.01)


def test_refusals():
    for bad in [(0, 1.0), (100, -1.0), (100, float("nan")), (100, float("inf"))]:
        with pytest.raises(ValueError):
            expected_max_sharpe(*bad)
    with pytest.raises(ValueError):
        null_var_sr(1)


def test_verdict_text():
    assert verdict(0.97, 40) == "survives deflation"
    assert verdict(0.95, 40) == "survives deflation"
    assert verdict(0.9, 1340) == "does not survive deflation (N = 1,340 tries)"
    assert verdict(float("nan"), 3) == "not computable (no return volatility)"


def _study_survival(settings, t, studies, seed, planted_sr=None):
    """Share of seeded studies whose best setting survives deflation; with a planted edge,
    also require the planted column (0) to be the best."""
    rng = np.random.default_rng(seed)
    survived = 0
    for _ in range(studies):
        r = rng.standard_normal((t, settings))
        if planted_sr is not None:
            r[:, 0] += planted_sr
        srs = r.mean(axis=0) / r.std(axis=0)
        best = int(np.argmax(srs))
        dsr = deflated_sharpe(pd.Series(r[:, best]), settings, float(np.var(srs)))
        if dsr >= SURVIVES and (planted_sr is None or best == 0):
            survived += 1
    return survived / studies


def test_calibration_small():
    assert _study_survival(50, 1000, 300, seed=7) <= 0.08


@pytest.mark.slow
def test_calibration_noise_survives_at_most_5_percent():
    assert _study_survival(50, 1000, 2000, seed=11) <= 0.06


def test_planted_edge_survives():
    assert _study_survival(50, 1000, 300, seed=3, planted_sr=0.20) >= 0.80


def _correlated_survival(rho, studies, seed, settings=50, t=1000):
    """Final review: real grids are correlated (EMA 20/50 and 21/50 move together). Each
    setting is sqrt(rho) x a common noise + sqrt(1 - rho) x its own: still zero edge."""
    rng = np.random.default_rng(seed)
    survived = 0
    for _ in range(studies):
        common = rng.standard_normal((t, 1))
        r = math.sqrt(rho) * common + math.sqrt(1 - rho) * rng.standard_normal((t, settings))
        srs = r.mean(axis=0) / r.std(axis=0)
        best = int(np.argmax(srs))
        survived += deflated_sharpe(pd.Series(r[:, best]), settings, float(np.var(srs))) \
            >= SURVIVES
    return survived / studies


def test_calibration_correlated_settings_small():
    assert _correlated_survival(0.9, 300, seed=21) <= 0.08


@pytest.mark.slow
def test_calibration_correlated_settings_stays_under_5_percent():
    # measured 2026-11-09 (1,000 studies, seed 21): rho 0.5 1.6%, 0.9 4.1%, 0.99 4.7%
    for rho in (0.5, 0.9, 0.99):
        assert _correlated_survival(rho, 1000, seed=21) <= 0.06


def test_a_bool_is_not_a_try_count():
    with pytest.raises(ValueError, match="whole number"):
        expected_max_sharpe(True, 0.01)


def test_a_try_count_too_large_to_compute_is_refused_plainly():
    with pytest.raises(ValueError, match="too large"):
        expected_max_sharpe(10 ** 16, 0.01)
    assert expected_max_sharpe(10 ** 12, 1.0) > expected_max_sharpe(10 ** 9, 1.0)


# --- 3.0.1, audit finding 4: too few active bars -------------------------------------------

def _sparse_false_pass(active, t, n, studies, seed, fat=False):
    """Zero-edge settings that are in the market on exactly ``active`` of ``t`` bars."""
    rng = np.random.default_rng(seed)
    passed = 0
    for _ in range(studies):
        r = np.zeros((t, n))
        for c in range(n):
            rows = rng.choice(t, size=active, replace=False)
            r[rows, c] = rng.standard_t(3, size=active) if fat else rng.standard_normal(active)
        srs = r.mean(axis=0) / r.std(axis=0)
        best = int(np.argmax(srs))
        var = float(np.var(srs)) if n > 1 else 0.0
        passed += deflated_sharpe(pd.Series(r[:, best]), n, var) >= SURVIVES
    return passed / studies


def test_the_low_activity_thresholds_match_what_was_measured():
    """Measured 2026-11-11 (4,000 studies a cell): the false-pass rate follows the number
    of non-zero returns, not the bar count. 10 active: 10%; 30: 6 to 7%; 100: 5 to 6%
    (fat tails 7%). So under MIN_ACTIVE the verdict is withheld, under LOW_ACTIVE warned."""
    from plutobt.deflation import LOW_ACTIVE, MIN_ACTIVE

    assert (MIN_ACTIVE, LOW_ACTIVE) == (30, 100)
    assert _sparse_false_pass(10, 500, 1, 1500, seed=1) > 0.075          # the problem is real
    assert _sparse_false_pass(MIN_ACTIVE, 500, 1, 1500, seed=2) < 0.09
    assert _sparse_false_pass(LOW_ACTIVE, 500, 2, 1500, seed=3) < 0.065
    # fat tails still pass too often at 30, hence the warning up to LOW_ACTIVE
    assert _sparse_false_pass(MIN_ACTIVE, 500, 2, 1500, seed=4, fat=True) > 0.06


def test_verdict_is_withheld_under_30_active_bars():
    assert verdict(0.99, 5, active_bars=29) == \
        "not judged: only 29 bars with a non-zero return (it takes 30)"
    assert verdict(0.99, 5, active_bars=30) == "survives deflation"
    assert verdict(0.10, 5, active_bars=12) == \
        "not judged: only 12 bars with a non-zero return (it takes 30)"
    assert verdict(0.99, 5) == "survives deflation"            # the command has no bar count
    assert verdict(float("nan"), 3, active_bars=5) == "not computable (no return volatility)"
