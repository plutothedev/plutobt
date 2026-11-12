"""Probability of Backtest Overfitting via CSCV (3.0 spec §7; Bailey, Borwein, Lopez de
Prado and Zhu). Planted answers: noise near 0.5, a steady edge near 0, luck high."""
import math
from itertools import combinations

import numpy as np
import pytest

from plutobt.pbo import PBOResult, pbo


def noise(rows, cols, seed):
    return np.random.default_rng(seed).standard_normal((rows, cols))


def test_result_fields():
    res = pbo(noise(1600, 5, 0))
    assert isinstance(res, PBOResult)
    assert res.n_splits == 12_870 and res.logits.shape == (12_870,)
    assert res.dropped_rows == 0
    assert 0.0 <= res.pbo <= 1.0 and 0.0 <= res.prob_oos_loss <= 1.0


def test_n_splits_and_dropped_rows():
    res = pbo(noise(1605, 4, 1), blocks=16)
    assert res.n_splits == 12_870 and res.dropped_rows == 5
    assert pbo(noise(600, 4, 1), blocks=4).n_splits == 6


def test_noise_gives_about_one_half():
    vals = [pbo(noise(960, 20, seed), blocks=8).pbo for seed in range(30)]
    assert 0.4 <= float(np.mean(vals)) <= 0.6


def test_planted_steady_edge_gives_near_zero():
    r = noise(1600, 20, 5)
    r[:, 7] += 0.1
    res = pbo(r)
    assert res.pbo <= 0.05
    assert res.prob_oos_loss <= 0.05


def test_lucky_blocks_give_high_pbo():
    rng = np.random.default_rng(9)
    r = rng.standard_normal((1600, 16)) * 0.2 - 0.02
    # each setting is strong in exactly one block of 100 rows and losing elsewhere:
    # whichever wins in-sample got lucky there, and it loses out-of-sample
    for col in range(16):
        r[col * 100:(col + 1) * 100, col] += 1.0
    res = pbo(r)
    assert res.pbo >= 0.7
    # the lucky winner loses out-of-sample (the slope's sign is not guaranteed: the paper
    # only expects it negative "in most practical cases")
    assert res.prob_oos_loss >= 0.7
    assert np.isfinite(res.degradation_slope)


def _sharpe(x):
    sd = x.std()
    return 0.0 if sd == 0 else x.mean() / sd


def test_logit_convention_by_hand():
    """4 blocks of 30 rows, 3 settings: every split worked out the slow way."""
    rng = np.random.default_rng(42)
    r = rng.standard_normal((120, 3))
    blocks = [range(k * 30, (k + 1) * 30) for k in range(4)]
    logits = []
    for chosen in combinations(range(4), 2):
        ins = np.concatenate([list(blocks[k]) for k in chosen])
        outs = np.concatenate([list(blocks[k]) for k in range(4) if k not in chosen])
        is_sr = [_sharpe(r[ins, c]) for c in range(3)]
        oos_sr = [_sharpe(r[outs, c]) for c in range(3)]
        best = int(np.argmax(is_sr))
        rank = 1 + sum(s < oos_sr[best] for s in oos_sr) + 0.5 * (
            sum(s == oos_sr[best] for s in oos_sr) - 1)
        w = rank / 4
        logits.append(math.log(w / (1 - w)))
    res = pbo(r, blocks=4)
    np.testing.assert_allclose(res.logits, logits, rtol=1e-12)
    assert res.pbo == pytest.approx(np.mean(np.array(logits) <= 0))


def test_nan_sharpe_counts_as_zero():
    r = noise(600, 3, 3)
    r[:, 1] = 0.0                      # never trades: no volatility anywhere
    res = pbo(r, blocks=4)
    assert np.isfinite(res.logits).all()


def test_ties_take_the_average_rank():
    r = noise(600, 3, 4)
    r[:, 2] = r[:, 0]                  # an exact twin of setting 0
    res = pbo(r, blocks=4)
    assert np.isfinite(res.logits).all()


def test_one_setting_refused():
    with pytest.raises(ValueError, match="at least 2 settings"):
        pbo(noise(1600, 1, 0))


# --- 3.0.1, audit findings 5 and 6 ---------------------------------------------------------

def _edge_and_noise(rows=960, noise_cols=3, seed=4):
    rng = np.random.default_rng(seed)
    edge = rng.normal(0.2, 1.0, rows)
    return edge, rng.normal(0.0, 1.0, (rows, noise_cols))


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_a_table_with_a_non_finite_return_is_refused(bad):
    """A NaN first row (pct_change without dropna) used to give PBO 1.0 for a table whose
    clean PBO is 0.0."""
    edge, noise = _edge_and_noise()
    table = np.column_stack([edge, noise])
    table[0, 1] = bad
    with pytest.raises(ValueError, match="finite"):
        pbo(table)


def test_identical_settings_count_once():
    edge, noise = _edge_and_noise()
    clean = pbo(np.column_stack([edge, noise]))
    padded = pbo(np.column_stack([edge, edge, noise[:, 0], edge, noise[:, 1:], edge]))
    assert padded.collapsed == 3 and clean.collapsed == 0
    assert padded.pbo == clean.pbo
    assert np.array_equal(padded.logits, clean.logits)


def test_settings_that_are_all_identical_are_refused_by_name():
    from plutobt.pbo import IdenticalSettingsError

    edge, _ = _edge_and_noise()
    with pytest.raises(IdenticalSettingsError, match="identical"):
        pbo(np.column_stack([edge] * 10))


def test_a_rounding_error_tie_does_not_depend_on_column_order():
    """Two settings equal to 15 digits tie; which is listed first must not move the PBO."""
    edge, noise = _edge_and_noise()
    twin = edge * (1 + 1e-15)
    a = pbo(np.column_stack([edge, twin, noise]))
    b = pbo(np.column_stack([twin, edge, noise]))
    assert a.collapsed == 0                     # not bit-identical, so not collapsed
    assert a.pbo == b.pbo and np.array_equal(a.logits, b.logits)
