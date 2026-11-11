"""run_study (3.0 spec §6): the grid, the ledger count, Deflated Sharpe, PBO and the
out-of-sample blocks, end to end on synthetic bars."""
import json
import math
import re

import numpy as np
import pytest

from plutobt.config import BacktestConfig
from plutobt.instruments import get_instrument
from plutobt.ledger import (
    Ledger,
    code_sha256,
    config_key,
    resolved_settings,
    strategy_id,
    try_key,
)
from plutobt.study import StudyResult, expand_grid, run_study
from tests.study_fixtures import CFG, Coin, Crashes, Hold, NeedsArgs, walk

NQ = get_instrument("NQ")


def study(df, cls, grid, tmp_path, **kw):
    kw.setdefault("ledger", tmp_path / "trials.jsonl")
    return run_study(cls, grid, df, NQ, CFG, **kw)


def test_grid_expansion_and_refusals():
    assert expand_grid(Coin, {"seed": [0, 1], "bias": [0.5, 1.0]}) == [
        {"seed": 0, "bias": 0.5}, {"seed": 0, "bias": 1.0},
        {"seed": 1, "bias": 0.5}, {"seed": 1, "bias": 1.0}]
    for bad, match in [({"seeed": [1]}, "seeed"), ({}, "empty"), ({"seed": []}, "seed"),
                       ({"on_bar": [1]}, "on_bar")]:
        with pytest.raises(ValueError, match=match):
            expand_grid(Coin, bad)
    with pytest.raises(ValueError, match="no arguments"):
        expand_grid(NeedsArgs, {"x": [1]})
    df = walk(600)
    with pytest.raises(ValueError, match=r"6 settings.*max_configs"):
        run_study(Coin, {"seed": [0, 1, 2], "bias": [0.5, 0.6]}, df, NQ, CFG, ledger=False,
                  max_configs=5)


def test_numpy_and_integral_float_values_are_one_try():
    out = expand_grid(Coin, {"seed": [np.int64(3), 3.0, 3], "bias": [np.float64(0.5)]})
    assert len(out) == 1
    assert type(out[0]["seed"]) is int and type(out[0]["bias"]) is float  # noqa: E721 (exact type is the point)


def test_periods_per_year_required(tmp_path):
    with pytest.raises(ValueError, match="periods_per_year"):
        run_study(Hold, {"side": [1, -1]}, walk(600), NQ, BacktestConfig(), ledger=False)


def test_crash_stops_and_names_the_setting(tmp_path):
    led = tmp_path / "trials.jsonl"
    named = re.escape('setting 2 of 2 ({"boom": true}) failed: RuntimeError: boom')
    with pytest.raises(RuntimeError, match=named):
        study(walk(600), Crashes, {"boom": [False, True]}, tmp_path, ledger=led)
    lines = [json.loads(x) for x in led.read_text(encoding="utf-8").splitlines()]
    assert [x["params"] for x in lines] == [{"boom": False}]


def test_every_completed_run_is_recorded(tmp_path):
    res = study(walk(600, drift=0.3), Hold, {"side": [1, -1, 0]}, tmp_path)
    led = Ledger(tmp_path / "trials.jsonl")
    assert len(led.tries(strategy_id(Hold), res.bars_sha256)) == 3
    assert res.n_this == 3 and res.n_earlier == 0 and res.ledger_new


def test_n_split_this_and_earlier(tmp_path):
    df = walk(600, drift=0.3)
    study(df, Coin, {"seed": [0, 1], "bias": [0.5]}, tmp_path)
    res = study(df, Coin, {"seed": [1, 2], "bias": [0.5]}, tmp_path)
    assert (res.n_this, res.n_earlier) == (2, 1)
    assert res.deflation["n_trials"] == 3
    assert not res.ledger_new


def test_ledger_off(tmp_path):
    res = study(walk(600, drift=0.3), Hold, {"side": [1, -1]}, tmp_path, ledger=False)
    assert res.ledger_path is None and res.deflation["n_trials"] == 2
    assert not (tmp_path / "trials.jsonl").exists()


def test_flat_settings_count_but_never_best(tmp_path):
    res = study(walk(600, drift=-0.3), Hold, {"side": [0, 1]}, tmp_path)
    assert res.best == {"side": 1}
    assert res.deflation["n_trials"] == 2


def test_all_flat_refused(tmp_path):
    with pytest.raises(ValueError, match="never traded"):
        study(walk(600), Hold, {"side": [0]}, tmp_path)


def test_tie_goes_to_grid_order(tmp_path):
    res = study(walk(600, drift=0.3), Coin, {"seed": [5, 6], "bias": [1.0]}, tmp_path)
    assert res.best == {"seed": 5, "bias": 1.0} and res.tie


def test_too_little_data_refused_before_any_run(tmp_path):
    with pytest.raises(ValueError, match="at most"):
        study(walk(300), Hold, {"side": [1, -1]}, tmp_path)
    assert not (tmp_path / "trials.jsonl").exists()


def test_planted_edge_end_to_end(tmp_path):
    df = walk(1800, drift=1.0, seed=11)
    res = study(df, Coin, {"bias": [0.2, 0.4, 0.6, 0.8, 1.0], "seed": [0, 1, 2, 3]}, tmp_path)
    assert isinstance(res, StudyResult)
    assert res.best["bias"] == 1.0
    assert res.deflation["verdict"] == "survives deflation"
    assert res.pbo.pbo <= 0.2
    assert len(res.grid) == 20 and set(res.grid.columns) >= {"bias", "seed", "sharpe",
                                                             "sharpe_ann", "trades", "net_pnl"}
    assert res.walk_forward.sharpe > 0
    assert res.returns.shape == (1799, 20)


def test_noise_end_to_end(tmp_path):
    df = walk(1800, drift=0.0, seed=12)
    res = study(df, Coin, {"seed": list(range(20)), "bias": [0.5]}, tmp_path)
    assert res.pbo.pbo >= 0.3
    assert res.deflation["verdict"] != "survives deflation"


def test_deflation_reads_the_ledger_spread(tmp_path):
    df = walk(600, drift=0.3)
    res = study(df, Hold, {"side": [1, -1]}, tmp_path)
    tries = Ledger(tmp_path / "trials.jsonl").tries(strategy_id(Hold), res.bars_sha256)
    assert res.deflation["var_sr"] == pytest.approx(np.var(list(tries.values())))
    key = try_key(resolved_settings(Hold, {"side": 1}), code_sha256(Hold), config_key(CFG))
    assert tries[key] == pytest.approx(res.grid.sharpe.iloc[0])


# --- final review, finding 1: a try is the fully resolved setting AND the code version ---

def test_changed_default_is_a_new_try(tmp_path, monkeypatch):
    """Editing a class default that is not in the grid, then rerunning the same grid, is
    the commonest overfitting loop: every rerun must count."""
    df = walk(600, drift=0.3)
    study(df, Coin, {"seed": [1, 2, 3]}, tmp_path)
    monkeypatch.setattr(Coin, "bias", 0.8)
    res = study(df, Coin, {"seed": [1, 2, 3]}, tmp_path)
    assert (res.n_this, res.n_earlier) == (3, 3)
    assert res.deflation["n_trials"] == 6


def test_edited_code_is_a_new_try(tmp_path, monkeypatch):
    import plutobt.study as study_mod

    df = walk(600, drift=0.3)
    study(df, Coin, {"seed": [1, 2]}, tmp_path)
    monkeypatch.setattr(study_mod, "code_sha256", lambda cls: "edited")
    res = study(df, Coin, {"seed": [1, 2]}, tmp_path)
    assert (res.n_this, res.n_earlier) == (2, 2)


def test_naming_a_default_is_not_a_new_try(tmp_path):
    df = walk(600, drift=0.3)
    study(df, Coin, {"seed": [1]}, tmp_path)
    res = study(df, Coin, {"seed": [1], "bias": [0.5]}, tmp_path)
    assert (res.n_this, res.n_earlier) == (1, 0)
    assert res.deflation["n_trials"] == 1


def test_string_values_refused_not_split_into_characters(tmp_path):
    """Final review, finding 3: {"seed": "abc"} became three junk tries in the ledger."""
    for bad in ("abc", 5, None):
        with pytest.raises(ValueError, match="list"):
            expand_grid(Coin, {"seed": bad})
    assert expand_grid(Coin, {"seed": range(2)}) == [{"seed": 0}, {"seed": 1}]
    assert expand_grid(Coin, {"seed": np.array([3, 4])}) == [{"seed": 3}, {"seed": 4}]


def test_one_setting_is_deflated_against_the_ledger_without_pbo(tmp_path):
    """Final review, finding 4: the honest user with one pre-chosen setting got a crash
    after the full run. PBO needs two settings to choose between, so it is skipped."""
    df = walk(600, drift=0.3)
    study(df, Coin, {"seed": [1, 2]}, tmp_path)
    res = study(df, Coin, {"seed": [3]}, tmp_path)
    assert res.pbo is None
    assert (res.n_this, res.n_earlier) == (1, 2) and res.deflation["n_trials"] == 3
    assert len(res.walk_forward.windows) == 5


def test_huge_grid_refused_before_expansion(tmp_path):
    """Final review, finding 5: the max_configs refusal came after itertools.product
    expanded the whole grid, which hangs on a large one."""
    grid = {"seed": list(range(100_000)), "bias": [x / 100_000 for x in range(100_000)]}
    with pytest.raises(ValueError, match=r"10,000,000,000 settings.*max_configs"):
        run_study(Coin, grid, walk(600), NQ, CFG, ledger=False)


# --- 3.0.1, audit finding 4: too few active bars -------------------------------------------

def _sparse_study(every, tmp_path):
    from tests.study_fixtures import Sparse

    return run_study(Sparse, {"every": [every], "phase": [5, 6]}, walk(1800, drift=0.3), NQ,
                     CFG, ledger=False, wf_splits=2, cv_folds=2, blocks=2)


def test_a_rarely_trading_best_setting_is_not_judged(tmp_path):
    res = _sparse_study(150, tmp_path)
    best = int(np.nanargmax(res.grid.sharpe))
    active = int(np.count_nonzero(res.returns.iloc[:, best]))
    assert active < 30
    assert res.deflation["active_bars"] == active
    assert res.deflation["verdict"] == \
        f"not judged: only {active} bars with a non-zero return (it takes 30)"
    assert math.isfinite(res.deflation["dsr"])           # the number still prints


def test_a_sometimes_trading_best_setting_keeps_its_verdict(tmp_path):
    res = _sparse_study(40, tmp_path)
    assert 30 <= res.deflation["active_bars"] < 100
    assert not res.deflation["verdict"].startswith("not judged")


def test_an_always_in_setting_counts_every_moving_bar(tmp_path):
    res = study(walk(600, drift=0.3), Hold, {"side": [1, -1]}, tmp_path)
    assert res.deflation["active_bars"] > 500
