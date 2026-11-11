"""format_study (3.0 spec §6 step 7): the printed study, pinned as a snapshot."""
import numpy as np
import pytest

from plutobt.config import load_config
from plutobt.data import load_bars
from plutobt.instruments import get_instrument
from plutobt.report import format_study
from plutobt.strategies.ema_cross import EMACross
from plutobt.study import run_study
from tests.conftest import REPO_ROOT
from tests.study_fixtures import CFG, Coin, Hold, walk

SNAP = REPO_ROOT / "tests" / "expected_output" / "study_report.txt"
NQ = get_instrument("NQ")


@pytest.fixture(scope="module")
def sample_study(tmp_path_factory):
    df = load_bars(REPO_ROOT / "data" / "sample" / "nq_5m.json")
    cfg = load_config(str(REPO_ROOT / "configs" / "nq.toml"))
    import os

    here = os.getcwd()
    os.chdir(tmp_path_factory.mktemp("ledger"))      # a relative ledger path prints the same
    try:
        return run_study(EMACross, {"fast_span": [10, 20, 30], "slow_span": [50, 100]}, df,
                         NQ, cfg, ledger=os.path.join(".plutobt", "trials.jsonl"))
    finally:
        os.chdir(here)


def _pinned(study) -> str:
    # the report prints the ledger's full path, which is this machine's temp folder
    return format_study(study, "NQ").replace(study.ledger_path, "<ledger>")


def test_study_report_snapshot(sample_study):
    assert _pinned(sample_study) == SNAP.read_text(encoding="utf-8")


def test_study_report_numbers_are_the_result(sample_study):
    text = format_study(sample_study, "NQ")
    d = sample_study.deflation
    assert f"Deflated Sharpe = {d['dsr']:.3f}" in text
    assert f"PBO = {sample_study.pbo.pbo:.3f}" in text
    assert f"Verdict: {d['verdict']}" in text
    assert "IN-SAMPLE, best of a grid search; not predictive of live results." in text
    assert "OUT-OF-SAMPLE (rerun flat in each test window)." in text
    assert "N = 6 (new ledger)" in text
    assert "single pre-specified" not in text


def test_ledger_off_line(tmp_path):
    res = run_study(Hold, {"side": [1, -1]}, walk(600, drift=0.3), NQ, CFG, ledger=False)
    assert "N = 2 (ledger OFF: N counts this study only, earlier tries unknown)" in \
        format_study(res, "NQ")


def test_n_split_line(tmp_path):
    df = walk(600, drift=0.3)
    led = tmp_path / "t.jsonl"
    run_study(Coin, {"seed": [0, 1], "bias": [0.5]}, df, NQ, CFG, ledger=led)
    res = run_study(Coin, {"seed": [1, 2], "bias": [0.5]}, df, NQ, CFG, ledger=led)
    text = format_study(res, "NQ")
    assert "N = 3 (2 this study, 1 of them already in the ledger, 1 earlier)" in text


def test_tie_noted(tmp_path):
    res = run_study(Coin, {"seed": [5, 6], "bias": [1.0]}, walk(600, drift=0.3), NQ, CFG,
                    ledger=False)
    assert "tie: 2 settings share the best Sharpe; the first in grid order is used" in \
        format_study(res, "NQ")


def test_winner_changed_warning():
    n = 1800
    df = walk(n, drift=np.where(np.arange(n) < n // 2, 1.0, -1.0), seed=5)
    res = run_study(Hold, {"side": [1, -1]}, df, NQ, CFG, ledger=False)
    text = format_study(res, "NQ")
    assert "winner changed between windows" in text
    assert "capped" in text            # always in the market: the CV embargo is capped


def test_ledger_path_printed(sample_study):
    """Final review: a study run from another folder silently starts a separate count;
    the path on the report is how the user notices."""
    assert f"  Ledger: {sample_study.ledger_path}" in format_study(sample_study, "NQ")


def test_one_setting_report(tmp_path):
    res = run_study(Hold, {"side": [1]}, walk(600, drift=0.3), NQ, CFG, ledger=False)
    text = format_study(res, "NQ")
    assert "not computed: one setting, so there is nothing to choose between" in text


def _sparse(every):
    from tests.study_fixtures import Sparse

    return run_study(Sparse, {"every": [every], "phase": [5, 6]}, walk(1800, drift=0.3), NQ,
                     CFG, ledger=False, wf_splits=2, cv_folds=2, blocks=2)


def test_low_activity_lines():
    """3.0.1, audit finding 4."""
    few = format_study(_sparse(150), "NQ")
    assert "Verdict: not judged: only" in few
    some = _sparse(40)
    text = format_study(some, "NQ")
    assert (f"  WARNING: only {some.deflation['active_bars']} bars with a non-zero return: "
            "under 100, zero-edge strategies pass more often than the 5% this bar promises"
            ) in text
    assert "WARNING: only" not in format_study(
        run_study(Hold, {"side": [1, -1]}, walk(600, drift=0.3), NQ, CFG, ledger=False), "NQ")


# --- 3.0.1: report text ---------------------------------------------------------------------

def test_the_report_is_plain_ascii(sample_study):
    assert format_study(sample_study, "NQ").isascii()


def test_the_ledger_path_is_absolute(sample_study, tmp_path):
    import pathlib

    assert pathlib.PurePath(sample_study.ledger_path).is_absolute()
    res = run_study(Hold, {"side": [1, -1]}, walk(600, drift=0.3), NQ, CFG,
                    ledger=tmp_path / "t.jsonl")
    assert f"  Ledger: {(tmp_path / 't.jsonl').resolve().as_posix()}" in format_study(res, "NQ")


def test_a_rerun_of_the_same_settings_says_they_were_already_counted(tmp_path):
    df, led = walk(600, drift=0.3), tmp_path / "t.jsonl"
    run_study(Hold, {"side": [1, -1]}, df, NQ, CFG, ledger=led)
    text = format_study(run_study(Hold, {"side": [1, -1]}, df, NQ, CFG, ledger=led), "NQ")
    assert "N = 2 (2 this study, 2 of them already in the ledger, 0 earlier)" in text


def test_tries_on_other_bars_get_a_line(tmp_path):
    df, led = walk(601, drift=0.3), tmp_path / "t.jsonl"
    run_study(Coin, {"seed": [0, 1, 2], "bias": [0.5]}, df.iloc[:-1].reset_index(drop=True),
              NQ, CFG, ledger=led)
    text = format_study(run_study(Coin, {"seed": [0], "bias": [0.5]}, df, NQ, CFG, ledger=led),
                        "NQ")
    assert "  The ledger also holds 3 tries of this strategy on other bars: N does not " \
           "count them" in text
    assert "on other bars" not in format_study(
        run_study(Hold, {"side": [1, -1]}, df, NQ, CFG, ledger=False), "NQ")


def _falling_study(drift):
    from tests.study_fixtures import Big

    return run_study(Big, {"size": [4, 3]}, walk(600, drift=drift, sigma=2.0, seed=3), NQ,
                     CFG, ledger=False, wf_splits=2, wf_anchored=True, cv_folds=2,
                     embargo_bars=5)


def test_walk_forward_says_each_window_restarts(sample_study):
    assert "  Each window restarts flat from 100,000: its sizing and margin checks do not " \
           "follow the stitched curve" in format_study(sample_study, "NQ")


def test_margin_events_in_the_reruns_are_printed():
    res = _falling_study(-1.3)
    wf, text = res.walk_forward, format_study(res, "NQ")
    assert (f"  WARNING: {wf.margin_calls} margin call(s) and {wf.margin_refusals} refused "
            "or cut order(s) inside the walk-forward reruns") in text
    assert (f"  WARNING: {res.cv.margin_calls} margin call(s) and "
            f"{res.cv.margin_refusals} refused or cut order(s) inside the fold reruns") in text


def test_a_stitched_curve_below_its_margin_is_printed():
    res = _falling_study(-0.5)
    wf, text = res.walk_forward, format_study(res, "NQ")
    assert wf.below_margin_windows >= 1
    assert (f"  WARNING: in {wf.below_margin_windows} window(s) the stitched curve stood "
            "below the initial margin of the contracts held (its low: "
            f"{wf.equity.min():,.0f}): one account could not have carried them") in text
    assert "inside the walk-forward reruns" not in text
