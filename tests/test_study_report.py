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
    led = tmp_path_factory.mktemp("ledger") / "trials.jsonl"
    return run_study(EMACross, {"fast_span": [10, 20, 30], "slow_span": [50, 100]}, df, NQ,
                     cfg, ledger=led)


def test_study_report_snapshot(sample_study):
    assert format_study(sample_study, "NQ") == SNAP.read_text(encoding="utf-8")


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
    assert "N = 3 (2 this study, 1 earlier)" in format_study(res, "NQ")


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
