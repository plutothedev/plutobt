"""What one try is (3.0.1, audit findings 1 to 3): the strategy's code wherever it lives,
the config switches that change what trades, and the bars by content rather than format."""
import importlib
import json
import sys
import textwrap

import numpy as np
import pytest

from plutobt.config import BacktestConfig
from plutobt.data import bars_sha256, frame_sha256
from plutobt.instruments import get_instrument
from plutobt.ledger import Ledger, code_files, code_sha256, config_key, try_key
from plutobt.study import run_study
from tests.study_fixtures import CFG, Coin, Hold, walk

NQ = get_instrument("NQ")

STRAT = '''
    import numpy as np
    from plutobt.strategy import Strategy
    import {helper}
    from {helper} import cut

    THRESH = {thresh}


    def entry_ok(x):
        return x > {same_file_cut}


    class S(Strategy):
        def on_bar(self, ctx):
            if entry_ok(THRESH) and cut(1.0) and ctx.position.qty == 0:
                ctx.broker.buy(contracts=1)
'''
HELPER = '''
    def cut(x):
        return x > {value}
'''


@pytest.fixture
def project(tmp_path, monkeypatch):
    """A user project of two files, importable, and re-importable after an edit."""
    monkeypatch.syspath_prepend(str(tmp_path))
    names = {"strat": f"strat_{tmp_path.name}".replace("-", "_"),
             "helper": f"helper_{tmp_path.name}".replace("-", "_")}

    def load(thresh=1.0, same_file_cut=0.0, helper_value=0.0, extra=""):
        (tmp_path / f"{names['helper']}.py").write_text(
            textwrap.dedent(HELPER.format(value=helper_value)), encoding="utf-8")
        (tmp_path / f"{names['strat']}.py").write_text(
            textwrap.dedent(STRAT.format(helper=names["helper"], thresh=thresh,
                                         same_file_cut=same_file_cut)) + extra,
            encoding="utf-8")
        for name in names.values():
            sys.modules.pop(name, None)
        importlib.invalidate_caches()
        return importlib.import_module(names["strat"]).S

    yield load
    for name in names.values():
        sys.modules.pop(name, None)


def test_a_module_constant_edit_is_new_code(project):
    assert code_sha256(project(thresh=1.0)) != code_sha256(project(thresh=2.0))


def test_a_same_file_helper_edit_is_new_code(project):
    assert code_sha256(project(same_file_cut=0.0)) != code_sha256(project(same_file_cut=1.0))


def test_an_edit_in_another_user_module_is_new_code(project):
    assert code_sha256(project(helper_value=0.0)) != code_sha256(project(helper_value=5.0))


def test_the_same_code_is_the_same_hash(project):
    assert code_sha256(project()) == code_sha256(project())


def test_comments_and_blank_lines_are_not_new_code(project):
    assert code_sha256(project()) == code_sha256(project(extra="\n\n# a note to self\n"))


def test_only_the_users_files_are_hashed(project):
    files = code_files(project())
    assert len(files) == 2                       # the strategy's file and its helper
    assert not any("site-packages" in f or "numpy" in f for f in files)


def test_two_classes_in_one_file_still_differ():
    assert code_sha256(Coin) != code_sha256(Hold)


def test_a_library_strategy_hashes_its_own_file_only():
    from plutobt.strategies.ema_cross import EMACross

    (only,) = code_files(EMACross)
    assert only.replace("\\", "/").endswith("plutobt/strategies/ema_cross.py")


def test_config_key_holds_the_switches_that_change_what_trades():
    key = config_key(BacktestConfig(allow_short=False, commission_per_side=9.0))
    assert key == {"allow_short": False, "break_on": "gap", "carry_overnight": False,
                   "on_margin_call": "liquidate"}       # costs are not part of a try


def test_try_key_with_no_config_is_the_3_0_0_key():
    assert try_key({"a": 1}, "c") == try_key({"a": 1}, "c", {})
    assert try_key({"a": 1}, "c") != try_key({"a": 1}, "c", {"allow_short": False})


def test_a_3_0_0_line_without_config_still_reads(tmp_path):
    p = tmp_path / "t.jsonl"
    p.write_text(json.dumps({
        "v": 1, "time": "2026-11-09T00:00:00+00:00", "strategy": "m:S", "code_sha256": "c",
        "params": {"a": 1}, "data_sha256": "d", "bars": 10, "sharpe": 0.1,
        "plutobt": "3.0.0"}) + "\n", encoding="utf-8")
    assert Ledger(p).tries("m:S", "d") == {try_key({"a": 1}, "c"): 0.1}


def test_a_config_switch_rerun_counts_as_new_tries(tmp_path):
    """Audit finding 2: flipping allow_short changed every Sharpe and N stayed 2."""
    df, led = walk(600, drift=0.3), tmp_path / "t.jsonl"
    from dataclasses import replace

    first = run_study(Hold, {"side": [1, -1]}, df, NQ, CFG, ledger=led)
    second = run_study(Hold, {"side": [1, -1]}, df, NQ, replace(CFG, allow_short=False),
                       ledger=led)
    assert first.deflation["n_trials"] == 2
    assert second.deflation["n_trials"] == 4
    assert (second.n_this, second.n_earlier) == (2, 2)
    assert "config" in json.loads(led.read_text(encoding="utf-8").splitlines()[-1])


def test_a_cost_change_is_not_a_new_try(tmp_path):
    df, led = walk(600, drift=0.3), tmp_path / "t.jsonl"
    from dataclasses import replace

    run_study(Hold, {"side": [1, -1]}, df, NQ, CFG, ledger=led)
    again = run_study(Hold, {"side": [1, -1]}, df, NQ, replace(CFG, commission_per_side=2.5),
                      ledger=led)
    assert again.deflation["n_trials"] == 2


def test_bars_hash_ignores_format_and_sees_content():
    df = walk(200)
    base = bars_sha256(df)
    assert bars_sha256(df.assign(volume=df.volume.astype(float))) == base
    assert bars_sha256(df.assign(symbol="NQ")) == base
    assert bars_sha256(df[["close", "ts", "volume", "low", "high", "open"]]) == base
    assert frame_sha256(df.assign(symbol="NQ")) != frame_sha256(df)      # the old key did not
    moved = df.copy()
    moved.loc[5, "close"] += 0.25
    assert bars_sha256(moved) != base
    assert bars_sha256(df.iloc[:-1]) != base
    assert bars_sha256(df.assign(session=np.arange(len(df)) // 78)) != base


def test_reformatted_bars_keep_their_count(tmp_path):
    """Audit finding 3: float volume started a fresh count."""
    df, led = walk(600, drift=0.3), tmp_path / "t.jsonl"
    run_study(Coin, {"seed": [0, 1], "bias": [0.5]}, df, NQ, CFG, ledger=led)
    res = run_study(Coin, {"seed": [2, 3], "bias": [0.5]},
                    df.assign(volume=df.volume.astype(float)), NQ, CFG, ledger=led)
    assert res.deflation["n_trials"] == 4 and res.n_earlier == 2


def test_tries_on_other_bars_are_reported(tmp_path):
    """Audit finding 3: one new bar after several tries printed N = 1 with no hint."""
    df, led = walk(601, drift=0.3), tmp_path / "t.jsonl"
    run_study(Coin, {"seed": [0, 1, 2], "bias": [0.5]}, df.iloc[:-1].reset_index(drop=True),
              NQ, CFG, ledger=led)
    res = run_study(Coin, {"seed": [0], "bias": [0.5]}, df, NQ, CFG, ledger=led)
    assert res.deflation["n_trials"] == 1
    assert res.n_other_bars == 3


def test_a_rerun_says_how_many_tries_were_already_recorded(tmp_path):
    df, led = walk(600, drift=0.3), tmp_path / "t.jsonl"
    run_study(Coin, {"seed": [0, 1], "bias": [0.5]}, df, NQ, CFG, ledger=led)
    res = run_study(Coin, {"seed": [1, 2], "bias": [0.5]}, df, NQ, CFG, ledger=led)
    assert (res.n_this, res.n_repeat, res.n_earlier) == (2, 1, 1)
