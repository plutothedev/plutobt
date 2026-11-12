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
from tests.conftest import REPO_ROOT
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


def test_tries_a_3_0_0_ledger_holds_for_these_bars_still_count(tmp_path):
    """3.0.0 keyed bars by the exact frame. Its lines must keep counting for the same
    frame, or upgrading would quietly lower N."""
    df, led = walk(600, drift=0.3), tmp_path / "t.jsonl"
    old = {"v": 1, "time": "2026-11-09T00:00:00+00:00", "strategy": "tests.study_fixtures:Hold",
           "code_sha256": "the-3.0.0-class-hash", "data_sha256": frame_sha256(df),
           "bars": 600, "plutobt": "3.0.0"}
    lines = [dict(old, params={"side": 1}, sharpe=0.04),
             dict(old, params={"side": -1}, sharpe=-0.04),
             dict(old, params={"side": 0}, sharpe=None)]
    led.write_text("".join(json.dumps(x) + "\n" for x in lines), encoding="utf-8")
    res = run_study(Hold, {"side": [1, -1]}, df, NQ, CFG, ledger=led)
    assert res.n_earlier == 3 and res.deflation["n_trials"] == 5
    assert res.n_other_bars == 0


# --- whole-branch review of 3.0.1 ----------------------------------------------------------

@pytest.fixture
def layout(tmp_path, monkeypatch):
    """Write a small project from {relative path: source} and import its strategy `S`."""
    monkeypatch.syspath_prepend(str(tmp_path))
    tag = tmp_path.name.replace("-", "_")
    made = []

    def load(files, strategy_module):
        for rel, source in files.items():
            path = tmp_path / rel.format(tag=tag)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(textwrap.dedent(source).format(tag=tag), encoding="utf-8")
        for name in [m for m in sys.modules if tag in m]:
            del sys.modules[name]
        importlib.invalidate_caches()
        made.append(tag)
        return importlib.import_module(strategy_module.format(tag=tag)).S

    yield load
    for name in [m for m in sys.modules if tag in m]:
        del sys.modules[name]


BODY = '''

class S(Strategy):
    def on_bar(self, ctx):
        pass
'''


def test_a_constant_imported_by_name_from_another_file_is_code(layout):
    """Review finding 1: `from params import THRESH` reached builtins, never params.py."""
    def version(thresh):
        return layout({
            "params_{tag}.py": f"THRESH = {thresh}\nWINDOWS = [10, 20]\n",
            "strat_{tag}.py": "from plutobt.strategy import Strategy\n"
                              "from params_{tag} import THRESH, WINDOWS\n" + BODY,
        }, "strat_{tag}")

    a, b = code_sha256(version(1.0)), code_sha256(version(2.0))
    assert a != b
    assert len(code_files(version(2.0))) == 2


def test_a_module_in_a_package_without_init_is_code(layout):
    """Review finding 1: `import lib.helpers` binds `lib`, a namespace package with no
    file, so helpers.py was never reached."""
    def version(cut):
        return layout({
            "lib_{tag}/helpers.py": f"CUT = {cut}\n",
            "strat_{tag}.py": "from plutobt.strategy import Strategy\n"
                              "import lib_{tag}.helpers\n" + BODY,
        }, "strat_{tag}")

    assert code_sha256(version(1)) != code_sha256(version(2))


def test_a_relative_import_inside_a_package_is_code(layout):
    def version(cut):
        return layout({
            "pkg_{tag}/__init__.py": "",
            "pkg_{tag}/params.py": f"CUT = {cut}\n",
            "pkg_{tag}/strat.py": "from plutobt.strategy import Strategy\n"
                                  "from .params import CUT\n"
                                  "from . import params as p\n" + BODY,
        }, "pkg_{tag}.strat")

    assert code_sha256(version(1)) != code_sha256(version(2))
    assert len(code_files(version(2))) == 3            # strat, params, the package __init__


def test_an_import_inside_a_function_is_code_once_it_has_run(layout):
    def version(cut):
        cls = layout({
            "late_{tag}.py": f"CUT = {cut}\n",
            "strat_{tag}.py": "from plutobt.strategy import Strategy\n"
                              "def cut():\n"
                              "    from late_{tag} import CUT\n"
                              "    return CUT\n" + BODY,
        }, "strat_{tag}")
        return cls

    assert code_sha256(version(1)) != code_sha256(version(2))


def test_an_import_cycle_between_user_modules_ends(layout):
    cls = layout({
        "a_{tag}.py": "import b_{tag}\nX = 1\n",
        "b_{tag}.py": "import a_{tag}\nY = 2\n",
        "strat_{tag}.py": "from plutobt.strategy import Strategy\nimport a_{tag}\n" + BODY,
    }, "strat_{tag}")
    assert len(code_files(cls)) == 3


def test_an_object_that_raises_on_attribute_access_does_not_break_the_hash(layout):
    """Review finding 9: a proxy at module level raised from inside code_sha256."""
    cls = layout({
        "strat_{tag}.py": textwrap.dedent('''
            from plutobt.strategy import Strategy


            class Proxy:
                def __getattribute__(self, name):
                    raise RuntimeError("working outside of application context")


            current = Proxy()
        ''') + BODY,
    }, "strat_{tag}")
    assert len(code_sha256(cls)) == 64


def test_libraries_outside_the_default_install_paths_are_not_user_code(monkeypatch):
    """Review finding 10: with numpy under `pip install --user` or Debian's dist-packages
    the hash took in hundreds of library files and moved on every upgrade."""
    import site
    from pathlib import Path

    from plutobt import ledger

    roots = ledger._installed_roots()
    for extra in [site.getusersitepackages(), *site.getsitepackages()]:
        assert Path(extra).resolve() in roots
    assert not ledger._is_user_file(Path("/usr/lib/python3/dist-packages/numpy/core.py"), ())
    assert not ledger._is_user_file(Path("C:/x/Lib/site-packages/pandas/api.py"), ())
    assert ledger._is_user_file(Path("C:/work/research/strat.py"), ())


def test_a_numpy_bool_switch_does_not_break_the_ledger(tmp_path):
    """Review finding 2: accepted by the config, then json could not write it."""
    from dataclasses import replace

    cfg = replace(CFG, allow_short=np.bool_(False))
    assert config_key(cfg)["allow_short"] is False
    res = run_study(Hold, {"side": [1]}, walk(600, drift=0.3), NQ, cfg,
                    ledger=tmp_path / "t.jsonl")
    assert res.deflation["n_trials"] == 1


def test_a_study_call_in_the_strategys_own_file_is_part_of_its_code(layout):
    """Review finding 4, pinned as documented: the whole file is the code, so the README
    tells users to keep the study that runs a strategy in another file."""
    def version(commission):
        return layout({
            "strat_{tag}.py": "from plutobt.strategy import Strategy\n"
                              f"COMMISSION = {commission}   # the study call's cost\n" + BODY,
        }, "strat_{tag}")

    assert code_sha256(version(2.5)) != code_sha256(version(3.0))
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    assert "Keep the strategy in its own file." in readme


def test_a_docstring_is_code_and_a_comment_is_not(layout):
    def version(extra):
        return layout({
            "strat_{tag}.py": '"""' + extra + '"""\n'
                              "from plutobt.strategy import Strategy\n# note\n" + BODY,
        }, "strat_{tag}")

    assert code_sha256(version("one")) != code_sha256(version("two"))
