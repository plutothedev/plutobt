import sys
from pathlib import Path

import pytest

from plutobt.ledger import strategy_id
from plutobt.strategies.ema_cross import EMACross
from plutobt.strategy_loader import (
    BUILTINS,
    StrategyLoadError,
    load_strategy,
    make_strategy,
    parse_settings,
    settings,
)

LOADER = Path(__file__).parent / "strategies" / "loader"
ONE = LOADER / "one_class.py"
TWO = LOADER / "two_classes.py"


def test_builtins_table():
    assert BUILTINS == {
        "ema_cross": "plutobt.strategies.ema_cross:EMACross",
        "opening_range": "plutobt.strategies.opening_range:OpeningRange",
        "donchian": "plutobt.strategies.donchian:Donchian",
    }


def test_builtin_name():
    assert load_strategy("ema_cross").__name__ == "EMACross"


def test_file_single_class():
    assert load_strategy(str(ONE)).__name__ == "Only"


def test_file_named_class():
    assert load_strategy(f"{TWO}:Second").__name__ == "Second"


def test_file_two_classes_needs_name():
    with pytest.raises(StrategyLoadError, match="First, Second"):
        load_strategy(str(TWO))


def test_file_no_strategy():
    with pytest.raises(StrategyLoadError, match="no Strategy subclass"):
        load_strategy(str(LOADER / "no_strategy.py"))


def test_file_named_class_missing():
    with pytest.raises(StrategyLoadError, match="Nope"):
        load_strategy(f"{TWO}:Nope")


def test_missing_file():
    with pytest.raises(StrategyLoadError, match="no such file"):
        load_strategy(str(LOADER / "absent.py"))


def test_unknown_builtin_lists_builtins():
    with pytest.raises(StrategyLoadError, match="ema_cross"):
        load_strategy("nonsense")


def test_same_file_twice_same_id():
    a, b = load_strategy(str(ONE)), load_strategy(str(ONE))
    assert a is b
    assert strategy_id(a) == strategy_id(b) and a.__module__ in sys.modules
    assert a.__module__ == "plutobt_user_one_class"


def test_settings_collects_plain_values_only():
    assert settings(load_strategy(str(ONE))) == {
        "length": 5, "scale": 1.5, "on": True, "label": "x",
    }
    assert list(settings(EMACross)) == ["fast_span", "slow_span", "stop_ticks"]


def test_settings_includes_inherited_in_order():
    cls = load_strategy(str(LOADER / "inherit.py") + ":Child")
    assert list(settings(cls)) == ["p", "c"]


def test_parse_settings_types():
    assert parse_settings(EMACross, ["fast_span=10", "stop_ticks=12.5"]) == {
        "fast_span": 10, "stop_ticks": 12.5,
    }


def test_parse_settings_unknown_lists_known():
    with pytest.raises(StrategyLoadError, match="fast_span, slow_span, stop_ticks"):
        parse_settings(EMACross, ["bogus=1"])


def test_parse_settings_bad_value():
    with pytest.raises(StrategyLoadError, match="fast_span expects a whole number"):
        parse_settings(EMACross, ["fast_span=abc"])
    with pytest.raises(StrategyLoadError, match="stop_ticks expects a number"):
        parse_settings(EMACross, ["stop_ticks=abc"])


def test_parse_settings_missing_equals():
    with pytest.raises(StrategyLoadError, match="name=value"):
        parse_settings(EMACross, ["fast_span"])


def test_parse_settings_bool_and_str():
    cls = load_strategy(str(ONE))
    assert parse_settings(cls, ["on=No"]) == {"on": False}
    assert parse_settings(cls, ["on=TRUE", "label=hi"]) == {"on": True, "label": "hi"}
    with pytest.raises(StrategyLoadError, match="on expects"):
        parse_settings(cls, ["on=maybe"])


def test_make_strategy_sets_params():
    s = make_strategy(EMACross, {"fast_span": 7})
    assert isinstance(s, EMACross) and s.fast_span == 7 and s.slow_span == 50


def _write(path, body):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)
    return path


STRAT = "from plutobt.strategy import Strategy\n\n\nclass S(Strategy):\n    a: int = 1\n"


def test_same_stem_different_folder_refused(tmp_path):
    a = _write(tmp_path / "a" / "dupstem.py", STRAT)
    b = _write(tmp_path / "b" / "dupstem.py", STRAT)
    first = load_strategy(str(a))
    with pytest.raises(StrategyLoadError, match="Another strategy file named dupstem.py"):
        load_strategy(str(b))
    assert Path(sys.modules[first.__module__].__file__).samefile(a)


def test_syntax_error_file(tmp_path):
    p = _write(tmp_path / "badsyntax_x.py", "def (:\n")
    with pytest.raises(StrategyLoadError, match="failed while loading"):
        load_strategy(str(p))
    assert "plutobt_user_badsyntax_x" not in sys.modules


def test_import_error_file_unregisters(tmp_path):
    p = _write(tmp_path / "raises_x.py", "raise RuntimeError('boom')\n")
    with pytest.raises(StrategyLoadError, match="boom"):
        load_strategy(str(p))
    assert "plutobt_user_raises_x" not in sys.modules


def test_system_exit_at_import(tmp_path):
    p = _write(tmp_path / "exits_x.py", "raise SystemExit(3)\n")
    with pytest.raises(StrategyLoadError, match="exit"):
        load_strategy(str(p))
    assert "plutobt_user_exits_x" not in sys.modules


def test_keyboard_interrupt_cleans_up_and_reraises(tmp_path):
    p = _write(tmp_path / "interrupt_x.py", "raise KeyboardInterrupt\n")
    with pytest.raises(KeyboardInterrupt):
        load_strategy(str(p))
    assert "plutobt_user_interrupt_x" not in sys.modules


def test_path_with_spaces(tmp_path):
    p = _write(tmp_path / "my folder" / "spaced_x.py", STRAT)
    assert load_strategy(str(p)).__name__ == "S"


def test_relative_path(tmp_path, monkeypatch):
    _write(tmp_path / "rel_x.py", STRAT)
    monkeypatch.chdir(tmp_path)
    assert load_strategy("rel_x.py").__name__ == "S"
    assert load_strategy("./rel_x.py:S").__name__ == "S"


def test_uppercase_py_extension_with_class(tmp_path):
    p = _write(tmp_path / "Upper_x.PY", STRAT)
    assert load_strategy(f"{p}:S").__name__ == "S"


def test_float_rejects_nan_and_inf():
    for bad in ("nan", "inf", "-inf"):
        with pytest.raises(StrategyLoadError, match="stop_ticks expects a finite number"):
            parse_settings(EMACross, [f"stop_ticks={bad}"])


def test_cli_loaded_and_imported_class_share_a_strategy_id(tmp_path, monkeypatch):
    import importlib
    folder = tmp_path / "pool"
    folder.mkdir()
    (folder / "pooled_strat.py").write_text(
        "from plutobt.strategies.ema_cross import EMACross\n\n\n"
        "class Pooled(EMACross):\n    pass\n", encoding="utf-8")
    via_cli = load_strategy(str(folder / "pooled_strat.py"))
    monkeypatch.syspath_prepend(str(folder))
    imported = importlib.import_module("pooled_strat").Pooled
    assert via_cli is not imported
    assert strategy_id(via_cli) == strategy_id(imported) == "pooled_strat:Pooled"


def test_a_strategy_file_can_import_a_sibling_helper(tmp_path):
    folder = tmp_path / "with_helper"
    folder.mkdir()
    (folder / "sibling_helper_x.py").write_text("SPAN = 7\n", encoding="utf-8")
    (folder / "uses_helper_x.py").write_text(
        "from sibling_helper_x import SPAN\n"
        "from plutobt.strategies.ema_cross import EMACross\n\n\n"
        "class UsesHelper(EMACross):\n    fast_span: int = SPAN\n", encoding="utf-8")
    before = list(sys.path)
    cls = load_strategy(str(folder / "uses_helper_x.py"))
    assert cls.fast_span == 7
    assert sys.path == before
    from plutobt.ledger import code_sha256
    first = code_sha256(cls)
    (folder / "sibling_helper_x.py").write_text("SPAN = 8\n", encoding="utf-8")
    assert code_sha256(cls) != first   # the helper counts as a user module
