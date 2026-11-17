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
