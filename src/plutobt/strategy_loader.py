"""Load a strategy from a built-in name or a .py file, and read its typed settings."""
from __future__ import annotations

import importlib
import importlib.util
import inspect
import os
import sys

from plutobt.strategy import Strategy

BUILTINS: dict[str, str] = {
    "ema_cross": "plutobt.strategies.ema_cross:EMACross",
    "opening_range": "plutobt.strategies.opening_range:OpeningRange",
    "donchian": "plutobt.strategies.donchian:Donchian",
}

_TRUE = {"true", "1", "yes"}
_FALSE = {"false", "0", "no"}


class StrategyLoadError(ValueError):
    """A strategy or setting could not be loaded; the message says what to change."""


def _load_builtin(name: str) -> type[Strategy]:
    module_name, class_name = BUILTINS[name].split(":")
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise StrategyLoadError(
            f"built-in strategy {name} could not be loaded: {exc}"
        ) from exc
    return getattr(module, class_name)


def _same_path(a: str, b: str) -> bool:
    return os.path.normcase(os.path.abspath(a)) == os.path.normcase(os.path.abspath(b))


def _load_module(path: str):
    """Import a file once per path: a second load returns the registered module."""
    abs_path = os.path.abspath(path)
    stem = os.path.splitext(os.path.basename(abs_path))[0]
    module_name = f"plutobt_user_{stem}"
    existing = sys.modules.get(module_name)
    if existing is not None:
        existing_file = getattr(existing, "__file__", None)
        if existing_file and _same_path(existing_file, abs_path):
            return existing
    spec = importlib.util.spec_from_file_location(module_name, abs_path)
    if spec is None or spec.loader is None:
        raise StrategyLoadError(f"{path} could not be loaded as a Python file.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        sys.modules.pop(module_name, None)
        raise StrategyLoadError(
            f"{path} failed while loading: {type(exc).__name__}: {exc}"
        ) from exc
    return module


def load_strategy(spec: str) -> type[Strategy]:
    """``spec`` is a built-in name, ``path.py`` or ``path.py:ClassName``."""
    path, class_name = spec, None
    if ".py:" in spec:
        path, class_name = spec.rsplit(".py:", 1)
        path += ".py"
    if not path.lower().endswith(".py"):
        if spec in BUILTINS:
            return _load_builtin(spec)
        raise StrategyLoadError(
            f"{spec} is not a built-in strategy or a .py file. "
            f"Built-in strategies: {', '.join(BUILTINS)}."
        )
    if not os.path.isfile(path):
        raise StrategyLoadError(f"no such file: {path}")
    module = _load_module(path)
    found = {
        name: obj
        for name, obj in vars(module).items()
        if inspect.isclass(obj)
        and issubclass(obj, Strategy)
        and obj is not Strategy
        and obj.__module__ == module.__name__
    }
    if class_name is not None:
        if class_name not in found:
            raise StrategyLoadError(
                f"{path} has no Strategy subclass named {class_name}. "
                f"It has: {', '.join(found) or 'none'}."
            )
        return found[class_name]
    if not found:
        raise StrategyLoadError(f"{path} has no Strategy subclass.")
    if len(found) > 1:
        names = ", ".join(found)
        raise StrategyLoadError(
            f"{path} has more than one strategy ({names}). "
            f"Pick one with {path}:{next(iter(found))}."
        )
    return next(iter(found.values()))


def settings(cls: type) -> dict[str, object]:
    """Public int/float/bool/str class attributes, inherited ones first, in definition order."""
    out: dict[str, object] = {}
    for klass in reversed(cls.__mro__):
        if klass is object or klass is Strategy or not issubclass(klass, Strategy):
            continue
        for name, value in vars(klass).items():
            if name.startswith("_"):
                continue
            if isinstance(value, bool | int | float | str):
                out[name] = value
    return out


def _convert(name: str, default: object, text: str) -> object:
    if isinstance(default, bool):
        low = text.strip().lower()
        if low in _TRUE:
            return True
        if low in _FALSE:
            return False
        raise StrategyLoadError(f"{name} expects true or false (or yes/no, 1/0), not {text!r}.")
    if isinstance(default, int):
        try:
            return int(text)
        except ValueError:
            raise StrategyLoadError(f"{name} expects a whole number, not {text!r}.") from None
    if isinstance(default, float):
        try:
            return float(text)
        except ValueError:
            raise StrategyLoadError(f"{name} expects a number, not {text!r}.") from None
    return text


def parse_settings(cls: type, pairs: list[str]) -> dict[str, object]:
    """Parse ``name=value`` strings by the type of each setting's default."""
    known = settings(cls)
    out: dict[str, object] = {}
    for pair in pairs:
        if "=" not in pair:
            raise StrategyLoadError(f"{pair!r} should look like name=value.")
        name, text = pair.split("=", 1)
        name = name.strip()
        if name not in known:
            raise StrategyLoadError(
                f"{name} is not a setting of {cls.__name__}. "
                f"Settings: {', '.join(known) or 'none'}."
            )
        out[name] = _convert(name, known[name], text)
    return out


def make_strategy(cls: type[Strategy], params: dict) -> Strategy:
    obj = cls()
    for name, value in params.items():
        setattr(obj, name, value)
    return obj
