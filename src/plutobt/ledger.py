"""The run ledger: every setting a study runs, counted (3.0 spec §5).

The Deflated Sharpe Ratio is only honest if N, the number of tries behind the best
result, is honest. So the study tools record every completed run here, one JSON line
each, and N is read back from the file rather than reported by the user.

A try is the distinct (strategy, data, code, settings). The settings are fully resolved:
every public class attribute after the grid is applied, so editing a default that is not in
the grid and rerunning is a new try, and naming a default in the grid is not. The code is a
hash of the strategy's files and of the user modules they import (``code_sha256``), so
editing the logic and rerunning is a new try wherever the logic lives; comments and
formatting are not an edit, and any other change to those files is, which errs strict.
The config switches that change what trades are part of a try (``config_key``); costs and
margin are not, since a cost sweep is not a parameter sweep. Rerunning the same setting on
the same bars, code and switches is one try, and its latest Sharpe is the one used.
``Engine.run`` alone never writes here.

The strategy belongs in its own file: a ``run_study`` call in the same file makes every
edit to that call an edit to the strategy's code.

An honesty aid, not a lock: a strategy is identified by its module and class name, so
renaming the class starts a fresh count, and deleting the file forgets every try. Python
cannot stop code written to hide its tries; the ledger makes the honest path the easy one.
"""
from __future__ import annotations

import ast
import hashlib
import inspect
import json
import math
import os
import site
import sys
import sysconfig
import tokenize
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

DEFAULT_PATH = ".plutobt/trials.jsonl"
_SCHEMA = 1
_KEYS = ("v", "time", "strategy", "code_sha256", "params", "data_sha256", "bars", "sharpe",
         "plutobt")


class LedgerError(ValueError):
    """The ledger cannot be read or written as it stands."""


def strategy_id(cls: type) -> str:
    """A strategy's identity in the ledger: ``<module>:<qualname>``."""
    return f"{cls.__module__}:{cls.__qualname__}"


def _plain(name: str, value):
    """A setting value as plain JSON: numpy scalars become Python ones; anything that is
    not a number, string, bool or None is refused (it could not be written or compared)."""
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or isinstance(value, bool | int | float | str):
        return value
    raise LedgerError(
        f"setting {name!r} = {value!r} is a {type(value).__name__}: a study setting must "
        "be a number, string, bool or None so the ledger can record and compare it")


def _key_value(value):
    # Key only: 20.0 and 20 are the same setting (an int attribute set from a float grid
    # value). The value set on the strategy is never changed.
    if isinstance(value, float) and not isinstance(value, bool) and math.isfinite(value) \
            and value.is_integer():
        return int(value)
    return value


def params_key(params: dict) -> str:
    """The canonical text of a setting: sorted keys, numpy scalars as Python, integral
    floats as ints. Two settings with the same key are the same try."""
    plain = {k: _key_value(_plain(k, v)) for k, v in params.items()}
    return json.dumps(plain, sort_keys=True)


def try_key(params: dict, code: str, config: dict | None = None) -> str:
    """The identity of one try: the code version, the canonical settings, and the config
    switches (``config_key``). With no config the key is the one 3.0.0 wrote."""
    key = {"code": code, "params": json.loads(params_key(params))}
    if config:
        key["config"] = dict(sorted(config.items()))
    return json.dumps(key, sort_keys=True)


def _own_classes(cls: type) -> list[type]:
    # The strategy's own classes, base first: everything in the MRO above Strategy.
    from plutobt.strategy import Strategy

    out = [k for k in cls.__mro__ if k not in (Strategy, object)
           and not (isinstance(k, type) and issubclass(Strategy, k))]
    return list(reversed(out))


def _installed_roots() -> tuple[Path, ...]:
    # Where code that is not the user's lives: the standard library, every site-packages
    # folder Python knows (the user site and Debian's dist-packages included), and
    # plutobt itself.
    found = [p for key, p in sysconfig.get_paths().items()
             if key in ("stdlib", "platstdlib", "purelib", "platlib") and p]
    try:
        found += [*site.getsitepackages(), site.getusersitepackages()]
    except AttributeError:          # some embedded or old virtualenv builds lack these
        pass
    roots = {Path(p).resolve() for p in found if p}
    roots.add(Path(__file__).resolve().parent)
    return tuple(roots)


def _namespace(module) -> dict:
    """A module's own names, read without running any of its code: ``vars`` never calls a
    module-level ``__getattr__``. An object in ``sys.modules`` that is not a module and
    raises on access (a proxy) reads as empty."""
    try:
        names = vars(module)
    except Exception:
        return {}
    return names if isinstance(names, dict) else {}


def _source_file(module) -> Path | None:
    path = _namespace(module).get("__file__")
    if not isinstance(path, str) or not path.endswith(".py"):
        return None
    try:
        path = Path(path).resolve()
        return path if path.is_file() else None
    except OSError:
        return None


def _is_user_file(path: Path, roots: tuple[Path, ...]) -> bool:
    if {"site-packages", "dist-packages"} & set(path.parts):
        return False
    return not any(path == root or root in path.parents for root in roots)


def _modules_held(module) -> list[str]:
    """Names of the modules a loaded module's top level holds: modules themselves, and
    the home modules of the functions, classes and objects found there. This catches what
    the source's import statements do not show (``importlib.import_module``, and a
    notebook cell, which has no source file to read)."""
    names = set()
    for value in list(_namespace(module).values()):
        try:
            if inspect.ismodule(value):
                name = value.__name__
            else:
                name = getattr(value, "__module__", None)
                if not isinstance(name, str):
                    name = getattr(type(value), "__module__", None)
        except Exception:           # an object that raises on attribute access: skip it
            continue
        if isinstance(name, str) and name:
            names.add(name)
    return sorted(names)


def _imports_written(tree: ast.AST, package: str) -> list[str]:
    """Every module name a file's import statements can mean, wherever they stand (top
    level or inside a function), relative ones resolved against ``package``.
    ``from params import THRESH`` yields ``params`` and ``params.THRESH``: the second is
    tried as a submodule, since ``from pkg import mod`` reads the same."""
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                parts = package.split(".") if package else []
                if node.level - 1 > len(parts) or not parts:
                    continue
                home = ".".join(parts[:len(parts) - (node.level - 1)])
                base = f"{home}.{base}" if base else home
            if base:
                out.append(base)
                out += [f"{base}.{a.name}" for a in node.names if a.name != "*"]
    return out


def _find_on_disk(parts: list[str], locations: list[str], name: str) -> tuple[Path | None, str]:
    """Follow ``parts`` down from ``locations`` the way the import system would (a package
    folder, then a module file, in each location in turn; a bare folder is a namespace
    package), reading only the file system. (file, package) of what is found."""
    for i, part in enumerate(parts):
        last = i == len(parts) - 1
        hit, folders = None, []
        for loc in locations:
            base = Path(loc or ".")
            try:
                if (base / part / "__init__.py").is_file():
                    hit = (base / part / "__init__.py", True)
                    break
                if (base / f"{part}.py").is_file():
                    hit = (base / f"{part}.py", False)
                    break
                if (base / part).is_dir():
                    folders.append(str(base / part))
            except OSError:
                continue
        if hit is None:
            if last or not folders:
                return None, ""                 # not found, or a namespace package: no file
            locations = folders
            continue
        path, is_package = hit
        if last:
            return path.resolve(), (name if is_package else name.rpartition(".")[0])
        if not is_package:
            return None, ""                     # a plain module has no submodules
        locations = [str(path.parent)]
    return None, ""


def _locate(name: str) -> tuple[Path | None, str]:
    """(source file, package) of the module called ``name``, without importing it and
    without running any user code. A loaded module answers from its own names. One that
    is not loaded yet (an import inside a function that has not run) is found on disk,
    starting from its nearest loaded parent package, or from ``sys.path`` for a top-level
    name, so the code is the same before and after that import first runs."""
    module = sys.modules.get(name)
    if module is not None:
        names = _namespace(module)
        return _source_file(module), (name if "__path__" in names
                                      else name.rpartition(".")[0])
    parts = name.split(".")
    for depth in range(len(parts) - 1, 0, -1):
        parent = sys.modules.get(".".join(parts[:depth]))
        if parent is None:
            continue
        search = _namespace(parent).get("__path__")
        try:
            locations = [str(p) for p in search] if search is not None else None
        except Exception:
            locations = None
        if locations is None:
            return None, ""                     # its parent is a plain module
        return _find_on_disk(parts[depth:], locations, name)
    return _find_on_disk(parts, [p for p in sys.path if isinstance(p, str)], name)


def _with_parents(name: str) -> list[str]:
    parts = name.split(".")
    return [".".join(parts[:i]) for i in range(1, len(parts) + 1)]


def _code_modules(cls: type) -> tuple[list[tuple[type, object]], list[Path]]:
    """(each of the strategy's classes with its module, the files that make up its code).

    The files: every module that defines one of the classes, then every USER module those
    import, followed through. An import is found two ways: as written in the source
    (``import x``, ``from x import NAME``, relative imports, imports inside functions
    whether or not they have run yet, and the parent packages they run), and as held by
    the loaded module (which also catches ``importlib.import_module`` and is all a class
    with no source file, such as a notebook cell, has). Installed packages, the standard
    library and plutobt itself are never followed, nothing is imported to do this, and no
    user code runs."""
    roots = _installed_roots()
    owners, owner_files, other_files = [], [], set()
    seen: set[Path] = set()
    queue: list[tuple[Path | None, str, object]] = []

    def visit(name: str) -> None:
        """Queue the module called ``name`` if it is the user's."""
        path, package = _locate(name)
        if path is None or path in seen or not _is_user_file(path, roots):
            return
        seen.add(path)
        other_files.add(path)
        queue.append((path, package, sys.modules.get(name)))

    done = set()
    for k in _own_classes(cls):
        module = sys.modules.get(k.__module__)
        owners.append((k, module))
        if module is None or id(module) in done:
            continue
        done.add(id(module))
        path, package = _locate(k.__module__)
        if path is not None and path not in seen:
            seen.add(path)
            owner_files.append(path)
        queue.append((path, package, module))    # followed even with no file (a notebook)
        for parent in _with_parents(k.__module__)[:-1]:         # the packages it lives in
            visit(parent)

    while queue:
        path, package, module = queue.pop(0)
        tree = None
        if path is not None:
            try:
                with tokenize.open(path) as fh:
                    tree = ast.parse(fh.read())
            except (SyntaxError, ValueError, OSError):
                tree = None
        for name in (_imports_written(tree, package) if tree is not None else []):
            for part in _with_parents(name):
                visit(part)
        if module is not None:
            for name in _modules_held(module):
                for part in _with_parents(name):
                    visit(part)
    return owners, owner_files + sorted(other_files - set(owner_files))


def code_files(cls: type) -> list[str]:
    """The files ``code_sha256`` reads for this strategy: its own modules first, then the
    user modules they import, by path."""
    return [str(p) for p in _code_modules(cls)[1]]


def _normalized(path: Path) -> str:
    # Parsed and written back out, so comments, blank lines and formatting are not an
    # edit; text that does not parse is hashed as it stands.
    with tokenize.open(path) as fh:
        text = fh.read()
    try:
        return ast.unparse(ast.parse(text))
    except (SyntaxError, ValueError):
        return text.replace("\r\n", "\n")


def _class_fallback(h, k: type) -> None:
    # A class with no file (a notebook cell, ``exec``): its own source if Python kept it,
    # else its methods' bytecode and constants.
    try:
        h.update(inspect.getsource(k).encode("utf-8"))
    except (OSError, TypeError):
        for name, value in sorted(vars(k).items()):
            func = getattr(value, "__func__", value)
            code = getattr(func, "__code__", None)
            if code is not None:
                h.update(name.encode("utf-8") + code.co_code + repr(code.co_consts)
                         .encode("utf-8"))


def code_sha256(cls: type) -> str:
    """sha256 of the strategy's code, so an edit to the logic is a new try wherever the
    logic lives: the whole file of every module that defines one of the strategy's classes
    (a module-level constant or a helper function counts), plus every user module those
    files import, followed through (``code_files`` lists them). Comments
    and formatting are not an edit. A class with no file (a notebook cell, ``exec``) falls
    back to its own source or bytecode, so a helper edited in another notebook cell is
    not seen."""
    owners, files = _code_modules(cls)
    h = hashlib.sha256()
    own = []
    for k, module in owners:
        h.update(k.__qualname__.encode("utf-8") + b"\0")
        path = _source_file(module) if module is not None else None
        if path is None:
            _class_fallback(h, k)
        elif path not in own:
            own.append(path)
    for path in own:                            # the strategy's own files, base class first
        h.update(_normalized(path).encode("utf-8") + b"\0")
    # The imported files go in as a sorted set of digests: which one was found first
    # (and the folder the project sits in) must not change the hash.
    rest = sorted(hashlib.sha256(_normalized(p).encode("utf-8")).hexdigest()
                  for p in files if p not in own)
    for digest in rest:
        h.update(digest.encode("ascii") + b"\0")
    return h.hexdigest()


_TRY_CONFIG = ("allow_short", "break_on", "carry_overnight", "on_margin_call")


def config_key(config) -> dict:
    """The config switches that change what a strategy trades, which makes them part of a
    try. Costs are left out on purpose (a cost sweep is not a parameter sweep), and so are
    the look-ahead check switches, which change no result."""
    return {name: _plain(name, getattr(config, name)) for name in _TRY_CONFIG}


def resolved_settings(cls: type, params: dict) -> dict:
    """Every public, non-method class attribute of the strategy with ``params`` applied:
    what the run actually used. A value that is not plain JSON is kept as its repr."""
    out = {}
    for k in _own_classes(cls):
        for name, value in vars(k).items():
            if name.startswith("_") or callable(value) or isinstance(
                    value, staticmethod | classmethod | property):
                continue
            out[name] = value
    out.update(params)
    resolved = {}
    for name, value in out.items():
        try:
            resolved[name] = _plain(name, value)
        except LedgerError:
            resolved[name] = repr(value)
    return resolved


class Ledger:
    """One JSON-lines file of completed runs. ``is_new`` is whether the file was absent
    when this object was made."""

    def __init__(self, path: str | os.PathLike):
        self.path = Path(path)
        self.is_new = not self.path.exists()

    def record(self, strategy: str, params: dict, data_sha256: str, bars: int,
               sharpe: float, code: str = "", config: dict | None = None) -> None:
        """Append one completed run and flush, so a crash loses at most the run in flight.
        ``params`` are the resolved settings (``resolved_settings``), ``code`` the
        strategy's ``code_sha256``, ``config`` its ``config_key``."""
        from plutobt import __version__

        plain = {k: _plain(k, v) for k, v in params.items()}
        sharpe = float(sharpe)
        line = {
            "v": _SCHEMA,
            "time": datetime.now(UTC).isoformat(timespec="seconds"),
            "strategy": strategy,
            "code_sha256": code,
            "config": dict(sorted((config or {}).items())),
            "params": dict(sorted(plain.items())),
            "data_sha256": data_sha256,
            "bars": int(bars),
            "sharpe": sharpe if math.isfinite(sharpe) else None,
            "plutobt": __version__,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps(line, allow_nan=False) + "\n")
            fh.flush()

    def _lines(self):
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as fh:
            for no, text in enumerate(fh, 1):
                if not text.strip():
                    continue
                try:
                    line = json.loads(text)
                except json.JSONDecodeError as err:
                    raise LedgerError(
                        f"{self.path} line {no} is not valid JSON ({err.msg}). The ledger "
                        "never skips a line, since that would quietly lower N: fix or "
                        "remove that line by hand") from None
                if not isinstance(line, dict) or line.get("v") != _SCHEMA:
                    version = line.get("v") if isinstance(line, dict) else None
                    raise LedgerError(
                        f"{self.path} line {no} has schema version {version}; this "
                        f"plutobt reads version {_SCHEMA} only")
                missing = [k for k in _KEYS if k not in line]
                if missing:
                    raise LedgerError(f"{self.path} line {no} lacks {', '.join(missing)}")
                yield line

    def tries(self, strategy: str, data_sha256: str,
              also: tuple[str, ...] = ()) -> dict[str, float]:
        """Every distinct try ever recorded for ``strategy`` on these bars, keyed by
        ``try_key`` (code version, settings, config switches) and mapped to its latest
        per-bar Sharpe (NaN for a run with no return volatility). A line 3.0.0 wrote has
        no config and reads as a try of its own. ``also`` are further hashes that mean
        the same bars: 3.0.0 keyed bars by the exact frame (``frame_sha256``), so the
        study passes that too and an older ledger keeps counting."""
        same = {data_sha256, *also}
        out: dict[str, float] = {}
        for line in self._lines():
            if line["strategy"] == strategy and line["data_sha256"] in same:
                s = line["sharpe"]
                key = try_key(line["params"], line["code_sha256"], line.get("config"))
                out[key] = float("nan") if s is None else float(s)
        return out

    def tries_on_other_bars(self, strategy: str, data_sha256: str,
                            also: tuple[str, ...] = ()) -> int:
        """How many distinct tries the file holds for ``strategy`` on any OTHER bars. N
        never counts them (different bars are a different question), but a report that
        says nothing would hide twenty tries behind one new bar."""
        same = {data_sha256, *also}
        seen = set()
        for line in self._lines():
            if line["strategy"] == strategy and line["data_sha256"] not in same:
                seen.add((line["data_sha256"],
                          try_key(line["params"], line["code_sha256"], line.get("config"))))
        return len(seen)
