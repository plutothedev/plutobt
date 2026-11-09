"""The run ledger: every setting a study runs, counted (3.0 spec §5).

The Deflated Sharpe Ratio is only honest if N, the number of tries behind the best
result, is honest. So the study tools record every completed run here, one JSON line
each, and N is read back from the file rather than reported by the user.

A try is the distinct (strategy, data, settings): rerunning the same setting on the same
bars is one try, and its latest Sharpe is the one used. Costs and margin are not part of
the key, since a cost sweep is not a parameter sweep. ``Engine.run`` alone never writes
here.

An honesty aid, not a lock: a strategy is identified by its module and class name, so
renaming the class starts a fresh count, and deleting the file forgets every try. Python
cannot stop code written to hide its tries; the ledger makes the honest path the easy one.
"""
from __future__ import annotations

import json
import math
import os
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

DEFAULT_PATH = ".plutobt/trials.jsonl"
_SCHEMA = 1
_KEYS = ("v", "time", "strategy", "params", "data_sha256", "bars", "sharpe", "plutobt")


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


class Ledger:
    """One JSON-lines file of completed runs. ``is_new`` is whether the file was absent
    when this object was made."""

    def __init__(self, path: str | os.PathLike):
        self.path = Path(path)
        self.is_new = not self.path.exists()

    def record(self, strategy: str, params: dict, data_sha256: str, bars: int,
               sharpe: float) -> None:
        """Append one completed run and flush, so a crash loses at most the run in flight."""
        from plutobt import __version__

        plain = {k: _plain(k, v) for k, v in params.items()}
        sharpe = float(sharpe)
        line = {
            "v": _SCHEMA,
            "time": datetime.now(UTC).isoformat(timespec="seconds"),
            "strategy": strategy,
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

    def tries(self, strategy: str, data_sha256: str) -> dict[str, float]:
        """Every distinct setting ever recorded for ``strategy`` on these bars, mapped to
        its latest per-bar Sharpe (NaN for a run with no return volatility)."""
        out: dict[str, float] = {}
        for line in self._lines():
            if line["strategy"] == strategy and line["data_sha256"] == data_sha256:
                s = line["sharpe"]
                out[params_key(line["params"])] = float("nan") if s is None else float(s)
        return out
