"""Per-user local store of Databento bars a trader has already bought.

Layout under the root (default ``~/.plutobt/data``, override with the
``PLUTOBT_DATA_DIR`` environment variable):

    <root>/GLBX.MDP3/<schema>/<contract>.csv
    <root>/index.json   {"version": 1, "entries": {"GLBX.MDP3/<schema>/<contract>":
                                                   {"covered": [["2026-01-01", "2026-03-16"]]}}}

Dates are ``datetime.date`` and every range is half-open [start, end): it covers
the UTC calendar days start .. end-1. Covered ranges are kept sorted and merged.
ts is int64 unix seconds of the bar open, UTC. Contract codes are stored upper case.
Nothing here touches the network.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from datetime import date
from pathlib import Path

import pandas as pd

DATASET = "GLBX.MDP3"
COLUMNS = ["ts", "open", "high", "low", "close", "volume"]
_DTYPES = {"ts": "int64", "open": "float64", "high": "float64",
           "low": "float64", "close": "float64", "volume": "int64"}
_DAY = 86400


def data_dir() -> Path:
    env = os.environ.get("PLUTOBT_DATA_DIR")
    if env:
        return Path(env)
    return Path.home() / ".plutobt" / "data"


_NAME_OK = re.compile(r"[A-Za-z0-9._-]+")
_RESERVED = {"CON", "PRN", "AUX", "NUL",
             *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}
_SECONDS_MAX = 1e11


def _check_name(kind: str, value: str) -> str:
    if (not isinstance(value, str) or not _NAME_OK.fullmatch(value) or value.startswith(".")
            or ".." in value or value.split(".")[0].upper() in _RESERVED):
        raise ValueError(
            f"invalid {kind} {value!r}: use letters, digits, '.', '_' and '-' only, "
            "not starting with '.', and not a reserved file name")
    return value


def _day_ts(d: date) -> int:
    return int(pd.Timestamp(d, tz="UTC").timestamp())


def _merge(ranges: list[tuple[date, date]]) -> list[tuple[date, date]]:
    out: list[tuple[date, date]] = []
    for s, e in sorted(r for r in ranges if r[0] < r[1]):
        if out and s <= out[-1][1]:
            if e > out[-1][1]:
                out[-1] = (out[-1][0], e)
        else:
            out.append((s, e))
    return out


def _atomic_write(path: Path, writer) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    os.close(fd)
    try:
        writer(Path(tmp))
        with open(tmp, "r+b") as fh:
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


class Store:
    def __init__(self, root: Path | None = None):
        self.root = Path(root) if root is not None else data_dir()

    # -- paths / index -------------------------------------------------
    def _key(self, schema: str, contract: str) -> str:
        name = _check_name('contract', contract).upper()
        return f"{DATASET}/{_check_name('schema', schema)}/{name}"

    def _csv(self, schema: str, contract: str) -> Path:
        self._key(schema, contract)
        return self.root / DATASET / schema / f"{contract.upper()}.csv"

    @property
    def _index_path(self) -> Path:
        return self.root / "index.json"

    def _read_index(self) -> dict:
        p = self._index_path
        if not p.exists():
            # A CSV with no index entry counts as held nothing (the trader is
            # quoted again and must say yes), so a missing index is an empty store.
            return {"version": 1, "entries": {}}
        try:
            idx = json.loads(p.read_text(encoding="utf-8"))
            if not isinstance(idx, dict) or not isinstance(idx.get("entries"), dict):
                raise ValueError("bad shape")
            for ent in idx["entries"].values():
                for s, e in ent["covered"]:
                    date.fromisoformat(s)
                    date.fromisoformat(e)
            return idx
        except (OSError, ValueError, KeyError, TypeError):
            raise ValueError(
                f"The store index {p} is unreadable or corrupt. "
                "Not treating the store as empty, so you are not charged again without notice. "
                "Fix or remove index.json (the bars in the CSV files are untouched).") from None

    def _covered(self, idx: dict, key: str) -> list[tuple[date, date]]:
        ent = idx["entries"].get(key)
        if not ent:
            return []
        return _merge([(date.fromisoformat(s), date.fromisoformat(e)) for s, e in ent["covered"]])

    # -- public API ----------------------------------------------------
    def held(self, schema: str, contract: str) -> list[tuple[date, date]]:
        key = self._key(schema, contract)
        return self._covered(self._read_index(), key)

    def missing(
        self, schema: str, contract: str, start: date, end: date
    ) -> list[tuple[date, date]]:
        held = self.held(schema, contract)
        gaps: list[tuple[date, date]] = []
        cur = start
        for s, e in held:
            if e <= cur:
                continue
            if s >= end:
                break
            if s > cur:
                gaps.append((cur, min(s, end)))
            cur = max(cur, e)
            if cur >= end:
                break
        if cur < end:
            gaps.append((cur, end))
        return gaps

    def _read_csv(self, path: Path) -> pd.DataFrame:
        if not path.exists():
            return pd.DataFrame({c: pd.Series(dtype=t) for c, t in _DTYPES.items()})
        return pd.read_csv(path, dtype=_DTYPES)[COLUMNS]

    def add(
        self, schema: str, contract: str, start: date, end: date, frame: pd.DataFrame
    ) -> None:
        key = self._key(schema, contract)
        if start >= end:
            raise ValueError(f"cannot add {contract}: start {start} is not before end {end}")
        lo, hi = _day_ts(start), _day_ts(end)
        new = frame[COLUMNS].astype(_DTYPES)
        if len(new):
            if new["ts"].abs().max() >= _SECONDS_MAX:
                raise ValueError(
                    f"cannot add {contract}: ts must be int64 unix seconds of the bar open "
                    "(a value of 1e11 or more is another unit, such as milliseconds)")
            if new["ts"].min() < lo or new["ts"].max() >= hi:
                raise ValueError(
                    f"cannot add {contract}: bars fall outside [{start}, {end}) UTC; "
                    "refusing to record coverage the frame does not hold")
        idx = self._read_index()
        old = self._read_csv(self._csv(schema, contract))
        merged = pd.concat([old, new], ignore_index=True)
        merged = merged.drop_duplicates(subset="ts", keep="first")
        merged = merged.sort_values("ts", kind="stable").reset_index(drop=True)

        covered = _merge(self._covered(idx, key) + [(start, end)])
        idx["version"] = 1
        idx["entries"][key] = {"covered": [[s.isoformat(), e.isoformat()] for s, e in covered]}

        if not self._index_path.exists():
            # The index must exist before any CSV, so a crash between the two
            # writes leaves a readable (empty) index rather than a bricked store.
            _atomic_write(self._index_path, lambda p: p.write_text(
                json.dumps({"version": 1, "entries": {}}), encoding="utf-8"))
        # CSV first, then index: a crash in between leaves the index claiming less.
        _atomic_write(self._csv(schema, contract),
                      lambda p: merged.to_csv(p, index=False))
        _atomic_write(self._index_path,
                      lambda p: p.write_text(
                          json.dumps(idx, indent=2, sort_keys=True), encoding="utf-8"))

    def frame(
        self, schema: str, contract: str, start: date, end: date
    ) -> pd.DataFrame:
        key = self._key(schema, contract)
        gaps = self.missing(schema, contract, start, end)  # also refuses a corrupt index
        if gaps:
            days = ", ".join(f"{a} to {b}" for a, b in gaps)
            raise ValueError(
                f"{contract.upper()} is not fully held for [{start}, {end}); uncovered: {days}")
        path = self._csv(schema, contract)
        if not path.exists():
            raise ValueError(
                f"the index says {contract.upper()} is held but its file is missing; "
                f"delete the {key} entry in {self._index_path} or re-fetch")
        df = self._read_csv(path)
        lo, hi = _day_ts(start), _day_ts(end)
        df = df[(df["ts"] >= lo) & (df["ts"] < hi)]
        return df.sort_values("ts").reset_index(drop=True)
