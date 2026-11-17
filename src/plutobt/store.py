"""Per-user local store of Databento bars a trader has already bought.

Layout under the root (default ``~/.plutobt/data``, override with the
``PLUTOBT_DATA_DIR`` environment variable):

    <root>/GLBX.MDP3/<schema>/<contract>.csv
    <root>/index.json   {"version": 1, "entries": {"GLBX.MDP3/<schema>/<contract>":
                                                   {"covered": [["2026-01-01", "2026-03-16"]]}}}

Dates are ``datetime.date`` and every range is half-open [start, end): it covers
the UTC calendar days start .. end-1. Covered ranges are kept sorted and merged.
Nothing here touches the network.
"""

from __future__ import annotations

import json
import os
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


def _check_name(kind: str, value: str) -> str:
    if (not value or ".." in value or "/" in value or "\\" in value
            or os.sep in value):
        raise ValueError(f"invalid {kind} {value!r}: must not contain a path separator or '..'")
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
        return f"{DATASET}/{_check_name('schema', schema)}/{_check_name('contract', contract)}"

    def _csv(self, schema: str, contract: str) -> Path:
        self._key(schema, contract)
        return self.root / DATASET / schema / f"{contract}.csv"

    @property
    def _index_path(self) -> Path:
        return self.root / "index.json"

    def _has_csv(self) -> bool:
        base = self.root / DATASET
        return base.is_dir() and any(base.rglob("*.csv"))

    def _read_index(self) -> dict:
        p = self._index_path
        if not p.exists():
            if self._has_csv():
                raise ValueError(
                    f"The store index {p} is missing but bought data files exist. "
                    "Not treating the store as empty, so nothing is bought twice by mistake. "
                    "Restore index.json or move the data folder aside.")
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
        idx = self._read_index()
        new = frame[COLUMNS].astype(_DTYPES)
        old = self._read_csv(self._csv(schema, contract))
        merged = pd.concat([old, new], ignore_index=True)
        merged = merged.drop_duplicates(subset="ts", keep="first")
        merged = merged.sort_values("ts", kind="stable").reset_index(drop=True)

        covered = _merge(self._covered(idx, key) + [(start, end)])
        idx["version"] = 1
        idx["entries"][key] = {"covered": [[s.isoformat(), e.isoformat()] for s, e in covered]}

        # CSV first, then index: a crash in between leaves the index claiming less.
        _atomic_write(self._csv(schema, contract),
                      lambda p: merged.to_csv(p, index=False))
        _atomic_write(self._index_path,
                      lambda p: p.write_text(
                          json.dumps(idx, indent=2, sort_keys=True), encoding="utf-8"))

    def frame(
        self, schema: str, contract: str, start: date, end: date
    ) -> pd.DataFrame:
        self._key(schema, contract)
        self._read_index()  # refuse a corrupt index even on read
        df = self._read_csv(self._csv(schema, contract))
        lo, hi = _day_ts(start), _day_ts(end)
        df = df[(df["ts"] >= lo) & (df["ts"] < hi)]
        return df.sort_values("ts").reset_index(drop=True)
