"""Futures contract codes, the equity-index expiry calendar, and contract sets (spec §5).

A contract is a root (``NQ``), a delivery month and a year. ``parse_contract`` reads the
forms vendors use (``NQZ5``, ``NQZ25``, ``NQZ2025``, ``NQ 12-25``, ``NQ DEC25``) into one
canonical ``Contract`` whose ``code`` is ``NQZ2025``.

The expiry calendar covers the CME equity-index roots only (``CALENDAR_ROOTS``): their
quarterly contracts (March, June, September, December) expire on the third Friday of the
contract month unless an exchange holiday moves it (``_EXPIRY_EXCEPTIONS``), and CME's
customary roll date is the Monday before that third Friday (CME Group, "Equity Index Roll
Dates", cmegroup.com/trading/equity-index/rolldates.html, read 2026-11-06: "Equity
products roll date is the Monday prior to the third Friday of the expiration month"). Other roots
(CL, GC, ZN, ...) roll on their own exchange rules, which this module does not encode:
use the volume trigger or explicit roll dates for them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, timedelta
from decimal import Decimal

import numpy as np
import pandas as pd

MONTH_CODES: dict[str, int] = {
    "F": 1, "G": 2, "H": 3, "J": 4, "K": 5, "M": 6,
    "N": 7, "Q": 8, "U": 9, "V": 10, "X": 11, "Z": 12,
}
_CODE_FOR_MONTH = {m: c for c, m in MONTH_CODES.items()}
_MONTH_NAMES = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
    "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}

CALENDAR_ROOTS = frozenset({"NQ", "ES", "MNQ", "MES", "YM", "RTY"})
ROLL_DAYS_BEFORE_THIRD_FRIDAY = 4      # Monday of the expiry week
_QUARTERLY = (3, 6, 9, 12)

# Contracts whose last trading day is not the plain third Friday. CME's roll-date table
# (read 2026-11-06) lists June 2026 expiring Thursday 6/18: Friday 6/19 is Juneteenth.
# The roll day is still the Monday before the third Friday (6/15), as CME lists it.
_EXPIRY_EXCEPTIONS: dict[str, date] = {
    f"{root}M2026": date(2026, 6, 18) for root in sorted(CALENDAR_ROOTS)
}
_ROLL_DAY_EXCEPTIONS: dict[str, date] = {}

_SHORT = re.compile(r"^([A-Z]+)([FGHJKMNQUVXZ])(\d{1,2}|\d{4})$")
_NUMERIC = re.compile(r"^([A-Z]+)\s+(\d{1,2})-(\d{2}|\d{4})$")
_NAMED = re.compile(r"^([A-Z]+)\s+([A-Z]{3})(\d{2}|\d{4})$")


@dataclass(frozen=True, order=True)
class Contract:
    """One futures contract. Orders by (year, month), then root."""

    year: int = field(init=False, repr=False)
    month: int = field(init=False, repr=False)
    root: str = field(init=False, repr=False)

    def __init__(self, root: str, month: int, year: int) -> None:
        object.__setattr__(self, "root", root)
        object.__setattr__(self, "month", int(month))
        object.__setattr__(self, "year", int(year))

    def __repr__(self) -> str:
        return f"Contract({self.root!r}, {self.month}, {self.year})"

    @property
    def code(self) -> str:
        """Canonical code, e.g. ``NQZ2025``."""
        return f"{self.root}{_CODE_FOR_MONTH[self.month]}{self.year:04d}"


def tick_decimals(tick: float) -> int:
    """Decimal places of a tick size: 0.25 -> 2, 0.015625 -> 6, 5e-05 -> 5, 1.0 -> 0.
    Read through Decimal so a tick ``repr`` prints in scientific notation still
    counts its places."""
    return max(0, -Decimal(repr(float(tick))).normalize().as_tuple().exponent)


def snap_to_tick(values: np.ndarray, tick: float) -> np.ndarray:
    """Each price as the exact float of its nearest tick multiple's decimal."""
    return np.round(np.rint(values / tick) * tick, tick_decimals(tick))


def _bad(code: str, why: str) -> ValueError:
    return ValueError(
        f"not a futures contract code: {code!r} ({why}). Accepted forms: NQZ5, NQZ25, "
        "NQZ2025, 'NQ 12-25', 'NQ DEC25'"
    )


def _year(digits: str, code: str, near: date | None) -> int:
    if len(digits) == 4:
        return int(digits)
    if len(digits) == 2:
        return 2000 + int(digits)
    if near is None:
        raise ValueError(
            f"contract code {code!r} has a one-digit year, which needs near=<a date in "
            "the data> to tell 2015 from 2025"
        )
    d = int(digits)
    base = near.year - near.year % 10
    candidates = [base - 10 + d, base + d, base + 10 + d]
    dist = sorted((abs(y - near.year), y) for y in candidates)
    if dist[0][0] == dist[1][0]:
        raise ValueError(
            f"contract code {code!r}: the one-digit year is equally near {dist[0][1]} and "
            f"{dist[1][1]} for near={near}; pass a code with a two- or four-digit year"
        )
    return dist[0][1]


def parse_contract(code: str, *, near: date | None = None) -> Contract:
    """Parse a vendor contract code into a ``Contract``.

    ``near`` (a date inside the contract's data) resolves a one-digit year to the
    nearest matching year. Anything that is not a contract code raises ValueError.
    """
    text = code.strip().upper() if isinstance(code, str) else ""
    m = _SHORT.match(text)
    if m:
        root, month = m.group(1), MONTH_CODES[m.group(2)]
        return Contract(root, month, _year(m.group(3), code, near))
    m = _NUMERIC.match(text) or _NAMED.match(text)
    if m:
        root, mon = m.group(1), m.group(2)
        month = int(mon) if mon.isdigit() else _MONTH_NAMES.get(mon, 0)
        if not 1 <= month <= 12:
            raise _bad(code, f"no month {mon!r}")
        return Contract(root, month, _year(m.group(3), code, near))
    raise _bad(code, "unrecognised form")


def _third_friday(year: int, month: int) -> date:
    first = date(year, month, 1)
    offset = (4 - first.weekday()) % 7          # Friday is weekday 4
    return first + timedelta(days=offset + 14)


def expiry(contract: Contract) -> date:
    """Last trading day of an equity-index quarterly contract.

    The third Friday of the contract month, unless an exchange holiday moved it. Only
    ``CALENDAR_ROOTS`` are covered; any other root raises ValueError.
    """
    if contract.root not in CALENDAR_ROOTS:
        raise ValueError(
            f"no expiry calendar for {contract.code}: the calendar roll covers the "
            f"equity-index roots {sorted(CALENDAR_ROOTS)} only. Roll {contract.root} with "
            "trigger='volume' or pass roll_dates={'<contract>': 'YYYY-MM-DD'}"
        )
    if contract.month not in _QUARTERLY:
        raise ValueError(
            f"{contract.code}: equity-index futures list quarterly months only "
            "(H, M, U, Z)"
        )
    return _EXPIRY_EXCEPTIONS.get(contract.code, _third_friday(contract.year,
                                                                contract.month))


def calendar_roll_day(contract: Contract) -> date:
    """CME's customary roll day for ``contract``: the Monday before the third Friday of
    its expiry month (even when a holiday moved the expiry itself), unless CME published
    a different date (``_ROLL_DAY_EXCEPTIONS``)."""
    expiry(contract)                     # same root and month checks
    third = _third_friday(contract.year, contract.month)
    return _ROLL_DAY_EXCEPTIONS.get(contract.code,
                                    third - timedelta(days=ROLL_DAYS_BEFORE_THIRD_FRIDAY))


# ---------------------------------------------------------------------------
# Contract sets
# ---------------------------------------------------------------------------

_BAR_COLUMNS = ("ts", "open", "high", "low", "close", "volume")


@dataclass
class ContractSet:
    """One root's contracts, validated, sorted by expiry (spec §5).

    ``frames`` maps each canonical code (``NQH2026``) to its bars; ``interval`` is the
    shared bar interval in seconds; ``file_sha256`` maps each code to the sha256 of the
    file it was read from (empty when built from frames).
    """

    root: str
    contracts: list[Contract]
    frames: dict[str, object]
    interval: int
    file_sha256: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_frames(cls, frames: dict, file_sha256: dict[str, str] | None = None
                    ) -> ContractSet:
        """Validate and sort ``{code: bars}``. Each frame needs the canonical bar columns
        (ts in unix seconds of the bar open). Refuses, naming the problem: fewer than two
        contracts, mixed roots, a duplicate contract, an empty contract, a bad bar, a
        price off the tick grid, mixed bar intervals, a gap in an equity-index quarterly
        chain, and (other roots) consecutive contracts whose data never overlap."""
        from plutobt.data import ohlcv_problem  # local: data imports instruments
        from plutobt.instruments import REGISTRY

        if len(frames) < 2:
            raise ValueError(
                f"a contract set needs at least two contracts to roll between, got "
                f"{len(frames)}"
            )
        parsed: dict[str, tuple[Contract, object]] = {}
        sources: dict[str, str] = {}
        for key, df in frames.items():
            near = None
            if len(df) and "ts" in df.columns:
                near = date(1970, 1, 1) + timedelta(seconds=int(df["ts"].iloc[0]))
            c = parse_contract(key, near=near)
            if c.code in parsed:
                raise ValueError(
                    f"{c.code} appears twice in the contract set (as {sources[c.code]!r} "
                    f"and {key!r}); keep one file per contract"
                )
            parsed[c.code] = (c, df)
            sources[c.code] = key
        roots = sorted({c.root for c, _ in parsed.values()})
        if len(roots) > 1:
            raise ValueError(
                f"a contract set holds one root, got {roots}: "
                + ", ".join(sorted(parsed)) + ". Build one set per root"
            )
        root = roots[0]
        if root not in REGISTRY:
            raise ValueError(
                f"no instrument spec for root {root!r} (its tick size is needed to check "
                f"and adjust prices); known roots: {sorted(REGISTRY)}"
            )
        tick = REGISTRY[root].tick_size
        ordered = sorted(parsed.values(), key=lambda pair: pair[0])
        intervals: dict[str, int] = {}
        for c, df in ordered:
            missing = [col for col in _BAR_COLUMNS if col not in df.columns]
            if missing:
                raise ValueError(f"{c.code}: missing column(s) {missing}")
            if len(df) == 0:
                raise ValueError(f"{c.code} has no bars")
            if "contract" in df.columns:
                held = set(df["contract"].astype(str))
                if held != {c.code}:
                    raise ValueError(
                        f"{c.code}: its 'contract' column holds {sorted(held)}, not "
                        f"{c.code}"
                    )
            problem = ohlcv_problem(df)
            if problem is not None:
                raise ValueError(f"{c.code}: {problem}")
            ts = df["ts"].to_numpy()
            if len(ts) > 1 and not (ts[1:] > ts[:-1]).all():
                i = int(np.flatnonzero(ts[1:] <= ts[:-1])[0]) + 1
                raise ValueError(f"{c.code}: bar timestamps must strictly increase "
                                 f"(row {i}, ts {int(ts[i])})")
            for col in ("open", "high", "low", "close"):
                v = df[col].to_numpy(dtype="float64") / tick
                off = np.flatnonzero(np.abs(v - np.rint(v)) > 1e-6)
                if off.size:
                    i = int(off[0])
                    raise ValueError(
                        f"{c.code}: {col} at row {i} is {float(df[col].iloc[i])!r}, not "
                        f"on the {root} tick grid ({tick}); adjustment works in whole "
                        "ticks"
                    )
            if len(ts) > 1:
                diffs = pd.Series(ts[1:] - ts[:-1])
                intervals[c.code] = int(diffs.mode().iloc[0])
        distinct = sorted(set(intervals.values()))
        if len(distinct) > 1:
            listing = ", ".join(f"{code}={sec}s" for code, sec in intervals.items())
            raise ValueError(
                f"the contracts have different bar intervals ({listing}); resample them "
                "to one interval first"
            )
        contracts = [c for c, _ in ordered]
        snapped: dict[str, object] = {}
        for c, df in ordered:
            # On the grid within tolerance (checked above): store the exact floats, so
            # vendor float noise can never stop a stop resting at an exact level.
            out = df.reset_index(drop=True).copy()
            for col in ("open", "high", "low", "close"):
                out[col] = snap_to_tick(out[col].to_numpy(dtype="float64"), tick)
            snapped[c.code] = out
        for a, b in zip(contracts, contracts[1:], strict=False):
            if root in CALENDAR_ROOTS:
                y, m = a.year, a.month + 3
                if m > 12:
                    y, m = y + 1, m - 12
                if (b.year, b.month) != (y, m):
                    raise ValueError(
                        f"the quarterly chain has a gap: {a.code} is followed by "
                        f"{b.code}, so {Contract(root, m, y).code} is missing"
                    )
            else:
                a_last = int(parsed[a.code][1]["ts"].iloc[-1])
                b_first = int(parsed[b.code][1]["ts"].iloc[0])
                if b_first > a_last:
                    raise ValueError(
                        f"{a.code} and {b.code} do not overlap ({a.code} ends before "
                        f"{b.code} starts), so there is no bar where both trade to roll "
                        "at"
                    )
        return cls(
            root=root,
            contracts=contracts,
            frames={c.code: snapped[c.code] for c in contracts},
            interval=distinct[0] if distinct else 0,
            file_sha256=dict(file_sha256 or {}),
        )

    @classmethod
    def from_folder(cls, path: str, preset: str, root: str, **load_kwargs
                    ) -> ContractSet:
        """Load every ``.csv`` / ``.txt`` file in ``path`` with ``preset`` (keyword
        arguments go to ``load_csv``) and build the set. Every contract must be ``root``.
        ``file_sha256`` records the sha256 of each contract's source file."""
        import hashlib
        from pathlib import Path

        from plutobt.importers import load_csv_contracts

        files = sorted(p for p in Path(path).iterdir()
                       if p.is_file() and p.suffix.lower() in (".csv", ".txt"))
        if not files:
            raise ValueError(f"{path}: no .csv or .txt files to load")
        frames: dict[str, object] = {}
        hashes: dict[str, str] = {}
        for p in files:
            digest = hashlib.sha256(p.read_bytes()).hexdigest()
            for code, df in load_csv_contracts(str(p), preset, **load_kwargs).items():
                if parse_contract(code).root != root:
                    raise ValueError(
                        f"{p.name} holds {code}, which is not root {root!r}; keep one "
                        "root per folder"
                    )
                if code in frames:
                    raise ValueError(f"{code} appears twice in {path} (again in {p.name})")
                frames[code] = df
                hashes[code] = digest
        return cls.from_frames(frames, file_sha256=hashes)
