"""Continuous-contract roll: the schedule, the adjustment and its metadata (spec §6).

``roll_schedule`` decides when each contract hands over to the next, with one of three
triggers:

- ``calendar`` (equity-index roots): CME's customary roll day, the Monday before the
  third Friday of the expiry month (``contracts.calendar_roll_day``).
- ``volume``: the trading day after the first day on which the next contract's full-day
  volume beat the current one's. The decision uses day t's completed volume and acts on
  day t+1, so it never reads ahead.
- ``dates``: ``roll_dates={"CLZ5": "2025-11-14", ...}`` names each outgoing contract's
  roll day.

The roll happens at the first bar of the roll day at which BOTH contracts have a bar (the
same ts): the gap between them is measured there (spec §6.2).

A trading day is the CME trading day: it opens at 17:00 Chicago time the evening before
its date, so a bar at or after 17:00 CT belongs to the next day.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import asdict, dataclass, replace
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from plutobt.contracts import (
    ContractSet,
    calendar_roll_day,
    parse_contract,
    tick_decimals,
)
from plutobt.data import CT, frame_sha256
from plutobt.instruments import REGISTRY

TRIGGERS = ("calendar", "volume", "dates")
# The columns a continuous series' hash pins. Anything else (a signal column, the
# 'session' label tag_sessions adds) may be added before Engine.run.
PINNED_COLUMNS = ("ts", "open", "high", "low", "close", "volume", "contract",
                  "adj_offset", "raw_close")
_DAY_SHIFT = timedelta(hours=7)          # 17:00 CT + 7 h = 00:00 of the trading day


@dataclass(frozen=True)
class Roll:
    """One hand-over: ``from_code`` to ``to_code`` at bar ``roll_ts`` (unix seconds)."""

    from_code: str
    to_code: str
    roll_ts: int
    trigger: str


def trading_day(ts: int) -> date:
    """The CME trading day of a bar opening at unix second ``ts``."""
    local = datetime.fromtimestamp(int(ts), tz=UTC).astimezone(CT)
    return (local + _DAY_SHIFT).date()


def _days(ts: np.ndarray) -> np.ndarray:
    local = pd.to_datetime(ts, unit="s", utc=True).tz_convert(CT) + _DAY_SHIFT
    return np.asarray(local.date)


def _first_shared_ts(a: pd.DataFrame, b: pd.DataFrame, day: date) -> int | None:
    ta = a["ts"].to_numpy()
    tb = b["ts"].to_numpy()
    shared = np.intersect1d(ta[_days(ta) == day], tb[_days(tb) == day])
    return int(shared[0]) if shared.size else None


def _volume_roll_ts(a: pd.DataFrame, b: pd.DataFrame) -> int | None:
    """First shared bar of the trading day after the first day on which ``b``'s full-day
    volume exceeded ``a``'s; None if that has not happened (or that day has no shared
    bar yet). Only days both contracts traded are compared."""
    va = a.groupby(_days(a["ts"].to_numpy()))["volume"].sum()
    vb = b.groupby(_days(b["ts"].to_numpy()))["volume"].sum()
    shared = sorted(set(va.index) & set(vb.index))
    for i, day in enumerate(shared):
        if vb[day] > va[day]:
            if i + 1 == len(shared):
                return None
            return _first_shared_ts(a, b, shared[i + 1])
    return None


def _canonical_dates(roll_dates: dict | None, cs: ContractSet) -> dict[str, date]:
    out: dict[str, date] = {}
    for key, value in (roll_dates or {}).items():
        near = cs.contracts[0]
        code = parse_contract(key, near=date(near.year, near.month, 1)).code
        out[code] = value if isinstance(value, date) else date.fromisoformat(str(value))
    return out


def roll_schedule(cs: ContractSet, trigger: str = "calendar",
                  roll_dates: dict[str, str] | None = None) -> list[Roll]:
    """The roll between each pair of consecutive contracts in ``cs`` (spec §6.1).

    Refuses, naming the problem: an unknown trigger; the calendar trigger on a root
    without an expiry calendar; a ``dates`` trigger missing an outgoing contract; a
    volume trigger that never crosses; a roll day on which the two contracts share no
    bar; rolls out of order."""
    if trigger not in TRIGGERS:
        raise ValueError(f"unknown roll trigger {trigger!r}; use one of "
                         f"{', '.join(TRIGGERS)}")
    dates = _canonical_dates(roll_dates, cs) if trigger == "dates" else {}
    rolls: list[Roll] = []
    for a, b in zip(cs.contracts, cs.contracts[1:], strict=False):
        fa, fb = cs.frames[a.code], cs.frames[b.code]
        if trigger == "volume":
            ts = _volume_roll_ts(fa, fb)
            if ts is None:
                raise ValueError(
                    f"{b.code}'s daily volume never beat {a.code}'s on a day with a next "
                    f"shared trading day, so the volume trigger never rolls {a.code}; "
                    "use roll_dates for this pair"
                )
        else:
            if trigger == "calendar":
                day = calendar_roll_day(a)
            else:
                if a.code not in dates:
                    raise ValueError(f"roll_dates has no roll day for {a.code} (the "
                                     f"roll to {b.code})")
                day = dates[a.code]
            ts = _first_shared_ts(fa, fb, day)
            if ts is None:
                raise ValueError(
                    f"{a.code} and {b.code} share no bar on the roll day {day.isoformat()}"
                    ", so the gap between them cannot be measured at one moment; supply "
                    "data for both on that day, or roll on another day with roll_dates"
                )
        if rolls and ts <= rolls[-1].roll_ts:
            raise ValueError(f"the roll {a.code} -> {b.code} at ts {ts} is not after the "
                             f"previous roll at ts {rolls[-1].roll_ts}")
        rolls.append(Roll(a.code, b.code, int(ts), trigger))
    return rolls


# ---------------------------------------------------------------------------
# Adjustment (spec §6.3) and metadata (spec §6.4)
# ---------------------------------------------------------------------------

@dataclass
class RollMeta:
    """How a continuous series was built. Saved as ``<name>.roll.json`` beside it.

    ``rolls`` holds one ``{from, to, roll_ts, gap_ticks, gap_points, gap_dollars}`` per
    roll; ``inputs`` the sha256 of each contract's source file; ``frame_sha256`` the
    canonical hash of the series itself (``data.frame_sha256``), which ``Engine.run``
    checks; ``window`` the ``[first_ts, last_ts]`` of a slice (None for a full build)."""

    root: str
    roll_trigger: str
    splice_method: str
    adjustment_method: str
    as_of_date: str
    rolls: list[dict]
    inputs: dict[str, str]
    frame_sha256: str
    min_adjusted_price: float
    plutobt_version: str
    window: list[int] | None = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> RollMeta:
        return cls(**json.loads(text))



def _adjust(cs: ContractSet, rolls: list[Roll]) -> tuple[pd.DataFrame, list[dict], float]:
    """Splice ``cs`` at ``rolls`` and back-adjust additively, in whole ticks. The newest
    contract keeps its raw floats exactly; an older one is rebuilt from integer ticks."""
    inst = REGISTRY[cs.root]
    tick, dec = inst.tick_size, tick_decimals(inst.tick_size)
    codes = [c.code for c in cs.contracts]
    gaps: list[int] = []
    rows: list[dict] = []
    for r in rolls:
        a, b = cs.frames[r.from_code], cs.frames[r.to_code]
        a_open = float(a.loc[a["ts"] == r.roll_ts, "open"].iloc[0])
        b_open = float(b.loc[b["ts"] == r.roll_ts, "open"].iloc[0])
        gap = int(np.rint(b_open / tick)) - int(np.rint(a_open / tick))
        gaps.append(gap)
        rows.append({"from": r.from_code, "to": r.to_code, "roll_ts": int(r.roll_ts),
                     "gap_ticks": gap, "gap_points": round(gap * tick, dec),
                     "gap_dollars": round(gap * inst.tick_value, 6)})
    parts = []
    for i, code in enumerate(codes):
        f = cs.frames[code]
        ts = f["ts"].to_numpy()
        keep = np.ones(len(f), dtype=bool)
        if i > 0:
            keep &= ts >= rolls[i - 1].roll_ts
        if i < len(rolls):
            keep &= ts < rolls[i].roll_ts
        part = f.loc[keep, ["ts", "open", "high", "low", "close", "volume"]].copy()
        offset = sum(gaps[i:])
        raw_close = part["close"].to_numpy(dtype="float64").copy()
        if offset:
            for col in ("open", "high", "low", "close"):
                ticks = np.rint(part[col].to_numpy(dtype="float64") / tick).astype(np.int64)
                part[col] = np.round((ticks + offset) * tick, dec)
        part["contract"] = code
        part["adj_offset"] = round(offset * tick, dec) if offset else 0.0
        part["raw_close"] = raw_close
        parts.append(part)
    df = pd.concat(parts, ignore_index=True)
    df["ts"] = df["ts"].astype("int64")
    df["volume"] = df["volume"].astype("int64")
    df["adj_offset"] = df["adj_offset"].astype("float64")
    min_price = float(min(df["low"].min(), df["open"].min(), df["close"].min()))
    if min_price <= 0:
        for col in ("low", "open", "close", "high"):
            bad = np.flatnonzero(df[col].to_numpy() <= 0)
            if bad.size:
                i = int(bad[0])
                warnings.warn(
                    f"back-adjustment pushed the history to or below zero: {col} "
                    f"{float(df[col].iloc[i])!r} at ts {int(df['ts'].iloc[i])} "
                    f"({df['contract'].iloc[i]}). Point and dollar P&L stay exact; "
                    "Engine.run refuses this frame unless allow_nonpositive_prices = true",
                    UserWarning, stacklevel=3)
                break
    return df, rows, min_price


def build_continuous(cs: ContractSet, trigger: str = "calendar",
                     roll_dates: dict[str, str] | None = None
                     ) -> tuple[pd.DataFrame, RollMeta]:
    """Build a back-adjusted continuous series from ``cs`` (spec §6).

    Columns: ``ts, open, high, low, close, volume`` (adjusted; the active contract's
    volume), ``contract``, ``adj_offset`` (adjusted minus raw, constant within a
    contract) and ``raw_close``. The newest contract keeps its raw prices; each older
    one is shifted by the sum of the later gaps, each gap measured at the roll bar's
    open from both contracts. Warns if the history goes to or below zero."""
    from plutobt import __version__

    rolls = roll_schedule(cs, trigger, roll_dates)
    df, rows, min_price = _adjust(cs, rolls)
    meta = RollMeta(
        root=cs.root, roll_trigger=trigger, splice_method="backward",
        adjustment_method="additive",
        as_of_date=trading_day(int(df["ts"].iloc[-1])).isoformat(),
        rolls=rows, inputs=dict(cs.file_sha256), frame_sha256=series_sha256(df),
        min_adjusted_price=min_price, plutobt_version=__version__,
    )
    return df, meta


def series_sha256(df: pd.DataFrame) -> str:
    """The pin of a continuous series: ``frame_sha256`` of its ``PINNED_COLUMNS`` only,
    so columns added later (signals, 'session') leave it unchanged."""
    missing = [c for c in PINNED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"not a continuous series: missing column(s) {missing}")
    return frame_sha256(df[list(PINNED_COLUMNS)])


def check_pin(df: pd.DataFrame, meta: RollMeta, what: str) -> None:
    """Refuse a series whose pinned columns no longer hash to ``meta.frame_sha256``."""
    actual = series_sha256(df)
    if actual != meta.frame_sha256:
        raise ValueError(
            f"{what}: its bars hash to sha256 {actual}, but its metadata pins "
            f"{meta.frame_sha256}; the series was edited or does not belong to this "
            "metadata. Rebuild it with build_continuous")


def _meta_path(path: str) -> str:
    p = Path(path)
    return str(p.with_name(p.stem + ".roll.json"))


def save_continuous(df: pd.DataFrame, meta: RollMeta, path: str) -> tuple[str, str]:
    """Write the series to ``path`` (CSV) and its metadata to ``<stem>.roll.json``."""
    df.to_csv(path, index=False, lineterminator=chr(10))
    json_path = _meta_path(path)
    Path(json_path).write_text(meta.to_json(), encoding="utf-8")
    return str(path), json_path


def load_continuous(path: str) -> tuple[pd.DataFrame, RollMeta]:
    """Read a series written by ``save_continuous``. Refuses a series whose hash no
    longer matches its metadata (edited, or saved by something else)."""
    df = pd.read_csv(path, dtype={"contract": str}, float_precision="round_trip")
    df["ts"] = df["ts"].astype("int64")
    df["volume"] = df["volume"].astype("int64")
    for col in ("open", "high", "low", "close", "adj_offset", "raw_close"):
        df[col] = df[col].astype("float64")
    meta = RollMeta.from_json(Path(_meta_path(path)).read_text(encoding="utf-8"))
    check_pin(df, meta, str(path))
    return df, meta


def slice_continuous(df: pd.DataFrame, meta: RollMeta, start_ts: int | None = None,
                     end_ts: int | None = None) -> tuple[pd.DataFrame, RollMeta]:
    """The bars with ``start_ts <= ts <= end_ts`` and metadata re-pinned to them: only
    the rolls inside the slice (a roll on its first bar is outside, since no contract
    change is visible there), a fresh ``frame_sha256``, and ``window``. The series must
    still match ``meta`` first: slicing never launders an edited series."""
    check_pin(df, meta, "slice_continuous")
    ts = df["ts"].to_numpy()
    keep = np.ones(len(df), dtype=bool)
    if start_ts is not None:
        keep &= ts >= start_ts
    if end_ts is not None:
        keep &= ts <= end_ts
    out = df.loc[keep].reset_index(drop=True)
    if out.empty:
        raise ValueError(f"no bars between {start_ts} and {end_ts}")
    first, last = int(out["ts"].iloc[0]), int(out["ts"].iloc[-1])
    rows = [r for r in meta.rolls if first < r["roll_ts"] <= last]
    min_price = float(min(out["low"].min(), out["open"].min(), out["close"].min()))
    return out, replace(meta, rolls=rows, frame_sha256=series_sha256(out),
                        min_adjusted_price=min_price, window=[first, last])
