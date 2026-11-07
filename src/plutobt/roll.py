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

from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta

import numpy as np
import pandas as pd

from plutobt.contracts import ContractSet, calendar_roll_day, parse_contract
from plutobt.data import CT

TRIGGERS = ("calendar", "volume", "dates")
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
