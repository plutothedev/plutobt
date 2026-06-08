"""data.py — load JSON OHLCV, validate integrity, tag sessions, resample.

Spec §5.5 (timestamps/timezone/sessions) and §5.9 (data integrity).
Data is unix-seconds; bar ts marks the bar OPEN (verified empirically, §5.5).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import time
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from plutobt.instruments import Instrument

CT = ZoneInfo("America/Chicago")
_COLUMNS = ["ts", "open", "high", "low", "close", "volume"]


def load_bars(path: str) -> pd.DataFrame:
    """Load JSON [ts, o, h, l, c, v] arrays into a canonical DataFrame.

    Columns: ts(int64), open/high/low/close(float64), volume(int64).
    ts is kept as a column (default RangeIndex), per the DATA API contract.
    """
    with open(path, encoding="utf-8") as fh:
        rows = json.load(fh)
    if not isinstance(rows, list):
        raise ValueError(
            f"expected a JSON array of bars in {path!r}, got {type(rows).__name__}"
        )
    if len(rows) == 0:
        raise ValueError(f"no bars in {path!r}")
    for i, row in enumerate(rows):
        if not isinstance(row, list | tuple) or len(row) != 6:
            got = len(row) if isinstance(row, list | tuple) else type(row).__name__
            raise ValueError(
                f"row {i}: expected 6 fields [ts,o,h,l,c,v], got {got}"
            )
    arr = np.asarray(rows, dtype="float64")
    df = pd.DataFrame(
        {
            "ts": arr[:, 0].astype("int64"),
            "open": arr[:, 1],
            "high": arr[:, 2],
            "low": arr[:, 3],
            "close": arr[:, 4],
            "volume": arr[:, 5].astype("int64"),
        },
        columns=_COLUMNS,
    )
    return df.reset_index(drop=True)


@dataclass
class ValidationReport:
    """Result of validate_bars (§5.9).

    ok: True iff no errors. errors: human-readable messages. bad_rows:
    integer positional indices of offending bars. gaps: list of
    (row_index, gap_seconds) where the inter-bar delta exceeds the modal
    interval (populated in M3.3). n_bars / max_gap / modal_interval summarize.
    """

    ok: bool
    n_bars: int
    errors: list[str] = field(default_factory=list)
    bad_rows: list[int] = field(default_factory=list)
    gaps: list[tuple[int, int]] = field(default_factory=list)
    max_gap: int = 0
    modal_interval: int | None = None


def validate_bars(df: pd.DataFrame, strict: bool = False) -> ValidationReport:
    """Validate OHLCV integrity (§5.9).

    Checks: no NaN; high >= max(open, close); low <= min(open, close);
    high >= low; volume >= 0. Monotonic/unique ts + gap report included.
    Returns a ValidationReport. If strict=True and any error exists, raises
    ValueError after building the report.
    """
    errors: list[str] = []
    bad_rows: set[int] = set()
    n = len(df)

    o, h, low, c = df["open"], df["high"], df["low"], df["close"]
    vol = df["volume"]

    nan_mask = df[_COLUMNS].isna().any(axis=1)
    for i in df.index[nan_mask]:
        errors.append(f"row {i}: NaN in OHLCV")
        bad_rows.add(int(i))

    body_max = np.maximum(o, c)
    body_min = np.minimum(o, c)
    for i in df.index[h < body_max]:
        errors.append(f"row {i}: high < max(open, close)")
        bad_rows.add(int(i))
    for i in df.index[low > body_min]:
        errors.append(f"row {i}: low > min(open, close)")
        bad_rows.add(int(i))
    for i in df.index[h < low]:
        errors.append(f"row {i}: high < low")
        bad_rows.add(int(i))
    for i in df.index[vol < 0]:
        errors.append(f"row {i}: volume < 0")
        bad_rows.add(int(i))

    # --- timestamp monotonicity + uniqueness (§5.9) + gap report ---
    ts = df["ts"].to_numpy()
    gaps: list[tuple[int, int]] = []
    modal_interval: int | None = None
    max_gap = 0
    if n >= 2:
        deltas = np.diff(ts)
        for j, d in enumerate(deltas):
            pos = j + 1  # positional index of the second bar in the pair
            if d == 0:
                errors.append(f"row {pos}: duplicate ts {int(ts[pos])}")
                bad_rows.add(pos)
            elif d < 0:
                errors.append(
                    f"row {pos}: non-monotonic ts ({int(ts[pos])} < {int(ts[j])})"
                )
                bad_rows.add(pos)
        positive = deltas[deltas > 0]
        if positive.size:
            vals, counts = np.unique(positive, return_counts=True)
            modal_interval = int(vals[np.argmax(counts)])
            for j, d in enumerate(deltas):
                if d > modal_interval:
                    gap = int(d)
                    gaps.append((j + 1, gap))
                    if gap > max_gap:
                        max_gap = gap

    rep = ValidationReport(
        ok=len(errors) == 0,
        n_bars=n,
        errors=errors,
        bad_rows=sorted(bad_rows),
        gaps=gaps,
        max_gap=max_gap,
        modal_interval=modal_interval,
    )
    if strict and not rep.ok:
        raise ValueError(f"validate_bars: {len(errors)} error(s): {errors}")
    return rep


def _parse_hhmm(s: str) -> time:
    hh, mm = s.split(":")
    return time(int(hh), int(mm))


def verify_ts_is_open(df: pd.DataFrame, instrument: Instrument) -> bool:
    """Empirically verify ts marks the bar OPEN, not close (§5.5).

    Groups bars by America/Chicago calendar date and checks that the modal
    first-bar-of-day local time equals the instrument's session_open_ct.
    If ts marked the bar CLOSE, the first bar of an RTH day would land one
    interval past the open and this returns False.
    """
    expected_open = _parse_hhmm(instrument.session_open_ct)
    idx = pd.DatetimeIndex(
        pd.to_datetime(df["ts"], unit="s", utc=True)
    ).tz_convert(CT)
    local_dates = idx.date
    local_times = idx.time
    first_times: list[time] = []
    seen: set = set()
    for d, t in zip(local_dates, local_times, strict=False):
        if d not in seen:
            seen.add(d)
            first_times.append(t)
    if not first_times:
        return False
    matches = sum(1 for t in first_times if t == expected_open)
    return matches > len(first_times) / 2


def tag_sessions(df: pd.DataFrame, instrument: Instrument) -> pd.DataFrame:
    """Add a 'session' column ("RTH"/"ETH") in America/Chicago (§5.5).

    A bar is RTH iff its OPEN local time lies in the half-open interval
    [session_open_ct, session_close_ct). The window is per-instrument from
    the registry. The interval is half-open so a bar opening exactly at the
    close (e.g. 15:15 for the index session) is ETH, and the last RTH 5m
    bar opens 15:10 / closes 15:15 (§10.10). Returns a copy; does not mutate.
    """
    open_t = _parse_hhmm(instrument.session_open_ct)
    close_t = _parse_hhmm(instrument.session_close_ct)
    idx = pd.DatetimeIndex(
        pd.to_datetime(df["ts"], unit="s", utc=True)
    ).tz_convert(CT)
    local_times = idx.time
    is_rth = np.array([open_t <= t < close_t for t in local_times])
    out = df.copy()
    out["session"] = np.where(is_rth, "RTH", "ETH")
    return out


_PRICE_COLS = ("open", "high", "low", "close")


def assert_positive_prices(df: "pd.DataFrame") -> None:
    """Refuse %/log returns on a series with any non-positive price (§5.6).

    Additive (backwards-difference / Panama) continuous-contract adjustment can
    push historical prices to zero or negative, making percentage and log
    returns undefined or sign-flipped. Points/dollar P&L is still valid, but
    callers must not compute ratio returns on such a series. Raises ValueError
    naming the first offending (column, row).
    """
    for col in _PRICE_COLS:
        bad = df.index[df[col] <= 0]
        if len(bad) > 0:
            row = int(bad[0])
            val = df[col].iloc[df.index.get_loc(bad[0])]
            raise ValueError(
                f"non-positive price in column '{col}' at row {row} "
                f"(value={val!r}); percentage/log returns are undefined on "
                f"non-positive prices (§5.6). Use points/dollar P&L instead."
            )


def resample(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Resample OHLCV bars to a coarser cadence (e.g. "15min").

    open=first, high=max, low=min, close=last, volume=sum. Buckets are
    label='left'/closed='left' so the resampled ts marks the bar OPEN,
    consistent with §5.5. Empty buckets (gaps) are dropped. ts returns to a
    plain int64 column with a RangeIndex.

    Note: a 'session' label (if present) is NOT preserved; call tag_sessions
    on the result if session labels are needed at the coarser cadence.
    """
    dt = pd.to_datetime(df["ts"], unit="s", utc=True)
    work = df.set_index(dt)
    agg = (
        work.resample(rule, label="left", closed="left")
        .agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
        )
        .dropna(subset=["open"])
    )
    out = agg.reset_index()
    # The datetime column is named "ts" (inherits from the Series used in set_index).
    out["ts"] = out["ts"].astype("int64") // 1_000_000_000
    out["volume"] = out["volume"].astype("int64")
    return out[_COLUMNS].reset_index(drop=True)
