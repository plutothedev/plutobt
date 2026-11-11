"""data.py — load JSON OHLCV, validate integrity, tag sessions, resample.

Spec §5.5 (timestamps/timezone/sessions) and §5.9 (data integrity).
Data is unix-seconds; bar ts marks the bar OPEN (verified empirically, §5.5).
"""

from __future__ import annotations

import hashlib
import json
import math
import numbers
from dataclasses import dataclass, field
from datetime import time
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from plutobt.instruments import Instrument

CT = ZoneInfo("America/Chicago")
_COLUMNS = ["ts", "open", "high", "low", "close", "volume"]

# A CME Globex trading day opens at 17:00 CT the evening before its date and ends at the
# 16:00 CT close: daily bars bucket by it, never by the UTC (or CT) calendar day.
_TRADING_DAY_OPEN_CT_HOURS = 17


def _whole(value) -> bool:
    """True for a finite whole number (1700000000 or 1700000000.0); not bool or None."""
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        return False
    return math.isfinite(value) and float(value).is_integer()


# Unix seconds stay below 1e11 until the year 5138; a larger ts is another unit.
_SECONDS_MAX = 1e11
_TS_UNITS = ((1e14, "milliseconds", "1,000"), (1e17, "microseconds", "1,000,000"),
             (2.0 ** 63, "nanoseconds", "1,000,000,000"))


def ts_unit_problem(value) -> str | None:
    """Why ``value`` (a whole number) is not a ts in unix seconds, or None if it can be:
    a ts of 1e11 or more is in another unit, named with the division that converts it
    (1772461800000 is milliseconds), and one past what int64 holds is out of range."""
    if -_SECONDS_MAX < value < _SECONDS_MAX:
        return None
    if value > 0:
        for limit, unit, factor in _TS_UNITS:
            if value < limit:
                return (f"ts {value!r} looks like {unit}, not unix seconds (seconds stay "
                        f"below 1e11 until the year 5138): divide the ts by {factor}")
    return (f"ts {value!r} is out of range for unix seconds (a whole number below 1e11 "
            "in absolute value)")


def _num(value) -> str:
    """A price for a message: 100.0, 18000.25."""
    return repr(float(value))


def missing_column_hint(df: pd.DataFrame, name: str) -> str:
    """What to add after "the frame has no '<name>' column": the columns it does have, and
    the fix when the cause is a capitalised name or a datetime index."""
    cols = [str(c) for c in df.columns]
    out = f" (it has: {', '.join(cols) if cols else 'no columns'}"
    twin = next((c for c in cols if c.lower() == name and c != name), None)
    if twin is not None:
        out += (f"; {twin!r} looks like {name!r}: column names are lower case, e.g. "
                "df.columns = df.columns.str.lower()")
    elif name == "ts" and isinstance(df.index, pd.DatetimeIndex):
        out += ("; its index holds datetimes: the engine reads a 'ts' column of whole "
                "unix seconds, not the index")
    return out + ")"


def ohlcv_problem(df: pd.DataFrame) -> str | None:
    """The first problem with a frame's base bar values, naming row and column; None if
    there is none. open, high, low and close must be present; they and volume (when
    present) must be numeric columns holding finite numbers (a NaN compares false
    against every stop and limit, so a NaN low silently disables a stop); every bar
    must hold its body: low <= min(open, close) and high >= max(open, close)."""
    ts = df["ts"].tolist() if "ts" in df.columns else None

    def where(i: int) -> str:
        return f"row {i}" if ts is None else f"row {i} (ts {ts[i]})"

    for col in _PRICE_COLS:
        if col not in df.columns:
            return f"the frame has no {col!r} column" + missing_column_hint(df, col)
    for col in (*_PRICE_COLS, "volume"):
        if col not in df.columns:
            continue
        s = df[col]
        if pd.api.types.is_bool_dtype(s.dtype) or not pd.api.types.is_numeric_dtype(s.dtype):
            sample = s.iloc[0] if len(s) else None
            return (f"column {col!r} holds {s.dtype} values (e.g. {sample!r}), not numbers; "
                    "convert it with pd.to_numeric first")
        v = s.to_numpy(dtype="float64", na_value=np.nan)
        bad = np.flatnonzero(~np.isfinite(v))
        if bad.size:
            i = int(bad[0])
            return (f"column {col!r} at {where(i)} holds {s.iloc[i]!r}, not a finite "
                    "number")
    o, h, lo, c = (df[col].to_numpy(dtype="float64") for col in _PRICE_COLS)
    top, bottom = np.maximum(o, c), np.minimum(o, c)
    for i in np.flatnonzero((h < top) | (lo > bottom)).tolist()[:1]:
        if h[i] < top[i]:
            return f"{where(i)}: high {_num(h[i])} is below max(open, close) {_num(top[i])}"
        return f"{where(i)}: low {_num(lo[i])} is above min(open, close) {_num(bottom[i])}"
    return None


def load_bars(path: str) -> pd.DataFrame:
    """Load JSON [ts, o, h, l, c, v] arrays into a canonical DataFrame.

    Columns: ts(int64), open/high/low/close(float64), volume(int64).
    ts is kept as a column (default RangeIndex), per the DATA API contract.
    ts and volume must be whole numbers (a float like 1700000000.0 is fine): a null,
    NaN, fractional, boolean or text value is refused with a ValueError naming the row,
    since casting it to int64 would invent a value (null became INT64_MIN, 0.9 of a
    second was dropped). ts must be unix seconds: one of 1e11 or more is refused,
    naming the unit it looks like (milliseconds, microseconds, nanoseconds) and the
    division that converts it, and one past int64 as out of range. Prices must be
    finite numbers (a null, NaN, inf, boolean or text price is refused, naming the
    row), and every bar must hold its body (low <= min(open, close), high >=
    max(open, close)), so a file that loads is one Engine.run can fill against.
    """
    with open(path, encoding="utf-8") as fh:
        try:
            rows = json.load(fh)
        except json.JSONDecodeError as err:
            raise ValueError(
                f"{str(path)!r} is not JSON ({err.msg}, line {err.lineno}): load_bars reads "
                "PlutoBT's JSON bars, an array of [ts, open, high, low, close, volume]; "
                "for a vendor CSV use load_csv") from None
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
        if not _whole(row[0]):
            raise ValueError(f"row {i}: ts must be a whole number of unix seconds, "
                             f"got {row[0]!r}")
        unit = ts_unit_problem(row[0])
        if unit is not None:
            raise ValueError(f"row {i}: {unit}")
        for col, value in zip(_PRICE_COLS, row[1:5], strict=True):
            if isinstance(value, bool) or not isinstance(value, numbers.Real):
                raise ValueError(f"row {i}: {col} must be a number, got {value!r} "
                                 f"({type(value).__name__})")
            if not math.isfinite(value):
                raise ValueError(f"row {i}: {col} must be a finite number, got {value!r}")
        if not _whole(row[5]):
            raise ValueError(f"row {i}: volume must be a whole number, got {row[5]!r}")
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
    problem = ohlcv_problem(df)
    if problem is not None:
        raise ValueError(f"{path!r}: {problem}")
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


def assert_positive_prices(df: pd.DataFrame) -> None:
    """Refuse %/log returns on a series with any non-positive price (§5.6).

    Additive (backwards-difference / Panama) continuous-contract adjustment can
    push historical prices to zero or negative, making percentage and log
    returns undefined or sign-flipped. Points/dollar P&L is still valid, but
    callers must not compute ratio returns on such a series. Raises ValueError
    naming the first offending (column, row); the row is the position (0 = first
    bar), whatever the frame's index holds.
    """
    for col in _PRICE_COLS:
        values = df[col].to_numpy()
        bad = np.flatnonzero(values <= 0)
        if bad.size > 0:
            row = int(bad[0])
            val = values[row].item() if hasattr(values[row], "item") else values[row]
            raise ValueError(
                f"non-positive price in column '{col}' at row {row} "
                f"(value={val!r}); percentage/log returns are undefined on "
                f"non-positive prices. Use points/dollar P&L instead."
            )


def resample(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Resample OHLCV bars to a coarser cadence (e.g. "15min").

    open=first, high=max, low=min, close=last, volume=sum. Buckets are
    label='left'/closed='left' so the resampled ts marks the bar OPEN,
    consistent with §5.5. Empty buckets (gaps) are dropped. ts returns to a
    plain int64 column with a RangeIndex.

    A rule of a day or longer ("1D", "W", "ME", "QS", ...) buckets by CME trading day
    in America/Chicago, not by the UTC calendar: the trading day ends at the 16:00 CT
    close and a bar opening at 17:00 CT or later belongs to the next one. Each bucket
    holds exactly its trading days (pandas' own closed/label side for the rule, so an
    end-anchored "ME" or "W" keeps its last day), and its ts is the 17:00 CT open of
    its first trading day: Sunday evening for a daily Monday bar or a weekly bar.
    Intraday rules bucket on the UTC clock, label='left'/closed='left', as before.

    Note: a 'session' label (if present) is NOT preserved; call tag_sessions
    on the result if session labels are needed at the coarser cadence.

    A continuous series (a ``contract`` column, from ``roll.build_continuous``) is
    refused: its roll columns cannot be aggregated, and dropping them would run the
    series as a plain frame with no rolls. Resample each contract, then build.
    """
    if "contract" in df.columns:
        raise ValueError(
            "resample refuses a continuous series (it has a 'contract' column): its "
            "contract, adj_offset and raw_close columns cannot be aggregated, and "
            "dropping them would lose every roll. Resample each contract's bars, then "
            "build the series with build_continuous")
    offset = pd.tseries.frequencies.to_offset(rule)
    daily = (not isinstance(offset, pd.offsets.Tick)
             or pd.Timedelta(offset) >= pd.Timedelta(days=1))
    dt = pd.to_datetime(df["ts"], unit="s", utc=True)
    shift = pd.Timedelta(hours=24 - _TRADING_DAY_OPEN_CT_HOURS)
    aggs = dict(open=("open", "first"), high=("high", "max"), low=("low", "min"),
                close=("close", "last"), volume=("volume", "sum"))
    if daily:
        # The trading date, as a naive midnight: the CT wall clock moved on by the hours
        # from the 17:00 open to midnight, so 17:00 CT lands on the next date. Dates
        # take pandas' default sides for the rule: an end-anchored rule ("ME", "W") is
        # closed on the right, so its last trading day stays in its own bucket.
        tdate = (dt.dt.tz_convert(CT).dt.tz_localize(None) + shift).dt.normalize()
        work = df.set_index(tdate).assign(first_day=tdate.to_numpy())
        agg = (work.resample(rule).agg(**aggs, first_day=("first_day", "first"))
               .dropna(subset=["open"]))
        # The bucket's open: 17:00 CT the evening before its first trading date.
        opened = (agg["first_day"] - shift).dt.tz_localize(CT).dt.tz_convert("UTC")
        out = agg.reset_index(drop=True).assign(ts=opened.to_numpy())
    else:
        work = df.set_index(dt)
        agg = (work.resample(rule, label="left", closed="left").agg(**aggs)
               .dropna(subset=["open"]))
        # The datetime column is named "ts" (inherits from the Series used in set_index).
        out = agg.reset_index()
    out["ts"] = out["ts"].astype("int64") // 1_000_000_000
    out["volume"] = out["volume"].astype("int64")
    return out[_COLUMNS].reset_index(drop=True)


def frame_sha256(df: pd.DataFrame) -> str:
    """sha256 hex digest of a frame's canonical CSV bytes (§5.10): ``to_csv`` with no
    index and LF line ends, so the same bars hash the same on every OS. This is the
    manifest's ``data_sha256`` and a continuous series' ``frame_sha256``."""
    # lineterminator pinned: pandas defaults to os.linesep, so the same bars hashed
    # differently on Windows and Linux.
    raw = df.to_csv(index=False, lineterminator=chr(10)).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


_BARS_NUMBERS = ("open", "high", "low", "close", "volume", "adj_offset", "raw_close")
_BARS_ORDER = ("ts", "open", "high", "low", "close", "volume", "session", "contract",
               "adj_offset", "raw_close")


def bars_sha256(df: pd.DataFrame) -> str:
    """sha256 of the bars by CONTENT: only the columns the engine reads, in one fixed
    order, numbers as float64 and ``ts`` as int64. The same bars with volume stored as
    floats, an extra column or the columns reordered hash the same, which
    ``frame_sha256`` (the exact frame, format included) does not. The study ledger keys
    its tries by this. A frame this cannot canonicalize (the engine will refuse it
    anyway) falls back to ``frame_sha256``."""
    try:
        cols = {}
        for name in _BARS_ORDER:
            if name not in df.columns:
                continue
            col = df[name]
            if name == "ts":
                col = col.astype("int64")
            elif name in _BARS_NUMBERS:
                col = col.astype("float64")
            cols[name] = col.to_numpy()
        if "ts" not in cols:
            return frame_sha256(df)
        return frame_sha256(pd.DataFrame(cols))
    except (TypeError, ValueError):
        return frame_sha256(df)
