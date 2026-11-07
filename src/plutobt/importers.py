"""Read vendor bar files into PlutoBT's canonical frame (spec §4).

``load_csv(path, preset=...)`` returns the same columns ``load_bars`` does (``ts`` as
whole unix seconds of the bar OPEN, float ``open/high/low/close``, int ``volume``), plus
a ``contract`` column (canonical code, e.g. ``NQH2026``) when the contract is known from
``symbol=``, the file's symbol column, or its file name. Every file runs the same bar
checks ``load_bars`` does, so a file that loads is one the engine can fill against.

Presets (formats from the vendors' own docs, 2026-11-06 fact-check):

- ``databento``: Databento OHLCV CSV, columns read by name (``ts_event``, ``open`` ..
  ``volume``, ``symbol``). ``ts_event`` is the bar OPEN, as nanoseconds or ISO text in
  UTC; prices are decimals (``pretty_px``) or fixed-point integers scaled by 1e-9. A
  file holding several contracts goes through ``load_csv_contracts``.
- ``tradingview``: "Export chart data" CSV, columns read by name (``time``, ``open`` ..
  ``close``, ``volume`` any case); extra indicator columns are dropped. ``time`` is unix
  seconds or milliseconds, or ISO text with a zone. TradingView does not document whether
  ``time`` is the bar open; it is read as the open (pass ``ts_is="close"`` to override).
  No volume column gives volume 0.
- ``ninjatrader``: NinjaTrader 8 historical data export, ``yyyyMMdd HHmmss;o;h;l;c;v``
  with no header. NinjaTrader exports in UTC with END-of-bar stamps, so each stamp is
  moved back one bar interval to the open.
- ``generic``: any CSV. ``columns`` maps ``ts, open, high, low, close, volume`` to the
  file's names (default: the same names), and ``ts_is`` ("open" or "close") is required.

Time rules, refused rather than guessed: numeric timestamps are unix time (seconds,
milliseconds, microseconds or nanoseconds, told apart by size) and need no zone; text
timestamps without a zone need ``tz`` (an IANA name such as "America/Chicago"); text
timestamps are ISO 8601 unless ``ts_format`` gives the exact strftime format; a bar
stamped at its close is moved back by the file's bar interval, which must be clear (one
interval covering at least half the gaps).
"""

from __future__ import annotations

import re
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from plutobt.contracts import parse_contract
from plutobt.data import ohlcv_problem

PRESETS: tuple[str, ...] = ("databento", "tradingview", "ninjatrader", "generic")
_BAR = ("open", "high", "low", "close", "volume")
_ZONED = re.compile(r"(?:Z|[+-]\d{2}:?\d{2})$")
_PX_UNITS = 1_000_000_000   # Databento fixed-point prices: 1 unit = 1e-9


# ---------------------------------------------------------------------------
# Timestamps
# ---------------------------------------------------------------------------

def _epoch_seconds(values: np.ndarray, name: str) -> np.ndarray:
    v = values.astype("float64")
    big = np.nanmax(np.abs(v)) if len(v) else 0.0
    for limit, div, unit in ((1e17, 1e9, "nanoseconds"), (1e14, 1e6, "microseconds"),
                             (1e11, 1e3, "milliseconds")):
        if big >= limit:
            ints = values.astype("int64")
            if (ints % int(div)).any():
                i = int(np.flatnonzero(ints % int(div))[0])
                raise ValueError(f"{name}: row {i} timestamp {int(ints[i])} looks like "
                                 f"{unit} but is not a whole second")
            return ints // int(div)
    if not np.all(v == np.rint(v)):
        i = int(np.flatnonzero(v != np.rint(v))[0])
        raise ValueError(f"{name}: row {i} timestamp {values[i]!r} is not whole seconds")
    return v.astype("int64")


def _parse_text(text: pd.Series, name: str, ts_format: str | None, utc: bool):
    """Text timestamps: ISO 8601 by default, or exactly ``ts_format``. Never guessed
    per row: format="mixed" read day-first 02/03..06/03 as Feb 3 .. Jun 3."""
    try:
        if ts_format is not None:
            return pd.to_datetime(text, format=ts_format, utc=utc)
        return pd.to_datetime(text, format="ISO8601", utc=utc)
    except (ValueError, TypeError) as bad:
        how = (f"ts_format={ts_format!r}" if ts_format is not None
               else "ISO 8601 (e.g. 2026-03-05 08:30:00)")
        raise ValueError(
            f"{name}: timestamps such as {text.iloc[0]!r} do not parse as {how}. Pass "
            "ts_format='%d/%m/%Y %H:%M' (or the file's exact strftime format): PlutoBT "
            "never guesses whether 02/03 is February or March") from bad


def _to_unix_seconds(col: pd.Series, tz: str | None, name: str,
                     ts_format: str | None = None) -> np.ndarray:
    if pd.api.types.is_numeric_dtype(col.dtype):
        if col.isna().any():
            raise ValueError(f"{name}: row {int(np.flatnonzero(col.isna())[0])} has no "
                             "timestamp")
        return _epoch_seconds(col.to_numpy(), name)
    text = col.astype(str).str.strip()
    zoned = text.str.contains(_ZONED)
    if zoned.all():
        stamps = _parse_text(text, name, ts_format, utc=True)
    elif not zoned.any():
        if tz is None:
            raise ValueError(
                f"{name}: its timestamps carry no time zone (e.g. {text.iloc[0]!r}); pass "
                "tz='America/Chicago' (or whatever zone the file was written in). "
                "PlutoBT never guesses a zone: a wrong one shifts every bar"
            )
        naive = _parse_text(text, name, ts_format, utc=False)
        stamps = naive.dt.tz_localize(tz, ambiguous="raise", nonexistent="raise")
    else:
        i = int(np.flatnonzero(~zoned.to_numpy())[0])
        raise ValueError(f"{name}: some timestamps carry a zone and some do not (row {i}: "
                         f"{text.iloc[i]!r})")
    return (stamps.astype("int64") // 10**9).to_numpy()


def _shift_close_to_open(ts: np.ndarray, name: str) -> np.ndarray:
    if len(ts) < 2:
        raise ValueError(f"{name}: one bar is not enough to find the bar interval needed "
                         "to move close stamps to opens")
    diffs = pd.Series(ts[1:] - ts[:-1])
    counts = diffs.value_counts()
    interval, n = int(counts.index[0]), int(counts.iloc[0])
    if n * 2 < len(diffs) or interval <= 0:
        raise ValueError(
            f"{name}: no clear bar interval (most common gap {interval}s covers {n} of "
            f"{len(diffs)} gaps), so close stamps cannot be moved to opens safely"
        )
    return ts - interval


# ---------------------------------------------------------------------------
# Frames
# ---------------------------------------------------------------------------

def _finish(ts: np.ndarray, data: pd.DataFrame, name: str, contract: str | None
            ) -> pd.DataFrame:
    df = pd.DataFrame({"ts": pd.Series(ts, dtype="int64")})
    for col in ("open", "high", "low", "close"):
        df[col] = pd.to_numeric(data[col], errors="coerce").to_numpy(dtype="float64")
    vol = pd.to_numeric(data["volume"], errors="coerce").to_numpy(dtype="float64")
    bad = np.flatnonzero(~np.isfinite(vol) | (vol != np.rint(vol)))
    if bad.size:
        i = int(bad[0])
        raise ValueError(f"{name}: volume at row {i} is {data['volume'].iloc[i]!r}, not a "
                         "whole number")
    df["volume"] = vol.astype("int64")
    problem = ohlcv_problem(df)
    if problem is not None:
        raise ValueError(f"{name}: {problem}")
    t = df["ts"].to_numpy()
    back = np.flatnonzero(t[1:] <= t[:-1])
    if back.size:
        i = int(back[0]) + 1
        raise ValueError(f"{name}: bar timestamps must strictly increase (row {i}, ts "
                         f"{int(t[i])} is not after the bar before it)")
    if contract is not None:
        df["contract"] = contract
    return df


def _near(ts: np.ndarray) -> date | None:
    return date(1970, 1, 1) + timedelta(seconds=int(ts[0])) if len(ts) else None


def _code(text: str, near: date | None) -> str | None:
    try:
        return parse_contract(text, near=near).code
    except ValueError:
        return None


def _contract_from_name(path: str, near: date | None) -> str | None:
    stem = Path(path).name
    stem = re.sub(r"\.(csv|txt)$", "", stem, flags=re.IGNORECASE)
    stem = re.sub(r"\.(last|bid|ask)$", "", stem, flags=re.IGNORECASE)
    found = _code(stem, near)
    if found:
        return found
    for token in re.split(r"[^A-Za-z0-9]+", stem):
        found = _code(token, near) if token else None
        if found:
            return found
    return None


def _contract(symbol: str | None, path: str, ts: np.ndarray) -> str | None:
    near = _near(ts)
    if symbol is not None:
        return parse_contract(symbol, near=near).code
    return _contract_from_name(path, near)


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------

def _databento_prices(raw: pd.DataFrame, name: str) -> pd.DataFrame:
    kinds = {}
    for col in ("open", "high", "low", "close"):
        s = raw[col]
        if pd.api.types.is_integer_dtype(s.dtype):
            lo, hi = int(s.abs().min()), int(s.abs().max())
            kinds[col] = "fixed" if lo >= 10**8 else ("decimal" if hi < 10**8 else None)
        else:
            kinds[col] = "decimal"
        if kinds[col] is None:
            raise ValueError(f"{name}: column {col!r} mixes small and huge integers, so "
                             "it cannot tell decimal prices from fixed-point (1e-9) ones")
    if len(set(kinds.values())) > 1:
        listing = ", ".join(f"{c} looks {k}" for c, k in kinds.items())
        raise ValueError(f"{name}: the price columns disagree on their scale ({listing})")
    out = raw.copy()
    if kinds["open"] == "fixed":
        for col in ("open", "high", "low", "close"):
            # Divide, never multiply by 1e-9: an integer over 1e9 is correctly
            # rounded to the float of its decimal; x * 1e-9 is not (15000.00 came
            # out 15000.000000000002, and a stop resting at that low never fired).
            out[col] = raw[col].to_numpy(dtype="int64") / _PX_UNITS
    return out


def _read_databento(path: str) -> dict[str | None, pd.DataFrame]:
    name = Path(path).name
    raw = pd.read_csv(path, float_precision="round_trip")
    missing = [c for c in ("ts_event", *_BAR) if c not in raw.columns]
    if missing:
        raise ValueError(f"{name}: not a Databento OHLCV CSV, missing column(s) {missing}")
    raw = _databento_prices(raw, name)
    groups = raw.groupby("symbol", sort=False) if "symbol" in raw.columns else [(None, raw)]
    out: dict[str | None, pd.DataFrame] = {}
    for sym, part in groups:
        part = part.reset_index(drop=True)
        ts = _to_unix_seconds(part["ts_event"], None, name)
        code = parse_contract(str(sym), near=_near(ts)).code if sym is not None else None
        out[code] = _finish(ts, part, name, code)
    return out


def _read_tradingview(path: str, ts_is: str | None, tz: str | None,
                      ts_format: str | None = None) -> tuple:
    name = Path(path).name
    raw = pd.read_csv(path, float_precision="round_trip")
    raw.columns = [str(c).strip().lower() for c in raw.columns]
    missing = [c for c in ("time", "open", "high", "low", "close") if c not in raw.columns]
    if missing:
        raise ValueError(f"{name}: not a TradingView chart export, missing column(s) "
                         f"{missing}")
    if "volume" not in raw.columns:
        raw["volume"] = 0
    ts = _to_unix_seconds(raw["time"], tz, name, ts_format)
    if (ts_is or "open") == "close":
        ts = _shift_close_to_open(ts, name)
    return ts, raw


def _read_ninjatrader(path: str, ts_is: str | None) -> tuple:
    name = Path(path).name
    raw = pd.read_csv(path, sep=";", header=None, dtype={0: str},
                      names=["stamp", *_BAR], float_precision="round_trip")
    stamp = raw["stamp"].astype(str).str.strip()
    if not stamp.str.fullmatch(r"\d{8} \d{6}").all():
        i = int(np.flatnonzero(~stamp.str.fullmatch(r"\d{8} \d{6}").to_numpy())[0])
        raise ValueError(f"{name}: row {i} stamp {stamp.iloc[i]!r} is not the minute-bar "
                         "form 'yyyyMMdd HHmmss' (daily and tick exports are not read)")
    when = pd.to_datetime(stamp, format="%Y%m%d %H%M%S").dt.tz_localize("UTC")
    ts = (when.astype("int64") // 10**9).to_numpy()
    if (ts_is or "close") == "close":
        ts = _shift_close_to_open(ts, name)
    return ts, raw


def _read_generic(path: str, columns: dict | None, ts_is: str | None, tz: str | None,
                  ts_format: str | None = None) -> tuple:
    name = Path(path).name
    if ts_is not in ("open", "close"):
        raise ValueError(
            f"{name}: pass ts_is='open' or ts_is='close' to say whether each timestamp is "
            "the bar's open or its close. Reading a close stamp as an open shifts every "
            "bar one interval early, which is look-ahead"
        )
    mapping = {c: c for c in ("ts", *_BAR)}
    mapping.update(columns or {})
    raw = pd.read_csv(path, float_precision="round_trip")
    missing = [f"{k} (column {v!r})" for k, v in mapping.items() if v not in raw.columns]
    if missing:
        raise ValueError(f"{name}: missing {missing}; map them with columns={{...}}")
    data = pd.DataFrame({k: raw[v] for k, v in mapping.items()})
    ts = _to_unix_seconds(data["ts"], tz, name, ts_format)
    if ts_is == "close":
        ts = _shift_close_to_open(ts, name)
    return ts, data


def _check_preset(preset: str) -> None:
    if preset not in PRESETS:
        raise ValueError(f"unknown preset {preset!r}; use one of {', '.join(PRESETS)}")


def load_csv(path: str, preset: str = "generic", *, tz: str | None = None,
             ts_is: str | None = None, columns: dict[str, str] | None = None,
             symbol: str | None = None, ts_format: str | None = None) -> pd.DataFrame:
    """Load one contract's bars from a vendor file (see the module docstring)."""
    _check_preset(preset)
    path = str(path)
    if preset == "databento":
        frames = _read_databento(path)
        if len(frames) > 1:
            raise ValueError(
                f"{Path(path).name} holds {len(frames)} contracts "
                f"({', '.join(str(k) for k in frames)}); use load_csv_contracts"
            )
        (code, df), = frames.items()
        if symbol is not None:
            df = df.drop(columns="contract", errors="ignore")
            ts = df["ts"].to_numpy()
            df["contract"] = parse_contract(symbol, near=_near(ts)).code
        return df
    if preset == "tradingview":
        ts, data = _read_tradingview(path, ts_is, tz, ts_format)
    elif preset == "ninjatrader":
        ts, data = _read_ninjatrader(path, ts_is)
    else:
        ts, data = _read_generic(path, columns, ts_is, tz, ts_format)
    return _finish(ts, data, Path(path).name, _contract(symbol, path, ts))


def load_csv_contracts(path: str, preset: str = "databento", **kw) -> dict[str, pd.DataFrame]:
    """Load a file holding one or more contracts into ``{code: bars}``.

    A Databento file is split by its ``symbol`` column. Any other preset holds one
    contract, whose code must be known (``symbol=`` or the file name)."""
    _check_preset(preset)
    if preset == "databento":
        frames = _read_databento(str(path))
        if None in frames:
            raise ValueError(f"{Path(path).name} has no symbol column; pass it to load_csv "
                             "with symbol='NQH6'")
        return frames
    df = load_csv(path, preset, **kw)
    if "contract" not in df.columns:
        raise ValueError(f"{Path(path).name}: cannot tell which contract this is; pass "
                         "symbol='NQH6' or name the file after the contract")
    return {df["contract"].iloc[0]: df}
