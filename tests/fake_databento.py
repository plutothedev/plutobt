"""A fake of the databento 0.84.0 ``Historical`` client, for tests. Never touches the network.

Only the surface PlutoBT uses is modelled, shaped like the real package:

- ``Historical(key=None)``: a None key reads ``DATABENTO_API_KEY``; a None or blank key
  raises ValueError. ``.key`` holds the key.
- ``.metadata.get_cost(dataset=, symbols=, schema=, start=, end=, stype_in=) -> float``
- ``.metadata.get_dataset_range(dataset) -> {"start", "end", "schema": {schema: {...}}}``
- ``.symbology.resolve(dataset=, symbols=, stype_in=, stype_out=, start_date=, end_date=)
  -> {"result": {"NQ.v.0": [{"d0", "d1", "s"}, ...]}, ...}`` (d1 exclusive)
- ``.timeseries.get_range(dataset=, symbols=, schema=, start=, end=, stype_in=) -> store``
  whose ``to_df(price_type="float")`` is indexed by a tz-aware UTC DatetimeIndex named
  ``ts_event`` with columns rtype, publisher_id, instrument_id, open, high, low, close,
  volume (uint64), symbol.

A ``Market`` holds what the fake exchange lists, what each call costs, and injected
failures; every call is recorded in ``market.calls`` as ``(name, kwargs)``.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from datetime import date, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

CT = ZoneInfo("America/Chicago")
DATASET = "GLBX.MDP3"
_STEP = {"ohlcv-1m": 60, "ohlcv-1h": 3600, "ohlcv-1d": 86400}
_RTYPE = {"ohlcv-1m": 33, "ohlcv-1h": 34, "ohlcv-1d": 35}


@dataclass
class Listing:
    """One outright contract: its raw symbol, when Databento's ``.v.0`` calls it front
    ([front_from, front_to)), and its price level."""

    raw: str
    front_from: date
    front_to: date
    base: float
    tick: float
    instrument_id: int = 1

    @property
    def root(self) -> str:
        return self.raw[:-2]

    @property
    def trades_from(self) -> date:
        return self.front_from - timedelta(days=60)

    @property
    def trades_to(self) -> date:
        return self.front_to + timedelta(days=30)


@dataclass
class Market:
    listings: list[Listing]
    dataset_start: str = "2010-06-06T00:00:00.000000000Z"
    available_end: str = "2026-11-16T00:00:00.000000000Z"
    rate_per_day: float = 0.01                 # USD per contract-day
    cost_fn: object = None                     # optional (kwargs) -> value, overrides rate
    fail_get_cost: Exception | None = None
    fail_resolve: Exception | None = None
    fail_get_range: dict[int, Exception] = field(default_factory=dict)  # 1-based call no.
    spread_rows: bool = False
    calls: list[tuple[str, dict]] = field(default_factory=list)

    def count(self, name: str) -> int:
        return sum(1 for n, _ in self.calls if n == name)

    def listing(self, raw: str) -> Listing:
        for li in self.listings:
            if li.raw == raw:
                return li
        raise KeyError(raw)


def _day(value) -> date:
    return value if isinstance(value, date) else date.fromisoformat(str(value)[:10])


def _ts(value) -> pd.Timestamp:
    t = pd.Timestamp(str(value))
    return t.tz_localize("UTC") if t.tz is None else t.tz_convert("UTC")


def _globex_open(idx: pd.DatetimeIndex) -> np.ndarray:
    """CME Globex hours: Sunday 17:00 CT to Friday 16:00 CT, closed 16:00-17:00 daily."""
    local = idx.tz_convert(CT)
    wd = np.asarray(local.weekday)
    hour = np.asarray(local.hour)
    ok = hour != 16
    ok &= wd != 5                                     # Saturday
    ok &= ~((wd == 4) & (hour >= 16))                 # Friday after the close
    ok &= ~((wd == 6) & (hour < 17))                  # Sunday before the open
    return ok


def make_bars(li: Listing, schema: str, start, end) -> pd.DataFrame:
    """Databento-shaped bars for one listing over [start, end)."""
    step = _STEP[schema]
    lo = max(_ts(start), pd.Timestamp(li.trades_from, tz="UTC"))
    hi = min(_ts(end), pd.Timestamp(li.trades_to, tz="UTC"))
    if lo >= hi:
        idx = pd.DatetimeIndex([], tz="UTC")
    elif step == 86400:
        idx = pd.date_range(lo.ceil("D"), hi, freq="D", inclusive="left", tz="UTC")
        idx = idx[np.asarray(idx.weekday) < 5]
    else:
        idx = pd.date_range(lo.ceil(f"{step}s"), hi, freq=f"{step}s", inclusive="left")
        idx = idx[_globex_open(idx)]
    secs = (idx.asi8 // 1_000_000_000) if len(idx) else np.array([], dtype=np.int64)
    k = np.rint(40 * np.sin(secs / 86400.0 / 3.0)).astype(np.int64)
    dec = max(0, -int(math.floor(math.log10(li.tick))) + 1)
    o = np.round(li.base + k * li.tick, dec)
    c = np.round(o + ((secs // step) % 3 - 1) * li.tick, dec)
    h = np.round(np.maximum(o, c) + 2 * li.tick, dec)
    lo_ = np.round(np.minimum(o, c) - 2 * li.tick, dec)
    days = np.asarray(idx.date) if len(idx) else np.array([], dtype=object)
    front = np.array([li.front_from <= d < li.front_to for d in days], dtype=bool)
    vol = np.where(front, 1000, 50).astype(np.uint64)
    df = pd.DataFrame({
        "rtype": np.full(len(idx), _RTYPE[schema], dtype=np.uint8),
        "publisher_id": np.full(len(idx), 1, dtype=np.uint16),
        "instrument_id": np.full(len(idx), li.instrument_id, dtype=np.uint32),
        "open": o, "high": h, "low": lo_, "close": c, "volume": vol,
        "symbol": li.raw,
    }, index=pd.DatetimeIndex(idx, name="ts_event"))
    return df


class FakeDBNStore:
    def __init__(self, df: pd.DataFrame, market: Market):
        self._df = df
        self._market = market

    def to_df(self, price_type="fixed", pretty_ts=True, map_symbols=True, **kwargs):
        self._market.calls.append(("to_df", {"price_type": price_type}))
        if price_type != "float":
            raise AssertionError("PlutoBT must ask for float prices")
        return self._df.copy()


class _Metadata:
    def __init__(self, client: Historical):
        self._c = client

    def get_cost(self, dataset, symbols, schema, start=None, end=None, stype_in="raw_symbol",
                 **kwargs):
        kw = dict(dataset=dataset, symbols=symbols, schema=schema, start=start, end=end,
                  stype_in=stype_in, **kwargs)
        m = self._c.market
        m.calls.append(("get_cost", kw))
        if m.fail_get_cost is not None:
            raise m.fail_get_cost
        assert dataset == DATASET and schema in _STEP and stype_in == "raw_symbol"
        assert _day(start) < _day(end)
        if m.cost_fn is not None:
            return m.cost_fn(kw)
        days = (_day(end) - _day(start)).days
        return float(m.rate_per_day * days * len(list(symbols)))

    def get_dataset_range(self, dataset):
        m = self._c.market
        m.calls.append(("get_dataset_range", {"dataset": dataset}))
        assert dataset == DATASET
        sch = {s: {"start": m.dataset_start, "end": m.available_end} for s in _STEP}
        return {"start": m.dataset_start, "end": m.available_end, "schema": sch}


class _Symbology:
    def __init__(self, client: Historical):
        self._c = client

    def resolve(self, dataset, symbols, stype_in, stype_out, start_date, end_date=None,
                **kwargs):
        m = self._c.market
        kw = dict(dataset=dataset, symbols=symbols, stype_in=stype_in, stype_out=stype_out,
                  start_date=start_date, end_date=end_date, **kwargs)
        m.calls.append(("resolve", kw))
        if m.fail_resolve is not None:
            raise m.fail_resolve
        assert dataset == DATASET and stype_in == "continuous" and stype_out == "raw_symbol"
        lo, hi = _day(start_date), _day(end_date)
        result: dict[str, list[dict]] = {}
        not_found = []
        for sym in symbols:
            root = sym.split(".")[0]
            rows = []
            for li in sorted(m.listings, key=lambda x: x.front_from):
                if li.root != root:
                    continue
                d0, d1 = max(li.front_from, lo), min(li.front_to, hi)
                if d0 < d1:
                    rows.append({"d0": d0.isoformat(), "d1": d1.isoformat(), "s": li.raw})
            if rows:
                result[sym] = rows
            else:
                not_found.append(sym)
        return {"result": result, "symbols": list(symbols), "stype_in": stype_in,
                "stype_out": stype_out, "start_date": lo.isoformat(),
                "end_date": hi.isoformat(), "partial": [], "not_found": not_found,
                "message": "OK", "status": 0}


class _Timeseries:
    def __init__(self, client: Historical):
        self._c = client

    def get_range(self, dataset, start, symbols=None, schema=None, end=None,
                  stype_in="raw_symbol", **kwargs):
        m = self._c.market
        kw = dict(dataset=dataset, symbols=symbols, schema=schema, start=start, end=end,
                  stype_in=stype_in, **kwargs)
        m.calls.append(("get_range", kw))
        n = m.count("get_range")
        if n in m.fail_get_range:
            raise m.fail_get_range[n]
        assert dataset == DATASET and schema in _STEP and stype_in == "raw_symbol"
        parts = []
        for raw in symbols:
            li = m.listing(raw)
            df = make_bars(li, schema, start, end)
            if m.spread_rows and len(df):
                # Calendar-spread rows at the same timestamps, ahead of the outright's.
                spread = df.copy()
                spread["symbol"] = f"{raw}-{raw[:-2]}Z9"
                spread["instrument_id"] = 999
                for col in ("open", "high", "low", "close"):
                    spread[col] = 3.333
                parts.append(spread)
            parts.append(df)
        out = pd.concat(parts).sort_index(kind="stable")
        out.index.name = "ts_event"
        return FakeDBNStore(out, m)


DEFAULT_MARKET: Market | None = None


class Historical:
    def __init__(self, key: str | None = None, *, market: Market | None = None, **kwargs):
        if key is None:
            key = os.environ.get("DATABENTO_API_KEY")
        if key is None:
            raise ValueError("Missing API key. Set DATABENTO_API_KEY or pass key.")
        if not isinstance(key, str) or not key.strip():
            raise ValueError("API key cannot be empty.")
        self.key = key
        self.market = market if market is not None else (DEFAULT_MARKET or Market([]))
        self.metadata = _Metadata(self)
        self.symbology = _Symbology(self)
        self.timeseries = _Timeseries(self)


# -- ready-made markets -------------------------------------------------------

def nq_market(**kw) -> Market:
    """NQ 2024: Databento's front rolls H4 -> M4 on 2024-03-08 and M4 -> U4 on 2024-06-14.
    CME's calendar roll days are 2024-03-11 and 2024-06-17."""
    return Market([
        Listing("NQH4", date(2023, 12, 15), date(2024, 3, 8), 17000.0, 0.25, 101),
        Listing("NQM4", date(2024, 3, 8), date(2024, 6, 14), 17200.0, 0.25, 102),
        Listing("NQU4", date(2024, 6, 14), date(2024, 9, 13), 17400.0, 0.25, 103),
    ], **kw)


def cl_market(**kw) -> Market:
    """CL early 2024: Databento's front rolls G4 -> H4 on 2024-01-19 (by volume)."""
    return Market([
        Listing("CLG4", date(2023, 12, 19), date(2024, 1, 19), 72.00, 0.01, 201),
        Listing("CLH4", date(2024, 1, 19), date(2024, 2, 20), 72.50, 0.01, 202),
    ], **kw)
