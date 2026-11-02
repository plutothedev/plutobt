"""Deterministically generate the frozen SYNTHETIC 5-minute sample bars (§12).

Synthetic so it is unambiguously redistributable: no exchange or vendor data. A seeded
random walk laid on each instrument's real REGULAR TRADING HOURS session, taken from the
registry in src/plutobt/instruments.py (America/Chicago wall clock):

    NQ, ES  08:30-15:15 CT  -> 81 bars/day x 20 days = 1,620 bars
    CL      08:00-13:30 CT  -> 66 bars/day x 20 days = 1,320 bars

The 20 trading days run Monday 2026-03-02 to Friday 2026-03-27: four weekends, and the
US DST change on Sunday 2026-03-08 (CT goes from UTC-6 to UTC-5, so the same 08:30 CT
open lands an hour earlier in UTC). No bars outside RTH: each session ends at its close
and the next opens the following trading day, with an overnight gap in price as well as
time. ts is unix seconds marking the bar OPEN (§5.5). Under 2,000 bars per file, so the
engine's default causality check runs every prefix.

Per bar: close = open + N(0, sigma); high/low add |N(0, wick_sigma)| beyond the body;
volume |N(1500, 400)| + 1. Between sessions the price jumps by N(0, overnight_sigma).
Prices snap to the instrument tick and stay strictly positive.

Symbols generated (each with its own seed; re-running is fully deterministic)
-----------------
nq_5m.json  NQ E-mini Nasdaq-100, base 18 000, seed 20260607
es_5m.json  ES E-mini S&P 500,    base 5 000,  seed 20260608
cl_5m.json  CL WTI Crude Oil,     base 70,     seed 20260609

Files are written with LF line endings (data/sample/.gitattributes keeps them LF on
every checkout), so tests/test_sample_data.py can pin them byte for byte.

Run:  PYTHONPATH=src python scripts/gen_sample.py
"""
from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

from plutobt.instruments import get_instrument

CT = ZoneInfo("America/Chicago")
BAR_SECONDS = 300
FIRST_DAY = date(2026, 3, 2)   # Monday; the US DST change is Sunday 2026-03-08
N_DAYS = 20

SYMBOLS = [
    dict(name="nq", symbol="NQ", seed=20260607, start_price=18000.0,
         sigma=6.0, wick_sigma=3.0, overnight_sigma=36.0),
    dict(name="es", symbol="ES", seed=20260608, start_price=5000.0,
         sigma=2.0, wick_sigma=1.0, overnight_sigma=12.0),
    dict(name="cl", symbol="CL", seed=20260609, start_price=70.0,
         sigma=0.25, wick_sigma=0.12, overnight_sigma=1.5),
]


def trading_days(first: date = FIRST_DAY, n: int = N_DAYS) -> list[date]:
    """The first ``n`` weekdays from ``first`` (no exchange holiday falls in the range)."""
    days: list[date] = []
    d = first
    while len(days) < n:
        if d.weekday() < 5:
            days.append(d)
        d += timedelta(days=1)
    return days


def session_opens(day: date, open_ct: str, close_ct: str) -> list[int]:
    """Unix-second bar OPENS on ``day`` from ``open_ct`` up to the bar that closes at
    ``close_ct`` (CT wall clock; zoneinfo applies that day's own UTC offset)."""
    def at(hhmm: str) -> int:
        hh, mm = (int(x) for x in hhmm.split(":"))
        return int(datetime(day.year, day.month, day.day, hh, mm, tzinfo=CT).timestamp())

    return list(range(at(open_ct), at(close_ct), BAR_SECONDS))


def _snap(x: float, tick: float) -> float:
    return round(round(x / tick) * tick, 2)


def generate(spec: dict) -> list[list]:
    """The bars for one symbol: ``[ts, open, high, low, close, volume]`` rows."""
    inst = get_instrument(spec["symbol"])
    tick = inst.tick_size
    rng = np.random.default_rng(spec["seed"])
    rows: list[list] = []
    price = spec["start_price"]
    for i, day in enumerate(trading_days()):
        if i:   # the overnight move between one session's close and the next open
            price = max(tick, price + rng.normal(0.0, spec["overnight_sigma"]))
        for ts in session_opens(day, inst.session_open_ct, inst.session_close_ct):
            o = price
            c = max(tick, o + rng.normal(0.0, spec["sigma"]))
            hi = max(o, c) + abs(rng.normal(0.0, spec["wick_sigma"]))
            lo = max(tick, min(o, c) - abs(rng.normal(0.0, spec["wick_sigma"])))
            vol = int(abs(rng.normal(1500, 400))) + 1
            rows.append([ts, _snap(o, tick), _snap(hi, tick), _snap(lo, tick),
                         _snap(c, tick), vol])
            price = c
    return rows


def render(spec: dict) -> bytes:
    """The exact bytes of ``<name>_5m.json``: one bar per line, LF endings."""
    body = ",\n".join(json.dumps(r, separators=(",", ":")) for r in generate(spec))
    return ("[\n" + body + "\n]\n").encode("utf-8")


def main() -> None:
    sample_dir = Path(__file__).resolve().parents[1] / "data" / "sample"
    sample_dir.mkdir(parents=True, exist_ok=True)
    for spec in SYMBOLS:
        out = sample_dir / f"{spec['name']}_5m.json"
        out.write_bytes(render(spec))
        print(f"wrote {len(generate(spec))} bars -> {out}")


if __name__ == "__main__":
    main()
