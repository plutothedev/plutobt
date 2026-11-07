"""Deterministically generate the SYNTHETIC NQ 2026 quarterly sample contracts (2.0).

Four contract files, ``data/sample/nq_contracts/NQ{H,M,U,Z}2026.csv``, for the roll
walkthrough, the build_continuous example and the roll tests. Synthetic, so they are
redistributable: no exchange or vendor data.

One seeded random walk in whole NQ ticks (the "spot" path) runs on the real NQ regular
trading hours grid, 08:30 to 15:15 America/Chicago, in 15-minute bars, on every weekday
from Monday 2026-03-02 to Friday 2026-10-02 (exchange holidays are not modelled). Each
contract is that path plus its own constant basis in ticks, so every pair of contracts
moves together and the gap between them at any bar is the difference of their bases:

    NQH2026  basis   0 ticks   2026-03-02 .. 2026-03-20
    NQM2026  basis 100 ticks   2026-03-09 .. 2026-06-18   (H -> M gap +25.00 points)
    NQU2026  basis  40 ticks   2026-06-08 .. 2026-09-18   (M -> U gap -15.00 points)
    NQZ2026  basis 160 ticks   2026-09-08 .. 2026-10-02   (U -> Z gap +30.00 points)

Each pair overlaps for at least five trading days around CME's roll day (the Monday
before the third Friday: 03-16, 06-15, 09-14). Volume: the leading contract trades about
1,500 a bar and the other about 150; the lead passes to the next contract two trading
days before each calendar roll, so the volume trigger rolls one trading day early.

Files are CSV in the ``generic`` layout (``ts,open,high,low,close,volume``; ts = unix
seconds of the bar OPEN) with LF endings (data/sample/.gitattributes keeps them LF), so
tests/test_sample_contracts.py pins them byte for byte.

Run:  PYTHONPATH=src python scripts/make_sample_contracts.py
"""
from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

CT = ZoneInfo("America/Chicago")
TICK = 0.25
BAR_SECONDS = 900
SEED = 20261106
FIRST, LAST = date(2026, 3, 2), date(2026, 10, 2)
START_TICKS = 72_000                     # 18,000.00

CONTRACTS = {
    "NQH2026": dict(basis=0, first=date(2026, 3, 2), last=date(2026, 3, 20)),
    "NQM2026": dict(basis=100, first=date(2026, 3, 9), last=date(2026, 6, 18)),
    "NQU2026": dict(basis=40, first=date(2026, 6, 8), last=date(2026, 9, 18)),
    "NQZ2026": dict(basis=160, first=date(2026, 9, 8), last=date(2026, 10, 2)),
}
# The volume leader from each date on (two trading days before each calendar roll).
LEAD_FROM = {"NQH2026": FIRST, "NQM2026": date(2026, 3, 12),
             "NQU2026": date(2026, 6, 11), "NQZ2026": date(2026, 9, 10)}


def weekdays(first: date = FIRST, last: date = LAST) -> list[date]:
    days, d = [], first
    while d <= last:
        if d.weekday() < 5:
            days.append(d)
        d += timedelta(days=1)
    return days


def session_opens(day: date) -> list[int]:
    def at(hh: int, mm: int) -> int:
        return int(datetime(day.year, day.month, day.day, hh, mm, tzinfo=CT).timestamp())
    return list(range(at(8, 30), at(15, 15), BAR_SECONDS))


def spot_path() -> list[tuple[date, int, int, int, int, int]]:
    """(day, ts, open, high, low, close) in whole ticks for every bar of the sample."""
    rng = np.random.default_rng(SEED)
    rows, price = [], START_TICKS
    for i, day in enumerate(weekdays()):
        if i:
            price += int(round(rng.normal(0.0, 240.0)))      # overnight move
        for ts in session_opens(day):
            o = price
            c = o + int(round(rng.normal(0.0, 48.0)))
            hi = max(o, c) + int(abs(round(rng.normal(0.0, 16.0))))
            lo = min(o, c) - int(abs(round(rng.normal(0.0, 16.0))))
            rows.append((day, ts, o, hi, lo, c))
            price = c
    return rows


def _leader(day: date) -> str:
    return max((d, code) for code, d in LEAD_FROM.items() if d <= day)[1]


def generate(code: str) -> list[tuple[int, float, float, float, float, int]]:
    spec = CONTRACTS[code]
    rng = np.random.default_rng(SEED + 1 + list(CONTRACTS).index(code))
    out = []
    for day, ts, o, hi, lo, c in spot_path():
        if not spec["first"] <= day <= spec["last"]:
            continue
        b = spec["basis"]
        base = 1500.0 if _leader(day) == code else 150.0
        vol = int(abs(rng.normal(base, base / 5))) + 1
        out.append((ts, (o + b) * TICK, (hi + b) * TICK, (lo + b) * TICK,
                    (c + b) * TICK, vol))
    return out


def render(code: str) -> bytes:
    nl = chr(10)
    lines = ["ts,open,high,low,close,volume"]
    lines += [f"{ts},{o!r},{h!r},{lo!r},{c!r},{v}" for ts, o, h, lo, c, v in generate(code)]
    return (nl.join(lines) + nl).encode("utf-8")


def main() -> None:
    out_dir = Path(__file__).resolve().parents[1] / "data" / "sample" / "nq_contracts"
    out_dir.mkdir(parents=True, exist_ok=True)
    for code in CONTRACTS:
        path = out_dir / f"{code}.csv"
        path.write_bytes(render(code))
        print(f"wrote {len(generate(code))} bars -> {path}")


if __name__ == "__main__":
    main()
