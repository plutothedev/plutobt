"""Deterministically generate the frozen synthetic NQ 5m sample (§12).

Synthetic so it is unambiguously redistributable. Seeded GBM-style walk on the
RTH grid (08:30-15:15 CT, 81 bars/day) over 4 sessions. ts marks the bar OPEN
in unix seconds (§5.5 convention). Run once; the JSON is committed and frozen.
"""
import json
from pathlib import Path

import numpy as np

SEED = 20260607
BARS_PER_DAY = 81          # 405 min / 5, RTH 08:30-15:15 CT (§5.7)
N_DAYS = 4
BAR_SECONDS = 300
START_TS = 1717503000      # 2024-06-04 08:30:00 CT in unix seconds
START_PRICE = 18000.0
TICK = 0.25                # NQ tick; snap OHLC to a valid grid


def _snap(x: float) -> float:
    return round(round(x / TICK) * TICK, 2)


def generate() -> list[list]:
    rng = np.random.default_rng(SEED)
    rows: list[list] = []
    price = START_PRICE
    ts = START_TS
    for _ in range(N_DAYS * BARS_PER_DAY):
        ret = rng.normal(0.0, 6.0)          # ~6 pt per-bar sigma
        o = price
        c = max(TICK, o + ret)              # never non-positive
        hi = max(o, c) + abs(rng.normal(0.0, 3.0))
        lo = min(o, c) - abs(rng.normal(0.0, 3.0))
        lo = max(TICK, lo)
        vol = int(abs(rng.normal(1500, 400))) + 1
        rows.append([int(ts), _snap(o), _snap(hi), _snap(lo), _snap(c), vol])
        price = c
        ts += BAR_SECONDS
    return rows


def main() -> None:
    out = Path(__file__).resolve().parents[1] / "data" / "sample" / "nq_5m.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = generate()
    # Stable, diff-friendly bytes: one bar per line, trailing newline.
    body = ",\n".join(json.dumps(r, separators=(",", ":")) for r in rows)
    out.write_text("[\n" + body + "\n]\n", encoding="utf-8")
    print(f"wrote {len(rows)} bars -> {out}")


if __name__ == "__main__":
    main()
