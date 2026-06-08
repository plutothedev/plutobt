"""Deterministically generate frozen synthetic 5m sample bars (§12).

Synthetic so it is unambiguously redistributable. Seeded GBM-style walk on the
RTH grid (08:30-15:15 CT, 81 bars/day) over 4 sessions. ts marks the bar OPEN
in unix seconds (§5.5 convention). Run once; the JSON is committed and frozen.

Symbols generated
-----------------
nq_5m.json  — NQ E-mini Nasdaq-100,  base ~18 000, seed 20260607 (original, unchanged)
es_5m.json  — ES E-mini S&P 500,     base ~5 000,  seed 20260608
cl_5m.json  — CL WTI Crude Oil,      base ~70,     seed 20260609

Each file uses its own independent seed; re-running is fully deterministic.
The NQ file is byte-for-byte identical to the original (same seed/params/snap).
"""
import json
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Shared grid parameters
# ---------------------------------------------------------------------------

BARS_PER_DAY = 81          # 405 min / 5, RTH 08:30-15:15 CT (§5.7)
N_DAYS = 4
BAR_SECONDS = 300
START_TS = 1717503000      # 2024-06-04 08:30:00 CT in unix seconds


def _snap2(x: float) -> float:
    """Round to 2 decimal places (NQ/ES 0.25-tick grid)."""
    return round(round(x / 0.25) * 0.25, 2)


def _snap_cl(x: float) -> float:
    """Round to CL 0.01-tick grid, 2 decimal places."""
    return round(round(x / 0.01) * 0.01, 2)


# ---------------------------------------------------------------------------
# Per-symbol parameters
# ---------------------------------------------------------------------------

_SYMBOLS = [
    dict(
        name="nq",
        seed=20260607,
        start_price=18000.0,
        snap=_snap2,
        sigma=6.0,       # ~6 pt per-bar sigma for NQ
        wick_sigma=3.0,
    ),
    dict(
        name="es",
        seed=20260608,
        start_price=5000.0,
        snap=_snap2,
        sigma=2.0,       # ~2 pt per-bar sigma for ES
        wick_sigma=1.0,
    ),
    dict(
        name="cl",
        seed=20260609,
        start_price=70.0,
        snap=_snap_cl,
        sigma=0.25,      # ~0.25 pt per-bar sigma for CL
        wick_sigma=0.12,
    ),
]


def generate(seed: int, start_price: float, snap,
             sigma: float, wick_sigma: float) -> list[list]:
    rng = np.random.default_rng(seed)
    rows: list[list] = []
    price = start_price
    ts = START_TS
    tick = 0.25 if snap is _snap2 else 0.01
    for _ in range(N_DAYS * BARS_PER_DAY):
        ret = rng.normal(0.0, sigma)
        o = price
        c = max(tick, o + ret)              # never non-positive
        hi = max(o, c) + abs(rng.normal(0.0, wick_sigma))
        lo = min(o, c) - abs(rng.normal(0.0, wick_sigma))
        lo = max(tick, lo)
        vol = int(abs(rng.normal(1500, 400))) + 1
        rows.append([
            int(ts),
            snap(o),
            snap(hi),
            snap(lo),
            snap(c),
            vol,
        ])
        price = c
        ts += BAR_SECONDS
    return rows


def main() -> None:
    sample_dir = Path(__file__).resolve().parents[1] / "data" / "sample"
    sample_dir.mkdir(parents=True, exist_ok=True)

    for spec in _SYMBOLS:
        rows = generate(
            seed=spec["seed"],
            start_price=spec["start_price"],
            snap=spec["snap"],
            sigma=spec["sigma"],
            wick_sigma=spec["wick_sigma"],
        )
        out = sample_dir / f"{spec['name']}_5m.json"
        body = ",\n".join(json.dumps(r, separators=(",", ":")) for r in rows)
        out.write_text("[\n" + body + "\n]\n", encoding="utf-8")
        print(f"wrote {len(rows)} bars -> {out}")


if __name__ == "__main__":
    main()
