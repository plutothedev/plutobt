# PlutoBT

Hybrid futures backtesting engine in Python. Vectorized signals + event-driven
execution, realistic futures fills/margin, verified CME contract specs. MIT.

## What it is

PlutoBT is a single-instrument futures backtesting engine built for correctness
and auditability. Key properties:

- **Decision-at-t / fill-at-t+1**: strategy sees bar t; the market order fills at
  bar t+1's open. No look-ahead by construction.
- **Realistic fills**: directional slippage, gap-through stops, GTC protective
  stops, limit order fill-or-carry.
- **Performance-bond accounting**: MTM equity, initial/maintenance margin, open-at-end
  marking — the way futures actually work.
- **Van Tharp R-multiples**: attach a stop on entry and every trade's r_multiple
  and the portfolio avg_r populate automatically.
- **Verified contract specs**: NQ, ES, CL, GC, ZN, YM, RTY, MNQ, MES — all
  point values, tick sizes, and margins from CME Group publications.

## Install

```bash
pip install -e ".[dev]"
```

Requires Python 3.11+. The dev extras include pytest, ruff, and matplotlib.
The exact stack used for the golden run is pinned in `requirements.lock`.

## Run the demo

```bash
python examples/run_demo.py
```

Loads the synthetic 5-minute sample bars for NQ, ES, and CL, runs an
EMA(20/50)-cross strategy on each with per-instrument cost configs, and prints
three IN-SAMPLE-labeled performance reports. Demonstrates that the same engine
generalizes across instruments while costs do NOT auto-carry.

## Reproducible golden run

The golden regression run is pinned by content hash in `configs/golden_expected.json`.
Run the full suite to verify it:

```bash
pytest -q
```

To re-freeze after a deliberate engine/data/config change:

```bash
python scripts/freeze_golden.py
```

## Sample data

All files in `data/sample/` are **synthetic** (seeded numpy random walks) and
are freely redistributable. They are NOT real market data and are NOT predictive.

| File | Instrument | Base Price | Seed |
|------|-----------|------------|------|
| `nq_5m.json` | NQ E-mini Nasdaq-100 | ~18 000 | 20260607 |
| `es_5m.json` | ES E-mini S&P 500 | ~5 000 | 20260608 |
| `cl_5m.json` | CL WTI Crude Oil | ~70 | 20260609 |

Regenerate with `python scripts/gen_sample.py` (deterministic).
See `data/sample/NOTICE.md` for the full data license.

## Contract specs

Verified against CME Group (NQ, ES, CL, GC, ZN, etc.). See `docs/superpowers/specs/`.
