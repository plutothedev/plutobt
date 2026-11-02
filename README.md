# PlutoBT

Hybrid futures backtesting engine in Python. Vectorized signals + event-driven
execution, realistic futures fills/margin, verified CME contract specs. MIT.

## What it is

PlutoBT is a single-instrument futures backtesting engine built for correctness
and auditability. Key properties:

- **Decision-at-t / fill-at-t+1**: strategy sees bar t; the market order fills at
  bar t+1's open. No look-ahead by construction, and checked: before the first bar,
  `Engine.run` re-runs the strategy's `precompute` on every prefix of the data and
  refuses any column whose value at bar t changes when the later bars are removed
  (`check_causality`, on by default), or a precompute that rewrites the price bars.
  Above 2,000 bars the default (`causality_sample = "auto"`) checks 250 evenly spaced
  prefixes plus the first and last 20, and warns that it sampled; `None` checks all.
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
generalizes across instruments while costs do NOT auto-carry: each config sets its
own commission, slippage and `periods_per_year` (5-minute bars in the instrument's
RTH session x 252: NQ/ES 81 x 252 = 20412, CL 66 x 252 = 16632).

The strategy's parameters are the ones it ships with; they were not tuned on the
sample. Every number is in-sample and not predictive.

## Examples

Teaching props for the video, each runnable on its own (from the repo root, with the
package installed, or with `PYTHONPATH=src`):

| Script | What it shows |
|--------|---------------|
| `examples/run_demo.py` | The EMA(20/50) demo on NQ, ES and CL with per-instrument costs. |
| `examples/cold_open_lookahead.py` | One changed line (`position = signal` vs `signal.shift(1)`): trading a bar on its own close (impossible) against acting at the next bar's open (frictionless pandas, "bar hit rate", unlevered), then the same signal through the real engine with NQ costs (closed-trade win rate, gross - slippage - commission = net). Saves a chart to `examples/output/`. |
| `examples/causality_demo.py` | A strategy built on `close.shift(-1) > close` is refused by `Engine.run` with a `LookaheadError` naming the column; its honest twin (`close > close.shift(1)`) runs and prints its report. |
| `examples/cost_sweep.py` | A COST sweep, not a parameter sweep: the demo strategy, parameters unchanged, at 0, 1, 2 and 4 ticks of slippage per side with commission fixed. Final equity, Sharpe, PF and expectancy per level. |

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

The bars sit on each instrument's real regular-trading-hours session from the
registry (America/Chicago clock), 5 minutes apart, over the 20 trading days
2026-03-02 to 2026-03-27: no bars outside RTH, an overnight gap between sessions,
four weekends, and the US DST change on 2026-03-08 (the CT open stays put, the UTC
open moves an hour). Each file is under 2,000 bars, so the default look-ahead check
runs every prefix.

| File | Instrument | Session (CT) | Bars | Base price | Seed |
|------|-----------|--------------|------|------------|------|
| `nq_5m.json` | NQ E-mini Nasdaq-100 | 08:30-15:15, 81/day | 1,620 | 18 000 | 20260607 |
| `es_5m.json` | ES E-mini S&P 500 | 08:30-15:15, 81/day | 1,620 | 5 000 | 20260608 |
| `cl_5m.json` | CL WTI Crude Oil | 08:00-13:30, 66/day | 1,320 | 70 | 20260609 |

Regenerate with `PYTHONPATH=src python scripts/gen_sample.py` (deterministic: it
reproduces every byte, and `tests/test_sample_data.py` checks that). After a
deliberate change, re-freeze the golden run with `scripts/freeze_golden.py`.
See `data/sample/NOTICE.md` for the full data license.

## Contract specs

Verified against CME Group (NQ, ES, CL, GC, ZN, etc.). See `docs/superpowers/specs/`.
