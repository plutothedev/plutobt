# PlutoBT

Hybrid futures backtesting engine in Python. Vectorized signals + event-driven
execution, realistic futures fills/margin, verified CME contract specs. MIT.

## What it is

PlutoBT is a single-instrument futures backtesting engine built for correctness
and auditability. Key properties:

- **Decision-at-t / fill-at-t+1**: strategy sees bar t; the market order fills at
  bar t+1's open. No look-ahead by construction, and checked before the first bar
  (see [Look-ahead check](#look-ahead-check)).
- **Realistic fills**: directional slippage, gap-through stops, GTC protective
  stops, limit order fill-or-carry.
- **Performance-bond accounting**: MTM equity, initial/maintenance margin, open-at-end
  marking — the way futures actually work.
- **Van Tharp R-multiples**: attach a stop on entry and every trade's r_multiple
  and the portfolio avg_r populate automatically.
- **Verified contract specs**: NQ, ES, CL, GC, ZN, YM, RTY, MNQ, MES — all
  point values, tick sizes, and margins from CME Group publications.

## Look-ahead check

Fills are causal by construction: `on_bar` decides at bar t, the order fills at bar
t+1's open. The vectorized `precompute` is where look-ahead usually hides (a
`.shift(-1)`, a backfill, a centred window, a whole-sample mean), so `Engine.run`
checks it before the first bar (`check_causality`, on by default):

- **What is checked is what is traded.** The strategy is snapshotted with
  `copy.deepcopy` before any `precompute` runs. The traded frame is `precompute` on a
  fresh copy, on the full data. Each prefix `df[:t+1]` runs on its own fresh copy, and
  its rows must match the traded frame's first t+1 rows, or the run is refused with a
  `LookaheadError` naming the column. State one call saves on the strategy (a z-score
  fitted "once", an RNG's position) never reaches another call. `on_bar` runs on the
  strategy you passed, which then runs the full frame itself so its state is the
  traded call's; a leak found in the prefixes leaves it as you passed it. A strategy
  that cannot be deep-copied (it holds a lock or a socket) is refused, naming the
  switch.
- **Float noise is not a leak.** A float column matches when
  `|prefix - full| <= 1e-6 * |full| + 1e-6 * scale`, where `scale` is that column's
  largest finite value in the traded frame, so the floor follows the column: a leak in
  a feature near 1e-10 is as visible as one in a price. Integer, bool, text and date
  columns match exactly. A change smaller than about a part per million of the column
  is out of reach, by design.
- **The grid.** Every prefix up to 2,000 bars. Above that, the default
  (`causality_sample = "auto"`) checks 250 evenly spaced prefixes plus the first and
  last 20 and warns at your `Engine.run` line how many it checked; `None` checks all
  (the cost grows with the square of the bar count).
- **Proof, not a reviewer's word.** `tests/test_lookahead_property.py` sweeps seeded
  synthetic bars from 1e-3 to 1e5 in price and 50 to 400 bars long: every honest
  feature (rolling, EWM and expanding statistics, returns, RSI, session sums, an
  expanding z-score, skew and kurtosis of returns, a seeded RNG) must pass, and every
  planted leak (`shift(-1..-5)`, backfills, centred windows, whole-sample
  normalisation, a fit-once strategy, a leak in one row near the end) must be caught.
  `pytest --slow` runs the full sweep.

What it does not do, stated plainly:

- **It is not a sandbox.** It checks `precompute`. `on_bar` must decide from `ctx`
  only; code written to cheat (say, by storing the bars on the strategy and reading
  ahead from `on_bar`) is not caught, and no Python check could promise to.
- **State held before the run is a parameter.** Whatever the strategy holds when
  `Engine.run` is called is trusted: fit models on training data before the run,
  never inside `precompute` on the bars being tested. A function stored on the
  strategy is shared by the copies, so state it captures is outside the snapshot.
- **pandas' skew and kurtosis of raw price levels** subtract the whole input's rounded
  mean first, so on a series that wanders far from its mean they can move by more than
  a part per million between a prefix and the full frame (up to about 1% of the column
  on short windows), and such a column is refused although honest. On realistic futures
  bars (the NQ sample, 1,500 one-minute-like bars) `rolling(20)` passes; compute them
  on returns, where pandas' result does not move at all.
- **A sampled grid can miss** a leak confined to rows between its prefixes.

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
