# PlutoBT

Hybrid futures backtesting engine in Python. Vectorized signals + event-driven
execution, realistic futures fills/margin, CME contract specs with representative
margins. MIT.

## What it is

PlutoBT is a single-instrument futures backtesting engine built for correctness
and auditability. Key properties:

- **Decision-at-t / fill-at-t+1**: strategy sees bar t; the market order fills at
  bar t+1's open. No look-ahead by construction, and checked before the first bar
  (see [Look-ahead check](#look-ahead-check)).
- **Realistic fills**: directional slippage, gap-through stops, GTC protective
  stops, limit order fill-or-carry. Each bar is booked worst case, per bar: the
  engine tries both monotone paths (open, low, high and open, high, low), each with
  both orders for a stop and a limit at the same price, plus both orders at the open
  when a gap-through stop and a market (or marketable) order act there together, and
  books the one that leaves the lowest equity at that bar's close (the worst-case mode
  of Löw, Maier-Paape and Platen 2015, arXiv:1509.08248). Stops and resting limits
  fill in the order that path crosses them. The choice is made bar by bar among those
  orderings only: it is not a worst case over the whole run, nor over all the ways a
  real bar could have moved (it can wiggle more than two monotone legs). A
  conservative convention, not a guarantee.
  A limit already through the market at the open fills there like a market order,
  slippage included (capped by its limit).
- **Session breaks**: with `carry_overnight = false` (the default), entries never
  carry across a break (an overnight or weekend gap, or a change of session; `break_on`
  picks which): a pending entry is cancelled there. That is not "flat overnight": a
  pending exit survives the break and fills at the next session's open, a flip keeps
  only its closing contracts, a margin liquidation always goes through, and an open
  position is never force-flattened, so it is held across the break unless the
  strategy exits.
- **Performance-bond accounting**: MTM equity, initial/maintenance margin, open-at-end
  marking — the way futures actually work.
- **Van Tharp R-multiples**: attach a stop on entry and every trade's r_multiple
  and the portfolio avg_r populate automatically.
- **Contract specs**: NQ, ES, CL, GC, ZN, YM, RTY, MNQ, MES point values and tick
  sizes per the CME contract specs. Margins are representative defaults (as of
  2026-06-07), not live exchange or broker margins: they vary by broker,
  volatility and date, so check yours before relying on a margin call.
- **Continuous contracts (2.0)**: your own data from Databento, TradingView,
  NinjaTrader or any CSV, stitched across expiries with a back-adjusted series,
  rolls that cost real money, and a proof that the result equals trading the real
  contracts (see [Contract roll](#contract-roll)).
- **Overfitting toolkit (3.0)**: the Deflated Sharpe Ratio, the Probability of
  Backtest Overfitting, walk-forward and purged cross-validation, with a ledger that
  counts every setting you ever tried (see [Did you overfit?](#did-you-overfit)).

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
- **State the copies share is watched.** Class attributes are shared by every copy,
  so a `precompute` that changes one (a mean fitted "once" and stored on the class,
  a cache) is refused with a `LookaheadError` naming it, and the attribute is put
  back (`Strategy` itself included; an object edited in place is restored best-effort:
  a list, dict or set in place, anything else by a rebuilt copy). An attribute `precompute` sets on `self` (a full-sample quantile kept for
  `on_bar`) draws a `LookaheadWarning` naming it at your `Engine.run` line: the check
  compares columns, so if `on_bar` reads that value and it came from the bars, it is
  look-ahead nobody checked. A `cached_property` a fresh copy computes to the same
  value without the bars (a function of the parameters) is not warned. Compute
  per-bar values as columns instead.
- **A short prefix is part of the check.** The prefixes start one bar long, so a
  `precompute` that raises on a short frame is re-raised naming the prefix, with the
  way past it: guard the indicator's warm-up (NaN until it has enough bars), or set
  `check_causality = false`.
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
- **A randomized property sweep, not a reviewer's word.** `tests/test_lookahead_property.py` sweeps seeded
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
  strategy is shared by the copies, and module-level globals are not copied at all,
  so state either one holds is outside the snapshot and out of the check's reach.
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
| `examples/cold_open_lookahead.py` | One changed line (`position = signal` vs `signal.shift(1)`): trading a bar on its own close (impossible) against acting at the next bar's open (frictionless pandas, "bar hit rate", unlevered), then the same signal through the real engine with NQ costs (closed-trade win rate, gross - slippage - commission = net). Each Sharpe prints with the engine's standard error. Saves a chart to `examples/output/`. |
| `examples/causality_demo.py` | A strategy built on `close.shift(-1) > close` is refused by `Engine.run` with a `LookaheadError` naming the column; its honest twin (`close > close.shift(1)`) runs and prints its report. |
| `examples/overfit_study.py` | A 3.0 study of the demo strategy over six EMA settings on the synthetic sample: every setting, the Deflated Sharpe against every try in the ledger, PBO, walk-forward and purged CV. |
| `examples/cost_sweep.py` | A COST sweep, not a parameter sweep: the demo strategy, parameters unchanged, at 0, 1, 2 and 4 ticks of slippage per side with commission fixed. Final equity, Sharpe with its standard error, PF and expectancy per level, and whether any Sharpe's 95% CI excludes zero. |

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

The full printed output of every example above is pinned too
(`tests/test_example_snapshots.py` against `tests/expected_output/`), so a drift in
any on-camera number fails the suite. After a deliberate change to what an example
prints, regenerate with `python scripts/freeze_examples.py`, review the diff, and say
why in the commit; never to silence a failure.

## Sample data

All files in `data/sample/` are **synthetic** (seeded numpy random walks) and
are freely redistributable. They are NOT real market data and are NOT predictive.

The bars sit on each instrument's real regular-trading-hours session from the
registry (America/Chicago clock), 5 minutes apart, over the 20 trading days
2026-03-02 to 2026-03-27: no bars outside RTH, an overnight gap between sessions,
three weekends, and the US DST change on 2026-03-08 (the CT open stays put, the UTC
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

Point values and tick sizes follow the CME contract specs (NQ, ES, CL, GC, ZN,
etc.; `src/plutobt/instruments.py`). Initial and maintenance margins are
representative defaults as of 2026-06-07, not live values: check your broker's
before relying on margin behaviour. A plain frame never rolls: each run is one
series as given (the `*_5m.json` samples are a single synthetic expiry), and the
manifest says so (`adjustment_method`). To trade through expiries, build a
continuous series (next two sections).

## Your own data

`load_csv` reads one contract's bars from a vendor file into the frame the engine
runs on, with the same checks `load_bars` makes (whole timestamps, finite prices,
every bar's body inside its range, timestamps strictly increasing):

```python
from plutobt import load_csv

df = load_csv("NQZ2025.csv", preset="tradingview")
df = load_csv("bars.csv", preset="generic", tz="America/Chicago", ts_is="open",
              columns={"ts": "Date", "open": "O", "high": "H", "low": "L",
                       "close": "C", "volume": "V"})
```

| Preset | Reads | Notes |
|--------|-------|-------|
| `databento` | Databento OHLCV CSV | Columns by name. `ts_event` is the bar open, nanoseconds or ISO in UTC; prices decimal or fixed-point (1e-9). A file with several contracts: `load_csv_contracts` splits it by `symbol`. |
| `tradingview` | "Export chart data" CSV from one contract's chart | Columns by name, any case; indicator columns dropped. Unix seconds, milliseconds or ISO with a zone. TradingView does not document whether `time` is the bar open: it is read as the open (`ts_is="close"` overrides). |
| `ninjatrader` | NinjaTrader 8 historical export, `yyyyMMdd HHmmss;o;h;l;c;v` | NinjaTrader exports in UTC with END-of-bar stamps, so each stamp moves back one bar to the open. Minute bars only. |
| `generic` | Any CSV | Map the columns, and say `ts_is="open"` or `"close"`. |

Three time rules are refused rather than guessed. Text timestamps are read as ISO
8601; any other layout needs `ts_format=` (`"%d/%m/%Y %H:%M"`), because 02/03 can
be February or March. Timestamps written without a zone need `tz=`: a wrong zone
shifts every bar. And a close-stamped file read as
open-stamped puts every bar one interval early, which is look-ahead, so `generic`
makes you say which it is. The contract (`NQZ2025`) comes from `symbol=`, the file's
symbol column, or its name: `NQZ5`, `NQZ25`, `NQZ2025`, `NQ 12-25` and `NQ DEC25`
all parse.

The Databento column order and the TradingView header are read by name because the
vendors do not document them exactly; if a real export does not load, open an issue
with its first lines.

## Contract roll

A futures contract expires, so years of history means stitching contracts into one
continuous series. PlutoBT builds it from one file per contract:

```bash
python -m plutobt.continuous data/sample/nq_contracts --preset generic --root NQ --ts-is open -o nq_cont.csv
```

```
NQ: 4 contracts, 3 rolls (calendar trigger, backward additive)
roll day    from      to           gap pts       gap $
2026-03-16  NQH2026   NQM2026         25.0       500.0
2026-06-15  NQM2026   NQU2026        -15.0      -300.0
2026-09-14  NQU2026   NQZ2026         30.0       600.0
```

```python
from plutobt import Engine, BacktestConfig, get_instrument, load_continuous

df, meta = load_continuous("nq_cont.csv")
result = Engine(get_instrument("NQ"), BacktestConfig()).run(df, strategy, roll_meta=meta)
```

**When it rolls.** `calendar` (equity-index roots NQ, ES, MNQ, MES, YM, RTY): CME's
customary roll date, the Monday before the third Friday of the expiry month
([CME Group, Equity Index Roll Dates](https://www.cmegroup.com/trading/equity-index/rolldates.html)).
June 2026 expires Thursday 6/18 (Juneteenth); the roll stays Monday 6/15. `volume`:
the trading day after the next contract's full-day volume first beats the current
one's (day t's completed volume decides, day t+1 acts, so it never reads ahead).
`dates`: your own roll day per contract, which is how CL, GC and ZN roll on their
exchange rules today. The roll happens at the first bar of the roll day where both
contracts trade.

**How it adjusts.** Backward additive (Panama): the newest contract keeps its real
prices, and each older one is shifted by the sum of the later gaps, in whole ticks.
Each gap is measured at the roll bar's open from both contracts, the moment the roll
trade executes, which is what makes the series' P&L equal the real contracts' P&L
exactly. The series carries `contract`, `adj_offset` (adjusted minus raw) and
`raw_close`; its `.roll.json` pins the trigger, the method, every roll and gap, a
hash of every input file and of the series itself.

**What the engine does with it.** A frame with a `contract` column runs only with its
`roll_meta`, and the hashes must match: an edited or stale series is refused
(`slice_continuous` takes a date range and re-pins it). When the contract changes
under an open position, the contracts still held after that bar's open are rolled:
a `RollEvent` in `result.rolls`, one round trip of commission and slippage in cash
and in the trade's `roll_cost`, stops and levels unchanged. An exit, a flip's closing
half or a stop gapped through at that open trades the expiring contract at its open
and never rolls; a stop hit later in the bar was held into the new contract, so it
did. The report adds the rolls taken,
their cost, and the **roll yield**: the jump an unadjusted front-month chart would
show at each roll, which is carry, not P&L.

**What it warns about.** On an adjusted series old prices are not the prices that
traded, so a fixed level (`close > 18000`) or a percentage or log return reads the
adjustment, not the market. With the look-ahead check on, precompute runs again on
prices raised by 1,000 ticks; a signal column that is neither unchanged (RSI, a
cross flag) nor raised by exactly that much (an EMA) draws a `PriceLevelWarning`
naming it. Adjusted history can also go to or below zero; the builder warns and
`Engine.run` refuses it unless `allow_nonpositive_prices = true`.

**How it is proven.** `tests/test_roll_reconciliation.py` runs thousands of random
cases (random contracts, gaps of both signs and zero, roll days, entries with and
without stops, flips, stops gapping through on the roll bar) through the engine on
the adjusted series AND through `tests/reference/raw_contract_sim.py`, an independent
simulator that imports nothing from PlutoBT, holds the real contracts and sells one
and buys the next on roll day. Price P&L matches to the tick, commission and roll
cost to 1e-9: 2,000 cases in CI, 20,000 with `pytest --slow`. A planted one-tick
adjustment error makes it fail, so it can fail. `tests/test_roll_invariance.py`
adds a contract and checks every older price moves by exactly one constant while
every trade and dollar stays the same.

## Did you overfit?

The honest question about a backtest is not "what's your Sharpe". It is "how many
did you try before you found that one?" The best of 100 settings with no edge at all
lands about 2.5 standard errors above zero by luck alone; the best of 1,000, about
3.25. PlutoBT 3.0 measures that, two ways.

**No code: deflate a number you already have.**

```bash
python -m plutobt.deflate --sharpe 2.1 --periods-per-year 12 --returns 60 --trials 40
```

```
PlutoBT deflate: best Sharpe 2.10 (annualized, 12 periods a year) over 60 returns, best of 40 tries
Luck bar: the best of 40 zero-edge tries would show an annualized Sharpe of about 0.99 by luck alone
Deflated Sharpe: 0.988
Verdict: survives deflation
```

`--trials` is every version you tried, the one you kept included. The spread of those
tries is unknown here, so it assumes what zero-edge tries would show over that many
returns, and says so. TradingView users: the Strategy Tester computes its Sharpe from
monthly returns after taking off a risk-free rate (2% a year by default), so deflating
it against zero is slightly strict. TradingView does not document whether its figure
is annualized: if yours is, pass it as it is with `--periods-per-year 12 --returns
<months>`; if it is a plain monthly figure, multiply it by 3.46 (the square root of 12)
first.

**Python: run the grid and let PlutoBT count.**

```python
from plutobt import format_study, get_instrument, load_bars, load_config, run_study
from plutobt.strategies.ema_cross import EMACross

result = run_study(EMACross, {"fast_span": [10, 20, 30], "slow_span": [50, 100]},
                   load_bars("data/sample/nq_5m.json"), get_instrument("NQ"),
                   load_config("configs/nq.toml"))
print(format_study(result, "NQ"))
```

Every setting runs once through the same engine (same costs, margin and rolls). The
report then answers:

- **Deflated Sharpe** (Bailey and Lopez de Prado, 2014): the probability the best
  setting truly beats the luck bar, the Sharpe the best of N zero-edge tries reaches by
  luck. N is every distinct setting ever recorded for this strategy on these exact bars,
  read from the ledger, so a study rerun until something looks good still counts every
  try. "Survives deflation" means 0.95 or more.
- **PBO**, the Probability of Backtest Overfitting (Bailey, Borwein, Lopez de Prado
  and Zhu): the bars are cut into 16 blocks; for each of the 12,870 ways to call half of
  them in-sample, how often does the in-sample winner land in the bottom half on the
  other half? 0.5 is a coin flip. The paper treats anything above 0.05 as a warning.
- **Walk-forward**: pick the best on one window, trade it on the next, five times.
  Each pick is rerun flat in its test window (earlier bars only warm up the
  indicators), so the stitched curve could actually have been traded. A position open
  at a window's end is charged one exit.
- **Purged CV**: every block is the test block once. The bars right after it are
  left out of training (the embargo, by default the longest trade in bars), because a
  position carried out of the test block would leak its outcome into training. Folds
  trained on later data say so; walk-forward is the strictly real-time test.

**The ledger.** Every completed run is one line in `.plutobt/trials.jsonl` (change it
with `ledger=`, turn it off with `ledger=False`, which the report then stamps). It is an
honesty aid, not a lock: a strategy is known by its module and class name, so renaming
the class starts a fresh count, and deleting the file forgets every try. Counting
near-identical settings as separate tries overstates N, so the verdict errs strict.

**How it is proven.** On pure noise, 2,000 seeded studies of 50 settings: the best
setting survived deflation in 0.25% of them. With one setting carrying a real edge, it
was found and survived in 97.7% of 300. The luck bar matches a brute-force simulation
and the paper's worked example (Deflated Sharpe 0.9004); PBO gives about 0.5 on noise,
near 0 for a planted steady edge and above 0.7 for settings that only got lucky; a
planted regime change makes walk-forward fall below in-sample. All test data is
synthetic: no real-data sweep was run to build this.
