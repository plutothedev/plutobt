# PlutoBT — Onboarding

## 1. What it is (30 seconds)
A **futures backtesting engine** in Python. You give it price bars + a trading strategy; it simulates the trades the way a real futures account works (margin, slippage, gap-through stops) and hands back an equity curve + performance metrics. Single instrument per run. MIT-licensed.

It's two things at once: a **real tool** for Pluto's NQ trading, and the **subject of a YouTube video** ("use AI to build a backtesting engine"). That second purpose is why correctness is obsessive — the video's payoff is "naive backtests lie, here's one that doesn't."

## 2. The one idea you must hold in your head
> **Decision-at-t, fill-at-t+1.**

Your strategy looks at bar `t` and decides. The order it places fills at bar `t+1`'s open — never bar `t`. That single rule kills look-ahead bias by construction. Most amateur backtesters secretly let you trade on a bar using that same bar's close (impossible in reality). PlutoBT physically can't. See `src/plutobt/engine.py` (the `run` loop).

## 3. The other thing that makes it "honest"
Everything is **futures-accurate, not stock-approximate**:
- **Performance-bond margin** — initial/maintenance margin, margin calls, mark-to-market equity every bar (`portfolio.py`).
- **Realistic fills** — directional slippage, gap-through protective stops (price gaps past your stop → you fill at the gap, not the stop), GTC resting stops (`broker.py`).
- **Verified contract specs** — NQ/ES/CL/GC/ZN/YM/RTY/MNQ/MES point values, tick sizes, margins, all from CME publications. ZN even handles Treasury 32nds quoting (`110'085` = 110 + 8.5/32) (`instruments.py`).
- Every result ships a **manifest** with `data_sha256`: the sha256 of the loaded bars written as CSV (`df.to_csv(index=False)`), not of the file on disk, so `sha256sum nq_5m.json` gives a different value. To check the hash a report prints, from the repo root:
  ```bash
  PYTHONPATH=src python -c "import hashlib; from plutobt.data import load_bars; print(hashlib.sha256(load_bars('data/sample/nq_5m.json').to_csv(index=False).encode()).hexdigest())"
  ```
  The golden run (`configs/golden_expected.json`) pins that hash, and `pytest` checks it.

## 4. Repo map
```
plutobt/
├─ src/plutobt/          ← the engine (12 modules, ~1700 LOC)
├─ tests/                ← the test suite, one file per module (plus property and sample-data tests)
├─ examples/run_demo.py  ← the end-to-end demo (the "video payoff")
├─ scripts/              ← gen_sample.py (synthetic data), freeze_golden.py
├─ configs/              ← per-symbol cost configs (nq/es/cl.toml) + golden
├─ data/sample/          ← synthetic NQ/ES/CL bars (seeded, redistributable)
└─ docs/superpowers/     ← the design spec, the 9-milestone build plan, video plan
```

## 5. The modules (build order = dependency order)
| Module | Does |
|---|---|
| `instruments.py` | Contract registry. `get_instrument("NQ")` → specs. Price/tick/PnL math, 32nds parsing. |
| `data.py` | Load JSON `[ts,o,h,l,c,v]` → DataFrame; validate (OHLC sanity, dup/non-monotonic timestamps, gaps); tag RTH/ETH sessions (America/Chicago); resample. |
| `portfolio.py` | The account. Mark-to-market equity, VWAP position averaging, realized/unrealized P&L, margin checks, Van Tharp R-multiples. |
| `broker.py` | Order lifecycle. `buy/sell/close/amend_stop`, slippage, stop triggers, gap-through fills. |
| `engine.py` | The bar loop that wires it all together (the decision@t/fill@t+1 rule lives here). Defines `BacktestResult`. |
| `metrics.py` | Sharpe (+ standard error & confidence interval), PSR, Sortino, Calmar, win rate, profit factor, expectancy, avg R, drawdown, exposure. |
| `config.py` | `BacktestConfig` + `load_config` (TOML costs/params). |
| `report.py` | Renders the human-readable IN-SAMPLE performance report. |
| `plotting.py` | Equity-curve / trade plots (headless matplotlib). |
| `strategy.py` | The base class you subclass. |
| `lookahead.py` | Causality guards. |

## 6. The public API (this is all a user touches)
```python
from plutobt import (
    Engine, BacktestConfig, load_config,
    Strategy, Context,
    get_instrument, load_bars,
    compute_metrics, format_report,
)
```

Minimal run:
```python
inst   = get_instrument("NQ")
config = load_config("configs/nq.toml")
df     = load_bars("data/sample/nq_5m.json")

result = Engine(inst, config).run(df, MyStrategy())
result.metrics = compute_metrics(result, periods_per_year=result.manifest["config"]["periods_per_year"])
print(format_report(result, symbol="NQ"))
```

## 7. How you write a strategy
Subclass `Strategy`, override two methods:
```python
class EMACross(Strategy):
    def precompute(self, df):          # VECTORIZED, causal, stateless — add signal columns
        df["ema_fast"] = df["close"].ewm(span=20).mean()
        df["ema_slow"] = df["close"].ewm(span=50).mean()
        return df

    def on_bar(self, ctx):             # EVENT-DRIVEN — one bar at a time, place orders
        if ctx.bar.ema_fast > ctx.bar.ema_slow and ctx.position.qty == 0:
            ctx.broker.buy(1, stop=ctx.bar.close - 50 * ctx.tick)
```
- `precompute` runs once (fast, whole DataFrame). `on_bar` runs per bar (stateful decisions).
- `ctx` gives you `bar`, `position`, `broker`, `instrument`, `tick`, `equity`, and a `state` dict for your own memory.
- Order methods: `broker.buy(n, stop=, limit=)`, `sell(...)`, `close()`, `amend_stop(price)`.
- Attach a `stop=` on entry and R-multiples populate automatically (real EMACross lives at `src/plutobt/strategies/ema_cross.py`).

## 8. How to run it
```bash
pip install -e ".[dev]"           # Python 3.11+
python examples/run_demo.py       # runs EMACross on NQ, ES, CL → 3 reports
pytest -q                         # full suite + golden-run regression (pytest --slow adds the look-ahead property sweep)
python scripts/freeze_golden.py   # re-freeze golden after a deliberate change
```
The demo is the thesis in code: **same engine + strategy, swap symbol & cost config → costs do NOT auto-carry.** Proves generalization without pretending costs are universal.

## 9. Two gotchas a newcomer will hit
1. **Reports are labeled IN-SAMPLE on purpose.** No out-of-sample split in v1. The honesty defense (and the video) depends on *not* secretly parameter-sweeping off-camera. Don't "tune until it looks good."
2. **`parse_price` requires exactly-3-digit 32nds** (`"110'085"` works, `"110'08"` raises). Resolves a contradiction in the build plan. Harmless in practice — the real data path is decimal floats; 32nds strings are a display/edge case.

## 10. Status
**v1 is complete; pre-film audit fixes landed 2026-11-02** on branch `film/ready`. A pre-film audit found and fixed look-ahead in the equity curve, entry-bar stops, margin calls that never ran, scale-in margin, stale stops after flips, allow_short, overnight exits, the look-ahead check itself (now run by `Engine.run`, verified by a randomized property sweep), off-session sample data, and report honesty (CI and PSR on the same skew/kurtosis-adjusted basis). Every fix was reviewed by an independent agent with executable probes.

**What's left** (none is engine code): record the YouTube video (script in `docs/`); keep the data/no-sweep discipline; optional beyond-v1 (MinTRL, full continuous-contract back-adjustment — both deliberately cut from v1).

## Where to go deeper
- Design spec: `docs/superpowers/specs/2026-06-07-plutobt-engine-design.md`
- Build plan (9 milestones, TDD): `docs/superpowers/plans/2026-06-07-plutobt-engine.md`
- Video plan: `docs/superpowers/specs/2026-06-07-plutobt-video-plan.md`
