"""On-camera demo: the same strategy, the same data, rising execution cost.

This is a COST sweep, NOT a parameter sweep. The demo strategy (EMACross: EMA 20/50,
20-tick stop, long only) keeps exactly the parameters it ships with; nothing about the
strategy is varied or picked. Only the slippage charged on every fill changes, 0, 1, 2
and 4 ticks per side, with commission held at the NQ config's value. The question it
answers is "how much friction can this result survive?", not "which setting looks
best?".

Data: the synthetic NQ sample (data/sample/nq_5m.json); costs from configs/nq.toml.
Every row is IN-SAMPLE and not predictive of live results.

Run:  PYTHONPATH=src python examples/cost_sweep.py
"""
from __future__ import annotations

import math
import pathlib
from dataclasses import replace

import pandas as pd

from plutobt.config import load_config
from plutobt.data import load_bars
from plutobt.engine import Engine
from plutobt.instruments import get_instrument
from plutobt.metrics import compute_metrics, cost_sensitivity
from plutobt.strategies.ema_cross import EMACross

_ROOT = pathlib.Path(__file__).resolve().parents[1]
SAMPLE = _ROOT / "data" / "sample" / "nq_5m.json"
NQ_CONFIG = _ROOT / "configs" / "nq.toml"
SLIPPAGE_TICKS = [0.0, 1.0, 2.0, 4.0]


def run(slippages: list[float] = SLIPPAGE_TICKS) -> pd.DataFrame:
    """One row per slippage level: final equity, Sharpe, profit factor, expectancy,
    closed trades, positions open at the end, and fills (each closed trade has two, an
    open position its entry only: the fills every tick of slippage is charged on)."""
    base = load_config(str(NQ_CONFIG))
    instrument = get_instrument("NQ")
    df = load_bars(str(SAMPLE))

    def one(slippage_ticks: float) -> dict:
        cfg = replace(base, slippage_ticks=slippage_ticks)
        result = Engine(instrument, cfg).run(df, EMACross())
        m = compute_metrics(result, periods_per_year=cfg.periods_per_year)
        n_open = sum(1 for t in result.trades if t.open_at_end)
        return {
            "commission_per_side": cfg.commission_per_side,
            "final_equity": float(result.equity_curve.iloc[-1]),
            "sharpe": m["sharpe"],
            "profit_factor": m["profit_factor"],
            "expectancy": m["expectancy"],
            "n_trades": m["n_trades"],
            "n_open": n_open,
            "fills": sum((1 if t.open_at_end else 2) * t.contracts for t in result.trades),
        }

    return cost_sensitivity(one, slippages)


def _fmt(x: float, spec: str) -> str:
    return "n/a" if math.isnan(x) else format(x, spec)


def _trades(n_closed: int, n_open: int) -> str:
    """Closed trades, plus any position still open at the end ("73 + 1 open")."""
    return f"{n_closed} + {n_open} open" if n_open else f"{n_closed}"


def main() -> None:
    table = run()
    tick_value = get_instrument("NQ").tick_value
    commission = table["commission_per_side"].iloc[0]
    print("COST SWEEP (not a parameter sweep): EMACross 20/50, parameters unchanged")
    print(f"NQ synthetic sample, commission fixed at ${commission:.2f}/side, "
          "slippage varied per side")
    print("IN-SAMPLE, single pre-specified strategy; not predictive of live results.")
    print()
    print(f"{'slippage':>12}  {'final equity':>14}  {'Sharpe':>7}  {'PF':>6}  "
          f"{'expectancy':>11}  {'trades':>12}")
    for ticks, row in table.iterrows():
        print(f"{ticks:>6g} ticks  {row['final_equity']:>14,.2f}  "
              f"{_fmt(row['sharpe'], '.3f'):>7}  {_fmt(row['profit_factor'], '.3f'):>6}  "
              f"{_fmt(row['expectancy'], ',.2f'):>11}  "
              f"{_trades(int(row['n_trades']), int(row['n_open'])):>12}")
    print()
    print(f"Starting equity $100,000.00, 1 NQ contract (${tick_value:.2f} a tick).")
    if table["n_open"].any():
        print("\"+ 1 open\": a position still open at the end, marked to the last close.")
    fills = set(table["fills"])
    if len(fills) == 1:
        # The step between rows is one tick per fill: a closed trade has two fills, a
        # position open at the end only its entry.
        (n_fills,) = fills
        n_closed, n_open = int(table["n_trades"].iloc[0]), int(table["n_open"].iloc[0])
        open_part = f" + {n_open} open entr{'y' if n_open == 1 else 'ies'}" if n_open else ""
        print(f"Each tick of slippage per side costs ${tick_value:.2f} x {int(n_fills)} fills")
        print(f"  ({n_closed} round trips x 2{open_part}) = "
              f"${tick_value * n_fills:,.2f} of final equity.")


if __name__ == "__main__":
    main()
