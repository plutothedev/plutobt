"""On-camera demo: PlutoBT refuses a strategy that peeks at the future.

Two strategies, one line apart:

  LEAKY   tomorrow_up = close.shift(-1) > close   "will the NEXT bar close higher?"
  HONEST  up          = close > close.shift(1)    "did THIS bar close higher?"

The leaky column reads bar t+1's close while it fills bar t, so a backtest built on it
would trade on a price that does not exist yet. Engine.run catches it before a single
bar runs: it re-runs the strategy's precompute on every prefix of the data and refuses
any column whose value at bar t changes when the bars after t are removed, raising
LookaheadError naming the column. The honest version passes the same check and runs.

Data: the synthetic NQ sample (data/sample/nq_5m.json), NQ demo costs (configs/nq.toml).
This is a teaching prop, not part of the engine. The honest result is reported as it
comes out: nothing here was tuned.

Run:  PYTHONPATH=src python examples/causality_demo.py
"""
from __future__ import annotations

import pathlib

import pandas as pd

from plutobt.config import load_config
from plutobt.data import load_bars
from plutobt.engine import BacktestResult, Engine
from plutobt.instruments import get_instrument
from plutobt.lookahead import LookaheadError
from plutobt.metrics import compute_metrics
from plutobt.report import format_report
from plutobt.strategy import Strategy

_ROOT = pathlib.Path(__file__).resolve().parents[1]
SAMPLE = _ROOT / "data" / "sample" / "nq_5m.json"
NQ_CONFIG = _ROOT / "configs" / "nq.toml"


def _trade(ctx, signal: bool) -> None:
    """Long one contract while the signal holds, flat when it does not."""
    if signal and ctx.position.qty == 0:
        ctx.broker.buy(contracts=1)
    elif not signal and ctx.position.qty != 0:
        ctx.broker.close()


class LeakyTomorrow(Strategy):
    """Peeks: bar t's signal is built from bar t+1's close."""

    def precompute(self, df: pd.DataFrame) -> pd.DataFrame:
        df["tomorrow_up"] = df["close"].shift(-1) > df["close"]   # THE LEAK
        return df

    def on_bar(self, ctx) -> None:
        _trade(ctx, ctx.bar.tomorrow_up)


class HonestToday(Strategy):
    """Causal: bar t's signal uses bar t's close and the one before it."""

    def precompute(self, df: pd.DataFrame) -> pd.DataFrame:
        df["up"] = df["close"] > df["close"].shift(1)
        return df

    def on_bar(self, ctx) -> None:
        _trade(ctx, ctx.bar.up)


def _engine() -> tuple[Engine, float]:
    cfg = load_config(str(NQ_CONFIG))
    return Engine(get_instrument("NQ"), cfg), cfg.periods_per_year


def run_leaky() -> LookaheadError:
    """Run the leaky strategy; return the LookaheadError the engine raises."""
    engine, _ = _engine()
    try:
        engine.run(load_bars(str(SAMPLE)), LeakyTomorrow())
    except LookaheadError as refused:
        return refused
    raise AssertionError("the engine ran a strategy that reads the next bar's close")


def run_honest() -> BacktestResult:
    engine, ppy = _engine()
    result = engine.run(load_bars(str(SAMPLE)), HonestToday())
    result.metrics = compute_metrics(result, periods_per_year=ppy)
    return result


def main() -> None:
    rule = "=" * 78
    print(rule)
    print("1) LEAKY:  tomorrow_up = close.shift(-1) > close   (reads the NEXT bar)")
    print(rule)
    refused = run_leaky()
    print("Engine.run REFUSED it before the first bar:")
    print(f"  {type(refused).__name__}: {refused}")
    print()
    print(rule)
    print("2) HONEST: up = close > close.shift(1)             (this bar and the last)")
    print(rule)
    print("Engine.run accepted it. The result, as it comes out:")
    print()
    print(format_report(run_honest(), symbol="NQ"))


if __name__ == "__main__":
    main()
