"""EMACross: the canonical §4 example strategy, ported to the PlutoBT API."""
from __future__ import annotations

import pandas as pd

from plutobt.strategy import Context, Strategy


class EMACross(Strategy):
    """Fast/slow EMA crossover. Stateless signal in precompute, execution in on_bar.

    precompute is CAUSAL and STATELESS: ema_f/ema_s are recursive (adjust=False)
    chart-matching EMAs with min_periods warm-up so 'long' never fires under-warmed.
    """

    fast_span: int = 20
    slow_span: int = 50
    stop_ticks: float = 20.0

    def precompute(self, df: pd.DataFrame) -> pd.DataFrame:
        df["ema_f"] = df["close"].ewm(
            span=self.fast_span, adjust=False, min_periods=self.fast_span
        ).mean()
        df["ema_s"] = df["close"].ewm(
            span=self.slow_span, adjust=False, min_periods=self.slow_span
        ).mean()
        # NaN > NaN -> False, so 'long' is a clean bool and never fires during warm-up.
        df["long"] = df["ema_f"] > df["ema_s"]
        return df

    def on_bar(self, ctx: Context) -> None:
        if ctx.bar.long and ctx.position.qty == 0:
            ctx.broker.buy(
                contracts=1,
                stop=ctx.bar.close - self.stop_ticks * ctx.tick,
            )
        elif (not ctx.bar.long) and ctx.position.qty != 0:
            ctx.broker.close()
