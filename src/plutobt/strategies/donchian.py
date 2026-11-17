"""Donchian: channel breakout with a shorter exit channel (the Turtle System 1 shape)."""
from __future__ import annotations

import pandas as pd

from plutobt.strategy import Context, Strategy


class Donchian(Strategy):
    """Donchian channel breakout.

    Long when the close is above the highest high of the previous ``entry_bars`` bars;
    exit the long when the close is below the lowest low of the previous ``exit_bars``
    bars. The short side mirrors it and is used only when the run allows shorts
    (``allow_short``). No stop.

    There is no stop-and-reverse: an exit and an opposite breakout on the same bar only
    exits; a new entry needs a later bar.

    Source: Curtis Faith, "Way of the Turtle" (2007), System 1: a 20-day breakout
    entry and a 10-day opposite-breakout exit (``entry_bars = 20``, ``exit_bars = 10``),
    applied here to bars. The original rule that skips an entry after a winning
    System 1 trade is left out. The defaults are not numbers tuned on any data.

    Causality: every channel is shifted one bar, so bar t compares its close with the
    channel of bars before t.
    """

    entry_bars: int = 20
    exit_bars: int = 10

    def precompute(self, df: pd.DataFrame) -> pd.DataFrame:
        entry_high = df["high"].rolling(self.entry_bars).max().shift(1)
        entry_low = df["low"].rolling(self.entry_bars).min().shift(1)
        exit_low = df["low"].rolling(self.exit_bars).min().shift(1)
        exit_high = df["high"].rolling(self.exit_bars).max().shift(1)
        # NaN comparisons are False, so nothing fires during warm-up.
        df["long_entry"] = df["close"] > entry_high
        df["short_entry"] = df["close"] < entry_low
        df["long_exit"] = df["close"] < exit_low
        df["short_exit"] = df["close"] > exit_high
        return df

    def on_bar(self, ctx: Context) -> None:
        qty = ctx.position.qty
        bar = ctx.bar
        if qty == 0:
            if bar.long_entry:
                ctx.broker.buy(contracts=1)
            elif bar.short_entry and ctx.broker.config.allow_short:
                ctx.broker.sell(contracts=1)
        elif qty > 0 and bar.long_exit:
            ctx.broker.close()
        elif qty < 0 and bar.short_exit:
            ctx.broker.close()
