"""Synthetic bars and module-level strategies for the 3.0 study tests. Module level so
the ledger's strategy id (module:qualname) is stable. All data is synthetic and seeded:
no real-data sweep anywhere (3.0 spec §11)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from plutobt.config import BacktestConfig
from plutobt.strategy import Strategy

T0 = 1_767_277_800            # 2026-01-01 14:30 UTC
CFG = BacktestConfig(commission_per_side=0.0, slippage_ticks=0.0, periods_per_year=20412.0,
                     check_causality=False)


def walk(n: int = 1800, drift=0.0, sigma: float = 5.0, seed: int = 0) -> pd.DataFrame:
    """A 5-minute NQ-like random walk. ``drift`` (points a bar) may be a number or an array
    of length n (a regime change)."""
    rng = np.random.default_rng(seed)
    steps = rng.normal(0.0, sigma, n) + np.broadcast_to(np.asarray(drift, float), (n,))
    close = np.round((18_000.0 + np.cumsum(steps)) / 0.25) * 0.25
    open_ = np.concatenate([[18_000.0], close[:-1]])
    return pd.DataFrame({
        "ts": T0 + 300 * np.arange(n, dtype=np.int64),
        "open": open_,
        "high": np.maximum(open_, close) + 0.25,
        "low": np.minimum(open_, close) - 0.25,
        "close": close,
        "volume": np.full(n, 100, dtype=np.int64),
    })


class Hold(Strategy):
    """Always long (side 1), always short (side -1), or never trades (side 0)."""
    side = 1

    def on_bar(self, ctx):
        if ctx.position.qty == 0 and self.side > 0:
            ctx.broker.buy(contracts=1)
        elif ctx.position.qty == 0 and self.side < 0:
            ctx.broker.sell(contracts=1)


class Coin(Strategy):
    """Long while a seeded coin (probability ``bias``) says so, flat otherwise. The coin
    depends only on the bar's position, so it is causal."""
    seed = 0
    bias = 0.5

    def precompute(self, df):
        df["go"] = np.random.default_rng(self.seed).random(len(df)) < self.bias
        return df

    def on_bar(self, ctx):
        if ctx.bar.go and ctx.position.qty == 0:
            ctx.broker.buy(contracts=1)
        elif not ctx.bar.go and ctx.position.qty != 0:
            ctx.broker.close()


class StartsLate(Strategy):
    """Flat until bar ``start_bar``, long from then on."""
    start_bar = 0

    def precompute(self, df):
        df["late"] = np.arange(len(df)) >= self.start_bar
        return df

    def on_bar(self, ctx):
        if ctx.bar.late and ctx.position.qty == 0:
            ctx.broker.buy(contracts=1)


class Crashes(Strategy):
    """Raises on its first bar when ``boom`` is true."""
    boom = False

    def on_bar(self, ctx):
        if self.boom:
            raise RuntimeError("boom")


class NeedsArgs(Strategy):
    def __init__(self, x):
        self.x = x


class Sparse(Strategy):
    """Long for one bar every ``every`` bars, starting at bar ``phase``: a strategy that is
    rarely in the market, with its few trades spread evenly."""
    every = 100
    phase = 5

    def precompute(self, df):
        df["go"] = (np.arange(len(df)) % self.every) == self.phase
        return df

    def on_bar(self, ctx):
        if ctx.bar.go and ctx.position.qty == 0:
            ctx.broker.buy(contracts=1)
        elif not ctx.bar.go and ctx.position.qty != 0:
            ctx.broker.close()


class Inert(Hold):
    """Hold with a setting that does nothing: every ``dummy`` value trades identically."""
    dummy = 0


class Big(Strategy):
    """Long ``size`` contracts whenever flat: enough contracts to meet the margin rules."""
    size = 4

    def on_bar(self, ctx):
        if ctx.position.qty == 0:
            ctx.broker.buy(contracts=self.size)


class ScaleOut(Strategy):
    """Every 100 bars: long ``lots`` contracts, then sold one at a time near the cycle's
    end. Its closed trades are one contract each, though it held ``lots`` at once."""
    lots = 4

    def precompute(self, df):
        df["step"] = np.arange(len(df)) % 100
        return df

    def on_bar(self, ctx):
        step = int(ctx.bar.step)
        if step == 0 and ctx.position.qty == 0:
            ctx.broker.buy(contracts=self.lots)
        elif step in (70, 80, 90, 95) and ctx.position.qty > 0:
            ctx.broker.sell(contracts=1)


class ScaleIn(Strategy):
    """Adds one contract at bars 5, 10, 15 and 20 of every 100, closes all at bar 60: a
    pyramid. The engine books it as ONE trade of ``4`` contracts entered at the first
    fill, so the position at a bar cannot be read off the trades."""
    side = 1

    def precompute(self, df):
        df["step"] = np.arange(len(df)) % 100
        return df

    def on_bar(self, ctx):
        step = int(ctx.bar.step)
        if step in (5, 10, 15, 20):
            (ctx.broker.buy if self.side > 0 else ctx.broker.sell)(contracts=1)
        elif step == 60 and ctx.position.qty != 0:
            ctx.broker.close()
