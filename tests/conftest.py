"""Shared test fixtures and helpers for the PlutoBT suite."""
import pathlib

import pandas as pd

from plutobt.strategy import Strategy

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


class GoldenStrategy(Strategy):
    """Fixed, pre-specified EMA(20/50) cross — the §5.8 single strategy."""

    def precompute(self, df: pd.DataFrame) -> pd.DataFrame:
        df["ema_f"] = df.close.ewm(span=20, adjust=False, min_periods=20).mean()
        df["ema_s"] = df.close.ewm(span=50, adjust=False, min_periods=50).mean()
        df["long"] = df.ema_f > df.ema_s
        return df

    def on_bar(self, ctx) -> None:
        if ctx.bar.long and ctx.position.qty == 0:
            ctx.broker.buy(contracts=1, stop=ctx.bar.close - 20 * ctx.tick)
        elif (not ctx.bar.long) and ctx.position.qty != 0:
            ctx.broker.close()


def pytest_addoption(parser):
    parser.addoption("--slow", action="store_true", default=False,
                     help="also run tests marked slow (the full look-ahead property sweep)")


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: long-running; skipped unless --slow is given")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--slow"):
        return
    import pytest

    skip = pytest.mark.skip(reason="slow: run with --slow")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)
