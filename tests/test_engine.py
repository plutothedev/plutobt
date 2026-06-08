import numpy as np
import pandas as pd
import pytest

from plutobt.engine import Engine
from plutobt.strategy import Strategy, Context
from plutobt.config import BacktestConfig
from plutobt.instruments import get_instrument
from plutobt.broker import Broker


def _bars(closes, ts0=0, step=300):
    n = len(closes)
    ts = np.arange(ts0, ts0 + n * step, step, dtype="int64")
    c = np.asarray(closes, dtype="float64")
    return pd.DataFrame({
        "ts": ts, "open": c, "high": c + 1.0, "low": c - 1.0,
        "close": c, "volume": np.ones(n, dtype="int64"),
    })


class BuyOnceAtBar1(Strategy):
    def precompute(self, df):
        df = df.copy()
        df["go"] = df.index == 1   # fire decision on bar index 1
        return df

    def on_bar(self, ctx):
        if ctx.bar.go and ctx.position.qty == 0:
            ctx.broker.buy(contracts=1)


def test_engine_decision_at_t_fills_at_t_plus_1_open():
    # Decision fires reading bar 1's row; market order fills at bar 2's OPEN.
    nq = get_instrument("NQ")
    eng = Engine(nq, BacktestConfig(commission_per_side=0.0, slippage_ticks=0.0))
    df = _bars([20000.0, 20010.0, 20020.0, 20030.0])
    result = eng.run(df, BuyOnceAtBar1())
    # exactly one entry fill, at bar index 2 (ts=600), price = open of bar 2.
    fills = [t for t in result.trades]
    assert len(result.trades) == 1
    assert result.trades[0].entry_ts == 600
    assert result.trades[0].entry_price == pytest.approx(20020.0)


def test_engine_equity_curve_indexed_by_ts_and_full_length():
    nq = get_instrument("NQ")
    eng = Engine(nq, BacktestConfig())
    df = _bars([20000.0, 20010.0, 20020.0])
    result = eng.run(df, Strategy())  # no-op strategy
    assert list(result.equity_curve.index) == [0, 300, 600]
    # flat, no positions -> equity constant at starting cash default
    assert result.equity_curve.nunique() == 1


def test_engine_imports_backtestconfig_from_config_module():
    # Guard the mandatory correction: engine must not redefine BacktestConfig.
    import plutobt.engine as eng_mod
    from plutobt.config import BacktestConfig as CfgConfig
    assert eng_mod.BacktestConfig is CfgConfig
