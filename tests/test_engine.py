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


def _recursive_ema(values, span):
    alpha = 2.0 / (span + 1.0)
    out = [float("nan")] * len(values)
    # min_periods=span: first defined value at index span-1, seeded by SMA? No:
    # pandas adjust=False seeds the recursion at the first observation, but masks
    # outputs before min_periods. Reference mirrors that exactly.
    ema = None
    for i, v in enumerate(values):
        ema = v if ema is None else alpha * v + (1 - alpha) * ema
        if i >= span - 1:
            out[i] = ema
    return out


class EMACrossEngine(Strategy):
    def precompute(self, df):
        df = df.copy()
        df["ema_f"] = df["close"].ewm(span=3, adjust=False, min_periods=3).mean()
        df["long"] = df["ema_f"] > df["close"]
        return df

    def on_bar(self, ctx):
        if ctx.bar.long and ctx.position.qty == 0:
            ctx.broker.buy(contracts=1)


def test_engine_ewm_adjust_false_matches_recursive_reference():
    closes = [20000.0, 20010.0, 19990.0, 20020.0, 20005.0, 20030.0]
    df = _bars(closes)
    got = df["close"].ewm(span=3, adjust=False, min_periods=3).mean().to_numpy()
    ref = _recursive_ema(closes, span=3)
    np.testing.assert_allclose(got[2:], np.asarray(ref[2:]), rtol=1e-12)
    assert np.isnan(got[0]) and np.isnan(got[1])  # masked by min_periods


def test_engine_no_order_during_warmup_nan_window():
    # span=3 -> ema_f is NaN at bars 0,1; 'long' is False there; no fill can
    # occur from a warm-up bar's decision. Decision on bar 0/1 must be skipped.
    nq = get_instrument("NQ")
    eng = Engine(nq, BacktestConfig(slippage_ticks=0.0, commission_per_side=0.0))

    class _Spy(EMACrossEngine):
        bars_seen = []

        def on_bar(self, ctx):
            _Spy.bars_seen.append(int(ctx.bar.ts))
            super().on_bar(ctx)

    df = _bars([20000.0, 20010.0, 19990.0, 20020.0, 20005.0, 20030.0])
    result = eng.run(df, _Spy())
    # on_bar never called for ts 0 or 300 (warm-up NaN bars).
    assert 0 not in _Spy.bars_seen
    assert 300 not in _Spy.bars_seen
    # the earliest possible entry fill is at or after the first warmed bar's t+1.
    if result.trades:
        assert result.trades[0].entry_ts >= 900
