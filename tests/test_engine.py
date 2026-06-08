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


class BuyAndStayLong(Strategy):
    def precompute(self, df):
        df = df.copy()
        df["go"] = df.index == 0
        return df

    def on_bar(self, ctx):
        if ctx.bar.go and ctx.position.qty == 0:
            ctx.broker.buy(contracts=1)


def test_engine_open_position_marked_to_last_not_dropped():
    nq = get_instrument("NQ")
    eng = Engine(nq, BacktestConfig(slippage_ticks=0.0, commission_per_side=0.0))
    df = _bars([20000.0, 20010.0, 20050.0])  # entry fills bar1 open=20010
    result = eng.run(df, BuyAndStayLong())

    # exactly one trade, still open at end, flagged.
    assert len(result.trades) == 1
    tr = result.trades[0]
    assert tr.open_at_end is True
    assert tr.exit_ts is None and tr.exit_price is None

    # final equity reflects unrealized MTM to last close (20050) from fill 20010:
    # 1 contract * $20/pt * (20050 - 20010) = +$800 on top of starting cash.
    final_equity = result.equity_curve.iloc[-1]
    assert final_equity == pytest.approx(100_000.0 + 800.0)
    # the open position is NOT silently removed from the equity curve length.
    assert len(result.equity_curve) == 3


from plutobt.instruments import pnl_points


def test_buy_and_hold_within_expiry_identity():
    nq = get_instrument("NQ")
    eng = Engine(nq, BacktestConfig(slippage_ticks=0.0, commission_per_side=0.0))
    closes = [20000.0, 20010.0, 20025.0, 20040.0, 20100.0]
    df = _bars(closes)  # entry decision bar0 -> fill bar1 open=20010
    result = eng.run(df, BuyAndStayLong())

    entry_fill = 20010.0
    last_close = closes[-1]
    expected_pnl = pnl_points(entry_fill, last_close, contracts=1, instrument=nq)
    # = (20100 - 20010) * 20 * 1 = 90 * 20 = 1800.0
    assert expected_pnl == pytest.approx(1800.0)

    # engine equity gain == identity P&L, no costs.
    equity_gain = result.equity_curve.iloc[-1] - 100_000.0
    assert equity_gain == pytest.approx(expected_pnl)
    # and the open trade's pnl matches the same identity.
    assert result.trades[0].pnl == pytest.approx(expected_pnl)


def test_buy_and_hold_with_costs_deducts_commission_and_slippage():
    nq = get_instrument("NQ")
    # 1 tick slippage on entry buy = +0.25pt = fill 20010.25; $2.50/contract/side.
    eng = Engine(nq, BacktestConfig(slippage_ticks=1.0, commission_per_side=2.50))
    closes = [20000.0, 20010.0, 20100.0]
    df = _bars(closes)
    result = eng.run(df, BuyAndStayLong())

    entry_fill = 20010.0 + 1.0 * nq.tick_size  # 20010.25
    gross = pnl_points(entry_fill, closes[-1], contracts=1, instrument=nq)
    # one side of commission paid at entry (still open at end): -2.50
    expected = gross - 2.50
    equity_gain = result.equity_curve.iloc[-1] - 100_000.0
    assert equity_gain == pytest.approx(expected)


class BuyOnLastBeforeGap(Strategy):
    def precompute(self, df):
        df = df.copy()
        # fire on the bar immediately before the discontinuity (index 1).
        df["go"] = df.index == 1
        return df

    def on_bar(self, ctx):
        if ctx.bar.go and ctx.position.qty == 0:
            ctx.broker.buy(contracts=1)


def _bars_with_weekend_gap():
    # 5m bars (step=300s). Friday close then ~49h weekend gap before Sunday open.
    ts = [0, 300, 300 + 49 * 3600, 300 + 49 * 3600 + 300]
    closes = [20000.0, 20010.0, 20500.0, 20510.0]  # huge Sunday gap up
    c = np.asarray(closes, dtype="float64")
    return pd.DataFrame({
        "ts": np.asarray(ts, dtype="int64"),
        "open": c, "high": c + 1.0, "low": c - 1.0,
        "close": c, "volume": np.ones(4, dtype="int64"),
    })


def test_discontinuity_cancels_unfilled_entry_default_policy():
    # Decision on bar 1 (last before weekend); default carry_overnight=False
    # -> the pending entry is cancelled at the gap, NOT filled at Sunday's open.
    nq = get_instrument("NQ")
    eng = Engine(nq, BacktestConfig(carry_overnight=False, slippage_ticks=0.0))
    df = _bars_with_weekend_gap()
    result = eng.run(df, BuyOnLastBeforeGap())
    assert result.trades == []  # NO phantom weekend fill


def test_carry_overnight_true_fills_at_next_session_open():
    nq = get_instrument("NQ")
    eng = Engine(nq, BacktestConfig(carry_overnight=True, slippage_ticks=0.0,
                                    commission_per_side=0.0))
    df = _bars_with_weekend_gap()
    result = eng.run(df, BuyOnLastBeforeGap())
    # entry carried across the gap fills at the Sunday bar's open = 20500.
    assert len(result.trades) == 1
    assert result.trades[0].entry_price == pytest.approx(20500.0)


def test_held_position_keeps_resting_stop_across_gap():
    # Position already filled before the gap; its protective stop must persist.
    nq = get_instrument("NQ")
    eng = Engine(nq, BacktestConfig(carry_overnight=False, slippage_ticks=0.0))

    class _EnterEarly(Strategy):
        def precompute(self, df):
            df = df.copy()
            df["go"] = df.index == 0
            return df

        def on_bar(self, ctx):
            if ctx.bar.go and ctx.position.qty == 0:
                ctx.broker.buy(contracts=1, stop=19900.0)  # resting protective stop

    df = _bars_with_weekend_gap()
    result = eng.run(df, _EnterEarly())
    # filled at bar1 open=20010, held through the gap, stop NOT cancelled.
    assert result.trades and result.trades[0].entry_price == pytest.approx(20010.0)
    # the position survives the discontinuity (no spurious flat); open at end.
    assert result.trades[0].open_at_end is True
