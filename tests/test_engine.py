import pathlib

import numpy as np
import pandas as pd
import pytest

from plutobt.config import BacktestConfig
from plutobt.data import load_bars
from plutobt.engine import Engine
from plutobt.instruments import get_instrument, pnl_points
from plutobt.lookahead import assert_causal
from plutobt.metrics import compute_metrics
from plutobt.strategy import Strategy

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


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


def test_engine_strategy_precompute_is_causal_and_deterministic():
    nq = get_instrument("NQ")
    s = EMACrossEngine()
    df = _bars([20000.0 + i * 3.0 for i in range(40)])

    # the strategy's precompute must pass the causality detector...
    assert assert_causal(s.precompute, df, sample=12) is None

    # ...and the engine result must be reproducible (same input -> same output).
    eng1 = Engine(nq, BacktestConfig(slippage_ticks=1.0, commission_per_side=2.50))
    eng2 = Engine(nq, BacktestConfig(slippage_ticks=1.0, commission_per_side=2.50))
    r1 = eng1.run(df, EMACrossEngine())
    r2 = eng2.run(df, EMACrossEngine())
    np.testing.assert_allclose(r1.equity_curve.to_numpy(),
                               r2.equity_curve.to_numpy(), rtol=0, atol=0)
    assert r1.manifest["data_sha256"] == r2.manifest["data_sha256"]


def test_engine_rejects_a_leaky_strategy_via_assert_causal():
    class Leaky(Strategy):
        def precompute(self, df):
            df = df.copy()
            df["future"] = df["close"].shift(-1)  # leak
            df["long"] = df["future"] > df["close"]
            return df

    df = _bars([20000.0 + i * 2.0 for i in range(30)])
    with pytest.raises(AssertionError, match="non-causal"):
        assert_causal(Leaky().precompute, df)


# ---------------------------------------------------------------------------
# C1: attached_stop wires to Position.initial_stop → r_multiple/avg_r populate
# ---------------------------------------------------------------------------

class _StopProtectedEMA(Strategy):
    """EMA(20/50) cross that always attaches a stop on entry; used for §0.4 E2E."""

    def precompute(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["ema_f"] = df["close"].ewm(span=20, adjust=False, min_periods=20).mean()
        df["ema_s"] = df["close"].ewm(span=50, adjust=False, min_periods=50).mean()
        df["long"] = df["ema_f"] > df["ema_s"]
        return df

    def on_bar(self, ctx) -> None:
        if ctx.bar.long and ctx.position.qty == 0:
            # Attach a stop 20 ticks below the close so initial_stop is always set.
            stop = ctx.bar.close - 20 * ctx.tick
            ctx.broker.buy(contracts=1, stop=stop)
        elif (not ctx.bar.long) and ctx.position.qty != 0:
            ctx.broker.close()


def test_r_multiple_and_avg_r_populate_with_attached_stop():
    """§0.4 E2E: attached_stop must flow into Position.initial_stop so
    r_multiple / avg_r are non-None after running through the real engine."""
    sample = _REPO_ROOT / "data" / "sample" / "nq_5m.json"
    df = load_bars(str(sample))
    nq = get_instrument("NQ")
    cfg = BacktestConfig(
        commission_per_side=2.5,
        slippage_ticks=1.0,
        periods_per_year=20412.0,
        carry_overnight=False,
    )
    result = Engine(nq, cfg).run(df, _StopProtectedEMA())

    # There must be at least one closed trade for this assertion to be meaningful.
    closed = [t for t in result.trades if not t.open_at_end]
    assert len(closed) >= 1, "No closed trades — sample may be too short"

    # Every closed trade that was entered with a stop must carry an r_multiple.
    assert result.trades[0].r_multiple is not None, (
        "r_multiple is None; attached_stop was not wired to Position.initial_stop"
    )

    # compute_metrics must produce a real (non-None) avg_r.
    metrics = compute_metrics(result, periods_per_year=20412.0)
    assert metrics["avg_r"] is not None, "avg_r is None after wiring initial_stop"
    assert isinstance(metrics["avg_r"], float)
    assert not (metrics["avg_r"] != metrics["avg_r"])  # not NaN


# ---------------------------------------------------------------------------
# Equity timing: equity[t] is the book after bar t's own fills, marked at close t.
# A fill at bar t+1's open first shows at t+1 (no next-bar fill on the decision bar).
# ---------------------------------------------------------------------------

def _ohlc(rows, ts0=0, step=300):
    """rows: list of (open, high, low, close), one bar every ``step`` seconds."""
    a = np.asarray(rows, dtype="float64")
    n = len(rows)
    return pd.DataFrame({
        "ts": np.arange(ts0, ts0 + n * step, step, dtype="int64"),
        "open": a[:, 0], "high": a[:, 1], "low": a[:, 2], "close": a[:, 3],
        "volume": np.ones(n, dtype="int64"),
    })


class _Script(Strategy):
    """Runs ``actions[i](ctx)`` on the bar with row index i; records what it saw."""

    def __init__(self, actions):
        self.actions = actions
        self.seen = {}

    def precompute(self, df):
        df = df.copy()
        df["i"] = np.arange(len(df))
        return df

    def on_bar(self, ctx):
        i = int(ctx.bar.i)
        self.seen[i] = (ctx.position.qty, ctx.equity)
        if i in self.actions:
            self.actions[i](ctx)


def test_equity_at_decision_bar_is_not_moved_by_the_next_bars_fill():
    # Buy decided on bar 1; bar 2 gaps up 10 points and the buy fills at its open.
    # By hand: flat through bar 1's close -> 100,000 at bars 0 and 1. Bar 2: fill
    # 110 + 0.25 slip, fee 2.50, close 110 -> 100,000 - 2.50 - 0.25 * 20 = 99,992.50.
    nq = get_instrument("NQ")
    eng = Engine(nq, BacktestConfig(slippage_ticks=1.0, commission_per_side=2.5))
    df = _ohlc([(100, 100, 100, 100), (100, 100, 100, 100),
                (110, 110, 110, 110), (110, 110, 110, 110)])
    result = eng.run(df, _Script({1: lambda c: c.broker.buy(contracts=1)}))
    assert list(result.equity_curve.index) == [0, 300, 600, 900]
    assert list(result.equity_curve) == pytest.approx([100_000.0, 100_000.0,
                                                       99_992.5, 99_992.5])


def test_entry_fee_lands_on_the_fill_bar_not_the_decision_bar():
    nq = get_instrument("NQ")
    eng = Engine(nq, BacktestConfig(slippage_ticks=0.0, commission_per_side=5.0))
    df = _ohlc([(100, 101, 99, 100)] * 3)
    result = eng.run(df, _Script({0: lambda c: c.broker.buy(contracts=1)}))
    assert result.trades[0].entry_ts == 300
    # decision bar 0 still flat; the $5 fee first shows at the fill bar (ts 300)
    assert list(result.equity_curve) == pytest.approx([100_000.0, 99_995.0, 99_995.0])


def test_stop_exit_drop_lands_on_exit_bar_not_the_bar_before():
    # Entry fills at bar 1's open (100), stop 95 rests; bar 3 trades down to 90.
    nq = get_instrument("NQ")
    eng = Engine(nq, BacktestConfig(slippage_ticks=0.0, commission_per_side=0.0))
    df = _ohlc([(100, 101, 99, 100), (100, 101, 99, 100), (100, 101, 99, 100),
                (100, 101, 90, 92), (92, 93, 91, 92)])
    result = eng.run(df, _Script({0: lambda c: c.broker.buy(contracts=1, stop=95.0)}))
    tr = result.trades[0]
    assert tr.exit_ts == 900 and tr.exit_price == pytest.approx(95.0)
    eq = result.equity_curve
    assert eq.loc[600] == pytest.approx(100_000.0)          # before the exit bar
    assert eq.loc[900] == pytest.approx(100_000.0 - 5 * 20)  # at exit_ts
    assert list(eq) == pytest.approx([100_000.0, 100_000.0, 100_000.0, 99_900.0, 99_900.0])


def test_market_exit_gap_lands_on_exit_bar_not_the_decision_bar():
    # Long from bar 1's open (100); close() decided on bar 2 (close 100); bar 3 gaps
    # down to 90 and the exit fills there. The gap is bar 3's loss, not bar 2's.
    nq = get_instrument("NQ")
    eng = Engine(nq, BacktestConfig(slippage_ticks=0.0, commission_per_side=0.0))
    df = _ohlc([(100, 100, 100, 100), (100, 100, 100, 100), (100, 100, 100, 100),
                (90, 90, 90, 90), (90, 90, 90, 90)])
    result = eng.run(df, _Script({0: lambda c: c.broker.buy(contracts=1),
                                  2: lambda c: c.broker.close()}))
    assert result.trades[0].exit_ts == 900
    assert list(result.equity_curve) == pytest.approx(
        [100_000.0, 100_000.0, 100_000.0, 99_800.0, 99_800.0])


def test_on_bar_sees_the_fill_at_its_own_bars_open():
    # Unchanged by the equity-timing fix: on_bar(t) sees the book after t's open fill,
    # with ctx.equity marked at close t.
    nq = get_instrument("NQ")
    eng = Engine(nq, BacktestConfig(slippage_ticks=0.0, commission_per_side=0.0))
    df = _ohlc([(100, 100, 100, 100), (100, 100, 100, 105), (105, 105, 105, 105)])
    strat = _Script({0: lambda c: c.broker.buy(contracts=1)})
    eng.run(df, strat)
    assert strat.seen[0] == (0, pytest.approx(100_000.0))
    assert strat.seen[1] == (1, pytest.approx(100_000.0 + 5 * 20))


def test_flip_without_a_new_stop_leaves_no_stale_stop_in_the_engine():
    # Long 1 from bar 1's open (100) with stop 95; sell(2) on bar 1 flips short at bar
    # 2's open (100). The 95 sell-stop belonged to the long: as a buy-stop on the short
    # it would fire at once on bar 3 (high 101 >= 95). The short must ride to the end.
    nq = get_instrument("NQ")
    eng = Engine(nq, BacktestConfig(slippage_ticks=0.0, commission_per_side=0.0))
    df = _ohlc([(100, 101, 99, 100)] * 5)
    result = eng.run(df, _Script({0: lambda c: c.broker.buy(contracts=1, stop=95.0),
                                  1: lambda c: c.broker.sell(contracts=2)}))
    assert [(t.side, t.exit_ts, t.open_at_end) for t in result.trades] == [
        ("long", 600, False), ("short", None, True)]


def test_market_flip_fills_at_the_open_before_the_old_stop_in_the_engine():
    # Long 1 @100 stop 95; sell(2) decided on bar 1. Bar 2 opens 98, then trades to 90:
    # the flip closes the long at 98 and the short rides; the 95 stop never fills.
    nq = get_instrument("NQ")
    eng = Engine(nq, BacktestConfig(slippage_ticks=0.0, commission_per_side=0.0))
    df = _ohlc([(100, 101, 99, 100), (100, 101, 99, 100), (98, 99, 90, 92),
                (92, 93, 91, 92)])
    result = eng.run(df, _Script({0: lambda c: c.broker.buy(contracts=1, stop=95.0),
                                  1: lambda c: c.broker.sell(contracts=2)}))
    assert [(t.side, t.entry_price, t.exit_price, t.open_at_end)
            for t in result.trades] == [("long", 100.0, 98.0, False),
                                        ("short", 98.0, None, True)]


def test_entry_bar_stop_pierce_fires_in_the_engine():
    # Entry fills at bar 1's open 100 with stop 95; bar 1's low (90) trades through it.
    nq = get_instrument("NQ")
    eng = Engine(nq, BacktestConfig(slippage_ticks=0.0, commission_per_side=0.0))
    df = _ohlc([(100, 101, 99, 100), (100, 101, 90, 100),
                (100, 105, 99, 104), (104, 106, 103, 105)])
    result = eng.run(df, _Script({0: lambda c: c.broker.buy(contracts=1, stop=95.0)}))
    assert len(result.trades) == 1
    tr = result.trades[0]
    assert tr.entry_ts == 300 and tr.exit_ts == 300
    assert tr.exit_price == pytest.approx(95.0)
    assert tr.pnl == pytest.approx(-5 * 20)
    assert list(result.equity_curve) == pytest.approx(
        [100_000.0, 99_900.0, 99_900.0, 99_900.0])


def test_demo_configs_refuse_shorts_in_the_engine():
    # configs/nq.toml sets allow_short = false: a strategy's sell from flat opens nothing.
    from plutobt.config import load_config
    cfg = load_config(str(_REPO_ROOT / "configs" / "nq.toml"))
    assert cfg.allow_short is False
    df = _ohlc([(100, 101, 99, 100)] * 4)
    result = Engine(get_instrument("NQ"), cfg).run(
        df, _Script({1: lambda c: c.broker.sell(contracts=1)}))
    assert result.trades == []
    assert list(result.equity_curve) == pytest.approx([100_000.0] * 4)


def test_demo_strategy_is_long_only_on_every_demo_config():
    # allow_short changes nothing in the demo: run with shorts allowed, EMACross
    # never opens one, so the demo's numbers do not depend on the new refusal.
    from dataclasses import replace as dc_replace

    from plutobt.config import load_config
    from plutobt.strategies.ema_cross import EMACross
    for sym in ("nq", "es", "cl"):
        cfg = load_config(str(_REPO_ROOT / "configs" / f"{sym}.toml"))
        df = load_bars(str(_REPO_ROOT / "data" / "sample" / f"{sym}_5m.json"))
        inst = get_instrument(sym.upper())
        refused = Engine(inst, cfg).run(df, EMACross())
        allowed = Engine(inst, dc_replace(cfg, allow_short=True)).run(df, EMACross())
        assert not any(t.side == "short" for t in allowed.trades)
        assert [(t.entry_ts, t.exit_ts, t.pnl) for t in refused.trades] == [
            (t.entry_ts, t.exit_ts, t.pnl) for t in allowed.trades]
        assert list(refused.equity_curve) == list(allowed.equity_curve)


# ---------------------------------------------------------------------------
# Film-ready audit (2026-11-02): NaN is a warm-up only. A NaN after the first fully
# valid row used to skip on_bar silently on that bar, blocking exits (probe p06).
# ---------------------------------------------------------------------------

class _VolRatio(Strategy):
    """Buys once, exits when the close drops under 95; 'vr' is 0/0 on a zero-volume bar."""

    def precompute(self, df):
        df = df.copy()
        df["vr"] = df["volume"] / df["volume"].rolling(1).sum()
        return df

    def on_bar(self, ctx):
        if ctx.position.qty == 0 and not ctx.state.get("in"):
            ctx.state["in"] = True
            ctx.broker.buy(contracts=1)
        elif ctx.position.qty != 0 and ctx.bar.close < 95:
            ctx.broker.close()


_FALL = [(100, 101, 99, 100)] * 4 + [(90, 91, 80, 82), (82, 83, 70, 72),
                                     (72, 73, 60, 62), (62, 63, 61, 62)]


def test_a_nan_after_the_warm_up_is_refused_naming_column_and_ts():
    df = _ohlc(_FALL)
    df.loc[3:6, "volume"] = 0                        # vendor zero-volume bars: vr = 0/0
    nq = get_instrument("NQ")
    with pytest.raises(ValueError,
                       match=r"column 'vr' is NaN at row 3 \(ts 900\), after the warm-up"):
        Engine(nq, BacktestConfig(slippage_ticks=0.0)).run(df, _VolRatio())


def test_the_same_strategy_without_the_nan_exits_control():
    # Control: nonzero volume, so no NaN; the exit decided on bar 4 (close 82) fills at
    # bar 5's open, 82.
    nq = get_instrument("NQ")
    eng = Engine(nq, BacktestConfig(slippage_ticks=0.0, commission_per_side=0.0))
    (t,) = eng.run(_ohlc(_FALL), _VolRatio()).trades
    assert (t.exit_ts, t.exit_price, t.pnl) == (1500, 82.0, pytest.approx(-360.0))


def test_a_nan_from_the_precompute_is_a_frame_error_one_from_the_caller_is_not():
    from plutobt.broker import FrameError

    nq = get_instrument("NQ")
    df = _ohlc([(100, 101, 99, 100)] * 5)
    df["vwap"] = [100.0, 100.0, float("nan"), 100.0, 100.0]    # the caller's own column
    with pytest.raises(ValueError, match="column 'vwap' is NaN at row 2") as e:
        Engine(nq, BacktestConfig()).run(df, Strategy())
    assert not isinstance(e.value, FrameError)

    class Gappy(Strategy):
        def precompute(self, d):
            return d.assign(sig=np.where(d.index == 2, np.nan, 1.0))

    with pytest.raises(FrameError, match="column 'sig' is NaN at row 2"):
        Engine(nq, BacktestConfig()).run(df.drop(columns=["vwap"]), Gappy())


def test_a_leading_warm_up_of_different_lengths_still_skips_on_bar():
    # Unchanged: a 2-bar and a 4-bar warm-up; on_bar first runs on row 4.
    seen = []

    class Warm(Strategy):
        def precompute(self, d):
            d = d.copy()
            d["a"] = d["close"].rolling(3).mean()
            d["b"] = d["close"].rolling(5).mean()
            return d

        def on_bar(self, ctx):
            seen.append(int(ctx.bar.ts))

    nq = get_instrument("NQ")
    Engine(nq, BacktestConfig()).run(_ohlc([(100, 101, 99, 100)] * 7), Warm())
    assert seen == [1200, 1500, 1800]


def test_a_column_that_never_warms_up_never_runs_on_bar():
    # Unchanged: all warm-up, no fully valid row, so no error and no decisions.
    seen = []

    class Never(Strategy):
        def precompute(self, d):
            return d.assign(sig=float("nan"))

        def on_bar(self, ctx):
            seen.append(ctx.bar.ts)

    nq = get_instrument("NQ")
    result = Engine(nq, BacktestConfig()).run(_ohlc([(100, 101, 99, 100)] * 4), Never())
    assert seen == [] and len(result.equity_curve) == 4


# ---------------------------------------------------------------------------
# Film-ready audit (2026-11-02): clean ValueErrors for bad column names and ts types;
# non-positive prices refused unless the config allows them (probe p10).
# ---------------------------------------------------------------------------

class _Adds(Strategy):
    def __init__(self, name):
        self.name = name

    def precompute(self, df):
        df = df.copy()
        df[self.name] = 1.0
        return df


@pytest.mark.parametrize("name", ["ema-20", "my col", "_x", "class", 0])
def test_a_column_name_on_bar_cannot_read_is_refused(name):
    # Unchecked, namedtuple's own error ("Type names and field names must be valid
    # identifiers") came out of the middle of the run.
    from plutobt.broker import FrameError

    nq = get_instrument("NQ")
    df = _ohlc([(100, 101, 99, 100)] * 3)
    with pytest.raises(FrameError, match=f"column {name!r}") as e:
        Engine(nq, BacktestConfig()).run(df, _Adds(name))
    assert "Python identifier" in str(e.value)
    df[name] = 1.0                                         # the caller's own column
    with pytest.raises(ValueError, match=f"column {name!r}") as e:
        Engine(nq, BacktestConfig()).run(df, Strategy())
    assert not isinstance(e.value, FrameError)


def test_a_duplicate_column_name_is_refused():
    nq = get_instrument("NQ")
    df = _ohlc([(100, 101, 99, 100)] * 3)
    df = pd.concat([df, df[["close"]].rename(columns={"close": "x"}),
                    df[["close"]].rename(columns={"close": "x"})], axis=1)
    with pytest.raises(ValueError, match="column 'x' appears twice"):
        Engine(nq, BacktestConfig()).run(df, Strategy())


def test_a_none_ts_is_a_value_error_not_a_type_error():
    nq = get_instrument("NQ")
    df = _ohlc([(100, 101, 99, 100)] * 5).astype({"ts": object})
    df.loc[2, "ts"] = None
    with pytest.raises(ValueError, match=r"row 2 holds ts None \(NoneType\), not a number"):
        Engine(nq, BacktestConfig()).run(df, Strategy())


def test_a_datetime_ts_column_is_a_value_error():
    nq = get_instrument("NQ")
    df = _ohlc([(100, 101, 99, 100)] * 5)
    df["ts"] = pd.to_datetime(df["ts"], unit="s")
    with pytest.raises(ValueError, match=r"the 'ts' column holds datetimes \(datetime64"):
        Engine(nq, BacktestConfig()).run(df, Strategy())


@pytest.mark.parametrize("px", [0.0, -5.0])
def test_a_non_positive_price_frame_is_refused_by_default(px):
    # assert_positive_prices existed but was never called: a back-adjusted series that
    # went negative ran without a word.
    def never(df):
        raise AssertionError("precompute ran")

    class Never(Strategy):
        precompute = staticmethod(never)

    nq = get_instrument("NQ")
    df = _ohlc([(100, 101, 99, 100), (px, px + 1, px - 1, px), (100, 101, 99, 100)])
    with pytest.raises(ValueError, match="non-positive price in column 'open' at row 1") as e:
        Engine(nq, BacktestConfig()).run(df, Never())
    assert "allow_nonpositive_prices" in str(e.value)


def test_allow_nonpositive_prices_runs_a_frame_that_traded_below_zero():
    # CL settled at -37.63 on 2020-04-20; dollar P&L stays valid below zero.
    cl = get_instrument("CL")
    df = _ohlc([(10, 11, 9, 10), (10, 11, 9, 10), (-5, -4, -6, -5), (-5, -4, -6, -5)])
    eng = Engine(cl, BacktestConfig(allow_nonpositive_prices=True, slippage_ticks=0.0,
                                    commission_per_side=0.0, on_margin_call="record"))
    (t,) = eng.run(df, _Script({0: lambda c: c.broker.buy(contracts=1)})).trades
    assert t.entry_price == 10.0 and t.pnl == pytest.approx(-15.0 * cl.point_value)
