"""Margin calls in the engine (spec §5.2): maintenance is the floor, the call restores
to INITIAL, and the configured action runs (on_margin_call)."""
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from plutobt.config import BacktestConfig, load_config
from plutobt.data import load_bars
from plutobt.engine import BacktestResult, Engine
from plutobt.instruments import get_instrument
from plutobt.portfolio import MarginCall
from plutobt.strategy import Strategy
from tests.conftest import GoldenStrategy

_ROOT = Path(__file__).resolve().parents[1]

# Textbook crude example (Britannica, cited §5.2): initial $10,000, maintenance $7,000.
# A LOCAL fixture: the registry's CME-representative CL margins are not 10k/7k.
CL_TEXTBOOK = replace(get_instrument("CL"), initial_margin=10_000.0,
                      maintenance_margin=7_000.0)


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
    """Runs ``actions[i](ctx)`` on the bar with row index i."""

    def __init__(self, actions):
        self.actions = actions

    def precompute(self, df):
        df = df.copy()
        df["i"] = np.arange(len(df))
        return df

    def on_bar(self, ctx):
        i = int(ctx.bar.i)
        if i in self.actions:
            self.actions[i](ctx)


def _flat(px):
    return (px, px, px, px)


# The engine starts every run on $100,000. A 1-lot CL bought at 150.00 leaves equity
# 7,000 at 57.00 (exactly maintenance: no call) and 6,600 at 56.60 (call).
_CL_ROWS = [_flat(150.0), _flat(150.0), _flat(57.0), _flat(56.6), (56.0, 56.0, 56.0, 56.0),
            _flat(56.0)]


def _buy_one_at_bar0():
    return _Script({0: lambda c: c.broker.buy(contracts=1)})


def test_cl_call_restores_to_initial_3400_through_the_engine():
    """Equity 6,600 below maintenance 7,000: the call is 10,000 - 6,600 = 3,400, not 400."""
    eng = Engine(CL_TEXTBOOK, BacktestConfig(slippage_ticks=0.0, commission_per_side=0.0,
                                             on_margin_call="record"))
    result = eng.run(_ohlc(_CL_ROWS), _buy_one_at_bar0())
    assert result.equity_curve.loc[600] == pytest.approx(7_000.0)   # at maintenance: no call
    assert result.equity_curve.loc[900] == pytest.approx(6_600.0)
    first = result.margin_calls[0]
    assert isinstance(first, MarginCall)
    assert first.ts == 900
    assert first.contracts == 1
    assert first.equity == pytest.approx(6_600.0)
    assert first.maintenance == pytest.approx(7_000.0)
    assert first.initial == pytest.approx(10_000.0)
    assert first.amount == pytest.approx(3_400.0)
    assert first.amount != pytest.approx(400.0)     # restore to INITIAL, not maintenance


def test_record_keeps_the_position_and_logs_every_bar_in_breach():
    eng = Engine(CL_TEXTBOOK, BacktestConfig(slippage_ticks=0.0, commission_per_side=0.0,
                                             on_margin_call="record"))
    result = eng.run(_ohlc(_CL_ROWS), _buy_one_at_bar0())
    # Bars 3, 4, 5 close below maintenance (equity 6,600, 6,000, 6,000).
    assert [mc.ts for mc in result.margin_calls] == [900, 1200, 1500]
    assert [mc.amount for mc in result.margin_calls] == pytest.approx([3_400.0, 4_000.0,
                                                                       4_000.0])
    # No deposit is invented and nothing is sold: the position rides to the end.
    assert len(result.trades) == 1 and result.trades[0].open_at_end
    assert not result.trades[0].margin_liquidation
    assert result.equity_curve.iloc[-1] == pytest.approx(100_000.0 - 94.0 * 1_000)


def test_liquidate_is_the_default_and_exits_at_the_next_bars_open():
    assert BacktestConfig().on_margin_call == "liquidate"
    eng = Engine(CL_TEXTBOOK, BacktestConfig(slippage_ticks=0.0, commission_per_side=0.0))
    result = eng.run(_ohlc(_CL_ROWS), _buy_one_at_bar0())
    assert len(result.margin_calls) == 1
    assert result.margin_calls[0].ts == 900
    assert result.margin_calls[0].amount == pytest.approx(3_400.0)
    assert len(result.trades) == 1
    tr = result.trades[0]
    assert tr.margin_liquidation is True
    assert tr.open_at_end is False
    assert tr.exit_ts == 1200                       # the bar after the call
    assert tr.exit_price == pytest.approx(56.0)     # that bar's open, not the call bar's close
    assert tr.pnl == pytest.approx((56.0 - 150.0) * 1_000)
    assert result.equity_curve.iloc[-1] == pytest.approx(6_000.0)


def test_liquidation_pays_slippage_and_fees_like_any_market_exit():
    # Costs count toward the breach: entry 150.02 plus a $3.50 fee leaves equity
    # 100,000 - 93,020 - 3.50 = 6,976.50 at 57.00, under maintenance one bar earlier.
    eng = Engine(CL_TEXTBOOK, BacktestConfig(slippage_ticks=2.0, commission_per_side=3.5))
    result = eng.run(_ohlc(_CL_ROWS), _buy_one_at_bar0())
    call = result.margin_calls[0]
    assert call.ts == 600
    assert call.equity == pytest.approx(6_976.5)
    assert call.amount == pytest.approx(10_000.0 - 6_976.5)
    tr = result.trades[-1]
    assert tr.margin_liquidation is True
    assert tr.exit_ts == 900
    assert tr.exit_price == pytest.approx(56.6 - 2 * 0.01)   # next open less 2 ticks
    assert tr.commission == pytest.approx(7.0)


def test_no_call_while_equity_stays_at_or_above_maintenance():  # positive control
    rows = [_flat(150.0), _flat(150.0), _flat(57.0), _flat(57.0)]
    eng = Engine(CL_TEXTBOOK, BacktestConfig(slippage_ticks=0.0, commission_per_side=0.0))
    result = eng.run(_ohlc(rows), _buy_one_at_bar0())
    assert result.margin_calls == []
    assert len(result.trades) == 1 and result.trades[0].open_at_end
    assert not result.trades[0].margin_liquidation


def test_ordinary_exits_are_not_tagged_as_margin_liquidations():  # positive control
    rows = [_flat(150.0)] * 4
    eng = Engine(CL_TEXTBOOK, BacktestConfig(slippage_ticks=0.0, commission_per_side=0.0))
    result = eng.run(_ohlc(rows), _Script({0: lambda c: c.broker.buy(contracts=1),
                                           1: lambda c: c.broker.close()}))
    assert result.margin_calls == []
    assert [t.margin_liquidation for t in result.trades] == [False]


def test_liquidation_replaces_an_order_the_strategy_placed_on_the_call_bar():
    # On the call bar the strategy tries to add a contract; the forced exit wins.
    eng = Engine(CL_TEXTBOOK, BacktestConfig(slippage_ticks=0.0, commission_per_side=0.0))
    result = eng.run(_ohlc(_CL_ROWS), _Script({0: lambda c: c.broker.buy(contracts=1),
                                               3: lambda c: c.broker.buy(contracts=1)}))
    assert [(t.contracts, t.margin_liquidation, t.exit_ts) for t in result.trades] == [
        (1, True, 1200)]


def test_liquidation_survives_a_gap_break():
    # Gap mode cancels a pending order at a break, exits included; a forced
    # liquidation is not the strategy's order and fills at the next session's open.
    rows = _CL_ROWS[:4] + [(56.0, 56.0, 56.0, 56.0), _flat(56.0)]
    df = _ohlc(rows)
    df.loc[4:, "ts"] += 86_400                      # a gap after the call bar (row 3)
    eng = Engine(CL_TEXTBOOK, BacktestConfig(slippage_ticks=0.0, commission_per_side=0.0))
    result = eng.run(df, _buy_one_at_bar0())
    tr = result.trades[0]
    assert tr.margin_liquidation is True
    assert tr.exit_ts == int(df.ts.iloc[4])
    assert tr.exit_price == pytest.approx(56.0)


def test_a_stop_that_flattens_first_leaves_no_liquidation_to_fill():
    # The stop at 55.00 is gapped through at the next open (54.00): it exits there and
    # the forced exit is stale. One exit, a stop exit, not a liquidation.
    rows = _CL_ROWS[:4] + [_flat(54.0), _flat(54.0)]
    eng = Engine(CL_TEXTBOOK, BacktestConfig(slippage_ticks=0.0, commission_per_side=0.0))
    result = eng.run(_ohlc(rows), _Script({0: lambda c: c.broker.buy(contracts=1,
                                                                    stop=55.0)}))
    assert len(result.margin_calls) == 1
    assert [(t.exit_price, t.margin_liquidation) for t in result.trades] == [
        (pytest.approx(54.0), False)]


def test_short_position_breach_is_called_too():
    # Short 1 CL at 50.00; a rally to 143.40 leaves equity 6,600.
    rows = [_flat(50.0), _flat(50.0), _flat(143.4), _flat(143.4), _flat(143.4)]
    eng = Engine(CL_TEXTBOOK, BacktestConfig(slippage_ticks=0.0, commission_per_side=0.0))
    result = eng.run(_ohlc(rows), _Script({0: lambda c: c.broker.sell(contracts=1)}))
    assert result.margin_calls[0].ts == 600
    assert result.margin_calls[0].amount == pytest.approx(3_400.0)
    assert result.trades[0].side == "short" and result.trades[0].margin_liquidation


def test_on_margin_call_is_validated_and_loadable(tmp_path):
    with pytest.raises(ValueError, match="on_margin_call"):
        Engine(CL_TEXTBOOK, BacktestConfig(on_margin_call="deposit")).run(
            _ohlc(_CL_ROWS), Strategy())
    p = tmp_path / "cfg.toml"
    p.write_text('on_margin_call = "record"\n', encoding="utf-8")
    assert load_config(str(p)).on_margin_call == "record"


def test_result_margin_calls_default_empty():
    r = BacktestResult(equity_curve=pd.Series([1.0]), trades=[], metrics={}, manifest={})
    assert r.margin_calls == []


def test_golden_run_never_hits_a_margin_call():
    df = load_bars(str(_ROOT / "data" / "sample" / "nq_5m.json"))
    cfg = load_config(str(_ROOT / "configs" / "golden.toml"))
    result = Engine(get_instrument("NQ"), cfg).run(df, GoldenStrategy())
    assert result.margin_calls == []
    assert not any(t.margin_liquidation for t in result.trades)
