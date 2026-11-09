"""Engine.run(trade_from_ts=...): earlier bars warm the indicators, no order before it
(3.0 spec §10). None keeps 2.0.1 exactly."""
import pandas as pd
import pytest

from plutobt.config import BacktestConfig
from plutobt.data import load_bars
from plutobt.engine import Engine
from plutobt.instruments import get_instrument
from plutobt.roll import slice_continuous
from plutobt.strategy import Strategy
from tests.conftest import REPO_ROOT, GoldenStrategy

NQ = get_instrument("NQ")
CFG = BacktestConfig(commission_per_side=2.5, slippage_ticks=1.0)


@pytest.fixture(scope="module")
def bars():
    return load_bars(REPO_ROOT / "data" / "sample" / "nq_5m.json")


def run(df, strategy=None, cfg=CFG, **kw):
    return Engine(NQ, cfg).run(df, strategy or GoldenStrategy(), **kw)


def test_no_orders_and_flat_equity_before(bars):
    start = int(bars.ts.iloc[800])
    res = run(bars, trade_from_ts=start)
    assert res.trades and all(t.entry_ts > start for t in res.trades)
    before = res.equity_curve[res.equity_curve.index < start]
    assert len(before) == 800 and (before == 100_000.0).all()


def test_first_bar_equals_a_plain_run(bars):
    plain = run(bars)
    same = run(bars, trade_from_ts=int(bars.ts.iloc[0]))
    pd.testing.assert_series_equal(plain.equity_curve, same.equity_curve)
    assert [vars(t) for t in plain.trades] == [vars(t) for t in same.trades]
    extra = dict(same.manifest)
    assert extra.pop("trade_from_ts") == int(bars.ts.iloc[0])
    assert extra == plain.manifest


class Records(GoldenStrategy):
    seen: dict = {}

    def on_bar(self, ctx):
        type(self).seen[int(ctx.bar.ts)] = float(ctx.bar.ema_s)
        super().on_bar(ctx)


def test_indicators_warm_on_the_prefix(bars):
    Records.seen = {}
    run(bars, Records(), cfg=BacktestConfig(check_causality=False))
    full = dict(Records.seen)
    Records.seen = {}
    start = int(bars.ts.iloc[900])
    run(bars.iloc[:1200].reset_index(drop=True), Records(),
        cfg=BacktestConfig(check_causality=False), trade_from_ts=start)
    assert min(Records.seen) == start
    assert all(Records.seen[ts] == full[ts] for ts in Records.seen)


def test_none_is_2_0_1(bars):
    res = run(bars, trade_from_ts=None)
    assert "trade_from_ts" not in res.manifest
    pd.testing.assert_series_equal(res.equity_curve, run(bars).equity_curve)


def test_ts_not_in_data_refused(bars):
    with pytest.raises(ValueError, match=r"trade_from_ts.*first bar.*last bar"):
        run(bars, trade_from_ts=int(bars.ts.iloc[10]) + 1)
    with pytest.raises(ValueError, match="trade_from_ts"):
        run(bars, trade_from_ts=int(bars.ts.iloc[-1]) + 300)


class HoldLong(Strategy):
    def on_bar(self, ctx):
        if ctx.position.qty == 0:
            ctx.broker.buy(contracts=1)


def test_continuous_series_with_trade_from():
    from tests.test_engine_roll import CFG as ROLL_CFG
    from tests.test_engine_roll import ROLL_ROW, series

    df, meta = series()
    end = int(df.ts.iloc[25])
    cut, cut_meta = slice_continuous(df, meta, end_ts=end)
    # trading starts after the roll: the position is opened on the new contract, so no roll
    after = run(cut, HoldLong(), cfg=ROLL_CFG, roll_meta=cut_meta,
                trade_from_ts=int(df.ts.iloc[ROLL_ROW + 2]))
    assert after.rolls == []
    # trading starts before the roll: the held position rolls once, as in a plain run
    before = run(cut, HoldLong(), cfg=ROLL_CFG, roll_meta=cut_meta,
                 trade_from_ts=int(df.ts.iloc[3]))
    assert len(before.rolls) == 1
