"""Flat reruns, walk-forward and purged CV (3.0 spec §8, §9)."""
import numpy as np
import pandas as pd
import pytest

from plutobt.config import BacktestConfig
from plutobt.instruments import get_instrument
from plutobt.oos import _pick, longest_trade_bars, purged_cv, rerun_window, walk_forward
from plutobt.study import _context
from tests.study_fixtures import CFG, Coin, Hold, StartsLate, walk

NQ = get_instrument("NQ")


def ctx_for(df, cls, settings, cfg=CFG, roll_meta=None):
    return _context(cls, settings, df, NQ, cfg, roll_meta)


def test_rerun_window_pnl_equals_equity_difference():
    df = walk(400, drift=0.5, seed=1)
    run = rerun_window(Hold, {"side": 1}, df, NQ, CFG, None, 100, 200)
    assert run.pnl == pytest.approx(run.equity.iloc[-1] - run.equity.iloc[0])
    assert run.equity.index[0] == df.ts.iloc[100] and run.equity.index[-1] == df.ts.iloc[200]
    assert run.equity.iloc[0] == 100_000.0
    # long 1 NQ from bar 101's open to bar 200's close
    assert run.pnl == pytest.approx((df.close.iloc[200] - df.open.iloc[101]) * 20.0)


def test_exit_charge_on_an_open_window_end():
    df = walk(400, seed=2)
    cfg = BacktestConfig(commission_per_side=2.5, slippage_ticks=1.0, periods_per_year=20412.0,
                         check_causality=False)
    run = rerun_window(Hold, {"side": 1}, df, NQ, cfg, None, 50, 150)
    assert run.open_at_end and run.exit_charge == pytest.approx(2.5 + 1.0 * 5.0)
    flat = rerun_window(Hold, {"side": 0}, df, NQ, cfg, None, 50, 150)
    assert not flat.open_at_end and flat.exit_charge == 0.0


def test_flat_rerun_books_a_roll_inside_the_window():
    from tests.test_engine_roll import CFG as ROLL_CFG
    from tests.test_engine_roll import ROLL_ROW, series

    df, meta = series()
    run = rerun_window(Hold, {"side": 1}, df, NQ, ROLL_CFG, meta, 2, 20)
    assert len(run.rolls) == 1 and run.rolls[0].ts == int(df.ts.iloc[ROLL_ROW])
    assert run.rolls[0].cost == pytest.approx(2 * 2.5 + 2 * 1.0 * 5.0)


def test_stitch_equals_sum_of_windows_less_charges():
    df = walk(1200, drift=0.3, seed=3)
    cfg = BacktestConfig(commission_per_side=2.5, slippage_ticks=1.0, periods_per_year=20412.0,
                         check_causality=False)
    ctx = ctx_for(df, Hold, [{"side": 1}, {"side": -1}], cfg=cfg)
    wf = walk_forward(ctx, 5, anchored=False)
    total = sum(w["pnl"] - w["exit_charge"] for w in wf.windows)
    assert wf.equity.iloc[-1] - 100_000.0 == pytest.approx(total)
    assert wf.open_windows == sum(w["open_at_end"] for w in wf.windows) == 5


def test_rolling_and_anchored_pick_by_train_slice():
    df = walk(1200, drift=0.5, seed=4)
    ctx = ctx_for(df, Hold, [{"side": -1}, {"side": 1}])
    for anchored in (False, True):
        wf = walk_forward(ctx, 5, anchored=anchored)
        assert [w["params"] for w in wf.windows] == [{"side": 1}] * 5
        assert not wf.winner_changed


def test_regime_change_degrades_oos():
    n = 1800
    drift = np.where(np.arange(n) < n // 2, 1.0, -1.0)
    df = walk(n, drift=drift, seed=5)
    ctx = ctx_for(df, Hold, [{"side": 1}, {"side": -1}])
    wf = walk_forward(ctx, 5, anchored=False)
    assert wf.winner_changed
    assert wf.sharpe < wf.is_sharpe_mean


def test_steady_edge_holds_oos():
    df = walk(1800, drift=1.0, seed=6)
    ctx = ctx_for(df, Hold, [{"side": 1}, {"side": -1}])
    wf = walk_forward(ctx, 5, anchored=False)
    assert wf.sharpe > 0.5 * wf.is_sharpe_mean > 0
    assert wf.psr > 0.95


def test_nan_train_sharpes_are_skipped_not_picked():
    df = walk(1200, drift=0.5, seed=7)
    # start_bar 0 trades everywhere; start_bar 600 is flat (NaN Sharpe) in early windows
    ctx = ctx_for(df, StartsLate, [{"start_bar": 600}, {"start_bar": 0}])
    wf = walk_forward(ctx, 5, anchored=False)
    assert wf.windows[0]["params"] == {"start_bar": 0}
    late = ctx_for(df, StartsLate, [{"start_bar": 600}, {"start_bar": 700}])
    with pytest.raises(ValueError, match="no setting traded"):
        walk_forward(late, 5, anchored=False)


def test_pick_skips_nan_and_takes_the_first_max():
    r = pd.DataFrame({"a": np.zeros(50), "b": np.r_[np.ones(25), -np.ones(25)] * 0.01 + 0.002,
                      "c": np.r_[np.ones(25), -np.ones(25)] * 0.01 + 0.002})
    idx, sr = _pick(r, np.arange(50))
    assert idx == 1 and sr > 0


def test_embargo_default_is_longest_trade():
    df = walk(1200, drift=0.2, seed=8)
    ctx = ctx_for(df, StartsLate, [{"start_bar": 900}, {"start_bar": 1000}])
    # start_bar 900 holds from bar 901 to the end (open at end): 1199 - 901 = 298 bars
    assert longest_trade_bars(ctx) == 298
    assert longest_trade_bars(ctx_for(df, Hold, [{"side": 0}, {"side": 0}])) == 1
    # purged_cv uses it when embargo is None and it fits
    short = ctx_for(df, Coin, [{"seed": 1, "bias": 0.5}, {"seed": 2, "bias": 0.5}])
    short_cv = purged_cv(short, 6, None)
    assert 1 < short_cv.embargo == longest_trade_bars(short) < 50 and not short_cv.capped


def test_default_embargo_is_capped_for_a_strategy_that_never_exits():
    """Always in the market: the longest trade is the whole history, which would leave the
    first fold nothing to train on. The default is cut to keep 30 training rows and the
    result says it was capped; an embargo passed in is never capped."""
    df = walk(1200, drift=0.2, seed=8)
    steady = ctx_for(df, Hold, [{"side": 1}, {"side": -1}])
    assert longest_trade_bars(steady) == 1198
    cv = purged_cv(steady, 6, None)
    assert cv.capped and cv.embargo == 1199 - 1199 // 6 - 30
    assert not purged_cv(steady, 6, 10).capped


def test_planted_leak_needs_the_embargo():
    """A setting with no edge of its own earns a steady edge in the 40 rows right after the
    test block (as a position carried out of the test block would). Without the embargo
    it is picked on that leak (Sharpe 0.166 against 0.082); with a 40-row embargo it is
    not (-0.033 against 0.101)."""
    rows = 600
    rng = np.random.default_rng(0)
    r = pd.DataFrame({"honest": rng.normal(0.0001, 0.001, rows),
                      "leaky": rng.normal(0.0, 0.001, rows)})
    test = range(100, 200)
    r.loc[200:239, "leaky"] += 0.003
    train_no_embargo = np.array([i for i in range(rows) if i not in test])
    train_embargo = np.array([i for i in train_no_embargo if not 200 <= i < 240])
    assert _pick(r, train_no_embargo)[0] == 1
    assert _pick(r, train_embargo)[0] == 0


def test_fold_before_training_is_flagged_not_real_time():
    df = walk(1200, drift=0.3, seed=9)
    cv = purged_cv(ctx_for(df, Hold, [{"side": 1}, {"side": -1}]), 6, 5)
    flags = [f["not_real_time"] for f in cv.folds]
    assert flags == [True] * 5 + [False]
    assert len(cv.sharpes) == 6 and cv.spread == pytest.approx(np.std(cv.sharpes))
