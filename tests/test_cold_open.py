"""Tests for the cold-open lookahead demo (examples/cold_open_lookahead.py).

The demo's point: the SAME "did this bar close green?" signal looks spectacular when bar
t is traded on bar t's own close (position = signal, impossible: the close does not
exist when the bar opens) and falls apart when it acts at the next bar's open
(position = signal.shift(1)), and further once the real engine charges NQ costs. These
pin the numbers the script prints on the regenerated NQ sample, so a drift in the data,
the metrics or the engine shows up here before it shows up on camera.

The truth line's Sharpe is +4.81 on this seed, not negative: 20 days of a random walk,
about 1.4 standard errors (sqrt(20412/1620) = 3.5) from zero. Nothing was tuned to move
it; the engine line, after costs, nets -$150.00.
"""

from __future__ import annotations

import importlib.util
import pathlib

import pandas as pd
import pytest

from plutobt.config import load_config
from plutobt.data import CT, load_bars

_ROOT = pathlib.Path(__file__).resolve().parents[1]
_DEMO = _ROOT / "examples" / "cold_open_lookahead.py"
_spec = importlib.util.spec_from_file_location("cold_open_lookahead", _DEMO)
cold_open = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cold_open)


@pytest.fixture(scope="module")
def res():
    return cold_open.run()


@pytest.fixture(scope="module")
def printed():
    import contextlib
    import io

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        cold_open.main()
    return buf.getvalue()


def test_the_lie_posts_the_sharpe_it_prints(res, printed):
    assert res["cheat_sharpe"] == pytest.approx(77.03, abs=0.01)
    assert f"annualized Sharpe = {res['cheat_sharpe']:>8.2f}" in printed
    assert res["cheat_hit"] > 0.95                      # nearly every bar held went up
    assert res["cheat_final"] == pytest.approx(1.25, abs=0.005)


def test_the_truth_is_a_fraction_of_the_lie(res, printed):
    # Not negative on this seed (see the module docstring): pinned as it comes out.
    assert res["honest_sharpe"] == pytest.approx(4.81, abs=0.01)
    assert res["cheat_sharpe"] > res["honest_sharpe"] + 70.0
    assert 0.5 < res["honest_hit"] < 0.56               # a coin, give or take
    assert res["honest_final"] == pytest.approx(1.01, abs=0.005)
    assert f"annualized Sharpe = {res['honest_sharpe']:>8.2f}" in printed


def test_the_engine_line_runs_the_same_signal_with_nq_costs(res, printed):
    eng = res["engine"]
    assert eng["commission_per_side"] == 2.5 and eng["slippage_ticks"] == 1.0
    assert eng["n_trades"] == 391
    assert eng["win_rate"] == pytest.approx(133 / 391)          # 133 winners
    assert eng["sharpe"] == pytest.approx(-0.07, abs=0.01)
    # 391 round trips: 782 fills x 1 tick ($5) of slippage and 391 x $5 commission.
    assert eng["slippage_usd"] == 3910.0 and eng["commission_usd"] == 1955.0
    assert eng["gross_pnl"] == 5715.0 and eng["net_pnl"] == -150.0
    assert "=== THE ENGINE: same signal through PlutoBT, NQ costs" in printed
    assert f"annualized Sharpe = {eng['sharpe']:>8.2f}    trades = 391" in printed
    assert "win rate = 34% (closed trades, net of costs)" in printed
    assert "= net -$150.00" in printed                  # a loss reads -$, never $-


def test_the_labels_say_what_the_math_does(printed):
    # The pandas lines count winning BARS, not trades, and are not the engine.
    assert printed.count("bar hit rate") == 2
    assert printed.count("win rate") == 1               # only the engine's closed trades
    assert "what PlutoBT enforces" not in printed
    assert "SAME bar's close" not in printed
    assert "(unlevered, no costs)" in printed
    assert "trade bar t on bar t's own close" in printed
    assert "act at the next bar's open" in printed
    assert cold_open.OUT_PNG.is_file()


def test_the_annualization_matches_the_data_it_runs_on():
    df = load_bars(str(cold_open.SAMPLE))
    days = pd.DatetimeIndex(pd.to_datetime(df["ts"], unit="s", utc=True)).tz_convert(CT)
    bars_per_day = set(pd.Series(days.date).value_counts())
    assert bars_per_day == {81}
    assert cold_open.PPY == 81 * 252 == load_config(str(cold_open.NQ_CONFIG)).periods_per_year


def test_the_toggle_is_one_shift():
    # Same inputs, the ONLY difference is signal.shift(1): prove the toggle is real.
    opens = pd.Series([100.0, 101.0, 100.0, 102.0, 103.0, 101.0], dtype="float64")
    close = opens.shift(-1).fillna(100.0)
    rets = (opens.shift(-1) / opens - 1.0).fillna(0.0)
    signal = (close > close.shift(1)).astype("float64")
    _, sr_cheat, r_cheat = cold_open.simulate(signal, rets, cheat=True)
    _, sr_honest, r_honest = cold_open.simulate(signal, rets, cheat=False)
    # cheat position == signal; honest position == signal shifted one bar
    assert not r_cheat.equals(r_honest)
    assert sr_cheat > sr_honest
