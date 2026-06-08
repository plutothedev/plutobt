"""Tests for the cold-open lookahead demo (examples/cold_open_lookahead.py).

The demo's whole point: the SAME signal looks great when you cheat (fill the
signal bar's own close) and collapses when you fill honestly (next bar). If that
gap ever disappears, the cold open is broken — so we pin it.
"""

from __future__ import annotations

import importlib.util
import pathlib

import pandas as pd

_DEMO = pathlib.Path(__file__).resolve().parents[1] / "examples" / "cold_open_lookahead.py"
_spec = importlib.util.spec_from_file_location("cold_open_lookahead", _DEMO)
cold_open = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cold_open)


def test_cheat_sharpe_dwarfs_honest_sharpe():
    res = cold_open.run()
    # The lie posts a big positive Sharpe; the truth does not come close.
    assert res["cheat_sharpe"] > 5.0
    assert res["cheat_sharpe"] > res["honest_sharpe"] + 10.0


def test_cheat_win_rate_is_inflated():
    res = cold_open.run()
    # "Buy every green bar at its own close" wins almost always; honest does not.
    assert res["cheat_win"] > 0.9
    assert res["honest_win"] < res["cheat_win"]


def test_the_toggle_is_one_shift():
    # Same inputs, the ONLY difference is signal.shift(1) — prove the toggle is real.
    close = pd.Series([100.0, 101.0, 100.0, 102.0, 103.0, 101.0], dtype="float64")
    rets = close.pct_change().fillna(0.0)
    signal = (close > close.shift(1)).astype("float64")
    _, sr_cheat, r_cheat = cold_open.simulate(signal, rets, cheat=True)
    _, sr_honest, r_honest = cold_open.simulate(signal, rets, cheat=False)
    # cheat position == signal; honest position == signal shifted one bar
    assert not r_cheat.equals(r_honest)
    assert sr_cheat > sr_honest
