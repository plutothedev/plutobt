"""The on-camera cost sweep (examples/cost_sweep.py): cost rises, money falls, and the
strategy itself is never touched."""
from __future__ import annotations

import contextlib
import importlib.util
import io
import pathlib

import numpy as np
import pytest

_DEMO = pathlib.Path(__file__).resolve().parents[1] / "examples" / "cost_sweep.py"
_spec = importlib.util.spec_from_file_location("cost_sweep", _DEMO)
demo = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(demo)


@pytest.fixture(scope="module")
def table():
    return demo.run()


def test_it_sweeps_slippage_0_1_2_4_with_commission_fixed(table):
    assert list(table.index) == [0.0, 1.0, 2.0, 4.0]
    assert set(table["commission_per_side"]) == {2.5}       # configs/nq.toml


def test_final_equity_never_rises_as_cost_rises(table):
    eq = table["final_equity"].to_numpy()
    assert (np.diff(eq) <= 0.0).all(), eq
    assert eq[0] > eq[-1]                                   # and cost does bite


def test_the_other_columns_degrade_with_cost_too(table):
    for col in ("sharpe", "profit_factor", "expectancy"):
        assert (np.diff(table[col].to_numpy()) <= 0.0).all(), col


def test_it_is_a_cost_sweep_not_a_parameter_sweep(table):
    from plutobt.strategies.ema_cross import EMACross

    assert "NOT a parameter sweep" in demo.__doc__
    # The demo strategy keeps the parameters it ships with on every row.
    assert (EMACross.fast_span, EMACross.slow_span, EMACross.stop_ticks) == (20, 50, 20.0)
    assert set(table["n_trades"]) == {68}


def test_the_one_tick_row_is_the_nq_demo_run(table):
    from examples.run_demo import run_symbol

    res = run_symbol("NQ", data_path="data/sample/nq_5m.json", config_path="configs/nq.toml")
    assert table.loc[1.0, "final_equity"] == float(res.equity_curve.iloc[-1])
    assert table.loc[1.0, "sharpe"] == res.metrics["sharpe"]


def test_the_printed_table_says_what_it_is(table, monkeypatch):
    monkeypatch.setattr(demo, "run", lambda: table)     # print the swept table, once
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        demo.main()
    out = buf.getvalue()
    assert "COST SWEEP (not a parameter sweep)" in out
    assert "IN-SAMPLE" in out
    for ticks in ("0 ticks", "1 ticks", "2 ticks", "4 ticks"):
        assert ticks in out
