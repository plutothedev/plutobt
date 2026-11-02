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
    assert set(table["n_trades"]) == {73}
    assert set(table["n_open"]) == {1}


# Every row as it prints (review, 2026-11-02: only the 1-tick row was pinned, through
# the demo run): final equity, Sharpe, PF, expectancy, closed trades, open at the end.
ROWS = {
    0.0: (102_907.50, 2.374, 1.401, 37.60),
    1.0: (102_172.50, 1.793, 1.269, 27.60),
    2.0: (101_437.50, 1.212, 1.158, 17.60),
    4.0: (99_967.50, 0.055, 0.981, -2.40),
}


@pytest.mark.parametrize("ticks", ROWS)
def test_every_row_is_pinned(table, ticks):
    equity, sharpe, pf, expectancy = ROWS[ticks]
    row = table.loc[ticks]
    assert row["final_equity"] == pytest.approx(equity, abs=0.005)
    assert row["sharpe"] == pytest.approx(sharpe, abs=0.0005)
    assert row["profit_factor"] == pytest.approx(pf, abs=0.0005)
    assert row["expectancy"] == pytest.approx(expectancy, abs=0.005)
    assert (row["n_trades"], row["n_open"], row["fills"]) == (73, 1, 147)


def test_each_tick_costs_one_tick_value_per_fill(table):
    # 73 round trips are 146 fills, plus the open position's entry: 147 fills. A tick
    # of slippage on each is 147 x $5 = $735 of final equity, the step between rows
    # (68 x 2 x $5 = $680 never matched the $685 step on screen before the fix).
    eq = table["final_equity"]
    step = 147 * 5.0
    assert eq[0.0] - eq[1.0] == pytest.approx(step)
    assert eq[1.0] - eq[2.0] == pytest.approx(step)
    assert eq[2.0] - eq[4.0] == pytest.approx(2 * step)


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
    # The open position is on screen, so the fills behind the step can be counted.
    assert out.count("73 + 1 open") == 4
    lines = out.splitlines()
    step = lines.index("Each tick of slippage per side costs $5.00 x 147 fills")
    assert lines[step + 1] == "  (73 round trips x 2 + 1 open entry) = $735.00 of final equity."
    assert '"+ 1 open": a position still open at the end' in out


def test_each_sharpe_carries_the_engines_standard_error(table):
    # Final pre-film audit, item 6: the Sharpes printed bare, with an SE of about 3.5.
    # The SE is compute_metrics' own (the 1-tick row is the NQ demo run).
    from examples.run_demo import run_symbol

    res = run_symbol("NQ", data_path="data/sample/nq_5m.json", config_path="configs/nq.toml")
    assert table.loc[1.0, "sharpe_se"] == res.metrics["sharpe_se"]
    assert table.loc[1.0, "sharpe_ci"] == res.metrics["sharpe_ci"]


def _print(table, monkeypatch):
    monkeypatch.setattr(demo, "run", lambda: table)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        demo.main()
    return buf.getvalue()


def test_the_printed_table_shows_the_se_and_a_computed_verdict(table, monkeypatch):
    out = _print(table, monkeypatch)
    header = next(ln for ln in out.splitlines() if "slippage" in ln and "Sharpe" in ln)
    assert header.index("Sharpe") < header.index("SE") < header.index("PF")
    for _, row in table.iterrows():
        assert f"{row['sharpe']:.3f}  {row['sharpe_se']:>6.3f}" in out
    # Every CI on this sample straddles zero, and the footer says so from the CIs.
    assert all(lo <= 0.0 <= hi for lo, hi in table["sharpe_ci"])
    assert ("Every Sharpe above has a 95% CI that includes 0: none is "
            "distinguishable from zero.") in out


def test_the_verdict_is_computed_not_hardcoded(table, monkeypatch):
    t = table.copy()
    t["sharpe_ci"] = [(0.5, 4.0), (-1.0, 3.0), (-2.0, 2.0), (-3.0, 1.0)]
    out = _print(t, monkeypatch)
    assert "none is distinguishable from zero" not in out
    assert "95% CI excludes 0 at: 0 ticks" in out
