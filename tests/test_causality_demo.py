"""The on-camera causality demo (examples/causality_demo.py): the leak is refused by
name, the honest twin runs, and the refusal comes from the look-ahead guard itself."""
from __future__ import annotations

import contextlib
import importlib.util
import io
import pathlib
from dataclasses import replace

import pytest

from plutobt.lookahead import LookaheadError

_DEMO = pathlib.Path(__file__).resolve().parents[1] / "examples" / "causality_demo.py"
_spec = importlib.util.spec_from_file_location("causality_demo", _DEMO)
demo = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(demo)


@pytest.fixture(scope="module")
def printed():
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        demo.main()
    return buf.getvalue()


def test_the_leaky_strategy_is_refused_naming_its_column():
    refused = demo.run_leaky()
    assert isinstance(refused, LookaheadError)
    assert "'tomorrow_up'" in str(refused)


def test_the_refusal_is_the_guard_not_something_else():
    # Positive control: with the guard off the same leaky strategy runs, and the leak
    # pays (it trades on the next bar's close), so the refusal above was the guard.
    from plutobt.data import load_bars
    from plutobt.metrics import compute_metrics

    engine, ppy = demo._engine()
    assert engine.config.check_causality is True
    engine.config = replace(engine.config, check_causality=False)
    result = engine.run(load_bars(str(demo.SAMPLE)), demo.LeakyTomorrow())
    assert compute_metrics(result, periods_per_year=ppy)["sharpe"] > 20.0


def test_the_honest_twin_runs_and_reports_as_it_comes_out():
    result = demo.run_honest()
    m = result.metrics
    assert m["n_trades"] == 391
    assert m["sharpe"] == pytest.approx(-0.07, abs=0.01)
    lo, hi = m["sharpe_ci"]
    assert lo < 0.0 < hi


def test_the_printed_demo_shows_the_refusal_then_the_report(printed):
    assert "REFUSED" in printed
    assert "LookaheadError:" in printed and "'tomorrow_up'" in printed
    assert printed.index("LookaheadError") < printed.index("=== PlutoBT report: NQ ===")
    assert "IN-SAMPLE" in printed
    assert "Trades        = 391   (closed)" in printed
