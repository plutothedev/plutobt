"""Report lines for a continuous series: rolls, roll cost, roll yield (spec §7.3, proof 3)."""
import pytest

from plutobt.metrics import compute_metrics
from plutobt.report import format_report
from tests.test_engine_roll import GAP, ROLL_ROW, run


def hold(side, n=1):
    def enter(c):
        (c.broker.buy if side == "long" else c.broker.sell)(n)
    return {2: enter, 15: lambda c: c.broker.close()}


def metrics(res):
    res.metrics = compute_metrics(res, periods_per_year=20412.0)
    return res.metrics


def test_roll_yield_long():
    _, _, res = run(hold("long"))
    m = metrics(res)
    assert m["rolls_taken"] == 1
    assert m["roll_cost_total"] == pytest.approx(2 * 2.5 + 2 * 5.0)
    assert m["roll_yield"] == pytest.approx(-GAP * 20.0) == -500.0


def test_roll_yield_short():
    _, _, res = run(hold("short", n=2))
    m = metrics(res)
    assert m["roll_yield"] == pytest.approx(GAP * 20.0 * 2) == 1000.0


def test_flat_through_roll_has_zero_rolls():
    _, _, res = run({2: lambda c: c.broker.buy(1), ROLL_ROW - 4: lambda c: c.broker.close()})
    m = metrics(res)
    assert (m["rolls_taken"], m["roll_cost_total"], m["roll_yield"]) == (0, 0.0, 0.0)


def test_report_roll_lines():
    _, _, res = run(hold("long"))
    metrics(res)
    text = format_report(res, "NQ")
    assert "Rolls taken   = 1" in text
    assert "Roll cost     = 15.00" in text
    assert "Roll yield    = -500.00" in text
    assert ("Roll yield: the jump an unadjusted front-month chart would show at each "
            "roll, which is carry, not P&L.") in text


def test_metrics_without_roll_meta_have_no_roll_keys():
    from plutobt.config import BacktestConfig
    from plutobt.data import load_bars
    from plutobt.engine import Engine
    from plutobt.instruments import get_instrument
    from plutobt.strategy import Strategy

    res = Engine(get_instrument("NQ"), BacktestConfig()).run(
        load_bars("data/sample/nq_5m.json").iloc[:200].reset_index(drop=True), Strategy())
    m = compute_metrics(res, periods_per_year=20412.0)
    assert "rolls_taken" not in m and "roll_yield" not in m


def test_roll_yield_with_an_unregistered_instrument():
    """Review #7: compute_metrics looked the point value up in REGISTRY by symbol, so a
    custom Instrument crashed it. The engine's own instrument prices the roll yield."""
    from dataclasses import replace

    from plutobt.engine import Engine
    from plutobt.instruments import get_instrument
    from tests.test_engine_roll import CFG, Script, series

    df, meta = series()
    feed = replace(get_instrument("NQ"), symbol="NQ_MYFEED")
    res = Engine(feed, CFG).run(df, Script(hold("long")), roll_meta=meta)
    m = metrics(res)
    assert m["roll_yield"] == pytest.approx(-GAP * 20.0)
    (ev,) = res.rolls
    assert ev.roll_yield == pytest.approx(-GAP * 20.0)
    assert "Roll yield" in format_report(res, "NQ_MYFEED")
