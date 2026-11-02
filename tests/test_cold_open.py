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
    assert (f"annualized Sharpe = {eng['sharpe']:>8.2f} (SE {eng['sharpe_se']:.2f})"
            "    trades = 391") in printed
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


def test_nobody_is_flat_overnight_and_the_docstring_says_so():
    # Review finding (2026-11-02): the docstring said pending orders were cancelled
    # across the overnight gap, read as "flat overnight". Nobody is: the engine keeps
    # an open position (and its exit) across a break, cancelling only entries, and
    # each pandas line holds whenever its signal is on at a session's last bar. These
    # pin the counts the docstring states.
    import json

    from plutobt.engine import Engine
    from plutobt.instruments import get_instrument

    rows = json.loads(cold_open.SAMPLE.read_text(encoding="utf-8"))
    ts = pd.Series([r[0] for r in rows])
    close = pd.Series([r[4] for r in rows], dtype="float64")
    signal = (close > close.shift(1)).astype("float64")
    truth = signal.shift(1).fillna(0.0)
    last = (ts.shift(-1) - ts) > 300                       # a session's last bar
    first = last.shift(1, fill_value=False)                # the next session's first
    assert int(last.sum()) == 19
    assert int((truth[last] == 1.0).sum()) == 10           # truth holds across 10 gaps
    assert int((signal[last] == 1.0).sum()) == 9           # the lie across 9
    opened = (truth == 1.0) & first & (truth.shift(1) == 0.0)
    assert int(opened.sum()) == 5                          # entries the engine cancels

    df = load_bars(str(cold_open.SAMPLE))
    cfg = load_config(str(cold_open.NQ_CONFIG))
    result = Engine(get_instrument("NQ"), cfg).run(df, cold_open.GreenBar())
    day = pd.to_datetime(df["ts"], unit="s", utc=True).dt.tz_convert(CT).dt.date
    day_of = dict(zip(df["ts"], day, strict=True))
    overnight = [t for t in result.trades
                 if t.exit_ts is not None and day_of[t.entry_ts] != day_of[t.exit_ts]]
    assert len(overnight) == 10                            # the engine holds 10 overnight
    engine_gaps = {x for x in ts[last] for t in overnight if t.entry_ts <= x < t.exit_ts}
    assert engine_gaps == set(ts[last & (truth == 1.0)])  # over the truth line's 10
    firsts = set(ts[first])
    assert not any(t.entry_ts in firsts for t in result.trades)   # and opens none there

    doc = " ".join(cold_open.__doc__.split())
    assert "cancelled across the overnight gap" not in doc
    for claim in ("holds 10 positions overnight", "on 10 of the 19 overnight gaps",
                  "the lie on 9", "5 times"):
        assert claim in doc, claim


def test_every_sharpe_carries_the_engines_standard_error(res, printed):
    # Final pre-film audit, item 6: three Sharpes printed bare, each with an SE of
    # about 3.5. The SE comes from metrics.sharpe_stats (compute_metrics' own), and
    # the verdict under them is computed from the CIs, not written in.
    from plutobt.metrics import sharpe_stats

    for key in ("cheat", "honest"):
        assert f"{res[key + '_sharpe']:>8.2f} (SE {res[key + '_se']:.2f})" in printed
    eng = res["engine"]
    assert f"{eng['sharpe']:>8.2f} (SE {eng['sharpe_se']:.2f})" in printed
    # The pandas lines' SE is for the very returns their Sharpe is computed from.
    _, sr, rets = cold_open.simulate(*cold_open.inputs(), cheat=True)
    st = sharpe_stats(rets, cold_open.PPY)
    assert st["sharpe"] == sr == res["cheat_sharpe"] and st["sharpe_se"] == res["cheat_se"]
    assert res["cheat_ci"][0] > 0.0                     # the lie looks "significant"
    for ci in (res["honest_ci"], eng["sharpe_ci"]):
        assert ci[0] <= 0.0 <= ci[1]
    assert "  95% CI excludes 0: the lie" in printed
    assert ("  95% CI includes 0 (not distinguishable from zero): the truth, the engine"
            in printed)


def test_the_saved_chart_path_is_repo_relative(printed):
    # Final pre-film audit, item 7: it printed the full local path on camera.
    assert "Saved before/after chart -> examples/output/cold_open_lie_vs_truth.png" in printed
    assert str(_ROOT) not in printed
