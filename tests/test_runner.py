"""run_backtest: one call from a frame and a strategy class to a report, ledger line, chart."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from plutobt.config import BacktestConfig
from plutobt.data import load_bars
from plutobt.lookahead import LookaheadError
from plutobt.runner import RunOutcome, run_backtest
from plutobt.strategies.ema_cross import EMACross
from plutobt.strategy import Strategy

ROOT = Path(__file__).resolve().parent.parent
SAMPLE = str(ROOT / "data" / "sample" / "nq_5m.json")


@pytest.fixture
def df():
    return load_bars(SAMPLE)


@pytest.fixture
def here(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _run(df, here, **kw):
    kw.setdefault("data_note", "sample nq_5m.json")
    kw.setdefault("ledger", str(here / "trials.jsonl"))
    kw.setdefault("chart_dir", here / "runs")
    return run_backtest(df, "NQ", EMACross, {}, **kw)


def test_run_prints_report_and_extras(df, here):
    out = _run(df, here)
    assert isinstance(out, RunOutcome)
    assert "IN-SAMPLE" in out.text
    assert "Data: sample nq_5m.json" in out.text
    assert out.chart.exists() and out.chart.suffix == ".png"
    assert out.chart.name.startswith("NQ_EMACross_")


def test_run_default_costs_line(df, here):
    out = _run(df, here)
    assert out.default_costs is True
    assert ("Costs: default, representative public retail costs, not a live account: "
            "$2.50/side commission, 1 tick(s) slippage") in out.text


def test_run_writes_one_ledger_line_and_counts_tries(df, here):
    a = _run(df, here)
    assert a.tries == 1
    assert (here / "trials.jsonl").read_text().count("\n") == 1
    b = _run(df, here)
    assert b.tries == 1  # same code, settings and bars is the same try
    assert (here / "trials.jsonl").read_text().count(chr(10)) == 2
    c = run_backtest(df, "NQ", EMACross, {"fast_span": 10}, data_note="x",
                     ledger=str(here / "trials.jsonl"), chart_dir=here / "runs")
    assert c.tries == 2
    assert "Tries of this strategy on these bars: 2 (ledger " in c.text


def test_run_ledger_off(df, here):
    out = _run(df, here, ledger=False)
    assert out.tries is None
    assert "Ledger: off" in out.text
    assert not (here / "trials.jsonl").exists()


class _Leaky(Strategy):
    def precompute(self, df):
        df["peek"] = df["close"].shift(-1)
        return df

    def on_bar(self, ctx):
        pass


def test_run_refuses_lookahead_strategy(df, here):
    with pytest.raises(LookaheadError, match="peek"):
        run_backtest(df, "NQ", _Leaky, {}, data_note="x",
                     ledger=str(here / "trials.jsonl"), chart_dir=here / "runs")
    assert not (here / "trials.jsonl").exists()
    assert not (here / "runs").exists() or not list((here / "runs").glob("*.png"))


def test_report_states_session_and_basis(df, here):
    out = _run(df, here)
    assert "Hours: regular trading hours only" in out.text
    assert "Bars per year: 20412 (measured from these bars: median bars per trading day" in out.text
    assert out.periods_per_year == 20412.0


def _full_day(df):
    base = int(df["ts"].iloc[0])
    n = 288 * 4
    rng = np.random.default_rng(1)
    close = 18000 + np.cumsum(rng.normal(0, 3, n))
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) + 1.0
    low = np.minimum(open_, close) - 1.0
    return pd.DataFrame({"ts": base + 300 * np.arange(n, dtype="int64"), "open": open_,
                         "high": high, "low": low, "close": close,
                         "volume": np.full(n, 100, dtype="int64")})


def test_session_all_keeps_eth_rows(here):
    full = _full_day(load_bars(SAMPLE))
    rth = _run(full, here, session="rth", ledger=False)
    allh = _run(full, here, session="all", ledger=False)
    assert len(allh.result.equity_curve) > len(rth.result.equity_curve)
    assert "Hours: full day (--session all)" in allh.text
    assert "Hours: regular trading hours only" in rth.text
    assert "session" not in full.columns


def test_config_periods_per_year_wins(df, here):
    cfg = BacktestConfig(commission_per_side=1.0, slippage_ticks=1.0, periods_per_year=20412.0)
    out = _run(df, here, config=cfg, cost_note="from configs/nq.toml")
    assert "(from your config)" in out.text
    assert "Costs: from configs/nq.toml" in out.text
    assert out.default_costs is False


def test_daily_bars_not_filtered(here):
    n = 400
    rng = np.random.default_rng(2)
    close = 18000 + np.cumsum(rng.normal(0, 30, n))
    open_ = np.r_[close[0], close[:-1]]
    d = pd.DataFrame({"ts": 1700000000 + 86400 * np.arange(n, dtype="int64"),
                      "open": open_, "high": np.maximum(open_, close) + 5,
                      "low": np.minimum(open_, close) - 5, "close": close,
                      "volume": np.full(n, 1000, dtype="int64")})
    out = _run(d, here, ledger=False)
    assert "Hours: daily bars, no session filter" in out.text
    assert len(out.result.equity_curve) == n


def test_as_built_session_hours_line(df, here):
    out = _run(df, here, session="as_built", ledger=False)
    assert "Hours: continuous file run as built" in out.text
    assert len(out.result.equity_curve) == len(df)


def _eth_only():
    # 72 five-minute bars at 02:00 CT on a weekday (ETH)
    base = 1772461800 - 1772461800 % 86400 + 8 * 3600
    n = 72
    close = 18000 + np.arange(n) * 0.5
    return pd.DataFrame({"ts": base + 300 * np.arange(n, dtype="int64"), "open": close,
                         "high": close + 1, "low": close - 1, "close": close,
                         "volume": np.full(n, 10, dtype="int64")})


@pytest.mark.parametrize("with_config", [False, True])
def test_eth_only_input_is_refused_clearly(here, with_config):
    cfg = (BacktestConfig(commission_per_side=1.0, slippage_ticks=1.0,
                          periods_per_year=20412.0) if with_config else None)
    with pytest.raises(ValueError, match="no regular-trading-hours bars in this data; "
                                         "use --session all to keep the full day"):
        _run(_eth_only(), here, config=cfg)
    assert not (here / "trials.jsonl").exists()
    assert not (here / "runs").exists()


def test_unknown_session_is_refused(df, here):
    with pytest.raises(ValueError, match="session must be one of"):
        _run(df, here, session="RTH")


def test_chart_leaves_pyplot_alone(df, here):
    import matplotlib.pyplot as plt

    fig = plt.figure()
    backend = plt.get_backend()
    try:
        _run(df, here, ledger=False)
        assert plt.fignum_exists(fig.number)
        assert plt.get_backend() == backend
    finally:
        plt.close(fig)


def test_two_quick_runs_make_two_charts(df, here):
    a = _run(df, here, ledger=False)
    b = _run(df, here, ledger=False)
    assert a.chart != b.chart
    assert len(list((here / "runs").glob("*.png"))) == 2


def test_roll_meta_frame_is_never_filtered(here, monkeypatch):
    from tests.test_engine_roll import series

    def boom(*a, **k):
        raise AssertionError("a roll_meta frame must not be session-tagged")

    monkeypatch.setattr("plutobt.runner.tag_sessions", boom)

    frame, meta = series()
    out = run_backtest(frame, "NQ", EMACross, {"fast_span": 3, "slow_span": 6},
                       roll_meta=meta, session="rth", data_note="x", ledger=False,
                       chart_dir=here / "runs")
    assert len(out.result.equity_curve) == len(frame)


# -- whole-branch review fixes (3.1) -----------------------------------------------

def test_hours_line_names_the_window_it_filtered_on(df, here):
    out = _run(df, here, ledger=False)
    assert ("Hours: regular trading hours only (08:30-15:15 America/Chicago, by bar open "
            "time)") in out.text


def test_one_try_has_no_tries_line_more_name_deflate(df, here):
    one = _run(df, here)
    assert "python -m plutobt.deflate" not in one.text
    out = run_backtest(df, "NQ", EMACross, {"fast_span": 5}, data_note="x",
                       ledger=str(here / "trials.jsonl"), chart_dir=here / "runs")
    assert out.tries == 2
    line = [ln for ln in out.text.splitlines() if ln.startswith("This strategy has ")]
    assert len(line) == 1
    assert line[0].startswith("This strategy has 2 tries on these bars; the report above "
                              "treats this run alone. To account for the tries: python -m "
                              "plutobt.deflate --sharpe ")
    assert "--periods-per-year 20412 --returns " in line[0] and line[0].endswith("--trials 2")


def test_config_periods_per_year_far_from_the_bars_warns(df, here):
    cfg = BacktestConfig(commission_per_side=1.0, slippage_ticks=1.0, periods_per_year=6804.0)
    out = _run(df, here, config=cfg, cost_note="from my.toml", ledger=False)
    assert ("WARNING: your config says 6804 bars per year; these bars measure 20412. The "
            "Sharpe uses 6804.") in out.text
    close = BacktestConfig(commission_per_side=1.0, slippage_ticks=1.0,
                           periods_per_year=20412.0 * 1.15)
    assert "WARNING: your config says" not in _run(df, here, config=close, ledger=False).text
