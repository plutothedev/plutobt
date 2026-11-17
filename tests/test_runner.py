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


def test_as_built_with_roll_meta_hours_line(df, here):
    out = _run(df, here, session="as_built", ledger=False)
    assert "Hours: continuous file run as built" in out.text
    assert len(out.result.equity_curve) == len(df)
