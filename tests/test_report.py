import pandas as pd
import pytest

from plutobt.config import load_config
from plutobt.engine import BacktestConfig, BacktestResult
from plutobt.portfolio import Trade
from plutobt.report import format_report


@pytest.mark.parametrize(
    "path,slippage",
    [
        ("configs/nq.toml", 1.0),
        ("configs/es.toml", 1.0),
        ("configs/cl.toml", 3.0),
    ],
)
def test_per_instrument_cost_configs_load(path, slippage):
    cfg = load_config(path)
    assert isinstance(cfg, BacktestConfig)
    assert cfg.slippage_ticks == slippage
    # Costs do NOT auto-carry: each config sets its own commission per side.
    assert cfg.commission_per_side > 0.0
    # All three share the 5m-RTH annualization basis: 81 bars/day * 252 = 20412.
    assert cfg.periods_per_year == 20412.0
    assert cfg.allow_short is False  # EMACross demo is long-only


def test_costs_do_not_auto_carry_between_instruments():
    nq = load_config("configs/nq.toml")
    cl = load_config("configs/cl.toml")
    # CL is a wider, more expensive contract: distinct slippage and commission.
    assert cl.slippage_ticks != nq.slippage_ticks
    assert cl.commission_per_side != nq.commission_per_side


def _fake_result():
    eq = pd.Series([100000.0, 100120.0, 100050.0, 100300.0], index=[0, 300, 600, 900])
    trades = [
        Trade(entry_ts=0, exit_ts=600, side="buy", contracts=1,
              entry_price=100.0, exit_price=106.0, pnl=120.0, r_multiple=1.5),
    ]
    metrics = {
        "sharpe": 1.234,
        "sharpe_ci": (0.10, 2.37),
        "sortino": 1.80,
        "calmar": 0.95,
        "profit_factor": 2.50,
        "expectancy": 60.0,
        "win_rate": 1.0,
        "max_drawdown_depth": -0.0007,
        "max_drawdown_duration": 1,
        "periods_per_year": 20412.0,
        "T": 4,
    }
    manifest = {
        "data_sha256": "deadbeef",
        "version": "0.1.0",
        "periods_per_year": 20412.0,
        "config": {"slippage_ticks": 1.0},
    }
    return BacktestResult(equity_curve=eq, trades=trades, metrics=metrics, manifest=manifest)


def test_format_report_is_labeled_in_sample_and_lists_metrics():
    text = format_report(_fake_result(), symbol="NQ")
    assert "IN-SAMPLE" in text
    assert "not predictive of live results" in text
    assert "NQ" in text
    # the formatter reads these exact metric keys
    assert "Sharpe" in text and "1.234" in text
    assert "Max DD depth" in text
    assert "Max DD duration" in text
    assert "T = 4" in text
    assert "data_sha256" in text
    assert "deadbeef" in text


def test_format_report_reads_required_keys():
    # Removing a required key must raise (no silent blanks) -> proves the keys are read.
    res = _fake_result()
    del res.metrics["max_drawdown_depth"]
    with pytest.raises(KeyError):
        format_report(res, symbol="NQ")
