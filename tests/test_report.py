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
        "n_trades": 1,
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


def test_format_report_warns_on_low_sample_and_omits_when_enough():
    from plutobt.metrics import MIN_TRADE_SAMPLE

    # Below the threshold -> the insufficient-sample warning is surfaced.
    low = _fake_result()
    low.metrics["n_trades"] = MIN_TRADE_SAMPLE - 1
    low_text = format_report(low, symbol="NQ")
    assert "insufficient sample" in low_text
    assert f"n_trades={MIN_TRADE_SAMPLE - 1}" in low_text

    # At/above the threshold -> no warning line.
    enough = _fake_result()
    enough.metrics["n_trades"] = MIN_TRADE_SAMPLE
    enough_text = format_report(enough, symbol="NQ")
    assert "insufficient sample" not in enough_text


def test_format_report_renders_real_compute_metrics_output():
    # §0.2: pipe REAL compute_metrics output THROUGH format_report (never a hand-built
    # dict). A tiny real Engine.run -> compute_metrics -> format_report must render.
    from plutobt.data import load_bars
    from plutobt.engine import Engine
    from plutobt.instruments import get_instrument
    from plutobt.metrics import compute_metrics
    from plutobt.strategies.ema_cross import EMACross

    instrument = get_instrument("NQ")
    cfg = load_config("configs/nq.toml")
    df = load_bars("data/sample/nq_5m.json")
    result = Engine(instrument, cfg).run(df, EMACross())
    ppy = result.manifest["config"]["periods_per_year"]
    result.metrics = compute_metrics(result, periods_per_year=ppy)

    # Renders without KeyError on the real key set and carries the IN-SAMPLE label.
    text = format_report(result, symbol="NQ")
    assert "IN-SAMPLE" in text
    assert "not predictive of live results" in text
    assert "PlutoBT report: NQ" in text


def _line(text: str, prefix: str) -> str:
    """The one report line starting with ``prefix`` (fails loudly if absent or doubled)."""
    hits = [ln for ln in text.splitlines() if ln.startswith(prefix)]
    assert len(hits) == 1, (prefix, hits)
    return hits[0]


def test_trades_line_counts_closed_trades_and_discloses_open_at_end():
    # One closed trade plus one position still open at the last bar. The trade stats
    # (win rate / PF / expectancy, n_trades) count only the closed one, so the Trades
    # line must too, and the open position gets its own disclosure line.
    res = _fake_result()
    res.trades.append(
        Trade(entry_ts=600, exit_ts=None, side="long", contracts=1,
              entry_price=103.0, exit_price=None, pnl=40.0, open_at_end=True)
    )
    res.metrics["n_trades"] = 1
    text = format_report(res, symbol="NQ")
    assert _line(text, "Trades ").split("=")[1].split()[0] == "1"
    assert _line(text, "Open at end").split("=")[1].strip().startswith("1 ")
    assert "excluded from trade stats" in _line(text, "Open at end")


def test_trades_line_agrees_with_low_sample_warning_on_real_es_run():
    # ES demo: 5 closed trades + 1 open at the end. It used to print "Trades = 6" beside
    # "n_trades=5".
    from examples.run_demo import run_symbol

    res = run_symbol("ES", data_path="data/sample/es_5m.json", config_path="configs/es.toml")
    n_open = sum(t.open_at_end for t in res.trades)
    assert n_open == 1
    text = format_report(res, symbol="ES")
    n_closed = res.metrics["n_trades"]
    assert _line(text, "Trades ").split("=")[1].split()[0] == str(n_closed)
    assert f"n_trades={n_closed} " in text
    assert _line(text, "Open at end").split("=")[1].strip().startswith(f"{n_open} ")
