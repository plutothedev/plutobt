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
    # Each config annualizes on its own 5m RTH session: bars/day * 252 (NQ/ES 81, CL 66).
    assert cfg.periods_per_year == {"configs/cl.toml": 16632.0}.get(path, 20412.0)
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
        "sharpe_raw": 1.234 / 20412.0 ** 0.5,
        "sharpe_se": 0.579,
        "sharpe_ci": (0.10, 2.37),
        "psr": 0.987,
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


def test_report_prints_the_annualization_factor_and_the_raw_sharpe():
    # Segment 8: "print the factor". The per-bar Sharpe, the factor and the annualized
    # Sharpe sit together so a reader can do the multiplication.
    text = format_report(_fake_result(), symbol="NQ")
    ann = _line(text, "Annualization")
    assert "sqrt(20412)" in ann and "142.871" in ann
    assert "0.00864" in _line(text, "Sharpe/bar")
    assert "142.871" in _line(text, "Sharpe        =")


def test_report_prints_sharpe_se_and_psr():
    # "print the error bar": the Lo (2002) SE beside the CI it builds, and PSR.
    text = format_report(_fake_result(), symbol="NQ")
    se = _line(text, "Sharpe SE")
    assert "0.579" in se and "Lo 2002" in se and "[0.100, 2.370]" in se
    assert "0.987" in _line(text, "PSR")


def test_report_flags_loudly_when_the_sharpe_ci_includes_zero():
    flag = "Sharpe CI includes 0: not statistically distinguishable from zero"
    assert flag not in format_report(_fake_result(), symbol="NQ")  # CI [0.10, 2.37]
    res = _fake_result()
    res.metrics["sharpe_ci"] = (-0.50, 2.97)
    text = format_report(res, symbol="NQ")
    assert "⚠ " + flag in text


@pytest.mark.parametrize(
    "pf,n_trades,expected",
    [
        (float("inf"), 3, "inf (no losing trades)"),
        (float("nan"), 0, "n/a (no trades)"),
        (float("nan"), 2, "n/a (no winning or losing trades)"),
        (2.5, 3, "2.500"),
    ],
)
def test_report_labels_a_non_finite_profit_factor(pf, n_trades, expected):
    res = _fake_result()
    res.metrics["profit_factor"] = pf
    res.metrics["n_trades"] = n_trades
    text = format_report(res, symbol="NQ")
    assert _line(text, "Profit factor").split("=", 1)[1].strip() == expected
    assert "nan" not in text
    assert not any(ln.rstrip().endswith("= inf") for ln in text.splitlines())


def test_report_labels_expectancy_with_no_trades():
    res = _fake_result()
    res.metrics["expectancy"] = float("nan")
    res.metrics["n_trades"] = 0
    text = format_report(res, symbol="NQ")
    assert _line(text, "Expectancy").split("=", 1)[1].strip() == "n/a (no trades)"


def test_real_demo_report_shows_factor_se_and_the_zero_flag():
    # The NQ demo's CI straddles zero, so the flag must fire on the real pipeline, and
    # the printed per-bar Sharpe times the printed factor is the printed Sharpe.
    from examples.run_demo import run_symbol

    res = run_symbol("NQ", data_path="data/sample/nq_5m.json", config_path="configs/nq.toml")
    m = res.metrics
    lo, hi = m["sharpe_ci"]
    assert lo < 0.0 < hi
    text = format_report(res, symbol="NQ")
    assert "Sharpe CI includes 0" in text
    assert f"{m['sharpe_se']:.3f}" in _line(text, "Sharpe SE")
    assert f"{m['psr']:.3f}" in _line(text, "PSR")
    assert f"{m['sharpe_raw']:.5f}" in _line(text, "Sharpe/bar")


def _flat_result():
    from plutobt.metrics import compute_metrics

    eq = pd.Series([100_000.0] * 50, index=range(0, 50 * 300, 300))
    res = BacktestResult(equity_curve=eq, trades=[], metrics={},
                         manifest={"data_sha256": "deadbeef", "version": "0.1.0"})
    res.metrics = compute_metrics(res, periods_per_year=20412.0)
    return res


def test_flat_zero_trade_report_prints_na_never_a_fake_number():
    # A flat curve used to print Sortino/Calmar = inf, Win rate = 0.000 and a
    # fake 95% CI [-15.58, 15.58] built from Lo's SE at zero volatility.
    text = format_report(_flat_result(), symbol="NQ")
    assert "inf" not in text and "nan" not in text
    assert "15.58" not in text
    se = _line(text, "Sharpe SE")
    assert "n/a" in se and "95% CI n/a" in se and "zero return volatility" in se
    assert "CI includes 0" not in text  # no interval, so no claim about it
    assert _line(text, "Sortino").split("=", 1)[1].strip().startswith("n/a")
    assert _line(text, "Calmar").split("=", 1)[1].strip().startswith("n/a")
    assert _line(text, "Win rate").split("=", 1)[1].strip() == "n/a (no trades)"
    assert _line(text, "Profit factor").split("=", 1)[1].strip() == "n/a (no trades)"


@pytest.mark.parametrize(
    "key,expected",
    [("sortino", "inf (no downside)"), ("calmar", "inf (no drawdown)")],
)
def test_report_labels_an_infinite_sortino_or_calmar(key, expected):
    res = _fake_result()
    res.metrics[key] = float("inf")
    text = format_report(res, symbol="NQ")
    assert _line(text, key.capitalize()).split("=", 1)[1].strip() == expected
