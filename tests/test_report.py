import math

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
    assert "0.00863719" in _line(text, "Sharpe/bar")      # 6 significant digits
    assert "142.871" in _line(text, "Sharpe        =")


def _printed_sharpe_parts(text):
    """(per-bar, factor, annualized) as printed: the three numbers a viewer multiplies."""
    bar = _line(text, "Sharpe/bar").split("=", 1)[1].split()[0]
    factor = _line(text, "Annualization").rsplit("=", 1)[1].split()[0]
    ann_line = _line(text, "Sharpe        =")
    ann = ann_line.split("=", 1)[1].split()[0]
    assert f"Sharpe/bar x {factor})" in ann_line          # the same factor, both lines
    return bar, factor, ann


def _check_the_arithmetic(text):
    bar, factor, ann = _printed_sharpe_parts(text)
    # per-bar x factor, from the PRINTED digits, is the printed Sharpe to its last digit
    assert format(float(bar) * float(factor), ".3f") == ann, (bar, factor, ann)
    digits = bar.replace("-", "").replace(".", "").lstrip("0")
    assert len(digits) >= 6, bar                          # at least 6 significant


@pytest.mark.parametrize("sharpe", [1.234, 1.7927259306154941, -0.0678878045, 77.03,
                                    0.0004999, -12.3456789, 1e-7, 250.0])
def test_printed_per_bar_sharpe_times_printed_factor_is_the_printed_sharpe(sharpe):
    # Review finding (2026-11-02): Sharpe/bar printed to 5 decimals (0.01255) times
    # 142.871 is 1.793..., which need not round to the printed Sharpe. The on-screen
    # arithmetic must check out to the last printed digit.
    res = _fake_result()
    res.metrics["sharpe"] = sharpe
    res.metrics["sharpe_raw"] = sharpe / 20412.0 ** 0.5
    _check_the_arithmetic(format_report(res, symbol="NQ"))


@pytest.mark.parametrize("sym", ["NQ", "ES", "CL"])
def test_the_demo_reports_arithmetic_checks_out(sym):
    from examples.run_demo import run_symbol

    low = sym.lower()
    res = run_symbol(sym, data_path=f"data/sample/{low}_5m.json",
                     config_path=f"configs/{low}.toml")
    _check_the_arithmetic(format_report(res, symbol=sym))


def test_report_prints_sharpe_se_and_psr():
    # "print the error bar": the SE beside the CI it builds, and PSR, each labelled
    # with what it does and does not adjust for (the SE used to say "Lo 2002", the
    # normal-only formula, beside a PSR that adjusts for skew and kurtosis).
    text = format_report(_fake_result(), symbol="NQ")
    se = _line(text, "Sharpe SE")
    assert "0.579" in se and "[+0.100, +2.370]" in se
    assert "(iid, skew/kurtosis-adjusted; no autocorrelation adjustment)" in se
    assert "Lo 2002" not in se
    psr_line = _line(text, "PSR")
    assert "0.987" in psr_line and "skew/kurtosis-adjusted" in psr_line


@pytest.mark.parametrize("lo,shown,flagged", [
    (0.0004, "+0.0004", False),      # used to print as 0.000 with no flag
    (-0.0004, "-0.0004", True),      # used to print as -0.000
    (0.0, "+0.000", True),           # exactly 0 is included
    (-3e-12, "-3.000e-12", True),
    (0.1, "+0.100", False),
])
def test_the_ci_never_shows_a_rounded_zero_without_saying_which_side(lo, shown, flagged):
    # Review finding (2026-11-02): the flag compared unrounded values but the CI printed
    # to 3 decimals, so [0.000, ...] could appear with no flag. The bound now prints
    # signed, with digits enough to show it is not zero.
    flag = "Sharpe CI includes 0"
    res = _fake_result()
    res.metrics["sharpe_ci"] = (lo, 2.37)
    text = format_report(res, symbol="NQ")
    assert f"95% CI [{shown}, +2.370]" in _line(text, "Sharpe SE")
    assert (flag in text) is flagged


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
    bar, _, ann = _printed_sharpe_parts(text)
    assert float(bar) == pytest.approx(m["sharpe_raw"], rel=1e-5)
    assert ann == f"{m['sharpe']:.3f}"


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
    # The Sharpe itself is 0/0 here: it used to print 0.000 next to "SE n/a".
    assert _line(text, "Sharpe        =").split("=", 1)[1].strip() == (
        "n/a (zero return volatility)")
    assert _line(text, "Sharpe/bar").split("=", 1)[1].strip() == (
        "n/a (zero return volatility)")
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


def test_a_constant_return_curve_reports_zero_volatility_not_a_huge_sharpe():
    # Review finding (2026-11-02): a constant 0.1%/bar curve has a float-noise std of
    # about 1e-16, which the exact std > 0 check turned into a Sharpe of about 1e15.
    import numpy as np

    from plutobt.metrics import compute_metrics

    eq = pd.Series(100_000.0 * np.cumprod(np.r_[1.0, np.full(50, 1.001)]),
                   index=range(0, 51 * 300, 300))
    res = BacktestResult(equity_curve=eq, trades=[], metrics={},
                         manifest={"data_sha256": "deadbeef", "version": "0.1.0"})
    res.metrics = compute_metrics(res, periods_per_year=20412.0)
    text = format_report(res, symbol="NQ")
    for prefix in ("Sharpe/bar", "Sharpe        =", "Sharpe SE"):
        assert "n/a (zero return volatility)" in _line(text, prefix), prefix
    assert _line(text, "PSR").split("=", 1)[1].strip().startswith(
        "n/a (zero return volatility)")
    assert "e+" not in text and "nan" not in text


def test_an_all_scratch_run_has_no_win_rate():
    # Review finding (2026-11-02): every closed trade a scratch (neither won nor lost)
    # printed "Win rate = 0.000", as if every trade had lost.
    res = _fake_result()
    res.metrics.update(win_rate=0.0, profit_factor=float("nan"), n_trades=3)
    text = format_report(res, symbol="NQ")
    assert _line(text, "Win rate").split("=", 1)[1].strip() == (
        "n/a (no winning or losing trades)")
    # A real 0% (losers, no winners) still prints its number.
    res.metrics.update(profit_factor=0.0)
    assert _line(format_report(res, symbol="NQ"), "Win rate").endswith("= 0.000")


# ---------------------------------------------------------------------------
# Final pre-film audit (2026-11-02): labels that say what the numbers are.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("sym,days", [("NQ", "20"), ("ES", "20"), ("CL", "20")])
def test_calmar_is_not_annualized_from_a_month(sym, days):
    # Calmar annualizes the mean bar return (x periods_per_year): from 20 trading days
    # that is a 12.6x extrapolation (CL implied -145.8%/yr). Under a year it prints n/a
    # and says why; the value stays in the metrics dict.
    from examples.run_demo import run_symbol

    low = sym.lower()
    res = run_symbol(sym, data_path=f"data/sample/{low}_5m.json",
                     config_path=f"configs/{low}.toml")
    assert math.isfinite(res.metrics["calmar"])
    text = format_report(res, symbol=sym)
    assert _line(text, "Calmar").split("=", 1)[1].strip() == (
        f"n/a (needs >= 1 year; this run covers {days} trading days)")


def test_calmar_prints_its_number_once_the_run_covers_a_year():
    res = _fake_result()                     # 4 bars
    res.metrics["periods_per_year"] = 4.0    # 4 bars is a year here
    text = format_report(res, symbol="NQ")
    assert _line(text, "Calmar").split("=", 1)[1].strip() == "0.950"
    res.metrics["periods_per_year"] = 5.0    # 4 of 5: under a year
    text = format_report(res, symbol="NQ")
    assert _line(text, "Calmar").split("=", 1)[1].strip() == (
        "n/a (needs >= 1 year; this run covers 201.6 trading days)")


def _dd_result(values):
    from plutobt.metrics import compute_metrics

    eq = pd.Series([float(v) for v in values], index=range(0, len(values) * 300, 300))
    res = BacktestResult(equity_curve=eq, trades=[], metrics={},
                         manifest={"data_sha256": "deadbeef", "version": "0.1.0"})
    res.metrics = compute_metrics(res, periods_per_year=20412.0)
    return res


@pytest.mark.parametrize("values,flagged", [
    ([100, 110, 90, 95], True),             # never regains 110
    ([100, 110, 90, 111], False),           # recovered on the last bar
    ([100, 110, 80, 111, 105], False),      # deepest recovered; a later shallow one did not
    ([100, 101, 102], False),               # no drawdown at all
])
def test_an_unrecovered_drawdown_duration_says_so(values, flagged):
    text = format_report(_dd_result(values), symbol="NQ")
    dur = _line(text, "Max DD duration")
    assert ("(not recovered by the last bar)" in dur) is flagged


def test_max_drawdown_recovered_helper():
    from plutobt.metrics import max_drawdown_recovered

    assert max_drawdown_recovered(pd.Series([100.0, 110.0, 90.0, 95.0])) is False
    assert max_drawdown_recovered(pd.Series([100.0, 110.0, 90.0, 110.0])) is True
    assert max_drawdown_recovered(pd.Series([100.0, 110.0, 80.0, 111.0, 105.0])) is True
    assert max_drawdown_recovered(pd.Series([], dtype=float)) is True


def test_psr_label_does_not_claim_a_probability_of_the_true_sharpe():
    psr_line = _line(format_report(_fake_result(), symbol="NQ"), "PSR")
    assert "P[true Sharpe > 0]" not in psr_line
    assert ("(Probabilistic Sharpe Ratio vs 0: 1 minus the one-sided p-value; "
            "skew/kurtosis-adjusted, Bailey & Lopez de Prado)") in psr_line


def test_expectancy_shows_its_unit():
    text = format_report(_fake_result(), symbol="NQ")
    assert _line(text, "Expectancy").split("=", 1)[1].strip() == (
        "60.000   ($ per closed trade, net of costs)")


def test_low_sample_warning_names_every_trade_based_stat():
    res = _fake_result()
    res.metrics["n_trades"] = 5
    warn = _line(format_report(res, symbol="NQ"), "⚠ insufficient sample")
    for stat in ("profit factor", "expectancy", "win rate", "Avg R"):
        assert stat in warn, stat


def test_the_hash_line_says_what_it_hashes():
    line = _line(format_report(_fake_result(), symbol="NQ"), "data_sha256")
    assert "deadbeef" in line
    assert "(of the loaded bars as CSV, not the file)" in line


def test_the_documented_hash_recipe_reproduces_the_manifest():
    # ONBOARDING's one-liner: sha256 of load_bars(path).to_csv(index=False).
    import hashlib

    from examples.run_demo import run_symbol

    from plutobt.data import load_bars

    path = "data/sample/nq_5m.json"
    res = run_symbol("NQ", data_path=path, config_path="configs/nq.toml")
    recipe = hashlib.sha256(load_bars(path).to_csv(index=False).encode()).hexdigest()
    assert recipe == res.manifest["data_sha256"]
