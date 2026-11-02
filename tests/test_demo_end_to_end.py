import pandas as pd
import pytest

from plutobt.engine import BacktestResult


def test_run_symbol_completes_and_populates_result():
    from examples.run_demo import run_symbol

    result = run_symbol("NQ", data_path="data/sample/nq_5m.json",
                        config_path="configs/nq.toml")
    assert isinstance(result, BacktestResult)
    # equity curve is a populated, ts-indexed Series
    assert isinstance(result.equity_curve, pd.Series)
    assert len(result.equity_curve) > 0
    # metrics dict assembled, manifest pinned with the data hash
    assert isinstance(result.metrics, dict)
    assert "sharpe" in result.metrics
    assert result.metrics["periods_per_year"] == 20412.0
    assert "data_sha256" in result.manifest
    assert result.manifest["data_sha256"]  # non-empty hash
    # trades is a list (possibly empty on this sample, but the type is fixed)
    assert isinstance(result.trades, list)


def test_run_symbol_uses_per_instrument_costs():
    from examples.run_demo import run_symbol

    res_nq = run_symbol("NQ", data_path="data/sample/nq_5m.json",
                        config_path="configs/nq.toml")
    # The slippage used is recorded in the manifest config and equals the NQ config.
    assert res_nq.manifest["config"]["slippage_ticks"] == 1.0


def test_generalization_sweep_runs_all_three_symbols():
    from examples.run_demo import run_demo_all

    results = run_demo_all()
    assert set(results.keys()) == {"NQ", "ES", "CL"}
    for _symbol, result in results.items():
        assert isinstance(result, BacktestResult)
        assert len(result.equity_curve) > 0
        assert "sharpe" in result.metrics
    # Each symbol annualizes on its own RTH session: 5m bars/day x 252.
    assert results["NQ"].metrics["periods_per_year"] == 20412.0      # 81 x 252
    assert results["ES"].metrics["periods_per_year"] == 20412.0      # 81 x 252
    assert results["CL"].metrics["periods_per_year"] == 16632.0      # 66 x 252


@pytest.mark.parametrize("sym", ["nq", "es", "cl"])
def test_each_config_annualizes_on_its_instruments_rth_session(sym):
    # periods_per_year = 5-minute bars in the registry session x 252 trading days.
    from datetime import datetime

    from plutobt.config import load_config
    from plutobt.instruments import get_instrument

    inst = get_instrument(sym.upper())
    span = (datetime.strptime(inst.session_close_ct, "%H:%M")
            - datetime.strptime(inst.session_open_ct, "%H:%M"))
    bars_per_day = span.total_seconds() / 300
    assert load_config(f"configs/{sym}.toml").periods_per_year == bars_per_day * 252


def test_sweep_applies_distinct_per_instrument_costs():
    from examples.run_demo import run_demo_all

    results = run_demo_all()
    nq_slip = results["NQ"].manifest["config"]["slippage_ticks"]
    cl_slip = results["CL"].manifest["config"]["slippage_ticks"]
    # Costs do NOT auto-carry: CL ran with its own (wider) slippage.
    assert nq_slip == 1.0
    assert cl_slip == 3.0
    assert nq_slip != cl_slip


def test_main_prints_in_sample_labeled_reports(capsys):
    from examples.run_demo import main

    main()
    out = capsys.readouterr().out
    # One IN-SAMPLE-labeled report block per symbol.
    assert out.count("IN-SAMPLE") == 3
    assert "PlutoBT report: NQ" in out
    assert "PlutoBT report: ES" in out
    assert "PlutoBT report: CL" in out
    # The honesty disclaimer is present in the printed output.
    assert "not predictive of live results" in out
