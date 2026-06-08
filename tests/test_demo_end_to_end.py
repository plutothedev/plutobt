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
