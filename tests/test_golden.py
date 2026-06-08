"""Golden-run regression tests: manifest, equity, metrics, label (M8)."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from plutobt.config import BacktestConfig, load_config
from plutobt.data import load_bars
from plutobt.engine import Engine
from plutobt.instruments import get_instrument
from plutobt.metrics import compute_metrics

from tests.conftest import GoldenStrategy

SAMPLE = Path(__file__).resolve().parents[1] / "data" / "sample" / "nq_5m.json"
GOLDEN_CFG = Path(__file__).resolve().parents[1] / "configs" / "golden.toml"
EXPECTED = Path(__file__).resolve().parents[1] / "configs" / "golden_expected.json"

# Stated tolerance (§5.10): float64 ewm/reduction order varies across
# numpy/pandas/BLAS/OS, so compare within tolerance, never bit-equality.
RTOL = 1e-9
ATOL = 1e-6


def _run():
    df = load_bars(str(SAMPLE))
    inst = get_instrument("NQ")
    cfg = BacktestConfig(commission_per_side=2.5, slippage_ticks=1.0)
    return Engine(inst, cfg).run(df, GoldenStrategy())


def _golden_result():
    df = load_bars(str(SAMPLE))
    inst = get_instrument("NQ")
    cfg = load_config(str(GOLDEN_CFG))
    return Engine(inst, cfg).run(df, GoldenStrategy())


# ---------------------------------------------------------------------------
# M8.5: roll metadata keys in manifest
# ---------------------------------------------------------------------------


def test_manifest_carries_data_sha256():
    result = _run()
    assert "data_sha256" in result.manifest
    assert isinstance(result.manifest["data_sha256"], str)
    assert len(result.manifest["data_sha256"]) == 64  # hex sha256


def test_manifest_records_roll_metadata_keys():
    result = _run()
    for key in ("roll_trigger", "splice_method", "adjustment_method", "as_of_date"):
        assert key in result.manifest, f"manifest missing roll key: {key}"


def test_manifest_roll_metadata_documents_v1_defaults():
    result = _run()
    m = result.manifest
    assert m["adjustment_method"] == "backwards_difference"  # additive Panama (§5.6 default)
    assert m["splice_method"] == "none_single_expiry"        # synthetic single expiry
    assert m["roll_trigger"] == "none_single_expiry"
    assert m["as_of_date"] == "2026-06-07"


# ---------------------------------------------------------------------------
# M8.6: golden.toml config
# ---------------------------------------------------------------------------


def test_golden_config_exists_and_loads():
    assert GOLDEN_CFG.is_file(), f"missing golden config: {GOLDEN_CFG}"
    cfg = load_config(str(GOLDEN_CFG))
    assert isinstance(cfg, BacktestConfig)


def test_golden_config_pins_explicit_costs_not_defaults():
    cfg = load_config(str(GOLDEN_CFG))
    # Explicitly pinned, NOT BacktestConfig() defaults (§5.10):
    assert cfg.commission_per_side == 2.5      # representative NQ per-side (§9)
    assert cfg.slippage_ticks == 1.0           # 1 tick/side minimum (§5.4)
    assert cfg.periods_per_year == 20412.0     # 81 bars/day * 252 (§5.7)
    assert cfg.carry_overnight is False        # cancel-by-default (§5.5)
    assert cfg.allow_short is True


# ---------------------------------------------------------------------------
# M8.7: frozen golden expectations within assert_allclose tolerance
# ---------------------------------------------------------------------------


def test_golden_expected_file_exists():
    assert EXPECTED.is_file(), (
        f"missing golden expectations: {EXPECTED}; run scripts/freeze_golden.py"
    )


def test_golden_equity_curve_within_tolerance():
    result = _golden_result()
    exp = json.loads(EXPECTED.read_text(encoding="utf-8"))
    got = result.equity_curve
    exp_ts = np.asarray(exp["equity_ts"], dtype=np.int64)
    exp_eq = np.asarray(exp["equity_values"], dtype=np.float64)
    # index (ts) must match exactly; values within tolerance.
    np.testing.assert_array_equal(np.asarray(got.index, dtype=np.int64), exp_ts)
    np.testing.assert_allclose(got.to_numpy(dtype=np.float64), exp_eq,
                               rtol=RTOL, atol=ATOL)


def test_golden_metrics_within_tolerance():
    result = _golden_result()
    cfg = load_config(str(GOLDEN_CFG))
    metrics = compute_metrics(result, periods_per_year=cfg.periods_per_year)
    exp = json.loads(EXPECTED.read_text(encoding="utf-8"))["metrics"]
    for key, exp_val in exp.items():
        if isinstance(exp_val, (int, float)) and not isinstance(exp_val, bool):
            np.testing.assert_allclose(
                float(metrics[key]), float(exp_val),
                rtol=RTOL, atol=ATOL, err_msg=f"metric '{key}' drifted",
            )


def test_golden_manifest_matches_frozen():
    result = _golden_result()
    exp = json.loads(EXPECTED.read_text(encoding="utf-8"))["manifest"]
    m = result.manifest
    assert m["data_sha256"] == exp["data_sha256"]   # data file unchanged
    assert m["adjustment_method"] == exp["adjustment_method"]
    assert m["splice_method"] == exp["splice_method"]
    assert m["roll_trigger"] == exp["roll_trigger"]
    assert m["as_of_date"] == exp["as_of_date"]
    assert m["config"]["commission_per_side"] == 2.5
    assert m["config"]["slippage_ticks"] == 1.0


# ---------------------------------------------------------------------------
# M8.8: IN-SAMPLE label regression test
# ---------------------------------------------------------------------------


def test_metrics_first_key_is_in_sample_label():
    result = _golden_result()
    cfg = load_config(str(GOLDEN_CFG))
    metrics = compute_metrics(result, periods_per_year=cfg.periods_per_year)
    first_key = next(iter(metrics))
    label = metrics[first_key]
    assert "IN-SAMPLE" in str(label), (
        "first metrics key must carry the in-sample honesty label (§5.8)"
    )
    assert "not predictive" in str(label).lower()


def test_in_sample_label_is_stable_string():
    result = _golden_result()
    cfg = load_config(str(GOLDEN_CFG))
    metrics = compute_metrics(result, periods_per_year=cfg.periods_per_year)
    label = str(metrics[next(iter(metrics))])
    expected = (
        "IN-SAMPLE, single pre-specified strategy; not predictive of live results."
    )
    assert label == expected
