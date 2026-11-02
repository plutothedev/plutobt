"""Provenance, sample-data integrity, and reproducibility tests (M8)."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from plutobt.data import assert_positive_prices, load_bars, validate_bars

# ---------------------------------------------------------------------------
# M8.2: assert_positive_prices guard
# ---------------------------------------------------------------------------


def _bars(closes):
    n = len(closes)
    return pd.DataFrame(
        {
            "ts": np.arange(n, dtype=np.int64) * 300,
            "open": closes,
            "high": [c + 1 for c in closes],
            "low": [c - 1 for c in closes],
            "close": closes,
            "volume": [100] * n,
        }
    )


def test_assert_positive_prices_passes_on_strictly_positive():
    df = _bars([100.0, 101.0, 102.0])
    assert_positive_prices(df)  # no raise, returns None


def test_assert_positive_prices_raises_on_zero():
    df = _bars([100.0, 0.0, 102.0])
    with pytest.raises(ValueError, match="non-positive price"):
        assert_positive_prices(df)


def test_assert_positive_prices_raises_on_negative_from_back_adjustment():
    df = _bars([5.0, 1.0, -3.0])  # additive Panama can push history negative
    with pytest.raises(ValueError, match="non-positive price"):
        assert_positive_prices(df)


def test_assert_positive_prices_message_names_offending_column_and_row():
    df = _bars([100.0, 101.0, 102.0])
    df.loc[1, "low"] = -0.5
    with pytest.raises(ValueError, match=r"non-positive price.*low.*row 1"):
        assert_positive_prices(df)


def test_assert_positive_prices_reports_the_position_on_a_datetime_index():
    # Review fix: int() of the index label raised TypeError on a DatetimeIndex.
    df = _bars([100.0] * 8)
    df.index = pd.to_datetime(df["ts"], unit="s")
    df.iloc[5, df.columns.get_loc("low")] = -1.0
    with pytest.raises(ValueError, match=r"column 'low' at row 5 \(value=-1.0\)"):
        assert_positive_prices(df)


def test_assert_positive_prices_reports_the_position_on_a_slice():
    # A slice keeps its labels (100..); the row named is the position, as everywhere else.
    df = _bars([100.0] * 120).iloc[100:]
    df.iloc[5, df.columns.get_loc("close")] = 0.0
    with pytest.raises(ValueError, match=r"column 'close' at row 5 \(value=0.0\)"):
        assert_positive_prices(df)


# ---------------------------------------------------------------------------
# M8.3: frozen synthetic sample
# ---------------------------------------------------------------------------

SAMPLE = Path(__file__).resolve().parents[1] / "data" / "sample" / "nq_5m.json"


def test_sample_file_exists():
    assert SAMPLE.is_file(), f"missing frozen sample: {SAMPLE}"


def test_sample_is_json_array_of_six_field_bars():
    rows = json.loads(SAMPLE.read_text(encoding="utf-8"))
    assert isinstance(rows, list)
    assert len(rows) >= 200  # enough bars to warm up EMA(50) and trade
    for r in rows[:5]:
        assert len(r) == 6  # [ts, o, h, l, c, v]


def test_sample_loads_and_validates_clean():
    df = load_bars(str(SAMPLE))
    report = validate_bars(df)  # raises on violation per M2 contract
    assert report is not None


def test_sample_prices_are_strictly_positive():
    df = load_bars(str(SAMPLE))
    for col in ("open", "high", "low", "close"):
        assert (df[col] > 0).all()


# ---------------------------------------------------------------------------
# M8.4: NOTICE.md
# ---------------------------------------------------------------------------

NOTICE = Path(__file__).resolve().parents[1] / "data" / "sample" / "NOTICE.md"


def test_notice_file_exists():
    assert NOTICE.is_file(), f"missing data notice: {NOTICE}"


def test_notice_states_synthetic_provenance_and_redistribution():
    text = NOTICE.read_text(encoding="utf-8").lower()
    assert "synthetic" in text            # provenance: not vendor/CME data
    assert "redistribut" in text          # redistribution rights stated
    assert "scripts/gen_sample.py" in text  # how it was produced
    assert "as-of" in text or "as of" in text


def test_notice_disclaims_mit_covering_data():
    text = NOTICE.read_text(encoding="utf-8").lower()
    assert "mit" in text  # explicitly notes MIT covers code, not this data


# ---------------------------------------------------------------------------
# M8.10: requirements.lock pins exact reproducibility stack
# ---------------------------------------------------------------------------

LOCK = Path(__file__).resolve().parents[1] / "requirements.lock"


def test_requirements_lock_pins_exact_reproducibility_stack():
    assert LOCK.is_file(), f"missing lockfile: {LOCK}"
    text = LOCK.read_text(encoding="utf-8")
    required = {
        "numpy==1.26.4",
        "pandas==2.2.2",
        "matplotlib==3.8.4",
        "pytest==8.2.0",
        "ruff==0.4.4",
    }
    for pin in required:
        assert pin in text, f"lockfile missing exact pin: {pin}"
