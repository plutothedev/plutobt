import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from plutobt.data import load_bars


def _write_json(tmp_path: Path, rows: list[list]) -> str:
    p = tmp_path / "bars.json"
    p.write_text(json.dumps(rows))
    return str(p)


def test_load_bars_parses_arrays_into_canonical_columns(tmp_path):
    # two 5-minute bars, unix-seconds; [ts, o, h, l, c, v]
    rows = [
        [1700000000, 100.0, 101.0, 99.5, 100.5, 1200],
        [1700000300, 100.5, 102.0, 100.0, 101.75, 900],
    ]
    path = _write_json(tmp_path, rows)

    df = load_bars(path)

    assert list(df.columns) == ["ts", "open", "high", "low", "close", "volume"]
    # ts kept as an int column (not the index) per the DATA API contract
    assert df.index.tolist() == [0, 1]
    assert df["ts"].dtype == np.dtype("int64")
    for col in ("open", "high", "low", "close"):
        assert df[col].dtype == np.dtype("float64")
    assert df["volume"].dtype == np.dtype("int64")
    assert df.loc[0, "ts"] == 1700000000
    assert df.loc[1, "close"] == 101.75
    assert df.loc[1, "volume"] == 900
    assert len(df) == 2


def test_load_bars_rejects_wrong_arity_row(tmp_path):
    rows = [[1700000000, 100.0, 101.0, 99.5, 100.5]]  # 5 fields, missing volume
    path = _write_json(tmp_path, rows)
    with pytest.raises(ValueError, match="expected 6 fields"):
        load_bars(path)


def test_load_bars_empty_file_raises(tmp_path):
    path = _write_json(tmp_path, [])
    with pytest.raises(ValueError, match="no bars"):
        load_bars(path)


# ---------------------------------------------------------------------------
# M3.2 — ValidationReport + validate_bars OHLC/volume/NaN checks
# ---------------------------------------------------------------------------

from plutobt.data import ValidationReport, validate_bars


def _good_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ts": [1700000000, 1700000300, 1700000600],
            "open": [100.0, 100.5, 101.0],
            "high": [101.0, 102.0, 101.5],
            "low": [99.5, 100.0, 100.5],
            "close": [100.5, 101.75, 101.25],
            "volume": [1200, 900, 1100],
        }
    )


def test_validate_bars_clean_data_returns_ok_report():
    rep = validate_bars(_good_df())
    assert isinstance(rep, ValidationReport)
    assert rep.ok is True
    assert rep.errors == []
    assert rep.n_bars == 3


def test_validate_bars_flags_high_below_body():
    df = _good_df()
    df.loc[1, "high"] = 101.0  # below close 101.75 -> violates high>=max(o,c)
    rep = validate_bars(df)
    assert rep.ok is False
    assert any("high < max(open, close)" in e for e in rep.errors)
    assert 1 in rep.bad_rows


def test_validate_bars_flags_low_above_body():
    df = _good_df()
    df.loc[2, "low"] = 101.10  # above close 101.25? no -> set above min(o,c)=101.0
    df.loc[2, "low"] = 101.20  # above min(open=101.0, close=101.25)=101.0
    rep = validate_bars(df)
    assert rep.ok is False
    assert any("low > min(open, close)" in e for e in rep.errors)
    assert 2 in rep.bad_rows


def test_validate_bars_flags_high_below_low():
    df = _good_df()
    df.loc[0, "high"] = 98.0  # below low 99.5
    rep = validate_bars(df)
    assert rep.ok is False
    assert any("high < low" in e for e in rep.errors)


def test_validate_bars_flags_negative_volume():
    df = _good_df()
    df.loc[1, "volume"] = -1
    rep = validate_bars(df)
    assert rep.ok is False
    assert any("volume < 0" in e for e in rep.errors)
    assert 1 in rep.bad_rows


def test_validate_bars_flags_nan():
    df = _good_df()
    df.loc[2, "close"] = np.nan
    rep = validate_bars(df)
    assert rep.ok is False
    assert any("NaN" in e for e in rep.errors)
    assert 2 in rep.bad_rows


def test_validate_bars_raises_when_strict():
    df = _good_df()
    df.loc[1, "volume"] = -5
    with pytest.raises(ValueError, match="validate_bars: 1 error"):
        validate_bars(df, strict=True)
