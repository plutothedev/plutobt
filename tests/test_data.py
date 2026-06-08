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
