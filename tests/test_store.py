import json
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from plutobt.store import Store, data_dir

S, C = "ohlcv-1m", "NQH2026"


def _day(d: date) -> int:
    return int(pd.Timestamp(d, tz="UTC").timestamp())


def _bars(start: date, end: date, step=3600):
    """One bar every `step` seconds across [start, end) UTC days."""
    ts = list(range(_day(start), _day(end), step))
    n = len(ts)
    return pd.DataFrame({
        "ts": pd.Series(ts, dtype="int64"),
        "open": [float(i) for i in range(n)],
        "high": [float(i) + 1 for i in range(n)],
        "low": [float(i) - 1 for i in range(n)],
        "close": [float(i) + 0.5 for i in range(n)],
        "volume": pd.Series(range(n), dtype="int64"),
    })


def test_empty_store_missing_is_whole_range(tmp_path):
    st = Store(tmp_path)
    got = st.missing(S, C, date(2026, 1, 1), date(2026, 2, 1))
    assert got == [(date(2026, 1, 1), date(2026, 2, 1))]
    assert st.held(S, C) == []


def test_add_then_missing_is_empty_and_frame_round_trips(tmp_path):
    st = Store(tmp_path)
    f = _bars(date(2026, 1, 1), date(2026, 1, 3))
    st.add(S, C, date(2026, 1, 1), date(2026, 1, 3), f)
    assert st.missing(S, C, date(2026, 1, 1), date(2026, 1, 3)) == []
    out = st.frame(S, C, date(2026, 1, 1), date(2026, 1, 3))
    pd.testing.assert_frame_equal(out, f)
    sub = st.frame(S, C, date(2026, 1, 2), date(2026, 1, 3))
    assert sub["ts"].min() == _day(date(2026, 1, 2))
    assert list(sub.index) == list(range(len(sub)))
    assert (tmp_path / "GLBX.MDP3" / S / f"{C}.csv").exists()
    idx = json.loads((tmp_path / "index.json").read_text())
    covered = idx["entries"][f"GLBX.MDP3/{S}/{C}"]["covered"]
    assert covered == [["2026-01-01", "2026-01-03"]]


def test_partial_cover_missing_is_only_the_gap(tmp_path):
    st = Store(tmp_path)
    jan = _bars(date(2026, 1, 1), date(2026, 2, 1))
    st.add(S, C, date(2026, 1, 1), date(2026, 2, 1), jan)
    got = st.missing(S, C, date(2026, 1, 1), date(2026, 4, 1))
    assert got == [(date(2026, 2, 1), date(2026, 4, 1))]


def test_missing_gaps_in_middle_sorted(tmp_path):
    st = Store(tmp_path)
    st.add(S, C, date(2026, 1, 5), date(2026, 1, 10), _bars(date(2026, 1, 5), date(2026, 1, 10)))
    assert st.missing(S, C, date(2026, 1, 1), date(2026, 1, 15)) == [
        (date(2026, 1, 1), date(2026, 1, 5)), (date(2026, 1, 10), date(2026, 1, 15))]


def test_overlapping_add_dedupes_ts_and_unions_ranges(tmp_path):
    st = Store(tmp_path)
    a = _bars(date(2026, 1, 1), date(2026, 1, 5))
    st.add(S, C, date(2026, 1, 1), date(2026, 1, 5), a)
    b = _bars(date(2026, 1, 3), date(2026, 1, 8))
    b["close"] = 999.0
    st.add(S, C, date(2026, 1, 3), date(2026, 1, 8), b)
    assert st.held(S, C) == [(date(2026, 1, 1), date(2026, 1, 8))]
    out = st.frame(S, C, date(2026, 1, 1), date(2026, 1, 8))
    assert out["ts"].is_unique and out["ts"].is_monotonic_increasing
    held_ts = a["ts"].iloc[-1]
    assert out.loc[out["ts"] == held_ts, "close"].iloc[0] != 999.0
    assert (out.loc[out["ts"] >= _day(date(2026, 1, 5)), "close"] == 999.0).all()


def test_adjacent_ranges_merge(tmp_path):
    st = Store(tmp_path)
    st.add(S, C, date(2026, 1, 1), date(2026, 1, 3), _bars(date(2026, 1, 1), date(2026, 1, 3)))
    st.add(S, C, date(2026, 1, 3), date(2026, 1, 5), _bars(date(2026, 1, 3), date(2026, 1, 5)))
    assert st.held(S, C) == [(date(2026, 1, 1), date(2026, 1, 5))]


def test_index_survives_reopen(tmp_path):
    f = _bars(date(2026, 1, 1), date(2026, 1, 3))
    Store(tmp_path).add(S, C, date(2026, 1, 1), date(2026, 1, 3), f)
    st = Store(tmp_path)
    assert st.held(S, C) == [(date(2026, 1, 1), date(2026, 1, 3))]
    assert len(st.frame(S, C, date(2026, 1, 1), date(2026, 1, 3))) == 48


def test_corrupt_index_with_data_raises(tmp_path):
    f = _bars(date(2026, 1, 1), date(2026, 1, 3))
    Store(tmp_path).add(S, C, date(2026, 1, 1), date(2026, 1, 3), f)
    (tmp_path / "index.json").write_text("{not json")
    with pytest.raises(ValueError, match="index.json"):
        Store(tmp_path).missing(S, C, date(2026, 1, 1), date(2026, 1, 3))


def test_missing_index_with_no_data_is_empty(tmp_path):
    assert Store(tmp_path / "nope").held(S, C) == []


def test_path_separators_refused(tmp_path):
    st = Store(tmp_path)
    for bad in ("../x", "a/b", "a" + chr(92) + "b", ".."):
        with pytest.raises(ValueError):
            st.missing(S, bad, date(2026, 1, 1), date(2026, 1, 2))
        with pytest.raises(ValueError):
            st.missing(bad, C, date(2026, 1, 1), date(2026, 1, 2))


def test_data_dir_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("PLUTOBT_DATA_DIR", str(tmp_path / "x"))
    assert data_dir() == tmp_path / "x"
    assert Store().root == tmp_path / "x"


def test_default_data_dir_is_in_home(monkeypatch):
    monkeypatch.delenv("PLUTOBT_DATA_DIR", raising=False)
    assert data_dir() == Path.home() / ".plutobt" / "data"
