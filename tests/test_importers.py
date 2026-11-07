"""load_csv and its vendor presets (spec §4)."""

from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from plutobt.importers import PRESETS, load_csv, load_csv_contracts

FIX = Path(__file__).parent / "fixtures" / "import"
T = 1772721000                       # 2026-03-05 14:30:00 UTC = 08:30 CST
CT = ZoneInfo("America/Chicago")


def expected(ts, rows, contract=None):
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close", "volume"])
    df.insert(0, "ts", pd.Series(ts, dtype="int64"))
    df["volume"] = df["volume"].astype("int64")
    if contract is not None:
        df["contract"] = contract
    return df


TWO = [(18000.25, 18001.0, 17999.75, 18000.5, 120), (18000.5, 18002.0, 18000.25, 18001.75, 98)]


def test_presets_listed():
    assert PRESETS == ("databento", "tradingview", "ninjatrader", "generic")


def test_databento_pretty_splits_by_symbol():
    frames = load_csv_contracts(str(FIX / "databento_pretty.csv"), preset="databento")
    assert list(frames) == ["NQH2026", "NQM2026"]
    pd.testing.assert_frame_equal(frames["NQH2026"], expected([T, T + 60], TWO, "NQH2026"))
    assert frames["NQM2026"]["close"].tolist() == [18080.25, 18081.5]


def test_databento_multi_symbol_refused_by_load_csv():
    with pytest.raises(ValueError, match="load_csv_contracts"):
        load_csv(str(FIX / "databento_pretty.csv"), preset="databento")


def test_databento_raw_fixed_point():
    df = load_csv(str(FIX / "databento_raw.csv"), preset="databento")
    pd.testing.assert_frame_equal(df, expected([T, T + 60], TWO, "NQH2026"))


def test_databento_ambiguous_price_scale_refused(tmp_path):
    p = tmp_path / "amb.csv"
    p.write_text("ts_event,open,high,low,close,volume,symbol\n"
                 "1772721000000000000,18000,500000000000,1,18000,5,NQH6\n")
    with pytest.raises(ValueError, match="open"):
        load_csv(str(p), preset="databento")


def test_tradingview_loads_and_drops_indicators():
    df = load_csv(str(FIX / "tradingview.csv"), preset="tradingview", symbol="NQH2026")
    pd.testing.assert_frame_equal(df, expected([T, T + 60], TWO, "NQH2026"))


def test_tradingview_milliseconds_converted(tmp_path):
    p = tmp_path / "CME_MINI_NQH2026, 1.csv"
    p.write_text("time,open,high,low,close,Volume\n"
                 "1772721000000,18000.25,18001,17999.75,18000.5,120\n"
                 "1772721060000,18000.5,18002,18000.25,18001.75,98\n")
    df = load_csv(str(p), preset="tradingview")
    pd.testing.assert_frame_equal(df, expected([T, T + 60], TWO, "NQH2026"))


def test_tradingview_iso_with_zone(tmp_path):
    p = tmp_path / "tv.csv"
    p.write_text("time,open,high,low,close,Volume\n"
                 "2026-03-05T14:30:00Z,18000.25,18001,17999.75,18000.5,120\n"
                 "2026-03-05T08:31:00-06:00,18000.5,18002,18000.25,18001.75,98\n")
    pd.testing.assert_frame_equal(load_csv(str(p), preset="tradingview"),
                                  expected([T, T + 60], TWO))


def test_ninjatrader_close_stamp_shifted():
    df = load_csv(str(FIX / "NQ 03-26.Last.txt"), preset="ninjatrader")
    assert df["ts"].tolist() == [T, T + 60, T + 120]
    assert df["contract"].iloc[0] == "NQH2026"
    assert df["close"].tolist() == [18000.5, 18001.75, 18002.25]


def test_ninjatrader_no_dominant_interval_refused(tmp_path):
    p = tmp_path / "x.txt"
    p.write_text("20260305 143100;1;1;1;1;1\n20260305 143200;1;1;1;1;1\n"
                 "20260305 143500;1;1;1;1;1\n20260305 144500;1;1;1;1;1\n")
    with pytest.raises(ValueError, match="interval"):
        load_csv(str(p), preset="ninjatrader")


def test_naive_text_needs_tz():
    with pytest.raises(ValueError, match="tz"):
        load_csv(str(FIX / "generic.csv"), columns=_GENERIC_COLS, ts_is="open")


_GENERIC_COLS = {"ts": "Date", "open": "O", "high": "H", "low": "L", "close": "C",
                 "volume": "V"}


def test_generic_mapping():
    df = load_csv(str(FIX / "generic.csv"), columns=_GENERIC_COLS, ts_is="open",
                  tz="America/Chicago", symbol="NQH6")
    pd.testing.assert_frame_equal(df, expected([T, T + 60], TWO, "NQH2026"))


def test_generic_needs_ts_is():
    with pytest.raises(ValueError, match="ts_is"):
        load_csv(str(FIX / "generic.csv"), columns=_GENERIC_COLS, tz="America/Chicago")


def test_generic_close_stamp_shifts():
    df = load_csv(str(FIX / "generic.csv"), columns=_GENERIC_COLS, ts_is="close",
                  tz="America/Chicago")
    assert df["ts"].tolist() == [T - 60, T]


def test_epoch_seconds_need_no_tz(tmp_path):
    p = tmp_path / "e.csv"
    p.write_text("ts,open,high,low,close,volume\n1772721000,1,1,1,1,1\n1772721060,1,1,1,1,1\n")
    assert load_csv(str(p), ts_is="open")["ts"].tolist() == [T, T + 60]


@pytest.mark.parametrize("row, needle", [
    ("1772721060,1,1,1,nan,1", "close"),
    ("1772721060,1,0.5,1,1,1", "high"),
    ("1772721000,1,1,1,1,1", "increase"),
])
def test_bad_rows_refused(tmp_path, row, needle):
    p = tmp_path / "b.csv"
    p.write_text(f"ts,open,high,low,close,volume\n1772721000,1,1,1,1,1\n{row}\n")
    with pytest.raises(ValueError, match=needle):
        load_csv(str(p), ts_is="open")


def test_dst_week_chicago():
    df = load_csv(str(FIX / "chicago_dst.csv"), ts_is="open", tz="America/Chicago")
    local = pd.to_datetime(df["ts"], unit="s", utc=True).dt.tz_convert(CT)
    firsts = local.groupby(local.dt.date).min()
    assert len(firsts) == 6
    assert all(t.hour == 8 and t.minute == 30 for t in firsts)
    # the switch is real: UTC hour moves from 14 to 13 across 2026-03-08
    utc = pd.to_datetime(df["ts"], unit="s", utc=True)
    assert utc.iloc[0].hour == 14 and utc.iloc[-1].hour == 13


def test_unknown_preset_refused():
    with pytest.raises(ValueError, match="databento"):
        load_csv(str(FIX / "generic.csv"), preset="sierra")
