import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from plutobt.data import (
    CT,
    ValidationReport,
    load_bars,
    resample,
    tag_sessions,
    validate_bars,
    verify_ts_is_open,
)
from plutobt.instruments import get_instrument

# ---------------------------------------------------------------------------
# Helpers shared across all M3 task groups
# ---------------------------------------------------------------------------


def _write_json(tmp_path: Path, rows: list[list]) -> str:
    p = tmp_path / "bars.json"
    p.write_text(json.dumps(rows))
    return str(p)


def _ct_ts(y, m, d, hh, mm) -> int:
    return int(datetime(y, m, d, hh, mm, tzinfo=CT).timestamp())


def _rth_day_5m(y, m, d) -> list[list]:
    # RTH index session 08:30 -> 15:15 CT, 81 five-minute bars, ts = OPEN
    rows = []
    t = _ct_ts(y, m, d, 8, 30)
    for _ in range(81):
        rows.append([t, 100.0, 101.0, 99.0, 100.5, 10])
        t += 300
    return rows


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


def _seq_df(ts: list[int]) -> pd.DataFrame:
    n = len(ts)
    return pd.DataFrame(
        {
            "ts": ts,
            "open": [100.0] * n,
            "high": [101.0] * n,
            "low": [99.0] * n,
            "close": [100.0] * n,
            "volume": [10] * n,
        }
    )


# ---------------------------------------------------------------------------
# M3.1 — load_bars
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# M3.3 — validate_bars ts monotonic+unique guard + gap report (§10.12 partial)
# ---------------------------------------------------------------------------


def test_validate_bars_duplicate_ts_guard_fires():  # §10.12 partial
    df = _seq_df([1700000000, 1700000300, 1700000300, 1700000600])
    rep = validate_bars(df)
    assert rep.ok is False
    assert any("duplicate ts" in e for e in rep.errors)
    assert 2 in rep.bad_rows  # the repeated row


def test_validate_bars_nonmonotonic_ts_guard_fires():  # §10.12 partial
    df = _seq_df([1700000000, 1700000600, 1700000300])  # row 2 goes backwards
    rep = validate_bars(df)
    assert rep.ok is False
    assert any("non-monotonic ts" in e for e in rep.errors)
    assert 2 in rep.bad_rows


def test_validate_bars_strict_raises_on_nonmonotonic():  # §10.12 partial
    df = _seq_df([1700000000, 1700000600, 1700000300])
    with pytest.raises(ValueError, match="validate_bars"):
        validate_bars(df, strict=True)


def test_validate_bars_gap_report():
    # 5-min cadence (300s); one weekend-sized gap after row 2
    ts = [1700000000, 1700000300, 1700000600, 1700000600 + 49 * 3600]
    df = _seq_df(ts)
    rep = validate_bars(df)
    assert rep.ok is True  # a gap is reported, not an error
    assert rep.modal_interval == 300
    assert rep.max_gap == 49 * 3600
    # gap recorded at the row that opens after the gap (index 3)
    assert (3, 49 * 3600) in rep.gaps


def test_validate_bars_no_gaps_on_regular_series():
    df = _seq_df([1700000000, 1700000300, 1700000600])
    rep = validate_bars(df)
    assert rep.modal_interval == 300
    assert rep.gaps == []
    assert rep.max_gap == 0


# ---------------------------------------------------------------------------
# M3.4 — verify_ts_is_open: empirical ts-convention check (§5.5)
# ---------------------------------------------------------------------------


def test_verify_ts_is_open_true_when_first_bar_at_session_open():
    nq = get_instrument("NQ")
    # two trading days of RTH 5m bars; first bar each day opens 08:30 CT
    rows = _rth_day_5m(2025, 6, 2) + _rth_day_5m(2025, 6, 3)
    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
    assert verify_ts_is_open(df, nq) is True


def test_verify_ts_is_open_false_when_ts_marks_close():
    nq = get_instrument("NQ")
    # shift every ts forward 5 min so the first bar-of-day "opens" 08:35 -> ts = CLOSE convention
    rows = _rth_day_5m(2025, 6, 2) + _rth_day_5m(2025, 6, 3)
    for r in rows:
        r[0] += 300
    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
    assert verify_ts_is_open(df, nq) is False


# ---------------------------------------------------------------------------
# M3.5 — tag_sessions RTH/ETH + 15:10/15:15 boundary (§10.10)
# ---------------------------------------------------------------------------


def test_tag_sessions_adds_session_column():
    nq = get_instrument("NQ")
    rows = _rth_day_5m(2025, 6, 2)
    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
    out = tag_sessions(df, nq)
    assert "session" in out.columns
    assert set(out["session"].unique()) == {"RTH"}
    # original df not mutated
    assert "session" not in df.columns


def test_tag_sessions_rth_boundary_1510_open_1515_close():  # §10.10
    nq = get_instrument("NQ")
    assert nq.session_open_ct == "08:30"
    assert nq.session_close_ct == "15:15"
    rows = _rth_day_5m(2025, 6, 2)  # 81 bars, last opens 15:10 CT
    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
    out = tag_sessions(df, nq)

    idx = pd.DatetimeIndex(
        pd.to_datetime(out["ts"], unit="s", utc=True)
    ).tz_convert(CT)
    # the LAST RTH bar opens 15:10 ...
    last_rth = out.index[out["session"] == "RTH"][-1]
    assert idx[last_rth].strftime("%H:%M") == "15:10"
    assert out.loc[last_rth, "session"] == "RTH"
    # ... and closes 15:15 (open + one 5-min interval)
    close_ct = idx[last_rth] + pd.Timedelta(minutes=5)
    assert close_ct.strftime("%H:%M") == "15:15"


def test_tag_sessions_1515_open_bar_is_eth_not_rth():  # §10.10 upper edge
    nq = get_instrument("NQ")
    rows = _rth_day_5m(2025, 6, 2)
    # append a bar OPENING exactly at 15:15 CT -> must be ETH (interval is half-open)
    t1515 = _ct_ts(2025, 6, 2, 15, 15)
    rows.append([t1515, 100.0, 101.0, 99.0, 100.5, 10])
    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
    out = tag_sessions(df, nq)
    assert out.iloc[-1]["session"] == "ETH"


def test_tag_sessions_overnight_bar_is_eth():
    nq = get_instrument("NQ")
    t_overnight = _ct_ts(2025, 6, 2, 2, 0)  # 02:00 CT -> ETH
    rows = [[t_overnight, 100.0, 101.0, 99.0, 100.5, 10]] + _rth_day_5m(2025, 6, 2)
    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
    out = tag_sessions(df, nq)
    assert out.iloc[0]["session"] == "ETH"


# ---------------------------------------------------------------------------
# M3.6 — resample OHLCV with left-labeled buckets (ts=open, 5.5)
# ---------------------------------------------------------------------------


def test_resample_5m_to_15m_ohlcv():
    # three consecutive 5-min bars -> one 15-min bar
    base = _ct_ts(2025, 6, 2, 8, 30)
    rows = [
        [base + 0, 100.0, 105.0, 99.0, 101.0, 10],
        [base + 300, 101.0, 106.0, 98.0, 102.0, 20],
        [base + 600, 102.0, 104.0, 97.0, 103.0, 30],
    ]
    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
    out = resample(df, "15min")

    assert list(out.columns) == ["ts", "open", "high", "low", "close", "volume"]
    assert len(out) == 1
    r = out.iloc[0]
    assert r["ts"] == base          # ts of the bucket = its OPEN (left label)
    assert r["open"] == 100.0       # first
    assert r["high"] == 106.0       # max
    assert r["low"] == 97.0         # min
    assert r["close"] == 103.0      # last
    assert r["volume"] == 60        # sum


def test_resample_drops_empty_buckets():
    base = _ct_ts(2025, 6, 2, 8, 30)
    # two bars 15 min apart -> two separate 15-min buckets, no empty NaN bucket between
    rows = [
        [base + 0, 100.0, 105.0, 99.0, 101.0, 10],
        [base + 900, 102.0, 104.0, 97.0, 103.0, 30],
    ]
    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
    out = resample(df, "15min")
    assert len(out) == 2
    assert out["ts"].tolist() == [base, base + 900]
    assert not out.isna().any().any()


# ---------------------------------------------------------------------------
# M3.7 — full pipeline round-trip
# ---------------------------------------------------------------------------


def test_data_pipeline_round_trip(tmp_path):
    nq = get_instrument("NQ")
    rows = _rth_day_5m(2025, 6, 2)  # 81 clean RTH 5m bars
    p = tmp_path / "day.json"
    p.write_text(json.dumps(rows))

    df = load_bars(str(p))
    rep = validate_bars(df)
    assert rep.ok is True
    assert rep.modal_interval == 300
    assert verify_ts_is_open(df, nq) is True

    tagged = tag_sessions(df, nq)
    assert (tagged["session"] == "RTH").all()

    fifteen = resample(df, "15min")
    # 81 five-min RTH bars -> 27 fifteen-min bars
    assert len(fifteen) == 27
    assert validate_bars(fifteen).ok is True


# ---------------------------------------------------------------------------
# Film-ready audit (2026-11-02): ts and volume are whole numbers or refused; daily
# bars are CME trading days in America/Chicago.
# ---------------------------------------------------------------------------

_GOOD = [1700000300, 100.0, 101.0, 99.0, 100.0, 1]


@pytest.mark.parametrize(("bad", "shown"), [
    (None, "None"), (1700000000.9, "1700000000.9"), (float("nan"), "nan"),
    (True, "True"), ("1700000000", "'1700000000'"),
], ids=["null", "fractional", "nan", "bool", "string"])
def test_load_bars_refuses_a_ts_that_is_not_whole_unix_seconds(tmp_path, bad, shown):
    # Unchecked, null loaded as INT64_MIN and 1700000000.9 as 1700000000 (probe p07).
    path = _write_json(tmp_path, [[bad, 100.0, 101.0, 99.0, 100.0, 1], _GOOD])
    with pytest.raises(ValueError, match=rf"row 0: ts must be a whole number of unix "
                                         rf"seconds, got {shown}"):
        load_bars(path)


@pytest.mark.parametrize("bad", [None, 2.5], ids=["null", "fractional"])
def test_load_bars_refuses_a_volume_that_is_not_a_whole_number(tmp_path, bad):
    # The same int64 cast turned a null volume into INT64_MIN.
    path = _write_json(tmp_path, [_GOOD, [1700000600, 100.0, 101.0, 99.0, 100.0, bad]])
    with pytest.raises(ValueError, match="row 1: volume must be a whole number"):
        load_bars(path)


def test_load_bars_accepts_whole_numbers_written_as_floats(tmp_path):
    # Control: 1700000000.0 is a whole second; it loads exactly.
    path = _write_json(tmp_path, [[1700000000.0, 100.0, 101.0, 99.0, 100.0, 3.0], _GOOD])
    df = load_bars(path)
    assert df["ts"].tolist() == [1700000000, 1700000300]
    assert df["volume"].tolist() == [3, 1]


@pytest.mark.parametrize(("ts", "unit", "factor"), [
    (1772461800000, "milliseconds", "1,000"),
    (1772461800000000, "microseconds", "1,000,000"),
    (1772461800000000000, "nanoseconds", "1,000,000,000"),
], ids=["ms", "us", "ns"])
def test_load_bars_names_the_unit_of_a_ts_that_is_not_seconds(tmp_path, ts, unit, factor):
    # Milliseconds loaded silently as seconds in the year 58,000 (final audit B).
    path = _write_json(tmp_path, [[ts, 100.0, 101.0, 99.0, 100.0, 1]])
    with pytest.raises(ValueError, match=rf"row 0: ts {ts} looks like {unit}, not unix "
                                         rf"seconds .*divide the ts by {factor}"):
        load_bars(path)


def test_load_bars_refuses_a_ts_past_int64(tmp_path):
    # 2**63 + 10 wrapped to a negative int64.
    path = _write_json(tmp_path, [[2**63 + 10, 100.0, 101.0, 99.0, 100.0, 1]])
    with pytest.raises(ValueError, match="row 0: ts 9223372036854775818 is out of range"):
        load_bars(path)


def test_load_bars_accepts_a_ts_just_under_the_seconds_bound(tmp_path):  # control
    path = _write_json(tmp_path, [[99_999_999_999, 100.0, 101.0, 99.0, 100.0, 1]])
    assert load_bars(path)["ts"].tolist() == [99_999_999_999]


@pytest.mark.parametrize(("bad", "why"), [
    ("100", r"must be a number, got '100' \(str\)"),
    (True, r"must be a number, got True \(bool\)"),
    (None, r"must be a number, got None \(NoneType\)"),
    (float("nan"), "must be a finite number, got nan"),
    (float("inf"), "must be a finite number, got inf"),
], ids=["text", "bool", "null", "nan", "inf"])
@pytest.mark.parametrize(("col", "k"), [("open", 1), ("low", 3)])
def test_load_bars_refuses_a_price_that_is_not_a_finite_number(tmp_path, bad, why, col, k):
    # Text prices were coerced ("100" became 100.0) and inf loaded as inf.
    row = [1700000600, 100.0, 101.0, 99.0, 100.0, 1]
    row[k] = bad
    path = _write_json(tmp_path, [_GOOD, row])
    with pytest.raises(ValueError, match=rf"row 1: {col} {why}"):
        load_bars(path)


def test_load_bars_refuses_a_literal_overflowing_to_inf(tmp_path):
    p = tmp_path / "bars.json"
    p.write_text("[[1700000000, 1e400, 1, 1, 1, 5]]", encoding="utf-8")
    with pytest.raises(ValueError, match="row 0: open must be a finite number, got inf"):
        load_bars(str(p))


@pytest.mark.parametrize(("row", "what"), [
    ([1700000600, 100.0, 99.5, 99.0, 100.0, 1], r"high 99\.5 is below max\(open, close\)"),
    ([1700000600, 100.0, 101.0, 100.5, 101.0, 1], r"low 100\.5 is above min\(open, close\)"),
], ids=["high", "low"])
def test_load_bars_refuses_a_bar_that_does_not_hold_its_body(tmp_path, row, what):
    path = _write_json(tmp_path, [_GOOD, row])
    with pytest.raises(ValueError, match=rf"row 1 \(ts 1700000600\): {what}"):
        load_bars(path)


def _hourly(t0: int, t1: int) -> pd.DataFrame:
    ts = np.arange(t0, t1, 3600, dtype="int64")
    n = len(ts)
    return pd.DataFrame({"ts": ts, "open": np.arange(n, dtype="float64"),
                         "high": np.arange(n, dtype="float64") + 1.0,
                         "low": np.arange(n, dtype="float64") - 1.0,
                         "close": np.arange(n, dtype="float64") + 0.5,
                         "volume": np.ones(n, dtype="int64")})


@pytest.mark.parametrize("day", [(2025, 1, 14), (2025, 7, 15)], ids=["CST", "CDT"])
def test_resample_daily_keeps_one_cme_trading_day_in_one_bar(day):
    # 17:00 CT the day before -> 16:00 CT: one trading day. Bucketed in UTC it split at
    # UTC midnight, 18:00 or 19:00 CT (probe p08).
    y, m, d = day
    df = _hourly(_ct_ts(y, m, d - 1, 17, 0), _ct_ts(y, m, d, 16, 0))
    out = resample(df, "1D")
    assert len(out) == 1
    r = out.iloc[0]
    assert r["ts"] == _ct_ts(y, m, d - 1, 17, 0)   # the trading day's open, CT
    assert (r["open"], r["high"], r["low"], r["close"], r["volume"]) == (
        0.0, 23.0, -1.0, 22.5, 23)


def test_resample_daily_splits_at_17_00_ct():
    # A bar opening 16:00 CT belongs to the day ending that afternoon; 17:00 CT opens the
    # next one. Three bars, two trading days.
    rows = [[_ct_ts(2025, 6, 3, 15, 0), 1.0, 2.0, 0.5, 1.5, 1],
            [_ct_ts(2025, 6, 3, 16, 0), 1.5, 2.5, 1.0, 2.0, 2],
            [_ct_ts(2025, 6, 3, 17, 0), 2.0, 3.0, 1.5, 2.5, 4]]
    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
    out = resample(df, "1D")
    assert out["ts"].tolist() == [_ct_ts(2025, 6, 2, 17, 0), _ct_ts(2025, 6, 3, 17, 0)]
    assert out["volume"].tolist() == [3, 4]


def test_resample_daily_puts_the_sunday_open_in_mondays_bar():
    # Thursday 17:00 -> Friday 16:00 is Friday's bar; Sunday 17:00 -> Monday 16:00 is
    # Monday's, opening Sunday 17:00 CT. Nothing for the weekend.
    friday = _hourly(_ct_ts(2025, 6, 5, 17, 0), _ct_ts(2025, 6, 6, 16, 0))
    monday = _hourly(_ct_ts(2025, 6, 8, 17, 0), _ct_ts(2025, 6, 9, 16, 0))
    out = resample(pd.concat([friday, monday], ignore_index=True), "1D")
    assert out["ts"].tolist() == [_ct_ts(2025, 6, 5, 17, 0), _ct_ts(2025, 6, 8, 17, 0)]
    assert out["volume"].tolist() == [23, 23]


def test_resample_intraday_still_buckets_on_the_utc_clock():
    # Unchanged: an hourly resample of 5-minute bars labels each bucket at the top of the
    # hour, crossing 17:00 CT with no trading-day handling.
    df = _seq_df(list(range(_ct_ts(2025, 6, 3, 16, 30), _ct_ts(2025, 6, 3, 17, 30), 300)))
    out = resample(df, "60min")
    assert out["ts"].tolist() == [_ct_ts(2025, 6, 3, 16, 0), _ct_ts(2025, 6, 3, 17, 0)]
    assert out["volume"].tolist() == [60, 60]


def _cme_week_hours():
    """Hourly CME bars 2026-02-22 (Sun) 17:00 CT .. 2026-04-10 (Fri) 15:00 CT: closed
    Saturday, Sunday before 17:00, Friday from 16:00, and the 16:00 CT halt daily."""
    idx = pd.date_range("2026-02-22 17:00", "2026-04-10 15:00", freq="h", tz=CT)
    keep = [t for t in idx if not (t.weekday() == 5 or (t.weekday() == 6 and t.hour < 17)
                                   or (t.weekday() == 4 and t.hour >= 16) or t.hour == 16)]
    ts = np.array([int(t.timestamp()) for t in keep], dtype="int64")
    n = len(ts)
    return pd.DataFrame({"ts": ts, "open": np.arange(n, dtype="float64"),
                         "high": np.arange(n) + 0.5, "low": np.arange(n) - 0.5,
                         "close": np.arange(n, dtype="float64"),
                         "volume": np.ones(n, dtype="int64")})


@pytest.mark.parametrize(("rule", "period"), [
    ("W", "W-SUN"), ("ME", "M"), ("MS", "M"), ("QE", "Q"), ("QS", "Q"), ("YE", "Y"),
    ("YS", "Y"),
])
def test_resample_longer_rules_hold_exactly_their_trading_days(rule, period):
    # Review fix: closed="left" on an end-anchored rule ("ME") put a month's last trading
    # day into the next month, and weekly bars were stamped Saturday 17:00 CT. Each bucket
    # now holds its own CME trading days, stamped at the 17:00 CT open of the first one.
    df = _cme_week_hours()
    ct = pd.to_datetime(df["ts"], unit="s", utc=True).dt.tz_convert(CT)
    tdate = (ct.dt.tz_localize(None) + pd.Timedelta(hours=7)).dt.normalize()
    groups = df.assign(day=tdate).groupby(tdate.dt.to_period(period))
    exp = groups.agg(open=("open", "first"), close=("close", "last"),
                     volume=("volume", "sum"), day=("day", "first"))
    out = resample(df, rule)
    assert out["open"].tolist() == exp["open"].tolist()
    assert out["close"].tolist() == exp["close"].tolist()
    assert out["volume"].tolist() == exp["volume"].tolist()
    opens = [_ct_ts(d.year, d.month, d.day, 17, 0) - 86400 for d in exp["day"]]
    assert out["ts"].tolist() == opens


def test_resample_weekly_bars_open_sunday_17_00_ct():
    out = resample(_cme_week_hours(), "W")
    opens = pd.to_datetime(out["ts"], unit="s", utc=True).dt.tz_convert(CT)
    assert set(opens.dt.weekday) == {6} and set(opens.dt.hour) == {17}
