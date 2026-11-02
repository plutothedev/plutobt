"""The shipped synthetic samples sit on each instrument's real RTH session grid.

Film-ready (2026-11-02): the old files were 324 continuous bars running through the
16:00-17:00 CT halt, so verify_ts_is_open was False on all three and no session break
ever fired in the demo. These pin the regenerated files: RTH bars only, the right count
per trading day, overnight gaps, a weekend and the US DST change inside the range, and
scripts/gen_sample.py reproducing every byte.
"""
from __future__ import annotations

import importlib.util
import json
from collections import Counter
from datetime import date, time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from plutobt.data import CT, load_bars, tag_sessions, validate_bars, verify_ts_is_open
from plutobt.instruments import get_instrument

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DIR = ROOT / "data" / "sample"
SYMBOLS = ["nq", "es", "cl"]
# Bars per RTH day from the registry session: NQ/ES 08:30-15:15 CT (405 min),
# CL 08:00-13:30 CT (330 min), at 5 minutes a bar.
BARS_PER_DAY = {"nq": 81, "es": 81, "cl": 66}
N_DAYS = 20
BAR = 300

_spec = importlib.util.spec_from_file_location("gen_sample", ROOT / "scripts" / "gen_sample.py")
gen_sample = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gen_sample)


def _local(df: pd.DataFrame) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.to_datetime(df["ts"], unit="s", utc=True)).tz_convert(CT)


def _load(sym: str) -> pd.DataFrame:
    return load_bars(str(SAMPLE_DIR / f"{sym}_5m.json"))


@pytest.mark.parametrize("sym", SYMBOLS)
def test_ts_marks_the_bar_open_on_every_shipped_file(sym):
    assert verify_ts_is_open(_load(sym), get_instrument(sym.upper())) is True


@pytest.mark.parametrize("sym", SYMBOLS)
def test_every_bar_lies_inside_its_instruments_session(sym):
    df = _load(sym)
    inst = get_instrument(sym.upper())
    assert set(tag_sessions(df, inst)["session"]) == {"RTH"}
    open_t = time.fromisoformat(inst.session_open_ct)
    close_t = time.fromisoformat(inst.session_close_ct)
    local = _local(df)
    # The bar OPENS at or after the session open and CLOSES at or before the close.
    ends = (local + pd.Timedelta(seconds=BAR)).time
    assert all(open_t <= t for t in local.time)
    assert all(e <= close_t for e in ends)


@pytest.mark.parametrize("sym", SYMBOLS)
def test_bar_counts_per_trading_day(sym):
    df = _load(sym)
    per_day = Counter(_local(df).date)
    assert len(per_day) == N_DAYS
    assert set(per_day.values()) == {BARS_PER_DAY[sym]}
    assert len(df) == N_DAYS * BARS_PER_DAY[sym]
    # Under the full causality grid ("auto" checks every prefix up to 2,000 bars).
    assert len(df) < 2000


@pytest.mark.parametrize("sym", SYMBOLS)
def test_bars_are_5_minutes_apart_inside_a_day_with_overnight_gaps_between(sym):
    df = _load(sym)
    local = _local(df)
    deltas = np.diff(df["ts"].to_numpy())
    same_day = local.date[1:] == local.date[:-1]
    assert (deltas[same_day] == BAR).all()
    assert (deltas[~same_day] > 12 * 3600).all()        # overnight / weekend halt
    assert int((~same_day).sum()) == N_DAYS - 1


@pytest.mark.parametrize("sym", SYMBOLS)
def test_range_spans_a_weekend_and_the_us_dst_change(sym):
    local = _local(_load(sym))
    days = sorted(set(local.date))
    assert days[0] == date(2026, 3, 2)                  # Monday
    assert all(d.weekday() < 5 for d in days)           # no weekend bars
    assert any((b - a).days == 3 for a, b in zip(days, days[1:], strict=False))
    # US DST starts Sunday 2026-03-08: CT is UTC-6 before it, UTC-5 after.
    offsets = {d: local[local.date == d][0].utcoffset() for d in days}
    assert offsets[date(2026, 3, 6)] == pd.Timedelta(hours=-6)
    assert offsets[date(2026, 3, 9)] == pd.Timedelta(hours=-5)
    # Same 08:30/08:00 CT wall-clock open either side, so the UTC open moved an hour.
    first = {d: local[local.date == d][0] for d in days}
    assert first[date(2026, 3, 6)].time() == first[date(2026, 3, 9)].time()


@pytest.mark.parametrize("sym", SYMBOLS)
def test_sample_validates_clean_on_the_tick_grid(sym):
    df = _load(sym)
    rep = validate_bars(df)
    assert rep.ok, rep.errors
    tick = get_instrument(sym.upper()).tick_size
    for col in ("open", "high", "low", "close"):
        assert (df[col] > 0).all()
        steps = df[col].to_numpy() / tick
        assert np.allclose(steps, np.round(steps), atol=1e-6)


@pytest.mark.parametrize("sym, base, sigma", [("nq", 18000.0, 6.0), ("es", 5000.0, 2.0),
                                              ("cl", 70.0, 0.25)])
def test_price_process_keeps_the_old_per_bar_volatility(sym, base, sigma):
    df = _load(sym)
    assert df["open"].iloc[0] == base
    local = _local(df)
    same_day = local.date[1:] == local.date[:-1]
    intraday = np.diff(df["close"].to_numpy())[same_day]
    assert 0.8 * sigma < intraday.std() < 1.2 * sigma


@pytest.mark.parametrize("sym", SYMBOLS)
def test_gen_sample_regenerates_the_shipped_file_byte_identical(sym):
    spec = next(s for s in gen_sample.SYMBOLS if s["name"] == sym)
    assert gen_sample.render(spec) == (SAMPLE_DIR / f"{sym}_5m.json").read_bytes()


def test_each_symbol_keeps_its_own_seed():
    seeds = {s["name"]: s["seed"] for s in gen_sample.SYMBOLS}
    assert seeds == {"nq": 20260607, "es": 20260608, "cl": 20260609}


def test_notice_describes_the_session_grid_and_says_synthetic():
    text = (SAMPLE_DIR / "NOTICE.md").read_text(encoding="utf-8")
    assert "SYNTHETIC" in text
    assert "1,620" in text and "1,320" in text
    assert "08:30-15:15" in text and "08:00-13:30" in text
    rows = json.loads((SAMPLE_DIR / "nq_5m.json").read_text(encoding="utf-8"))
    assert len(rows) == 1620
