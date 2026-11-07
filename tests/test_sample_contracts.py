"""The shipped synthetic NQ 2026 sample contracts (spec §9)."""
from __future__ import annotations

import importlib.util
from datetime import date, time
from pathlib import Path

import pandas as pd
import pytest

from plutobt.contracts import ContractSet
from plutobt.data import CT
from plutobt.roll import build_continuous, trading_day

ROOT = Path(__file__).resolve().parents[1]
DIR = ROOT / "data" / "sample" / "nq_contracts"
CODES = ["NQH2026", "NQM2026", "NQU2026", "NQZ2026"]


def _gen():
    spec = importlib.util.spec_from_file_location(
        "make_sample_contracts", ROOT / "scripts" / "make_sample_contracts.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def cs():
    return ContractSet.from_folder(str(DIR), "generic", "NQ", ts_is="open")


@pytest.mark.parametrize("code", CODES)
def test_regeneration_is_byte_identical(code):
    assert _gen().render(code) == (DIR / f"{code}.csv").read_bytes()


def test_sample_builds_on_cme_roll_days(cs):
    df, meta = build_continuous(cs)
    days = [trading_day(r["roll_ts"]) for r in meta.rolls]
    assert days == [date(2026, 3, 16), date(2026, 6, 15), date(2026, 9, 14)]
    gaps = [r["gap_points"] for r in meta.rolls]
    assert gaps == [25.0, -15.0, 30.0]
    assert any(g > 0 for g in gaps) and any(g < 0 for g in gaps)
    assert meta.min_adjusted_price > 0


def test_volume_trigger_rolls_a_day_early(cs):
    _, meta = build_continuous(cs, "volume")
    days = [trading_day(r["roll_ts"]) for r in meta.rolls]
    assert days == [date(2026, 3, 13), date(2026, 6, 12), date(2026, 9, 11)]


def test_sample_on_session_grid(cs):
    assert cs.interval == 900
    for code, df in cs.frames.items():
        local = pd.to_datetime(df["ts"], unit="s", utc=True).dt.tz_convert(CT)
        assert local.dt.time.min() == time(8, 30), code
        assert local.dt.time.max() == time(15, 0), code


def test_neighbours_overlap_five_days(cs):
    for a, b in zip(CODES, CODES[1:], strict=False):
        da = {trading_day(t) for t in cs.frames[a]["ts"]}
        db = {trading_day(t) for t in cs.frames[b]["ts"]}
        assert len(da & db) >= 5, (a, b)


def test_notice_covers_the_contracts():
    text = (ROOT / "data" / "sample" / "NOTICE.md").read_text(encoding="utf-8")
    assert "nq_contracts" in text and "SYNTHETIC" in text
