"""Backward additive adjustment, RollMeta, save/load and slicing (spec §6.2 to §6.4)."""

import json
import warnings

import numpy as np
import pandas as pd
import pytest

from plutobt import __version__
from plutobt.contracts import ContractSet
from plutobt.data import frame_sha256, load_bars
from plutobt.engine import _SAMPLE_SHA256
from plutobt.roll import (
    RollMeta,
    build_continuous,
    load_continuous,
    save_continuous,
    slice_continuous,
)

T0 = 1_767_277_800


def seg(start, n, price, tick=0.25, step=1):
    px = price + np.arange(n) * tick * step
    return pd.DataFrame({
        "ts": start + 300 * np.arange(n, dtype=np.int64),
        "open": px, "high": px + tick, "low": px - tick, "close": px + tick,
        "volume": np.full(n, 10, dtype=np.int64),
    })


def two_contracts():
    """CLF2026 bars 0..19, CLG2026 bars 10..29 (10 shared). Roll on bar 12 (dates)."""
    a = seg(T0, 20, 60.0, tick=0.01)
    b = seg(T0 + 3000, 20, 60.5, tick=0.01)
    return ContractSet.from_frames({"CLF2026": a, "CLG2026": b})


def roll_day_of(ts):
    from plutobt.roll import trading_day
    return trading_day(ts).isoformat()


def build_two():
    cs = two_contracts()
    roll_ts = T0 + 12 * 300
    return cs, build_continuous(cs, "dates", roll_dates={"CLF6": roll_day_of(roll_ts)})


def test_gap_measured_at_roll_bar_open():
    a = seg(T0, 20, 18000.0)
    b = seg(T0 + 3000, 20, 18050.25 - 2 * 0.25)   # opens 18050.25 at the shared bar 2
    cs = ContractSet.from_frames({"NQH2026": a, "NQM2026": b})
    roll_ts = T0 + 12 * 300
    # roll_ts falls on the 13th bar of a (opens 18003.0) and the 3rd of b (18050.25)
    df, meta = build_continuous(cs, "dates", roll_dates={"NQH6": roll_day_of(roll_ts)})
    (r,) = meta.rolls
    first_shared = int(df["ts"][df["contract"] == "NQM2026"].iloc[0])
    a_open = float(a.loc[a["ts"] == first_shared, "open"].iloc[0])
    b_open = float(b.loc[b["ts"] == first_shared, "open"].iloc[0])
    assert r["roll_ts"] == first_shared
    assert r["gap_ticks"] == round((b_open - a_open) / 0.25)
    assert r["gap_points"] == pytest.approx(b_open - a_open)
    assert r["gap_dollars"] == pytest.approx((b_open - a_open) * 20.0)
    assert (r["from"], r["to"]) == ("NQH2026", "NQM2026")


def test_backward_additive_newest_raw():
    cs, (df, meta) = build_two()
    new = df[df["contract"] == "CLG2026"].reset_index(drop=True)
    raw_new = cs.frames["CLG2026"]
    raw_new = raw_new[raw_new["ts"] >= new["ts"].iloc[0]].reset_index(drop=True)
    for col in ("open", "high", "low", "close"):
        assert new[col].tolist() == raw_new[col].tolist()
    old = df[df["contract"] == "CLF2026"].reset_index(drop=True)
    raw_old = cs.frames["CLF2026"].iloc[: len(old)].reset_index(drop=True)
    gap = meta.rolls[0]["gap_points"]
    assert np.allclose(old["close"] - raw_old["close"], gap, atol=1e-9)
    assert set(np.round(old["adj_offset"], 9)) == {round(gap, 9)}
    assert set(new["adj_offset"]) == {0.0}
    assert np.allclose(df["raw_close"], df["close"] - df["adj_offset"], atol=1e-9)
    assert list(df.columns) == ["ts", "open", "high", "low", "close", "volume",
                                "contract", "adj_offset", "raw_close"]


def test_splice_drops_overlap():
    _, (df, meta) = build_two()
    roll_ts = meta.rolls[0]["roll_ts"]
    assert (df.loc[df["contract"] == "CLF2026", "ts"] < roll_ts).all()
    assert (df.loc[df["contract"] == "CLG2026", "ts"] >= roll_ts).all()
    assert (np.diff(df["ts"].to_numpy()) > 0).all()


def test_tick_exact_over_500_rolls():
    rng = np.random.default_rng(3)
    frames, price = {}, 18000.0
    codes = [f"NQ{'HMUZ'[q % 4]}{2026 + q // 4}" for q in range(501)]
    for i, code in enumerate(codes):
        frames[code] = seg(T0 + i * 3000, 20, price)
        price += int(rng.integers(-400, 401)) * 0.25
    cs = ContractSet.from_frames(frames)
    days = {code: roll_day_of(T0 + i * 3000 + 12 * 300) for i, code in enumerate(codes)}
    # several rolls share one trading day here, so roll on the shared bars directly
    from plutobt.roll import Roll, _adjust
    rolls = [Roll(a, b, T0 + i * 3000 + 12 * 300, "dates")
             for i, (a, b) in enumerate(zip(codes, codes[1:], strict=False))]
    df = _adjust(cs, rolls)[0]
    assert days                                   # (days unused: rolls built directly)
    for col in ("open", "high", "low", "close"):
        ticks = df[col].to_numpy() / 0.25
        assert np.array_equal(ticks, np.rint(ticks)), col


def test_negative_history_warns():
    a = seg(T0, 20, 1000.0, step=400)             # rises 100 points a bar: 2000 at bar 10
    b = seg(T0 + 3000, 20, 900.0)                  # a -1100-point gap at the roll
    cs = ContractSet.from_frames({"NQH2026": a, "NQM2026": b})
    roll_ts = T0 + 12 * 300
    with pytest.warns(UserWarning, match="ts"):
        _, meta = build_continuous(cs, "dates",
                                   roll_dates={"NQH6": roll_day_of(roll_ts)})
    assert meta.min_adjusted_price <= 0


def test_meta_fields_and_round_trip():
    _, (df, meta) = build_two()
    assert meta.root == "CL"
    assert meta.roll_trigger == "dates"
    assert meta.splice_method == "backward"
    assert meta.adjustment_method == "additive"
    assert meta.plutobt_version == __version__
    assert meta.frame_sha256 == frame_sha256(df)
    assert meta.window is None
    assert RollMeta.from_json(meta.to_json()) == meta
    json.loads(meta.to_json())


def test_save_load_round_trip(tmp_path):
    _, (df, meta) = build_two()
    csv_path, json_path = save_continuous(df, meta, str(tmp_path / "cl_cont.csv"))
    assert json_path.endswith("cl_cont.roll.json")
    df2, meta2 = load_continuous(csv_path)
    pd.testing.assert_frame_equal(df2, df)
    assert meta2 == meta


def test_load_refuses_edited_file(tmp_path):
    _, (df, meta) = build_two()
    csv_path, _ = save_continuous(df, meta, str(tmp_path / "c.csv"))
    text = open(csv_path).read().replace("60.5", "60.75", 1)
    open(csv_path, "w").write(text)
    with pytest.raises(ValueError, match="sha256"):
        load_continuous(csv_path)


def test_slice_rehashes_and_keeps_inside_rolls():
    _, (df, meta) = build_two()
    roll_ts = meta.rolls[0]["roll_ts"]
    after, m_after = slice_continuous(df, meta, start_ts=roll_ts)
    assert m_after.rolls == []
    assert m_after.frame_sha256 == frame_sha256(after)
    assert m_after.window == [int(after["ts"].iloc[0]), int(after["ts"].iloc[-1])]
    around, m_around = slice_continuous(df, meta, start_ts=roll_ts - 600)
    assert m_around.rolls == meta.rolls


def test_engine_manifest_hash_unchanged():
    for name in ("nq_5m.json", "es_5m.json", "cl_5m.json"):
        assert frame_sha256(load_bars(f"data/sample/{name}")) in _SAMPLE_SHA256


def test_no_warning_for_positive_history():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        build_two()


@pytest.mark.parametrize("tick, places", [(0.25, 2), (0.01, 2), (0.015625, 6),
                                          (5e-05, 5), (5e-07, 7), (1.0, 0), (0.1, 1)])
def test_tick_decimals(tick, places):
    """Review #2: repr(5e-05) has no '.', so the old helper gave 0 places."""
    from plutobt.contracts import tick_decimals

    assert tick_decimals(tick) == places


def test_scientific_tick_root_adjusts(monkeypatch):
    """Review #2: a 0.00005-tick root (a 6E-like contract) keeps its gap and decimals."""
    from dataclasses import replace

    from plutobt import instruments

    eu = replace(instruments.REGISTRY["NQ"], symbol="EU", tick_size=5e-05,
                 tick_value=6.25, point_value=125_000.0)
    monkeypatch.setitem(instruments.REGISTRY, "EU", eu)
    a = seg(T0, 20, 1.0800, tick=5e-05)
    b = seg(T0 + 3000, 20, 1.0817, tick=5e-05)
    cs = ContractSet.from_frames({"EUF2026": a, "EUG2026": b})
    df, meta = build_continuous(cs, "dates", roll_dates={"EUF6": roll_day_of(T0)})
    gap = meta.rolls[0]["gap_points"]
    assert gap == pytest.approx(b["open"].iloc[0] - a["open"].iloc[10], abs=1e-12)
    assert gap != 0.0
    old = df[df["contract"] == "EUF2026"]
    assert (old["close"] - old["raw_close"]).round(10).tolist() == [round(gap, 10)] * len(old)
