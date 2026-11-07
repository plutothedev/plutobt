"""Roll schedule: calendar, volume and dates triggers (spec §6.1, §6.2, proof 4)."""

from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from plutobt.contracts import ContractSet
from plutobt.roll import TRIGGERS, Roll, _volume_roll_ts, roll_schedule, trading_day

CT = ZoneInfo("America/Chicago")


def ct(y, m, d, hh=8, mm=30) -> int:
    return int(datetime(y, m, d, hh, mm, tzinfo=CT).timestamp())


def day_bars(days, price=18000.0, vol=100, start=(8, 30), n=6, tick=0.25):
    """n 5-minute bars from ``start`` CT on each date in ``days``."""
    rows = []
    for d in days:
        t0 = ct(d.year, d.month, d.day, *start)
        v = vol(d) if callable(vol) else vol
        for i in range(n):
            px = price + i * tick
            rows.append((t0 + 300 * i, px, px + tick, px - tick, px, v))
    return pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])


def weekdays(a: date, b: date):
    d, out = a, []
    while d <= b:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def test_triggers():
    assert TRIGGERS == ("calendar", "volume", "dates")


def test_trading_day_rolls_at_17_ct():
    assert trading_day(ct(2026, 3, 12, 16, 55)) == date(2026, 3, 12)
    assert trading_day(ct(2026, 3, 12, 17, 0)) == date(2026, 3, 13)


def test_calendar_roll_ts_is_first_shared_bar_of_roll_day():
    # NQH2026: third Friday 2026-03-20, so CME's roll day is Monday 2026-03-16.
    old = day_bars(weekdays(date(2026, 3, 2), date(2026, 3, 20)))
    new = pd.concat([day_bars(weekdays(date(2026, 3, 2), date(2026, 3, 13)), 18100.0),
                     day_bars(weekdays(date(2026, 3, 16), date(2026, 3, 27)), 18100.0,
                              start=(8, 35))], ignore_index=True)
    cs = ContractSet.from_frames({"NQH2026": old, "NQM2026": new})
    assert roll_schedule(cs) == [Roll("NQH2026", "NQM2026", ct(2026, 3, 16, 8, 35),
                                      "calendar")]


def test_no_shared_bar_on_roll_day_refused():
    old = day_bars(weekdays(date(2026, 3, 2), date(2026, 3, 20)))
    new = day_bars(weekdays(date(2026, 3, 9), date(2026, 3, 27)), 18100.0)
    new = new[(new["ts"] < ct(2026, 3, 16, 0, 0)) | (new["ts"] >= ct(2026, 3, 17, 0, 0))]
    cs = ContractSet.from_frames({"NQH2026": old, "NQM2026": new.reset_index(drop=True)})
    with pytest.raises(ValueError, match="2026-03-16") as err:
        roll_schedule(cs)
    assert "NQH2026" in str(err.value) and "NQM2026" in str(err.value)


def _cl_set():
    days = weekdays(date(2025, 11, 3), date(2025, 11, 28))
    return ContractSet.from_frames({
        "CLZ2025": day_bars(days, 60.0, tick=0.01, start=(8, 0),
                            vol=lambda d: 500 if d < date(2025, 11, 17) else 100),
        "CLF2026": day_bars(days, 60.5, tick=0.01, start=(8, 0),
                            vol=lambda d: 100 if d < date(2025, 11, 17) else 500),
    })


def test_calendar_on_cl_refused():
    with pytest.raises(ValueError) as err:
        roll_schedule(_cl_set())
    assert "volume" in str(err.value) and "roll_dates" in str(err.value)


def test_dates_trigger_uses_given_day():
    rolls = roll_schedule(_cl_set(), "dates", roll_dates={"CLZ5": "2025-11-14"})
    assert rolls == [Roll("CLZ2025", "CLF2026", ct(2025, 11, 14, 8, 0), "dates")]


def test_dates_missing_entry_refused():
    with pytest.raises(ValueError, match="CLZ2025"):
        roll_schedule(_cl_set(), "dates", roll_dates={})


def test_volume_trigger_next_day():
    # CLF2026 out-trades CLZ2025 from Monday 2025-11-17: roll on Tuesday's first bar.
    rolls = roll_schedule(_cl_set(), "volume")
    assert rolls == [Roll("CLZ2025", "CLF2026", ct(2025, 11, 18, 8, 0), "volume")]


def test_volume_never_crosses_refused():
    days = weekdays(date(2025, 11, 3), date(2025, 11, 28))
    cs = ContractSet.from_frames({"CLZ2025": day_bars(days, 60.0, tick=0.01, vol=500),
                                  "CLF2026": day_bars(days, 60.5, tick=0.01, vol=100)})
    with pytest.raises(ValueError, match="never"):
        roll_schedule(cs, "volume")


def test_unknown_trigger_refused():
    with pytest.raises(ValueError, match="calendar"):
        roll_schedule(_cl_set(), "oi")


def test_volume_trigger_is_causal():
    """Proof 4b: the volume decision on any prefix of the data never differs from the
    full-data decision once the prefix reaches the roll, and is None before it."""
    rng = np.random.default_rng(7)
    days = weekdays(date(2025, 11, 3), date(2025, 11, 21))
    va = {d: int(rng.integers(100, 1000)) for d in days}
    vb = {d: int(rng.integers(100, 1000)) for d in days}
    a = day_bars(days, 60.0, tick=0.01, vol=lambda d: va[d])
    b = day_bars(days, 60.5, tick=0.01, vol=lambda d: vb[d])
    full = _volume_roll_ts(a, b)
    assert full is not None
    for cut in a["ts"]:
        got = _volume_roll_ts(a[a["ts"] <= cut], b[b["ts"] <= cut])
        assert got == (full if cut >= full else None), cut


def test_calendar_schedule_matches_cme_roll_days():
    """Proof 4a: a 2025-2026 NQ chain on CME's own roll table (tests/test_contracts.py)."""
    from tests.test_contracts import CME_TABLE

    codes = [f"NQ{'HMUZ'[m // 3 - 1]}{y}" for y, m, _, _ in CME_TABLE if y < 2027]
    frames = {}
    for i, code in enumerate(codes):
        y, m = CME_TABLE[i][0], CME_TABLE[i][1]
        start = date(y, m, 1) - timedelta(days=100)
        frames[code] = day_bars(weekdays(start, CME_TABLE[i][2]), 18000.0 + 50 * i, n=2)
    rolls = roll_schedule(ContractSet.from_frames(frames))
    assert [trading_day(r.roll_ts) for r in rolls] == [row[3] for row in CME_TABLE[:7]]
