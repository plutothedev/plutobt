import numpy as np
import pandas as pd
import pytest

from plutobt.config import BacktestConfig
from plutobt.engine import Engine
from plutobt.instruments import get_instrument
from plutobt.strategies.donchian import Donchian
from plutobt.strategies.opening_range import OpeningRange
from plutobt.strategy_loader import load_strategy

CT = "America/Chicago"


def _day_ts(day, start="08:30", end="15:10", step=5):
    """Bar-open unix seconds for one Chicago day."""
    idx = pd.date_range(f"{day} {start}", f"{day} {end}", freq=f"{step}min", tz=CT)
    return (idx.tz_convert("UTC").asi8 // 10**9).astype("int64")


def _frame(ts, close, spread=0.5):
    close = np.asarray(close, dtype=float)
    return pd.DataFrame({
        "ts": ts, "open": close, "high": close + spread, "low": close - spread,
        "close": close, "volume": np.full(len(ts), 100.0),
    })


def _orb_day(day, breakout="up"):
    """Range 08:30-08:55 (high 100.5, low 99.5); the break starts at the 09:00 bar."""
    ts = _day_ts(day)  # RTH: the last bar opens 15:10, as `plutobt run` leaves it
    close = np.full(len(ts), 100.0)
    if breakout == "up":
        close[6:] = 103.0
    elif breakout == "down":
        close[6:] = 97.0
    return _frame(ts, close)


def _run(df, strat, **cfg):
    config = BacktestConfig(check_causality=True, causality_sample=None, **cfg)
    return Engine(get_instrument("NQ"), config).run(df, strat)


def test_orb_buys_break_above_range_with_stop_at_low():
    df = _orb_day("2024-03-05")
    close = df["close"].to_numpy().copy()
    close[10:] = 99.75  # opens above the range low, trades through it: stop fills at 99.5
    df = _frame(df["ts"].to_numpy(), close)
    res = _run(df, OpeningRange(), allow_short=False)
    ts = df["ts"].to_numpy()
    assert len(res.trades) == 1
    t = res.trades[0]
    assert t.side == "long" and t.contracts == 1
    assert t.entry_ts == int(ts[7])      # signal on the 09:00 bar, fill at the next open
    assert t.exit_ts == int(ts[10])
    assert abs(t.exit_price - 99.5) <= 0.5   # the range low, not the bar low (99.25)
    assert t.exit_price > 99.25 - 1e-9 and t.r_multiple < 0


def test_orb_short_stop_at_range_high():
    df = _orb_day("2024-03-05", breakout="down")
    close = df["close"].to_numpy().copy()
    close[10:] = 100.25  # opens below the range high (100.5), trades through it
    df = _frame(df["ts"].to_numpy(), close)
    res = _run(df, OpeningRange(), allow_short=True)
    t = res.trades[0]
    assert t.side == "short"
    assert abs(t.exit_price - 100.5) <= 0.5 and t.exit_price < 100.75 + 1e-9


def test_orb_range_includes_the_first_bar_of_the_frame():
    df = _orb_day("2024-03-05")
    high = df["high"].to_numpy().copy()
    high[0] = 105.5  # the 08:30 bar sets the range high above the 103 break
    df["high"] = high
    assert len(_run(df, OpeningRange()).trades) == 0


def test_orb_one_entry_per_day():
    # break up, get stopped out below the range, then break up again the same day
    df = _orb_day("2024-03-05")
    close = df["close"].to_numpy().copy()
    close[10:20] = 90.0
    close[20:] = 105.0
    df = _frame(df["ts"].to_numpy(), close)
    res = _run(df, OpeningRange(), allow_short=False)
    assert len(res.trades) == 1


def test_orb_enters_again_on_the_next_day():
    df = pd.concat([_orb_day("2024-03-05"), _orb_day("2024-03-06")], ignore_index=True)
    assert len(_run(df, OpeningRange()).trades) == 2


def _flat_by_close(df, n_per_day, days):
    res = _run(df, OpeningRange())
    ts = df["ts"].to_numpy()
    assert len(res.trades) == days
    for k, trade in enumerate(res.trades):
        assert not trade.open_at_end
        last_rth = int(_day_ts(["2024-03-05", "2024-03-06"][k])[-1])  # the 15:10 bar
        assert trade.exit_ts == last_rth
        assert res.position.iloc[int(np.searchsorted(ts, last_rth))] == 0
    return res


def test_orb_flat_by_session_close():
    # RTH frame, as `plutobt run` builds it: the last bar of each day opens 15:10
    days = ["2024-03-05", "2024-03-06"]
    df = pd.concat([_orb_day(d) for d in days], ignore_index=True)
    res = _flat_by_close(df, len(df) // 2, 2)
    n = len(df) // 2
    assert res.position.iloc[n - 2] == 1


def test_orb_flat_by_session_close_on_a_full_day_frame():
    days = ["2024-03-05", "2024-03-06"]
    parts = []
    for d in days:
        ts = _day_ts(d, end="16:00")
        close = np.full(len(ts), 100.0)
        close[6:] = 103.0
        parts.append(_frame(ts, close))
    _flat_by_close(pd.concat(parts, ignore_index=True), 0, 2)


def test_orb_prefix_starting_with_a_lone_late_bar_does_not_refuse():
    first = _frame(_day_ts("2024-03-04", start="15:10", end="15:10"), [100.0])
    df = pd.concat([first, _orb_day("2024-03-05")], ignore_index=True)
    assert len(_run(df, OpeningRange()).trades) == 1  # default guard runs every prefix


def test_orb_no_short_when_shorts_off():
    df = _orb_day("2024-03-05", breakout="down")
    assert len(_run(df, OpeningRange(), allow_short=False).trades) == 0
    shorted = _run(df, OpeningRange(), allow_short=True).trades
    assert len(shorted) == 1 and shorted[0].side == "short"


def test_orb_refuses_hourly_bars_with_one_stray_short_gap():
    ts = list(_day_ts("2024-03-05", step=60, end="14:30"))
    ts.insert(3, ts[2] + 300)
    df = _frame(np.array(ts, dtype="int64"), np.full(len(ts), 100.0))
    with pytest.raises(ValueError, match="60-minute bars"):
        _run(df, OpeningRange())


def test_orb_refuses_bars_longer_than_range():
    ts = _day_ts("2024-03-05", step=60, end="14:30")
    df = _frame(ts, np.full(len(ts), 100.0))
    with pytest.raises(ValueError, match="30 minutes or less; these are 60-minute bars"):
        _run(df, OpeningRange())


def test_donchian_enters_on_prior_20_high_and_exits_on_prior_10_low():
    ts = _day_ts("2024-03-05")
    close = np.full(len(ts), 100.0)
    close[25:] = 110.0      # breaks the prior 20-bar high at bar 25
    close[40:] = 100.0      # falls below the prior 10-bar low (109.5) at bar 40
    res = _run(_frame(ts, close), Donchian(), allow_short=False)
    assert len(res.trades) == 1
    t = res.trades[0]
    assert t.entry_ts == int(ts[26])
    assert t.exit_ts == int(ts[41])


def test_donchian_exit_and_opposite_breakout_on_one_bar_only_exits():
    ts = _day_ts("2024-03-05")
    close = np.full(len(ts), 100.0)
    close[25:] = 110.0
    close[40:] = 80.0       # below the 10-bar exit low and the 20-bar entry low at once
    res = _run(_frame(ts, close), Donchian(), allow_short=True)
    assert [t.side for t in res.trades] == ["long"]


def test_donchian_no_short_when_shorts_off():
    ts = _day_ts("2024-03-05")
    close = np.full(len(ts), 100.0)
    close[25:] = 90.0
    df = _frame(ts, close)
    assert len(_run(df, Donchian(), allow_short=False).trades) == 0
    assert len(_run(df, Donchian(), allow_short=True).trades) == 1


def test_builtins_pass_lookahead_guard():
    df = pd.concat([_orb_day("2024-03-05"), _orb_day("2024-03-06", "down")],
                   ignore_index=True)
    for cls in (OpeningRange, Donchian):
        _run(df, cls())  # check_causality=True, every prefix: raises if a column leaks


def test_defaults_are_the_cited_ones():
    assert (OpeningRange.range_minutes, Donchian.entry_bars, Donchian.exit_bars) == (
        30, 20, 10)


def test_loader_finds_both_builtins():
    assert load_strategy("opening_range").__name__ == "OpeningRange"
    assert load_strategy("donchian").__name__ == "Donchian"
