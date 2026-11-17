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
    ts = _day_ts(day, end="15:15")  # the 15:15 bar is the first after the close
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
    day = "2024-03-05"
    df = _orb_day(day)
    close = df["close"].to_numpy().copy()
    close[10:] = 99.0  # falls through the range low (99.5): the stop must be that low
    df = _frame(df["ts"].to_numpy(), close)
    res = _run(df, OpeningRange(), allow_short=False)
    ts = df["ts"].to_numpy()
    assert len(res.trades) == 1
    t = res.trades[0]
    assert t.side == "long" and t.contracts == 1
    assert t.entry_ts == int(ts[7])      # signal on the 09:00 bar, fill at the next open
    assert t.exit_ts == int(ts[10])      # stopped on the first bar that trades through 99.5
    assert t.exit_price == pytest.approx(99.5, abs=1.0)  # at the range low, less slippage


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


def test_orb_flat_by_session_close():
    days = ["2024-03-05", "2024-03-06"]
    df = pd.concat([_orb_day(d) for d in days], ignore_index=True)
    res = _run(df, OpeningRange())
    ts = df["ts"].to_numpy()
    n = len(ts) // 2
    assert len(res.trades) == 2
    for k, trade in enumerate(res.trades):
        assert not trade.open_at_end
        # the 15:10 bar (open + 5 min = the 15:15 close) decides; it fills at the 15:15 open
        assert trade.exit_ts == int(ts[k * n + n - 1])
    # held between the entry and the close decision, flat from the 15:15 bar on
    assert res.position.iloc[n - 2] == 1
    assert res.position.iloc[n - 1] == 0
    assert res.position.iloc[2 * n - 1] == 0


def test_orb_no_short_when_shorts_off():
    df = _orb_day("2024-03-05", breakout="down")
    assert len(_run(df, OpeningRange(), allow_short=False).trades) == 0
    shorted = _run(df, OpeningRange(), allow_short=True).trades
    assert len(shorted) == 1 and shorted[0].side == "short"


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
