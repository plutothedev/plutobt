"""OpeningRange: opening-range breakout, flat by the session close."""
from __future__ import annotations

import numpy as np
import pandas as pd

from plutobt.data import CT
from plutobt.roll import _DAY_SHIFT
from plutobt.strategy import Context, Strategy

# Trading days, each with a single bar, after which a frame is read as daily bars.
_ONE_BAR_DAYS = 5


def _minutes(hhmm: str) -> int:
    hours, minutes = hhmm.split(":")
    return int(hours) * 60 + int(minutes)


class OpeningRange(Strategy):
    """Opening-range breakout.

    The range is the high and low of the bars that open in
    ``[session_open, session_open + range_minutes)`` (America/Chicago, from the
    instrument) of each CME trading day. Once the range is complete, the first bar of
    the day that closes above the range high buys 1 contract with a stop at the range
    low; the first that closes below the range low sells 1 with a stop at the range
    high, only when the run allows shorts (``allow_short``). At most one entry per
    trading day. The position is closed so that it is flat by the session close: the
    exit is decided on the bar before the last in-session bar and fills at that last
    bar's open. No entry is taken from that bar on, so every entry has a bar left to
    exit on.

    Source: the concept is from Toby Crabel, "Day Trading with Short Term Price
    Patterns and Opening Range Breakout" (1990). The 30-minute window
    (``range_minutes = 30``) is a common convention, not Crabel's rule; opening-range
    lengths of 5, 15, 30 and 60 minutes are compared in Zarattini, Barbon and Aziz,
    "A Profitable Day Trading Strategy For The U.S. Equity Market" (2024, SSRN
    4729284). It is a default taken from the literature, not a number tuned on any
    data.

    Bars must be ``range_minutes`` long or shorter: the bar size is the most common
    gap between consecutive bars of the same trading day, and a longer one is refused.
    Daily or longer bars (five or more trading days of one bar each) are refused too.

    Causality: precompute adds only per-bar facts taken from the bars up to and
    including that bar (``ct_min`` minutes of day in Chicago, ``tday`` the CME trading
    day as an integer, ``bar_secs`` the most common same-day gap seen so far, 0 until
    one is seen). The range itself is built bar by bar in ``ctx.state``.
    """

    range_minutes: int = 30

    def precompute(self, df: pd.DataFrame) -> pd.DataFrame:
        ts = df["ts"].to_numpy(dtype="int64")
        local = pd.DatetimeIndex(pd.to_datetime(ts, unit="s", utc=True)).tz_convert(CT)
        df["ct_min"] = (local.hour * 60 + local.minute).to_numpy(dtype="int64")
        wall = (local + _DAY_SHIFT).tz_localize(None).normalize()
        day = np.asarray((wall - pd.Timestamp(0)).days, dtype="int64")
        df["tday"] = day
        # Gap to the previous bar, only when both bars are in the same trading day.
        gap = np.zeros(len(ts), dtype="int64")
        if len(ts) > 1:
            same = day[1:] == day[:-1]
            step = np.diff(ts)
            gap[1:] = np.where(same & (step > 0), step, 0)
        # Expanding mode of the same-day gaps (ties go to the smaller gap).
        sizes = np.unique(gap[gap > 0])
        bar_secs = np.zeros(len(ts), dtype="float64")
        best_count = np.zeros(len(ts), dtype="int64")
        for v in sizes:  # ascending, strict > : a tie keeps the smaller gap
            count = np.cumsum(gap == v)
            better = count > best_count
            bar_secs = np.where(better, float(v), bar_secs)
            best_count = np.where(better, count, best_count)
        df["bar_secs"] = bar_secs
        # No two bars share a trading day across several days: daily (or longer) bars,
        # where no opening range can form. A short frame is let through, because the
        # look-ahead check runs this on the first rows alone, and a file can open with a
        # lone bar or two before its intraday bars.
        if len(ts) and bar_secs.max() == 0 and len(np.unique(day)) >= _ONE_BAR_DAYS:
            raise ValueError(
                f"opening_range needs bars of {self.range_minutes} minutes or less; these "
                "bars are one per trading day (daily or longer)"
            )
        if bar_secs.max() > self.range_minutes * 60:
            raise ValueError(
                f"opening_range needs bars of {self.range_minutes} minutes or less; "
                f"these are {int(round(bar_secs.max() / 60))}-minute bars"
            )
        return df

    def on_bar(self, ctx: Context) -> None:
        bar = ctx.bar
        state = ctx.state
        if state.get("day") != bar.tday:
            state.update(day=bar.tday, hi=None, lo=None, entered=False)
        open_m = _minutes(ctx.instrument.session_open_ct)
        close_m = _minutes(ctx.instrument.session_close_ct)
        minute = int(bar.ct_min)

        if open_m <= minute < open_m + self.range_minutes:
            state["hi"] = bar.high if state["hi"] is None else max(state["hi"], bar.high)
            state["lo"] = bar.low if state["lo"] is None else min(state["lo"], bar.low)
            return
        if bar.bar_secs <= 0:  # no same-day gap seen yet: the interval is unknown
            return
        # The bar after this one opens at minute + interval, and an order decided now
        # fills there; "last" below means the last bar opening before the session close.
        next_is_last = minute + 2 * bar.bar_secs / 60.0 >= close_m

        if ctx.position.qty != 0:
            if next_is_last:
                ctx.broker.close()
            return

        if state["entered"] or state["hi"] is None:
            return
        if minute < open_m + self.range_minutes or next_is_last:
            return
        if bar.close > state["hi"]:
            ctx.broker.buy(contracts=1, stop=state["lo"])
            state["entered"] = True
        elif bar.close < state["lo"] and ctx.broker.config.allow_short:
            ctx.broker.sell(contracts=1, stop=state["hi"])
            state["entered"] = True
