"""OpeningRange: opening-range breakout, flat by the session close."""
from __future__ import annotations

import numpy as np
import pandas as pd

from plutobt.data import CT
from plutobt.roll import _days
from plutobt.strategy import Context, Strategy


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
    trading day. The position is closed on the first bar whose open plus the bar
    interval reaches the session close, so it is flat by the close.

    Default: ``range_minutes = 30``, the opening range of Toby Crabel, "Day Trading
    with Short Term Price Patterns and Opening Range Breakout" (1990). It is a
    published default, not a number tuned on any data. The citation is taken from the
    PlutoBT 3.1 plan and was not re-checked against the book.

    Causality: precompute adds only per-bar facts taken from each bar's own timestamp
    (``ct_min`` minutes of day in Chicago, ``tday`` the CME trading day as an integer,
    ``bar_secs`` the smallest positive gap between consecutive bars seen so far). The
    range itself is built bar by bar in ``ctx.state``.
    """

    range_minutes: int = 30

    def precompute(self, df: pd.DataFrame) -> pd.DataFrame:
        ts = df["ts"].to_numpy(dtype="int64")
        if len(ts) > 1:
            gaps = np.diff(ts)
            gaps = gaps[gaps > 0]
            if gaps.size and gaps.min() > self.range_minutes * 60:
                minutes = int(round(gaps.min() / 60))
                raise ValueError(
                    f"opening_range needs bars of {self.range_minutes} minutes or less; "
                    f"these are {minutes}-minute bars"
                )
        local = pd.DatetimeIndex(pd.to_datetime(ts, unit="s", utc=True)).tz_convert(CT)
        df["ct_min"] = (local.hour * 60 + local.minute).to_numpy(dtype="int64")
        df["tday"] = np.array([d.toordinal() for d in _days(ts)], dtype="int64")
        gap = pd.Series(np.diff(ts, prepend=ts[:1]).astype("float64"))
        gap[gap <= 0] = np.nan
        df["bar_secs"] = gap.cummin().to_numpy()
        return df

    def on_bar(self, ctx: Context) -> None:
        bar = ctx.bar
        state = ctx.state
        if state.get("day") != bar.tday:
            state.update(day=bar.tday, hi=None, lo=None, entered=False)
        open_m = _minutes(ctx.instrument.session_open_ct)
        close_m = _minutes(ctx.instrument.session_close_ct)
        minute = int(bar.ct_min)
        if bar.bar_secs != bar.bar_secs:  # first bar: no interval seen yet
            return
        ends = minute + bar.bar_secs / 60.0

        if open_m <= minute < open_m + self.range_minutes:
            state["hi"] = bar.high if state["hi"] is None else max(state["hi"], bar.high)
            state["lo"] = bar.low if state["lo"] is None else min(state["lo"], bar.low)
            return

        if ctx.position.qty != 0:
            if ends >= close_m:
                ctx.broker.close()
            return

        if state["entered"] or state["hi"] is None:
            return
        if minute < open_m + self.range_minutes or ends >= close_m:
            return
        if bar.close > state["hi"]:
            ctx.broker.buy(contracts=1, stop=state["lo"])
            state["entered"] = True
        elif bar.close < state["lo"] and ctx.broker.config.allow_short:
            ctx.broker.sell(contracts=1, stop=state["hi"])
            state["entered"] = True
