"""Default costs and the bars-per-year basis for the ``plutobt`` command.

Costs are representative public retail costs, not a live account: a commission is never
zero, and slippage is never below the instrument registry's own figure.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from plutobt.config import BacktestConfig
from plutobt.roll import _days

COST_LABEL = "representative public retail costs, not a live account"

# symbol -> (commission_per_side in dollars, slippage_ticks). Commission is an all-in
# per-side figure (exchange + clearing + NFA + broker). Slippage is the instrument
# registry's own ticks per side. Published per-side ranges, TopstepX / Tradeify:
#   TopstepX commissions and fees, retrieved 2026-11-16 (page dated 2026-09-30):
#     https://intercom.help/topstep-llc/en/articles/8284213-topstepx-commissions-and-fees
#   Tradeify trading commission fees, retrieved 2026-11-16 (page dated 2026-10-17,
#     round turn halved):
#     https://intercom.help/tradeify/en/articles/10468315-trading-commission-fees
DEFAULT_COSTS: dict[str, tuple[float, float]] = {
    "NQ": (2.50, 1.0),  # copies configs/nq.toml; published 1.89 / 2.88
    "MNQ": (0.75, 1.0),  # published 0.61 / 0.91
    "ES": (2.50, 1.0),  # copies configs/es.toml; published 1.89 / 2.88
    "MES": (0.75, 1.0),  # published 0.61 / 0.91
    "YM": (2.50, 1.0),  # published 1.89 / 2.88
    "RTY": (2.50, 1.0),  # published 1.89 / 2.88
    "CL": (3.50, 3.0),  # copies configs/cl.toml; published 2.01 / 2.99
    "GC": (3.50, 2.0),  # published 2.16 / 3.10
    "ZN": (2.50, 1.0),  # published 1.31 (TopstepX only; Tradeify does not list it)
}


def default_config(symbol: str, *, periods_per_year: float,
                   allow_short: bool = False) -> BacktestConfig:
    """BacktestConfig with the default costs for ``symbol``."""
    try:
        commission, slippage = DEFAULT_COSTS[symbol]
    except KeyError:
        raise KeyError(f"no default costs for {symbol!r}; pass --config FILE") from None
    return BacktestConfig(commission_per_side=commission, slippage_ticks=slippage,
                          periods_per_year=periods_per_year, allow_short=allow_short)


def bars_per_year(df: pd.DataFrame) -> float:
    """Bars a year measured on ``df``'s ``ts`` (see ``measure_bars_per_year``)."""
    return measure_bars_per_year(df)[0]


def measure_bars_per_year(df: pd.DataFrame) -> tuple[float, str]:
    """(bars a year, how it was measured), from ``df``'s ``ts``.

    Intraday bars: 252 times the median bars per CME trading day. Daily or longer bars
    (a median of one bar per trading day): 252 divided by the median gap between bars
    counted in weekdays (Monday to Friday), at least one. Daily bars give 252, weekly bars
    252 / 5 = 50.4, monthly bars about 12."""
    if len(df) == 0:
        raise ValueError("bars_per_year needs at least one bar")
    days = _days(df["ts"].to_numpy())
    per_day = pd.Series(1, index=days).groupby(level=0).sum()
    median = float(np.median(per_day.to_numpy()))
    if median > 1 or len(per_day) < 2:
        return 252.0 * median, "measured from these bars: median bars per trading day x 252"
    d = np.asarray(per_day.index, dtype="datetime64[D]")
    gaps = np.maximum(np.busday_count(d[:-1], d[1:]), 1)
    gap = float(np.median(gaps))
    return 252.0 / gap, (f"measured from these bars: 252 trading days a year / a median "
                         f"gap of {gap:g} weekday(s) between bars")
