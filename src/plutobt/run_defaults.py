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
# per-side figure (exchange + clearing + NFA + broker). NQ, ES and CL copy the repo's
# shipped configs exactly (configs/nq.toml, es.toml, cl.toml). The others follow typical
# published retail per-side schedules (2025/2026 NinjaTrader-style all-in rates; micros
# cost less than minis); slippage is the instrument registry's own ticks per side.
DEFAULT_COSTS: dict[str, tuple[float, float]] = {
    "NQ": (2.50, 1.0),    # shipped configs/nq.toml
    "MNQ": (0.75, 1.0),   # retail micro all-in per side, ~$0.75 (2025 public schedules)
    "ES": (2.50, 1.0),    # shipped configs/es.toml
    "MES": (0.75, 1.0),   # retail micro all-in per side, ~$0.75 (2025 public schedules)
    "YM": (2.50, 1.0),    # retail mini all-in per side, ~$2.50 (2025 public schedules)
    "RTY": (2.50, 1.0),   # retail mini all-in per side, ~$2.50 (2025 public schedules)
    "CL": (3.50, 3.0),    # shipped configs/cl.toml
    "GC": (3.50, 2.0),    # retail COMEX all-in per side, ~$3.50 (2025 public schedules)
    "ZN": (2.50, 1.0),    # retail CBOT all-in per side, ~$2.50 (2025 public schedules)
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
    """252 times the median bars per CME trading day, measured on ``df``'s ``ts``."""
    per_day = pd.Series(1, index=_days(df["ts"].to_numpy())).groupby(level=0).sum()
    return 252.0 * float(np.median(per_day.to_numpy()))
