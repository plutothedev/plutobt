"""Equity/drawdown + trade-marker plots (§3/§6)."""
from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt

if TYPE_CHECKING:
    import pandas as pd

    from plutobt.engine import BacktestResult


def plot_equity(result: BacktestResult, ax=None):
    """Plot the equity curve (and its running-peak drawdown shading).

    Parameters
    ----------
    result:
        BacktestResult from Engine.run.
    ax:
        Optional matplotlib Axes. Created if not supplied.

    Returns
    -------
    The matplotlib Axes with the equity curve drawn.
    """
    if ax is None:
        _, ax = plt.subplots()
    eq = result.equity_curve
    ax.plot(eq.index, eq.to_numpy(), label="equity", color="C0")
    peak = eq.cummax()
    ax.fill_between(eq.index, eq.to_numpy(), peak.to_numpy(),
                    where=(eq < peak).to_numpy(), color="C3", alpha=0.2,
                    label="drawdown")
    ax.set_xlabel("bar ts")
    ax.set_ylabel("equity")
    ax.legend(loc="best")
    return ax


def plot_trades(df: pd.DataFrame, result: BacktestResult, ax=None):
    """Plot close price with entry/exit markers from the trade log.

    Parameters
    ----------
    df:
        The OHLCV DataFrame passed to Engine.run (used for the price line).
    result:
        BacktestResult from Engine.run.
    ax:
        Optional matplotlib Axes. Created if not supplied.

    Returns
    -------
    The matplotlib Axes with price + trade markers drawn.
    """
    if ax is None:
        _, ax = plt.subplots()
    ax.plot(df["ts"].to_numpy(), df["close"].to_numpy(), label="close",
            color="C0", linewidth=0.8)
    for tr in result.trades:
        ax.scatter([tr.entry_ts], [tr.entry_price], marker="^", color="C2",
                   zorder=3)
        if tr.exit_ts is not None and tr.exit_price is not None:
            ax.scatter([tr.exit_ts], [tr.exit_price], marker="v", color="C3",
                       zorder=3)
    ax.set_xlabel("bar ts")
    ax.set_ylabel("price")
    ax.legend(loc="best")
    return ax
