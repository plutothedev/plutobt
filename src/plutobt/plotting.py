"""Equity/drawdown + trade-marker plots (§3/§6)."""
from __future__ import annotations

from typing import TYPE_CHECKING

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
    # Lazy import: never lock the backend at module import; the caller/test
    # picks the backend (e.g. Agg for headless) before importing pyplot.
    import matplotlib.pyplot as plt

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
    # Lazy import: never lock the backend at module import; the caller/test
    # picks the backend (e.g. Agg for headless) before importing pyplot.
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots()
    ax.plot(df["ts"].to_numpy(), df["close"].to_numpy(), label="close",
            color="C0", linewidth=0.8)
    # Colour = side (long green, short red); shape = what happened: an entry is a
    # triangle pointing the way the trade bets (a long buys "^", a short sells "v"),
    # and an exit is an "x" in its side's colour. All four kinds stay distinct.
    # Each kind is labelled once for the legend ("_" labels are skipped).
    labelled: set[str] = set()

    def _mark(ts, price, marker, color, label):
        ax.scatter([ts], [price], marker=marker, color=color, zorder=3,
                   label=label if label not in labelled else "_" + label)
        labelled.add(label)

    for tr in result.trades:
        short = tr.side == "short"
        color = "C3" if short else "C2"
        side = "short" if short else "long"
        _mark(tr.entry_ts, tr.entry_price, "v" if short else "^", color, f"{side} entry")
        if tr.exit_ts is not None and tr.exit_price is not None:
            _mark(tr.exit_ts, tr.exit_price, "x", color, f"{side} exit")
    ax.set_xlabel("bar ts")
    ax.set_ylabel("price")
    ax.legend(loc="best")
    return ax
