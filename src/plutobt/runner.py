"""The run core behind ``plutobt run``: one call from bars and a strategy to a report.

It loads nothing: the caller passes a frame and a strategy class. It applies the session
filter, the default (or given) costs, runs the engine, writes one ledger line and one
chart, and returns the text to print.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from plutobt.config import BacktestConfig
from plutobt.data import bars_sha256, frame_sha256, tag_sessions
from plutobt.engine import BacktestResult, Engine
from plutobt.instruments import get_instrument
from plutobt.ledger import (
    DEFAULT_PATH,
    Ledger,
    code_sha256,
    config_key,
    resolved_settings,
    strategy_id,
)
from plutobt.metrics import compute_metrics
from plutobt.oos import _bar_sharpe
from plutobt.report import format_report
from plutobt.roll import RollMeta
from plutobt.run_defaults import COST_LABEL, bars_per_year, default_config
from plutobt.strategy import Strategy
from plutobt.strategy_loader import make_strategy

# A bar interval of about a day or more. One hour is allowed off for a DST change.
_DAILY_SECONDS = 86400 - 3600


@dataclass
class RunOutcome:
    result: BacktestResult
    text: str
    chart: Path
    tries: int | None
    periods_per_year: float
    default_costs: bool


def _is_daily(df: pd.DataFrame) -> bool:
    """Daily bars: the median positive gap between bar times is at least 23 hours."""
    gaps = np.diff(df["ts"].to_numpy(dtype="int64"))
    gaps = gaps[gaps > 0]
    return bool(gaps.size) and float(np.median(gaps)) >= _DAILY_SECONDS


def _save_chart(result: BacktestResult, path: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from plutobt.plotting import plot_equity

    fig, ax = plt.subplots(figsize=(11, 6))
    plot_equity(result, ax=ax)
    ax.set_title(title)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def run_backtest(df: pd.DataFrame, symbol: str, strategy_cls: type[Strategy], params: dict,
                 *, config: BacktestConfig | None = None, cost_note: str | None = None,
                 session: str = "rth", allow_short: bool = False,
                 roll_meta: RollMeta | None = None, data_note: str,
                 ledger: str | bool = DEFAULT_PATH,
                 chart_dir: Path = Path("plutobt_runs")) -> RunOutcome:
    instrument = get_instrument(symbol)
    daily = _is_daily(df)

    # A roll_meta pins the frame's rows, so a frame that has one is never filtered.
    if roll_meta is None and session == "rth" and not daily:
        had_session = "session" in df.columns
        tagged = df if had_session else tag_sessions(df, instrument)
        df = tagged[tagged["session"] == "RTH"].reset_index(drop=True)
        if not had_session:
            df = df.drop(columns="session")

    if session == "as_built":
        hours = "Hours: continuous file run as built"
    elif daily:
        hours = "Hours: daily bars, no session filter"
    elif session == "rth":
        hours = "Hours: regular trading hours only"
    else:
        hours = "Hours: full day (--session all)"

    from_config = config is not None and config.periods_per_year is not None
    ppy = float(config.periods_per_year) if from_config else bars_per_year(df)
    default_costs = config is None
    if config is None:
        config = default_config(symbol, periods_per_year=ppy, allow_short=allow_short)

    result = Engine(instrument, config).run(df, make_strategy(strategy_cls, params),
                                            roll_meta=roll_meta)
    result.metrics = compute_metrics(result, ppy)

    tries = None
    led = None
    if ledger:
        sid = strategy_id(strategy_cls)
        bars = bars_sha256(df)
        led = Ledger(ledger)
        sharpe = _bar_sharpe(result.equity_curve.pct_change().dropna())
        led.record(sid, resolved_settings(strategy_cls, params), bars, len(df), sharpe,
                   code=code_sha256(strategy_cls), config=config_key(config))
        tries = len(led.tries(sid, bars, also=(frame_sha256(df),)))

    name = strategy_cls.__name__
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    chart = Path(chart_dir) / f"{symbol}_{name}_{stamp}.png"
    _save_chart(result, chart, f"{symbol} {name} (in-sample)")

    basis = ("from your config" if from_config else
             "measured from these bars: median bars per trading day x 252")
    if default_costs:
        costs = (f"Costs: default, {COST_LABEL}: ${config.commission_per_side:.2f}/side "
                 f"commission, {config.slippage_ticks:g} tick(s) slippage")
    else:
        costs = f"Costs: {cost_note or 'from your config'}"
    last = (f"Tries of this strategy on these bars: {tries} (ledger {led.path})"
            if led is not None else "Ledger: off")
    lines = [format_report(result, symbol), f"Data: {data_note}", hours,
             f"Bars per year: {ppy:g} ({basis})", costs, last]
    return RunOutcome(result=result, text="\n".join(lines), chart=chart, tries=tries,
                      periods_per_year=ppy, default_costs=default_costs)
