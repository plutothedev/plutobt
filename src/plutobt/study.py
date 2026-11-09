"""run_study: did you overfit? (3.0 spec §6)

Runs every setting in a grid once through the unchanged engine, records each run in the
ledger, and answers four questions about the best one:

    Deflated Sharpe   is it better than the best of N zero-edge tries would look by luck,
                      N being every try ever recorded for this strategy on these bars?
    PBO               how often does the in-sample winner fall to the bottom half
                      out-of-sample?
    walk-forward      what would picking the best on the past and trading it next have
                      made, rerun flat in each test window?
    purged CV         the same pick-then-test on every block, with an embargo against
                      positions leaking across block edges.

Everything about the full runs is IN-SAMPLE; the walk-forward and CV reruns are
out-of-sample. Nothing here is a claim about live results.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from plutobt.config import BacktestConfig
from plutobt.data import frame_sha256
from plutobt.deflate import deflated_sharpe, expected_max_sharpe, verdict
from plutobt.engine import Engine
from plutobt.ledger import DEFAULT_PATH, Ledger, _plain, params_key, strategy_id
from plutobt.oos import (
    PurgedCV,
    WalkForward,
    _bar_sharpe,
    _build,
    purged_cv,
    walk_forward,
)
from plutobt.pbo import PBOResult, pbo
from plutobt.splits import cscv_blocks, purged_folds, walk_forward_windows
from plutobt.strategy import Strategy


class StudyRunError(RuntimeError):
    """One setting's run failed; the study stops and names it."""


@dataclass
class StudyResult:
    strategy: str
    grid: pd.DataFrame            # one row per setting: params, sharpe, sharpe_ann, trades,
    #                               net_pnl
    best: dict
    tie: bool
    returns: pd.DataFrame         # rows x settings, row i = the return on bar i + 1
    periods_per_year: float
    data_sha256: str
    ledger_path: str | None
    ledger_new: bool
    n_this: int
    n_earlier: int
    deflation: dict               # dsr, n_trials, var_sr, luck_bar, luck_bar_ann, verdict
    pbo: PBOResult
    walk_forward: WalkForward
    cv: PurgedCV


@dataclass
class _Context:
    strategy_cls: type
    settings: list[dict]
    returns: pd.DataFrame
    sharpes: list[float]
    trades: list[list]
    net_pnl: list[float]
    df: pd.DataFrame
    instrument: object
    config: BacktestConfig
    roll_meta: object
    ppy: float


def expand_grid(strategy_cls: type, grid: dict) -> list[dict]:
    """Every combination of ``grid``'s values, in grid order, duplicates collapsed (20,
    20.0 and numpy's 20 are one setting). Refuses names the class does not have, methods,
    empty grids or value lists, and a class that cannot be built with no arguments."""
    if not (isinstance(strategy_cls, type) and issubclass(strategy_cls, Strategy)):
        raise ValueError(f"{strategy_cls!r} is not a Strategy subclass")
    try:
        strategy_cls()
    except TypeError as err:
        raise ValueError(
            f"{strategy_cls.__name__} cannot be built with no arguments ({err}): a study "
            "builds each setting as Cls() and then sets the grid's attributes, so give "
            "every setting a class-level default") from None
    if not grid:
        raise ValueError("the grid is empty: give at least one setting name and its values")
    for name, values in grid.items():
        if name.startswith("_") or not hasattr(strategy_cls, name):
            raise ValueError(
                f"{strategy_cls.__name__} has no setting {name!r}: a grid name must be a "
                "class attribute the strategy reads (a typo would quietly test the default)")
        if callable(getattr(strategy_cls, name)):
            raise ValueError(f"{name!r} is a method of {strategy_cls.__name__}, not a setting")
        if not list(values):
            raise ValueError(f"setting {name!r} has no values to try")
    names = list(grid)
    seen, out = set(), []
    for combo in itertools.product(*(list(grid[n]) for n in names)):
        params = {n: _plain(n, v) for n, v in zip(names, combo, strict=True)}
        key = params_key(params)
        if key not in seen:
            seen.add(key)
            out.append(params)
    return out


def _context(strategy_cls, settings, df, instrument, config, roll_meta,
             on_run=None) -> _Context:
    """Run every setting once over all the bars; ``on_run(params, bars, sharpe)`` after
    each completed run."""
    cols, sharpes, trades, pnl = {}, [], [], []
    for k, params in enumerate(settings):
        try:
            res = Engine(instrument, config).run(df, _build(strategy_cls, params),
                                                 roll_meta=roll_meta)
        except Exception as err:
            raise StudyRunError(
                f"setting {k + 1} of {len(settings)} ({params_key(params)}) failed: "
                f"{type(err).__name__}: {err}. The study stops here; settings run before "
                "it are in the ledger") from err
        r = res.equity_curve.pct_change().dropna()
        sr = _bar_sharpe(r)
        cols[k] = r.to_numpy()
        index = r.index
        sharpes.append(sr)
        trades.append(res.trades)
        pnl.append(float(res.equity_curve.iloc[-1] - res.equity_curve.iloc[0]))
        if on_run is not None:
            on_run(params, len(df), sr)
    returns = pd.DataFrame(cols, index=index)
    return _Context(strategy_cls, settings, returns, sharpes, trades, pnl, df, instrument,
                    config, roll_meta, float(config.periods_per_year))


def run_study(strategy_cls, grid, df, instrument, config=None, *, roll_meta=None,
              ledger=DEFAULT_PATH, max_configs=1000, blocks=16, wf_splits=5,
              wf_anchored=False, cv_folds=6, embargo_bars=None) -> StudyResult:
    """Run the grid and answer the overfitting questions; see the module docstring."""
    config = config or BacktestConfig()
    if config.periods_per_year is None:
        raise ValueError(
            "run_study needs config.periods_per_year (bars a year, e.g. 20412 for 5-minute "
            "RTH index bars) to annualize its Sharpes")
    settings = expand_grid(strategy_cls, grid)
    if len(settings) > max_configs:
        raise ValueError(
            f"the grid makes {len(settings):,} settings, over max_configs={max_configs:,}: "
            "each is one full backtest, so raise max_configs if you mean it")
    n_rows = len(df) - 1
    # Every split is checked before the first run: too little data fails in a second.
    cscv_blocks(n_rows, blocks)
    walk_forward_windows(n_rows, wf_splits, wf_anchored)
    purged_folds(n_rows, cv_folds, 0)

    sid, sha = strategy_id(strategy_cls), frame_sha256(df)
    led = Ledger(ledger) if ledger else None
    before = led.tries(sid, sha) if led else {}
    record = (lambda p, bars, sr: led.record(sid, p, sha, bars, sr)) if led else None
    ctx = _context(strategy_cls, settings, df, instrument, config, roll_meta, on_run=record)

    srs = np.array(ctx.sharpes, dtype=float)
    if not np.isfinite(srs).any():
        raise ValueError("every setting never traded (no return volatility in any full "
                         "run): there is nothing to pick")
    best_i = int(np.nanargmax(srs))
    tie = int(np.sum(srs == srs[best_i])) > 1

    this = {params_key(p): s for p, s in zip(settings, ctx.sharpes, strict=True)}
    tries = led.tries(sid, sha) if led else this
    values = np.array([v for v in tries.values() if math.isfinite(v)])
    var_sr = float(np.var(values)) if values.size > 1 else 0.0
    n_trials = len(tries)
    dsr = deflated_sharpe(ctx.returns.iloc[:, best_i], n_trials, var_sr)
    luck = expected_max_sharpe(n_trials, var_sr)
    root = math.sqrt(ctx.ppy)

    table = pd.DataFrame(settings)
    table["sharpe"] = srs
    table["sharpe_ann"] = srs * root
    table["trades"] = [len(t) for t in ctx.trades]
    table["net_pnl"] = ctx.net_pnl

    return StudyResult(
        strategy=sid, grid=table, best=dict(settings[best_i]), tie=tie,
        returns=ctx.returns, periods_per_year=ctx.ppy, data_sha256=sha,
        ledger_path=str(led.path) if led else None, ledger_new=led.is_new if led else False,
        n_this=len(this), n_earlier=len(set(before) - set(this)),
        deflation={"dsr": dsr, "n_trials": n_trials, "var_sr": var_sr, "luck_bar": luck,
                   "luck_bar_ann": luck * root, "verdict": verdict(dsr, n_trials)},
        pbo=pbo(ctx.returns.to_numpy(), blocks),
        walk_forward=walk_forward(ctx, wf_splits, wf_anchored),
        cv=purged_cv(ctx, cv_folds, embargo_bars),
    )
