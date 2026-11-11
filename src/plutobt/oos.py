"""Out-of-sample checks for a study: flat reruns, walk-forward and purged CV (3.0 spec
§8, §9).

A setting is PICKED on a slice of its full run (cheap: the full runs already exist), then
RERUN flat inside the test window: ``Engine.run(..., trade_from_ts=<window start>)`` over
every bar up to the window's end, so earlier bars only warm the indicators and no
position crosses into the window. The windows' P&L stitched together is a curve that could
actually have been traded.

Row convention (shared with ``splits``): returns-table row ``i`` is the return earned on
bar ``i + 1``. A row window ``[a, b)`` is traded by a rerun that starts on bar ``a`` (its
decision there fills on bar ``a + 1``) and its P&L is ``equity[b] - equity[a]``.

A position still open at a window's end is marked at the last close (the engine's
end-of-data rule) and charged one exit side, commission plus slippage, as if closed
there: without it the stitched curve would skip every exit's cost.

These functions take the study's context (``study._context``): the strategy class, its
settings, the full-run returns table and trades, the bars, instrument, config and roll
metadata.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from plutobt.engine import Engine
from plutobt.metrics import psr, sharpe, sharpe_stats
from plutobt.portfolio import RollEvent
from plutobt.splits import MIN_BARS, purged_folds, walk_forward_windows

_START = 100_000.0


@dataclass
class WindowRun:
    pnl: float                    # equity[b] - equity[a], before the exit charge
    equity: pd.Series             # the rerun's equity on bars a..b
    open_at_end: bool             # a position was still open on bar b
    exit_charge: float            # one exit side charged when open_at_end, else 0
    rolls: list[RollEvent] = field(default_factory=list)


@dataclass
class WalkForward:
    windows: list[dict]           # train, test, params, is_sharpe, oos_sharpe, open_at_end,
    #                               pnl, exit_charge
    equity: pd.Series             # stitched out-of-sample equity from the starting cash
    sharpe: float                 # per bar, out-of-sample
    sharpe_ann: float
    sharpe_se: float              # annualized, as format_report prints it
    psr: float
    open_windows: int
    winner_changed: bool
    is_sharpe_mean: float         # per bar: mean in-sample Sharpe of the picks


@dataclass
class PurgedCV:
    folds: list[dict]             # test, params, oos_sharpe, not_real_time
    sharpes: list[float]          # per bar, one per fold
    spread: float                 # population std of the fold Sharpes
    embargo: int
    capped: bool = False          # the default embargo was cut short to keep training rows


def _build(strategy_cls, params: dict):
    strategy = strategy_cls()
    for name, value in params.items():
        setattr(strategy, name, value)
    return strategy


def _bar_sharpe(returns) -> float:
    return sharpe(np.asarray(returns, dtype=float), 1.0)


def _pick(returns: pd.DataFrame, rows: np.ndarray) -> tuple[int | None, float]:
    """The column with the best per-bar Sharpe over ``rows``: NaN (no volatility) is
    skipped, ties go to the first. (None, NaN) when every column is NaN."""
    part = returns.iloc[rows]
    srs = np.array([_bar_sharpe(part.iloc[:, c]) for c in range(part.shape[1])])
    if not np.isfinite(srs).any():
        return None, float("nan")
    idx = int(np.nanargmax(srs))
    return idx, float(srs[idx])


def rerun_window(strategy_cls, params: dict, df: pd.DataFrame, instrument, config,
                 roll_meta, a: int, b: int) -> WindowRun:
    """Rerun one setting flat over rows ``[a, b)`` (bars ``a..b``)."""
    end_ts, start_ts = int(df["ts"].iloc[b]), int(df["ts"].iloc[a])
    if roll_meta is not None:
        from plutobt.roll import slice_continuous

        cut, meta = slice_continuous(df, roll_meta, end_ts=end_ts)
    else:
        cut, meta = df.iloc[: b + 1].reset_index(drop=True), None
    res = Engine(instrument, config).run(cut, _build(strategy_cls, params), roll_meta=meta,
                                         trade_from_ts=start_ts)
    equity = res.equity_curve.iloc[a: b + 1]
    last = res.trades[-1] if res.trades else None
    open_at_end = bool(last is not None and last.open_at_end)
    charge = 0.0
    if open_at_end:
        slip = config.slippage_ticks if config.slippage_ticks is not None \
            else instrument.slippage_ticks
        charge = last.contracts * (config.commission_per_side + slip * instrument.tick_value)
    return WindowRun(pnl=float(equity.iloc[-1] - equity.iloc[0]), equity=equity,
                     open_at_end=open_at_end, exit_charge=float(charge), rolls=list(res.rolls))


def _charged(run: WindowRun) -> np.ndarray:
    eq = run.equity.to_numpy(dtype=float).copy()
    eq[-1] -= run.exit_charge
    return eq


def _test_block(ctx, idx: int, test: range):
    run = rerun_window(ctx.strategy_cls, ctx.settings[idx], ctx.df, ctx.instrument,
                       ctx.config, ctx.roll_meta, test.start, test.stop)
    eq = _charged(run)
    return run, eq, _bar_sharpe(eq[1:] / eq[:-1] - 1.0)


def walk_forward(ctx, splits: int, anchored: bool) -> WalkForward:
    """Pick on each train window's slice, rerun the pick flat in its test window, stitch."""
    windows = []
    values, index = [], []
    level = _START
    for k, (train, test) in enumerate(walk_forward_windows(len(ctx.returns), splits,
                                                           anchored)):
        idx, is_sr = _pick(ctx.returns, np.arange(train.start, train.stop))
        if idx is None:
            raise ValueError(
                f"no setting traded in walk-forward train window {k + 1} (rows "
                f"{train.start}-{train.stop - 1}): every setting is flat there, so nothing "
                "can be picked; use fewer windows or a longer history")
        run, eq, oos_sr = _test_block(ctx, idx, test)
        steps = np.diff(eq)
        values.extend(level + np.cumsum(steps))
        level = values[-1]
        index.extend(run.equity.index[1:])
        windows.append({"train": train, "test": test, "params": dict(ctx.settings[idx]),
                        "is_sharpe": is_sr, "oos_sharpe": oos_sr,
                        "open_at_end": run.open_at_end, "pnl": run.pnl,
                        "exit_charge": run.exit_charge})
    equity = pd.Series(values, index=pd.Index(index, name="ts"), dtype=float)
    full = np.r_[_START, equity.to_numpy()]
    returns = pd.Series(full[1:] / full[:-1] - 1.0)
    stats = sharpe_stats(returns, ctx.ppy)
    keys = {tuple(sorted(w["params"].items())) for w in windows}
    return WalkForward(
        windows=windows, equity=equity, sharpe=float(stats["sharpe_raw"]),
        sharpe_ann=float(stats["sharpe"]), sharpe_se=float(stats["sharpe_se"]),
        psr=float(psr(returns)), open_windows=sum(w["open_at_end"] for w in windows),
        winner_changed=len(keys) > 1,
        is_sharpe_mean=float(np.mean([w["is_sharpe"] for w in windows])))


def longest_trade_bars(ctx) -> int:
    """The longest trade in the study's full runs, in bars (a trade still open at the end
    runs to the last bar); at least 1."""
    ts = ctx.df["ts"].to_numpy()
    last = len(ts) - 1
    longest = 1
    for trades in ctx.trades:
        for t in trades:
            entry = int(np.searchsorted(ts, t.entry_ts))
            out = last if (t.open_at_end or t.exit_ts is None) \
                else int(np.searchsorted(ts, t.exit_ts))
            longest = max(longest, out - entry)
    return longest


def purged_cv(ctx, folds: int, embargo: int | None) -> PurgedCV:
    """Each block is the test block once: pick on the rest (less the embargo after the
    test block), rerun the pick flat in the test block.

    The default embargo is the longest trade in bars, capped so every fold keeps at least
    ``MIN_BARS`` training rows: a strategy that holds for most of the history would
    otherwise leave the first fold nothing to train on. When the cap bites, ``capped`` is
    set and the report says so: positions longer than the embargo still leak, so the CV
    numbers carry some of it (walk-forward does not). An embargo passed in is never
    capped."""
    capped = False
    if embargo is None:
        n = len(ctx.returns)
        most = n - n // folds - MIN_BARS            # the first fold's training rows >= MIN_BARS
        longest = longest_trade_bars(ctx)
        embargo = max(1, min(longest, most))
        capped = embargo < longest
    else:
        embargo = int(embargo)
    out, sharpes = [], []
    for k, (train, test) in enumerate(purged_folds(len(ctx.returns), folds, embargo)):
        idx, _ = _pick(ctx.returns, train)
        if idx is None:
            raise ValueError(
                f"no setting traded in the training rows of CV fold {k + 1}: every setting "
                "is flat there, so nothing can be picked")
        _, _, oos_sr = _test_block(ctx, idx, test)
        sharpes.append(oos_sr)
        out.append({"test": test, "params": dict(ctx.settings[idx]), "oos_sharpe": oos_sr,
                    "not_real_time": bool(train.max() >= test.stop)})
    return PurgedCV(folds=out, sharpes=sharpes, spread=float(np.std(sharpes)),
                    embargo=embargo, capped=capped)
