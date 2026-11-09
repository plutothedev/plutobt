"""Time splits for the overfitting tools (3.0 spec §7, §8, §9).

Everything here works on returns-table rows: row ``i`` is the return earned on bar
``i + 1`` (``equity.pct_change().dropna()``). Ranges are half-open. Pure index
arithmetic; nothing here runs the engine.

Every block, window and fold must hold at least ``MIN_BARS`` rows: a Sharpe ratio over
fewer is noise, so a split that would produce one is refused with the largest count the
data can support.
"""
from __future__ import annotations

import numpy as np

MIN_BARS = 30


class SplitError(ValueError):
    """A split the data cannot support."""


def _too_short(n_rows: int, what: str) -> SplitError:
    return SplitError(
        f"{n_rows} return rows are too short for {what}: every piece needs at least "
        f"{MIN_BARS} rows")


def cscv_blocks(n_rows: int, blocks: int) -> tuple[int, list[range]]:
    """``blocks`` equal blocks of rows for CSCV. The ``n_rows % blocks`` remainder rows
    are dropped from the START, so the most recent bars always count. Returns
    (rows dropped, the blocks)."""
    if blocks < 2 or blocks % 2:
        raise SplitError(f"blocks must be an even number of at least 2, got {blocks}")
    size = n_rows // blocks
    if size < MIN_BARS:
        most = (n_rows // MIN_BARS) // 2 * 2
        if most < 2:
            raise _too_short(n_rows, "the Probability of Backtest Overfitting")
        raise SplitError(
            f"{blocks} blocks of {n_rows} return rows leave {size} rows a block, under "
            f"{MIN_BARS}: this data supports at most {most} blocks (pass blocks={most})")
    dropped = n_rows % blocks
    return dropped, [range(dropped + k * size, dropped + (k + 1) * size)
                     for k in range(blocks)]


def _chunks(n_rows: int, count: int) -> list[range]:
    # Equal chunks, the remainder joining the last.
    size = n_rows // count
    edges = [k * size for k in range(count)] + [n_rows]
    return [range(edges[k], edges[k + 1]) for k in range(count)]


def walk_forward_windows(n_rows: int, splits: int, anchored: bool) -> list[tuple[range, range]]:
    """``splits + 1`` equal chunks (remainder to the last); window i tests chunk i + 1 and
    trains on chunk i (rolling) or chunks 0..i (anchored). Returns (train, test) pairs."""
    if splits < 1:
        raise SplitError(f"walk-forward needs at least 1 split, got {splits}")
    if n_rows // (splits + 1) < MIN_BARS:
        most = n_rows // MIN_BARS - 1
        if most < 1:
            raise _too_short(n_rows, "walk-forward")
        raise SplitError(
            f"{splits} walk-forward windows over {n_rows} return rows leave chunks of "
            f"{n_rows // (splits + 1)} rows, under {MIN_BARS}: this data supports at most "
            f"{most} windows (pass wf_splits={most})")
    chunks = _chunks(n_rows, splits + 1)
    out = []
    for i in range(splits):
        train = range(0 if anchored else chunks[i].start, chunks[i].stop)
        out.append((train, chunks[i + 1]))
    return out


def purged_folds(n_rows: int, folds: int, embargo: int) -> list[tuple[np.ndarray, range]]:
    """``folds`` equal blocks (remainder to the last), each the test block once. Training
    is every other row except the ``embargo`` rows right after the test block, where a
    position carried out of the test block would leak its outcome. Returns (train rows,
    test range) pairs."""
    if folds < 2:
        raise SplitError(f"purged CV needs at least 2 folds, got {folds}")
    if embargo < 0:
        raise SplitError(f"embargo must be 0 or more bars, got {embargo}")
    if n_rows // folds < MIN_BARS:
        most = n_rows // MIN_BARS
        if most < 2:
            raise _too_short(n_rows, "purged cross-validation")
        raise SplitError(
            f"{folds} folds over {n_rows} return rows leave blocks of {n_rows // folds} "
            f"rows, under {MIN_BARS}: this data supports at most {most} folds "
            f"(pass cv_folds={most})")
    out = []
    rows = np.arange(n_rows)
    for k, test in enumerate(_chunks(n_rows, folds)):
        keep = (rows < test.start) | (rows >= test.stop + embargo)
        train = rows[keep]
        if train.size == 0:
            raise SplitError(
                f"fold {k + 1} of {folds} has no training rows left after its "
                f"{embargo}-bar embargo: lower embargo_bars or use fewer folds")
        out.append((train, test))
    return out
