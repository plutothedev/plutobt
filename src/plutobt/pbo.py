"""Probability of Backtest Overfitting via combinatorially symmetric cross-validation
(3.0 spec §7; Bailey, Borwein, Lopez de Prado and Zhu, "The Probability of Backtest
Overfitting").

The returns table (rows = bars, columns = settings) is cut into an even number of equal
blocks of time. For every way to call half of them in-sample, the setting with the best
in-sample Sharpe is found and ranked among all settings on the other half. PBO is the
share of those splits where the in-sample winner lands at or below the out-of-sample
median: 0.5 means picking the best in-sample was no better than a coin, 0 means the
winner kept winning.

Conventions (checked against the paper): rank 1 = worst, N = best; ``w = rank / (N + 1)``;
logit ``ln(w / (1 - w))``; PBO = share of logits ``<= 0``. Ties take the average rank
(the paper sets no rule). A Sharpe with no volatility (a setting that never traded in that
half) counts as 0: no return, no edge.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np

from plutobt.splits import cscv_blocks

_BATCH = 2048   # splits per vectorised batch; bounds memory at large N


@dataclass(frozen=True)
class PBOResult:
    pbo: float                    # share of splits whose in-sample winner fell to the bottom half
    logits: np.ndarray            # one logit per split
    degradation_slope: float      # slope of the winner's out-of-sample on in-sample Sharpe
    prob_oos_loss: float          # share of splits where the winner lost money out-of-sample
    n_splits: int
    dropped_rows: int             # remainder rows dropped from the start


def _sharpe(s: np.ndarray, q: np.ndarray, count: int) -> np.ndarray:
    """Per-bar Sharpe (population std) from column sums and sums of squares; 0 where the
    column has no volatility."""
    mean = s / count
    var = q / count - mean * mean
    with np.errstate(invalid="ignore", divide="ignore"):
        sr = mean / np.sqrt(var)
    flat = ~(var > 0) | ~np.isfinite(sr)
    sr[flat] = 0.0
    return sr


def pbo(returns: np.ndarray, blocks: int = 16) -> PBOResult:
    """PBO of picking the best column of ``returns`` (rows x settings) by Sharpe."""
    r = np.asarray(returns, dtype=float)
    if r.ndim != 2 or r.shape[1] < 2:
        raise ValueError("the Probability of Backtest Overfitting needs at least 2 settings "
                         "to choose between")
    n_rows, n = r.shape
    dropped, parts = cscv_blocks(n_rows, blocks)
    size = len(parts[0])
    sums = np.stack([r[p].sum(axis=0) for p in parts])          # blocks x N
    squares = np.stack([(r[p] ** 2).sum(axis=0) for p in parts])
    combos = np.array(list(combinations(range(blocks), blocks // 2)))
    half = size * (blocks // 2)

    logits, is_best, oos_best, oos_loss = [], [], [], []
    for start in range(0, len(combos), _BATCH):
        chosen = combos[start:start + _BATCH]
        mask = np.zeros((len(chosen), blocks))
        np.put_along_axis(mask, chosen, 1.0, axis=1)
        is_sr = _sharpe(mask @ sums, mask @ squares, half)
        oos_sum = (1.0 - mask) @ sums
        oos_sr = _sharpe(oos_sum, (1.0 - mask) @ squares, half)
        best = np.argmax(is_sr, axis=1)                          # first max on a tie
        rows = np.arange(len(chosen))
        v = oos_sr[rows, best][:, None]
        rank = 1 + (oos_sr < v).sum(axis=1) + 0.5 * ((oos_sr == v).sum(axis=1) - 1)
        w = rank / (n + 1)
        logits.append(np.log(w / (1 - w)))
        is_best.append(is_sr[rows, best])
        oos_best.append(oos_sr[rows, best])
        oos_loss.append(oos_sum[rows, best] < 0)

    logits = np.concatenate(logits)
    x, y = np.concatenate(is_best), np.concatenate(oos_best)
    slope = float(np.polyfit(x, y, 1)[0]) if np.ptp(x) > 0 else float("nan")
    return PBOResult(
        pbo=float(np.mean(logits <= 0)),
        logits=logits,
        degradation_slope=slope,
        prob_oos_loss=float(np.mean(np.concatenate(oos_loss))),
        n_splits=len(combos),
        dropped_rows=dropped,
    )
