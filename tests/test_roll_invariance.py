"""Proof 2 (2.0 spec §8): adding a contract rewrites the history of a back-adjusted
series, but only by one constant, and changes no dollar result.

The shipped sample is built twice: from NQH, NQM and NQU 2026, then with NQZ 2026 added.
Over the bars both builds share, every price must move by exactly the U -> Z gap (in
whole ticks), and EMACross must make the same trades for the same dollars on each.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from plutobt.config import BacktestConfig
from plutobt.contracts import ContractSet
from plutobt.engine import Engine
from plutobt.instruments import get_instrument
from plutobt.roll import build_continuous, slice_continuous
from plutobt.strategies.ema_cross import EMACross

DIR = Path(__file__).resolve().parents[1] / "data" / "sample" / "nq_contracts"
TICK = 0.25


@pytest.fixture(scope="module")
def builds():
    full = ContractSet.from_folder(str(DIR), "generic", "NQ", ts_is="open")
    three = ContractSet.from_frames({c: full.frames[c]
                                     for c in ("NQH2026", "NQM2026", "NQU2026")})
    df3, meta3 = build_continuous(three)
    df4, meta4 = build_continuous(full)
    z_roll = meta4.rolls[-1]["roll_ts"]
    return (df3, meta3), (df4, meta4), z_roll


def shared(builds):
    (df3, meta3), (df4, meta4), z_roll = builds
    end = z_roll - 1
    return slice_continuous(df3, meta3, end_ts=end), slice_continuous(df4, meta4, end_ts=end)


def test_history_shifts_by_one_constant(builds):
    (a, _), (b, _) = shared(builds)
    _, (_, meta4), _ = builds
    assert a["ts"].tolist() == b["ts"].tolist()
    assert a["contract"].tolist() == b["contract"].tolist()
    assert a["raw_close"].tolist() == b["raw_close"].tolist()
    gap = meta4.rolls[-1]["gap_ticks"]
    assert gap == 120                                     # +30.00 points, U -> Z
    for col in ("open", "high", "low", "close"):
        diff = np.rint(b[col].to_numpy() / TICK) - np.rint(a[col].to_numpy() / TICK)
        assert set(diff.astype(int)) == {gap}, col


def test_dollar_results_do_not_change(builds):
    (a, ma), (b, mb) = shared(builds)
    cfg = BacktestConfig(commission_per_side=2.5, slippage_ticks=1.0)
    ra = Engine(get_instrument("NQ"), cfg).run(a, EMACross(), roll_meta=ma)
    rb = Engine(get_instrument("NQ"), cfg).run(b, EMACross(), roll_meta=mb)
    assert len(ra.trades) > 5 and len(ra.rolls) >= 1
    key = [(t.entry_ts, t.exit_ts, t.side, t.contracts, t.pnl, t.commission,
            t.roll_cost, t.rolls) for t in ra.trades]
    assert key == [(t.entry_ts, t.exit_ts, t.side, t.contracts, t.pnl, t.commission,
                    t.roll_cost, t.rolls) for t in rb.trades]
    assert ra.equity_curve.tolist() == rb.equity_curve.tolist()
    assert [(e.ts, e.cost) for e in ra.rolls] == [(e.ts, e.cost) for e in rb.rolls]
