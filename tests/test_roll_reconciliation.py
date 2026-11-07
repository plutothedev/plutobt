"""Proof 1 (2.0 spec §8): the engine on a back-adjusted series reconciles with an
independent simulator that holds the real contracts, exactly, in ticks.

Each case: 2 to 5 MNQ quarterly contracts, one shared random walk plus a per-contract
basis (so the roll gaps are random, zero included, both signs) and small per-contract
noise, a random roll day inside each three-day overlap, and a random script of market
entries (with and without stops, scale-ins and flips) and exits. The engine trades the
continuous series built by build_continuous; tests/reference/raw_contract_sim.py trades
the raw contracts. Price P&L must match to the tick, commission and roll cost to 1e-9.
"""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pytest

from plutobt.config import BacktestConfig
from plutobt.contracts import ContractSet
from plutobt.engine import Engine
from plutobt.instruments import get_instrument
from plutobt.roll import build_continuous
from plutobt.strategy import Strategy
from tests.reference.raw_contract_sim import RawContractSim

INST = get_instrument("MNQ")
TICK, TV = INST.tick_size, INST.tick_value
COMM, SLIP = 0.74, 1
BARS_PER_DAY, DAYS_ALONE, OVERLAP = 8, 6, 3
CFG = BacktestConfig(commission_per_side=COMM, slippage_ticks=float(SLIP),
                     carry_overnight=True, check_causality=False,
                     on_margin_call="record")


def weekdays(n: int) -> list[date]:
    out, d = [], date(2026, 1, 5)
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def gen_case(seed: int):
    rng = np.random.default_rng(seed)
    k = int(rng.integers(2, 6))
    n_days = DAYS_ALONE * k + OVERLAP
    days = weekdays(n_days)
    ts = np.array([int(np.datetime64(d, "s").astype(np.int64)) + 14 * 3600 + 1800
                   + 300 * i for d in days for i in range(BARS_PER_DAY)], dtype=np.int64)
    steps = rng.normal(0, 6, len(ts))
    steps[::BARS_PER_DAY] = rng.normal(0, 40, n_days)          # overnight moves
    spot_close = 80_000 + np.cumsum(np.rint(steps)).astype(np.int64)
    spot_open = np.concatenate([[80_000], spot_close[:-1]])
    codes = [f"MNQ{'HMUZ'[q % 4]}{2026 + q // 4}" for q in range(k)]
    frames, basis = {}, 0
    for i, code in enumerate(codes):
        if i:
            basis += 0 if rng.random() < 0.15 else int(rng.integers(-400, 401))
        lo_day, hi_day = DAYS_ALONE * i, DAYS_ALONE * i + DAYS_ALONE + OVERLAP
        sl = slice(lo_day * BARS_PER_DAY, hi_day * BARS_PER_DAY)
        noise = rng.integers(-2, 3, size=(2, sl.stop - sl.start)) * (rng.random() < 0.5)
        o = spot_open[sl] + basis + noise[0]
        c = spot_close[sl] + basis + noise[1]
        h = np.maximum(o, c) + rng.integers(0, 4, len(o))
        lw = np.minimum(o, c) - rng.integers(0, 4, len(o))
        frames[code] = {"ts": ts[sl], "o": o, "h": h, "l": lw, "c": c}
    roll_dates = {}
    for i, code in enumerate(codes[:-1]):
        day = days[DAYS_ALONE * (i + 1) + int(rng.integers(0, OVERLAP))]
        roll_dates[code] = day.isoformat()
    return rng, codes, frames, roll_dates


def to_frame(f):
    import pandas as pd

    return pd.DataFrame({"ts": f["ts"], "open": f["o"] * TICK, "high": f["h"] * TICK,
                         "low": f["l"] * TICK, "close": f["c"] * TICK,
                         "volume": np.full(len(f["ts"]), 10, dtype=np.int64)})


class Replay(Strategy):
    def __init__(self, script):
        self.script = script

    def on_bar(self, ctx):
        a = self.script.get(int(ctx.bar.ts))
        if a is None:
            return
        if a[0] == "close":
            ctx.broker.close()
            return
        side, n, dist = a
        stop = None
        if dist is not None:
            stop = ctx.bar.close - dist * TICK if side == "buy" else ctx.bar.close + dist * TICK
        (ctx.broker.buy if side == "buy" else ctx.broker.sell)(n, stop=stop)


def one_case(seed: int) -> dict:
    rng, codes, frames, roll_dates = gen_case(seed)
    cs = ContractSet.from_frames({c: to_frame(frames[c]) for c in codes})
    df, meta = build_continuous(cs, "dates", roll_dates=roll_dates)
    script = {}
    for t in df["ts"].tolist():
        if rng.random() < 0.2:
            if rng.random() < 0.3:
                script[t] = ("close",)
            else:
                dist = None if rng.random() < 0.3 else int(rng.integers(2, 31))
                script[t] = (str(rng.choice(["buy", "sell"])), int(rng.integers(1, 4)), dist)
    res = Engine(INST, CFG).run(df, Replay(script), roll_meta=meta)

    bars = {c: {int(t): (int(o), int(h), int(lw), int(cl)) for t, o, h, lw, cl in
                zip(frames[c]["ts"], frames[c]["o"], frames[c]["h"], frames[c]["l"],
                    frames[c]["c"], strict=True)} for c in codes}
    rolls = [(r["roll_ts"], r["from"], r["to"]) for r in meta.rolls]
    sim = RawContractSim(bars, rolls, tick_value=TV, commission_per_side=COMM,
                         slippage_ticks=SLIP)
    state = sim.run(script)

    engine_ticks = sum(t.pnl for t in res.trades) / TV
    assert abs(engine_ticks - round(engine_ticks)) < 1e-6, (seed, engine_ticks)
    assert round(engine_ticks) == sim.price_pnl_ticks(state), (
        seed, round(engine_ticks), sim.price_pnl_ticks(state))
    assert sum(t.commission for t in res.trades) == pytest.approx(state.commission,
                                                                   abs=1e-9), seed
    assert sum(e.cost for e in res.rolls) == pytest.approx(state.roll_cost, abs=1e-9), seed
    assert len(res.rolls) == state.counters["rolls_held"], seed
    c = dict(state.counters)
    c["negative_gap"] = any(r["gap_ticks"] < 0 for r in meta.rolls)
    c["zero_gap"] = any(r["gap_ticks"] == 0 for r in meta.rolls)
    return c


def run_fuzz(n: int) -> None:
    seen = {"rolls_held": 0, "roll_bar_gap_stops": 0, "open_ties": 0,
            "negative_gap": 0, "zero_gap": 0}
    for seed in range(n):
        c = one_case(seed)
        for key in seen:
            seen[key] += int(bool(c[key]))
    # The fuzz must reach the cases that matter, or it proves less than it claims.
    assert all(v > 0 for v in seen.values()), seen


def test_reconciliation_fuzz():
    run_fuzz(2_000)


@pytest.mark.slow
def test_reconciliation_fuzz_slow():
    run_fuzz(20_000)


def test_reference_is_independent():
    src = open("tests/reference/raw_contract_sim.py", encoding="utf-8").read()
    assert "plutobt" not in src.replace("imports nothing from plutobt", "")


def test_fuzz_catches_a_one_tick_adjustment_error(monkeypatch):
    """Positive control: shift every adjusted (older) bar by one tick, as an off-by-one
    gap would, and the reconciliation must fail on some case. A proof that cannot fail
    proves nothing."""
    import plutobt.roll as roll

    real = roll._adjust

    def off_by_one(cs, rolls):
        df, rows, low = real(cs, rolls)
        older = df["adj_offset"] != 0.0
        for col in ("open", "high", "low", "close"):
            df.loc[older, col] = df.loc[older, col] + TICK
        return df, rows, low

    monkeypatch.setattr(roll, "_adjust", off_by_one)
    caught = 0
    for seed in range(200):
        try:
            one_case(seed)
        except AssertionError:
            caught += 1
    assert caught > 0
