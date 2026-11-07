"""Engine: roll metadata pinning and roll events (spec §7.1, §7.2)."""
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from plutobt.broker import FrameError
from plutobt.config import BacktestConfig
from plutobt.contracts import ContractSet
from plutobt.engine import Engine
from plutobt.instruments import get_instrument
from plutobt.roll import build_continuous, trading_day
from plutobt.strategy import Strategy

T0 = 1_767_277_800            # 2026-01-01 14:30 UTC
GAP = 25.0                    # NQM2026 trades 25 points over NQH2026
ROLL_ROW = 10                 # first shared bar: a's bar 10 = b's bar 0


def contracts(gap_down_at_roll: float = 0.0):
    n = 20
    o = 18000.0 + np.arange(n) * 1.0
    a = pd.DataFrame({"ts": T0 + 300 * np.arange(n, dtype=np.int64), "open": o,
                      "close": o + 0.5, "volume": np.full(n, 10, dtype=np.int64)})
    if gap_down_at_roll:
        a.loc[ROLL_ROW:, ["open", "close"]] -= gap_down_at_roll
    a["high"] = a[["open", "close"]].max(axis=1) + 0.25
    a["low"] = a[["open", "close"]].min(axis=1) - 0.25
    b = a.iloc[ROLL_ROW:].copy()
    b = pd.concat([b, b.iloc[-10:].assign(ts=b["ts"].iloc[-1] + 300 * np.arange(1, 11))])
    for col in ("open", "high", "low", "close"):
        b[col] = b[col] + GAP
    cols = ["ts", "open", "high", "low", "close", "volume"]
    return ContractSet.from_frames({"NQH2026": a[cols], "NQM2026": b[cols].reset_index(drop=True)})


def series(**kw):
    cs = contracts(**kw)
    return build_continuous(cs, "dates", roll_dates={"NQH6": trading_day(T0).isoformat()})


class Script(Strategy):
    """Runs ``script[bar_index](ctx)`` and records the resting stop on every bar."""

    def __init__(self, script=None):
        self.script = script or {}

    def on_bar(self, ctx):
        i = ctx.state.setdefault("i", 0)
        ctx.state.setdefault("stops", []).append(ctx.broker.resting_stop)
        if i in self.script:
            self.script[i](ctx)
        ctx.state["i"] = i + 1


CFG = BacktestConfig(commission_per_side=2.5, slippage_ticks=1.0, carry_overnight=True)


def run(script=None, cfg=CFG, **kw):
    df, meta = series(**kw)
    strat = Script(script)
    res = Engine(get_instrument("NQ"), cfg).run(df, strat, roll_meta=meta)
    return df, meta, res


def test_series_shape():
    df, meta = series()
    assert df["contract"].tolist().index("NQM2026") == ROLL_ROW
    assert meta.rolls[0]["gap_points"] == GAP


def test_roll_columns_are_base():
    df, meta, res = run()
    assert res.rolls == []

    class EditsRaw(Strategy):
        def precompute(self, frame):
            frame["raw_close"] = frame["raw_close"] + 1.0
            return frame

    with pytest.raises(FrameError, match="raw_close"):
        Engine(get_instrument("NQ"), CFG).run(df, EditsRaw(), roll_meta=meta)


def test_contract_column_without_meta_refused():
    df, _ = series()
    with pytest.raises(ValueError, match="roll_meta"):
        Engine(get_instrument("NQ"), CFG).run(df, Script())


def test_meta_hash_mismatch_refused():
    df, meta = series()
    bad = replace(meta, frame_sha256="0" * 64)
    with pytest.raises(ValueError, match="0" * 64) as err:
        Engine(get_instrument("NQ"), CFG).run(df, Script(), roll_meta=bad)
    assert meta.frame_sha256 in str(err.value)


def test_manifest_records_roll_meta():
    _, meta, res = run()
    man = res.manifest
    assert man["roll_trigger"] == "dates"
    assert man["splice_method"] == "backward"
    assert man["adjustment_method"] == "additive"
    assert man["as_of_date"] == meta.as_of_date
    assert man["rolls"] == 1


def hold_long_2(exit_at=15):
    return {2: lambda c: c.broker.buy(2), exit_at: lambda c: c.broker.close()}


def test_roll_cost_long_held():
    df, meta, res = run(hold_long_2())
    (ev,) = res.rolls
    assert (ev.ts, ev.from_contract, ev.to_contract) == (int(df["ts"][ROLL_ROW]),
                                                         "NQH2026", "NQM2026")
    assert (ev.side, ev.contracts, ev.gap_points) == ("long", 2, GAP)
    assert ev.cost == 2 * 2.5 * 2 + 2 * 1 * 5.0 * 2 == 30.0
    (trade,) = res.trades
    assert trade.rolls == 1 and trade.roll_cost == 30.0
    assert trade.net_pnl == pytest.approx(trade.pnl - trade.commission - 30.0)
    eq = res.equity_curve.to_numpy()
    move = (df["close"][ROLL_ROW] - df["close"][ROLL_ROW - 1]) * 20.0 * 2
    assert eq[ROLL_ROW] - eq[ROLL_ROW - 1] == pytest.approx(move - 30.0)


def test_flat_through_roll_no_event():
    _, _, res = run({2: lambda c: c.broker.buy(1), 5: lambda c: c.broker.close()})
    assert res.rolls == [] and res.trades[0].roll_cost == 0.0


def test_exit_at_roll_bar_open_pays_no_roll():
    """Review #3: an exit filling at the roll bar's open is a sale of the expiring
    contract at its open. A real trader pays that one side, never a roll round trip."""
    df, _, res = run(hold_long_2(exit_at=ROLL_ROW - 1))
    assert res.rolls == []
    (trade,) = res.trades
    assert trade.roll_cost == 0.0 and trade.rolls == 0
    assert trade.exit_ts == int(df["ts"][ROLL_ROW])
    assert trade.exit_price == df["open"][ROLL_ROW] - 0.25


def test_flip_at_roll_bar_open_pays_no_roll():
    """The old side closes on the expiring contract and the new side opens on the next:
    nothing is held through the roll."""
    script = {2: lambda c: c.broker.buy(2), ROLL_ROW - 1: lambda c: c.broker.sell(4),
              15: lambda c: c.broker.close()}
    _, _, res = run(script)
    assert res.rolls == []
    first, second = res.trades
    assert first.side == "long" and first.roll_cost == 0.0
    assert second.side == "short" and second.roll_cost == 0.0 and second.rolls == 0


def test_partial_exit_at_roll_bar_open_rolls_the_rest():
    script = {2: lambda c: c.broker.buy(2), ROLL_ROW - 1: lambda c: c.broker.sell(1),
              15: lambda c: c.broker.close()}
    _, _, res = run(script)
    (ev,) = res.rolls
    assert (ev.side, ev.contracts, ev.cost) == ("long", 1, 15.0)
    closed_at_open, rest = res.trades
    assert closed_at_open.contracts == 1 and closed_at_open.roll_cost == 0.0
    assert rest.contracts == 1 and rest.roll_cost == 15.0 and rest.rolls == 1


def test_scale_in_at_roll_bar_open_rolls_only_the_held():
    script = {2: lambda c: c.broker.buy(1), ROLL_ROW - 1: lambda c: c.broker.buy(2),
              15: lambda c: c.broker.close()}
    _, _, res = run(script)
    (ev,) = res.rolls
    assert (ev.contracts, ev.cost) == (1, 15.0)
    assert sum(t.roll_cost for t in res.trades) == pytest.approx(15.0)


def test_stop_gapped_at_roll_bar_open_pays_no_roll():
    """The expiring contract opens through the stop, so the stop fills on it at the
    open: the position never reaches the new contract."""
    stop = None

    def enter(c):
        nonlocal stop
        stop = float(c.bar.close) - 3.0
        c.broker.buy(1, stop=stop)

    df, _, res = run({2: enter}, gap_down_at_roll=12.0)
    assert df["open"][ROLL_ROW] < stop                       # the roll bar opens through it
    assert res.rolls == []
    (trade,) = res.trades
    assert trade.exit_ts == int(df["ts"][ROLL_ROW])
    assert trade.exit_price == df["open"][ROLL_ROW] - 0.25   # gap-through: open less slip
    assert trade.roll_cost == 0.0


def test_intrabar_stop_on_roll_bar_still_pays_the_roll():
    """Held through the open into the new contract, then stopped out inside the bar:
    the roll happened, so it is paid."""
    # The roll bar opens 18035.00 with a low of 18034.75: a stop there is not gapped
    # through at the open, but the bar trades down to it.
    script = {2: lambda c: c.broker.buy(1),
              ROLL_ROW - 1: lambda c: c.broker.amend_stop(18034.75)}
    df, _, res = run(script)
    roll_open = df["open"][ROLL_ROW]
    (ev,) = res.rolls
    (trade,) = res.trades
    assert trade.exit_ts == int(df["ts"][ROLL_ROW]) and trade.exit_price < roll_open
    assert trade.roll_cost == ev.cost == 2 * 2.5 + 2 * 5.0


def test_resting_stop_level_unchanged_across_roll():
    seen = {}

    def enter(c):
        c.broker.buy(1, stop=17990.0)

    def look(c):
        seen["stop"] = c.broker.resting_stop

    _, _, res = run({2: enter, ROLL_ROW + 2: look})
    assert len(res.rolls) == 1
    assert seen["stop"] == 17990.0


def test_added_columns_do_not_break_the_pin():
    """Review #5: the pin covers the series' own columns only, so a signal column or
    tag_sessions' 'session' can be added before Engine.run."""
    from plutobt.data import tag_sessions

    df, meta = series()
    df["sma"] = df["close"].rolling(3).mean()
    df = tag_sessions(df, get_instrument("NQ"))
    res = Engine(get_instrument("NQ"), CFG).run(df, Script(hold_long_2()), roll_meta=meta)
    assert len(res.rolls) == 1


def test_edited_bars_still_refused_with_extra_columns():
    df, meta = series()
    df["sma"] = 0.0
    df.loc[3, "close"] += 1.0
    with pytest.raises(ValueError, match="sha256"):
        Engine(get_instrument("NQ"), CFG).run(df, Script(), roll_meta=meta)


def test_slice_refuses_an_edited_series():
    """Review #5: slice_continuous re-pinned whatever it was handed, so an edited series
    passed after one documented call."""
    from plutobt.roll import slice_continuous

    df, meta = series()
    df.loc[3, ["open", "high", "low", "close"]] += 500.0
    with pytest.raises(ValueError, match="sha256"):
        slice_continuous(df, meta, start_ts=int(df["ts"].iloc[1]))
