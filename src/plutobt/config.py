"""Backtest configuration value type (the single home of BacktestConfig)."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, fields


@dataclass
class BacktestConfig:
    commission_per_side: float = 0.0
    slippage_ticks: float | None = None   # None => use instrument default
    risk_free: float = 0.0
    periods_per_year: float | None = None  # None => derived from data/session
    carry_overnight: bool = False
    allow_short: bool = True   # False: an order that would leave a net short is refused
    # What is a break, where a pending order is cancelled (unless carry_overnight):
    # "gap" (default): a gap longer than the usual bar interval, or any change of the
    #   frame's `session` column (RTH/ETH labels work here); exits are cancelled too.
    # "session": only a change of `session`; an exit survives it and fills at the next
    #   session's open. `session` must then hold a trading-session ID on every bar that
    #   never decreases (e.g. the CME trading day as an int YYYYMMDD), not RTH/ETH labels.
    break_on: str = "gap"
    # What a margin call does (spec §5.2): at a bar's close, equity below the
    # maintenance margin of the held contracts records a MarginCall (the call restores
    # to INITIAL margin; no deposit is ever invented), once per breach, then:
    # "liquidate" (default): a forced market exit of the whole position fills at the
    #   next bar's open; its Trade carries margin_liquidation=True.
    # "record": the event only; the position is kept, so the breach can be studied.
    on_margin_call: str = "liquidate"
    # Look-ahead guard: before the bar loop, Engine.run checks the strategy's precompute
    # with lookahead.assert_causal (each bar's values must not change when the later bars
    # are removed) and refuses a leak with a LookaheadError naming the column. Every
    # precompute call runs on its own copy.deepcopy of the strategy as passed in; a
    # strategy that cannot be copied (it holds a lock, socket or open file) is refused
    # with a TypeError, and false skips the guard (and the copies).
    check_causality: bool = True
    # The truncation grid. "auto" (default): every prefix up to 2,000 bars (about 0.3 s on
    # each 324-bar demo sample); above that 250 evenly spaced prefixes plus the first 20
    # and last 20, with a warning that the check was sampled. None forces every prefix
    # (exhaustive; the cost grows with the square of the bar count). An int k checks k
    # evenly spaced prefixes. A sampled grid CAN MISS a leak on a few rows between them.
    causality_sample: int | str | None = "auto"
    # Engine.run refuses a frame with any open/high/low/close <= 0 (the guard is
    # data.assert_positive_prices): it is usually a back-adjusted continuous series pushed
    # through zero, where percentage and log returns break. True runs it anyway, for an
    # instrument that really traded at or below zero (CL settled at -37.63 on 2020-04-20):
    # points/dollar P&L stays valid there, ratio returns on the prices do not.
    allow_nonpositive_prices: bool = False


def load_config(path: str) -> BacktestConfig:
    """Load a BacktestConfig from a TOML file (stdlib tomllib).

    Omitted keys fall back to the BacktestConfig dataclass defaults.
    Unknown keys raise ValueError so a typo never silently no-ops.
    """
    with open(path, "rb") as fh:
        raw = tomllib.load(fh)
    allowed = {f.name for f in fields(BacktestConfig)}
    for key in raw:
        if key not in allowed:
            raise ValueError(f"unknown config key: {key}")
    return BacktestConfig(**raw)
