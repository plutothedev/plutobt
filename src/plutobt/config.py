"""Backtest configuration value type (the single home of BacktestConfig)."""

from __future__ import annotations

import difflib
import math
import numbers
import tomllib
from dataclasses import dataclass, fields

_BOOLS = ("carry_overnight", "allow_short", "check_causality", "allow_nonpositive_prices")
_TEXTS = ("break_on", "on_margin_call")


def _is_number(value) -> bool:
    return isinstance(value, numbers.Real) and not isinstance(value, bool)


def _is_bool(value) -> bool:
    return isinstance(value, bool) or type(value).__name__ == "bool_"      # numpy's bool


@dataclass
class BacktestConfig:
    commission_per_side: float = 0.0
    slippage_ticks: float | None = None   # None => use instrument default
    risk_free: float = 0.0
    periods_per_year: float | None = None  # bars a year; None => pass it to compute_metrics
    # False (default): ENTRIES never carry across a break (see break_on): a pending order
    # that would open or add to a position is cancelled there. It is NOT "flat
    # overnight": a pending EXIT survives the break and fills at the next session's open
    # (a flip keeps only its closing contracts; a margin liquidation always survives),
    # and an open position is never force-flattened, so it is held across the break
    # unless the strategy exits. True carries every pending order across.
    carry_overnight: bool = False
    allow_short: bool = True   # False: an order that would leave a net short is refused
    # What is a break, where a pending entry is cancelled (unless carry_overnight); both
    # modes treat a break the same way:
    # "gap" (default): a gap longer than the usual bar interval, or any change of the
    #   frame's `session` column (RTH/ETH labels work here).
    # "session": only a change of `session`, which must then hold a trading-session ID
    #   on every bar that never decreases (e.g. the CME trading day as an int YYYYMMDD),
    #   not RTH/ETH labels.
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
    # The truncation grid. "auto" (default): every prefix up to 2,000 bars (about 1.7 s
    # for EMACross on the 1,620-bar NQ demo sample, measured 2026-11-02); above that 250
    # evenly spaced prefixes plus the first 20 and last 20, with a warning that the check
    # was sampled. None forces every prefix
    # (exhaustive; the cost grows with the square of the bar count). An int k checks k
    # evenly spaced prefixes. A sampled grid CAN MISS a leak on a few rows between them.
    causality_sample: int | str | None = "auto"
    # Engine.run refuses a frame with any open/high/low/close <= 0 (the guard is
    # data.assert_positive_prices): it is usually a back-adjusted continuous series pushed
    # through zero, where percentage and log returns break. True runs it anyway, for an
    # instrument that really traded at or below zero (CL settled at -37.63 on 2020-04-20):
    # points/dollar P&L stays valid there, ratio returns on the prices do not.
    allow_nonpositive_prices: bool = False

    def __post_init__(self) -> None:
        """Refuse a value of the wrong kind, by name. A quoted "false" in a TOML file is
        text, and text counts as true: ``carry_overnight = "false"`` used to run with
        carry ON. Whether ``break_on`` and ``on_margin_call`` hold a known word is still
        ``Engine``'s check."""
        for name in _BOOLS:
            value = getattr(self, name)
            if not _is_bool(value):
                raise ValueError(
                    f"config {name} = {value!r}: must be true or false (in a TOML file "
                    "write true or false without quotes; quoted text counts as true)")
        for name in _TEXTS:
            value = getattr(self, name)
            if not isinstance(value, str):
                raise ValueError(f"config {name} = {value!r}: must be text")
        for name, optional, least, strict in (
                ("commission_per_side", False, 0.0, False),
                ("slippage_ticks", True, 0.0, False),
                ("risk_free", False, None, False),
                ("periods_per_year", True, 0.0, True)):
            value = getattr(self, name)
            if value is None and optional:
                continue
            if not _is_number(value):
                raise ValueError(
                    f"config {name} = {value!r}: must be a number"
                    + (" or left out" if optional else ""))
            if not math.isfinite(value):
                raise ValueError(f"config {name} = {value!r}: must be a finite number")
            if least is not None and (value <= least if strict else value < least):
                raise ValueError(
                    f"config {name} = {value!r}: must be "
                    + (f"above {least:g}" if strict else f"{least:g} or more"))


def load_config(path: str) -> BacktestConfig:
    """Load a BacktestConfig from a TOML file (stdlib tomllib).

    Omitted keys fall back to the BacktestConfig dataclass defaults.
    Unknown keys raise ValueError so a typo never silently no-ops.
    """
    with open(path, "rb") as fh:
        raw = tomllib.load(fh)
    allowed = [f.name for f in fields(BacktestConfig)]
    for key in raw:
        if key not in allowed:
            close = difflib.get_close_matches(key, allowed, n=1)
            hint = f" (did you mean {close[0]!r}?)" if close else ""
            raise ValueError(
                f"unknown config key: {key}{hint}. The keys are: {', '.join(allowed)}")
    return BacktestConfig(**raw)
