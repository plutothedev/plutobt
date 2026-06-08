"""Verified futures contract registry and price/tick math (spec §5.3, §10.1)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Instrument:
    symbol: str
    point_value: float
    tick_size: float
    tick_value: float
    currency: str
    session_open_ct: str        # "08:30"
    session_close_ct: str       # "15:15"
    slippage_ticks: float
    initial_margin: float
    maintenance_margin: float
    settlement: str             # "cash" | "physical"
    price_format: str           # "decimal" | "thirty_seconds"


# §5.3 verified contract specs (vs CME, 2026). tick_value == tick_size * point_value.
# Margins are representative published CME initial/maintenance, as-of 2026-06-07,
# and VARY by broker/volatility/date (§5.2). Index RTH session = 08:30-15:15 CT (§5.5).
REGISTRY: dict[str, Instrument] = {
    "NQ": Instrument(
        symbol="NQ", point_value=20.0, tick_size=0.25, tick_value=5.0, currency="USD",
        session_open_ct="08:30", session_close_ct="15:15",
        slippage_ticks=1.0, initial_margin=23000.0, maintenance_margin=20900.0,
        settlement="cash", price_format="decimal",
    ),
    "MNQ": Instrument(
        symbol="MNQ", point_value=2.0, tick_size=0.25, tick_value=0.5, currency="USD",
        session_open_ct="08:30", session_close_ct="15:15",
        slippage_ticks=1.0, initial_margin=2300.0, maintenance_margin=2090.0,
        settlement="cash", price_format="decimal",
    ),
    "ES": Instrument(
        symbol="ES", point_value=50.0, tick_size=0.25, tick_value=12.5, currency="USD",
        session_open_ct="08:30", session_close_ct="15:15",
        slippage_ticks=1.0, initial_margin=13200.0, maintenance_margin=12000.0,
        settlement="cash", price_format="decimal",
    ),
    "MES": Instrument(
        symbol="MES", point_value=5.0, tick_size=0.25, tick_value=1.25, currency="USD",
        session_open_ct="08:30", session_close_ct="15:15",
        slippage_ticks=1.0, initial_margin=1320.0, maintenance_margin=1200.0,
        settlement="cash", price_format="decimal",
    ),
    "YM": Instrument(
        symbol="YM", point_value=5.0, tick_size=1.0, tick_value=5.0, currency="USD",
        session_open_ct="08:30", session_close_ct="15:15",
        slippage_ticks=1.0, initial_margin=9900.0, maintenance_margin=9000.0,
        settlement="cash", price_format="decimal",
    ),
    "RTY": Instrument(
        symbol="RTY", point_value=50.0, tick_size=0.10, tick_value=5.0, currency="USD",
        session_open_ct="08:30", session_close_ct="15:15",
        slippage_ticks=1.0, initial_margin=7150.0, maintenance_margin=6500.0,
        settlement="cash", price_format="decimal",
    ),
    "CL": Instrument(
        symbol="CL", point_value=1000.0, tick_size=0.01, tick_value=10.0, currency="USD",
        session_open_ct="08:00", session_close_ct="13:30",
        slippage_ticks=2.0, initial_margin=7700.0, maintenance_margin=7000.0,
        settlement="physical", price_format="decimal",
    ),
    "GC": Instrument(
        symbol="GC", point_value=100.0, tick_size=0.10, tick_value=10.0, currency="USD",
        session_open_ct="07:20", session_close_ct="12:30",
        slippage_ticks=2.0, initial_margin=12100.0, maintenance_margin=11000.0,
        settlement="physical", price_format="decimal",
    ),
    "ZN": Instrument(
        symbol="ZN", point_value=1000.0, tick_size=0.015625, tick_value=15.625, currency="USD",
        session_open_ct="07:20", session_close_ct="14:00",
        slippage_ticks=1.0, initial_margin=2200.0, maintenance_margin=2000.0,
        settlement="physical", price_format="thirty_seconds",
    ),
}


def get_instrument(symbol: str) -> Instrument:
    try:
        return REGISTRY[symbol]
    except KeyError:
        known = ", ".join(sorted(REGISTRY))
        raise KeyError(f"unknown symbol {symbol!r}; known symbols: {known}") from None
