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
    # --- Equities (v0.5). Cash equity: the multiplier IS point_value (1.0), so a
    # +1.00 move on N shares is exactly $N — no separate contract multiplier, and a
    # futures multiplier can never leak into the equity path (CONTRACT §3.1/§3.2).
    # tick_value == tick_size * point_value (0.01 == 0.01 * 1.0). Margins are 0.0:
    # no performance bond, so Portfolio.margin_ok always passes (NO buying-power
    # model in v0.5 — a documented limit, not an invariant). RTH 08:30-15:00 CT.
    "AAPL": Instrument(
        symbol="AAPL", point_value=1.0, tick_size=0.01, tick_value=0.01, currency="USD",
        session_open_ct="08:30", session_close_ct="15:00",
        slippage_ticks=1.0, initial_margin=0.0, maintenance_margin=0.0,
        settlement="cash", price_format="decimal",
    ),
    "MSFT": Instrument(
        symbol="MSFT", point_value=1.0, tick_size=0.01, tick_value=0.01, currency="USD",
        session_open_ct="08:30", session_close_ct="15:00",
        slippage_ticks=1.0, initial_margin=0.0, maintenance_margin=0.0,
        settlement="cash", price_format="decimal",
    ),
    "SPY": Instrument(
        symbol="SPY", point_value=1.0, tick_size=0.01, tick_value=0.01, currency="USD",
        session_open_ct="08:30", session_close_ct="15:00",
        slippage_ticks=1.0, initial_margin=0.0, maintenance_margin=0.0,
        settlement="cash", price_format="decimal",
    ),
}


def get_instrument(symbol: str) -> Instrument:
    try:
        return REGISTRY[symbol]
    except KeyError:
        known = ", ".join(sorted(REGISTRY))
        raise KeyError(f"unknown symbol {symbol!r}; known symbols: {known}") from None


def parse_price(raw: str | float, instrument: Instrument) -> float:
    """Convert a raw quote to decimal points.

    Decimal instruments pass through. For ``thirty_seconds`` (Treasury 32nds),
    a string ``"<handle>'<32nds>[half]"`` like ``"110'085"`` parses to
    ``110 + 8.5/32`` (the 3rd digit is the half-32nd: 0 or 5). Only prices the
    contract can trade at parse: whole 32nds 00..31 and a 3rd digit of 0 or 5, so the
    fractional part runs 000..315 on ZN's 1/64 grid; anything else raises ValueError.
    Numeric input on a 32nds instrument is treated as already-decimal points (spec §5.3).
    """
    if instrument.price_format != "thirty_seconds":
        return float(raw)

    if not isinstance(raw, str):
        return float(raw)

    if "'" not in raw:
        raise ValueError(f"malformed 32nds quote {raw!r}: expected \"<handle>'<32nds>\"")

    handle_str, _, frac_str = raw.partition("'")
    if not handle_str or len(frac_str) != 3 or not frac_str.isdigit():
        raise ValueError(
            f"malformed 32nds quote {raw!r}: fractional part must be exactly 3 digits"
        )

    # Refuse quotes off the half-32nd grid: "110'087" is not a ZN price (17.4 ticks),
    # and "110'990" is not 99/32 of a point but a typo.
    if int(frac_str[:2]) > 31 or frac_str[2] not in "05":
        raise ValueError(
            f"off-grid 32nds quote {raw!r}: {instrument.symbol} trades in halves of a "
            f"32nd, so the fractional part must be 000..315 ending in 0 or 5 "
            f"(e.g. 110'085 = 110 + 8.5/32)"
        )

    handle = float(handle_str)
    thirty_seconds = float(frac_str[:2])
    thirty_seconds += int(frac_str[2]) / 10.0   # e.g. '5' -> +0.5 of a 32nd, '0' -> +0.0
    return handle + thirty_seconds / 32.0


def ticks_to_price(ticks: float, instrument: Instrument) -> float:
    """Decimal price magnitude of ``ticks`` ticks (signed). ticks * tick_size."""
    return ticks * instrument.tick_size


def pnl_points(entry: float, exit: float, contracts: int, instrument: Instrument) -> float:
    """Signed P&L in account currency: (exit - entry) * point_value * contracts.

    ``contracts`` is signed (>0 long, <0 short). Inputs are decimal points
    (run raw quotes through ``parse_price`` first).
    """
    return (exit - entry) * instrument.point_value * contracts
