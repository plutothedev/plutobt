"""Futures contract codes, the equity-index expiry calendar, and contract sets (spec §5).

A contract is a root (``NQ``), a delivery month and a year. ``parse_contract`` reads the
forms vendors use (``NQZ5``, ``NQZ25``, ``NQZ2025``, ``NQ 12-25``, ``NQ DEC25``) into one
canonical ``Contract`` whose ``code`` is ``NQZ2025``.

The expiry calendar covers the CME equity-index roots only (``CALENDAR_ROOTS``): their
quarterly contracts (March, June, September, December) expire on the third Friday of the
contract month unless an exchange holiday moves it (``_EXPIRY_EXCEPTIONS``), and CME's
roll convention is ``ROLL_DAYS_BEFORE_EXPIRY`` calendar days before that. Other roots
(CL, GC, ZN, ...) roll on their own exchange rules, which this module does not encode:
use the volume trigger or explicit roll dates for them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, timedelta

MONTH_CODES: dict[str, int] = {
    "F": 1, "G": 2, "H": 3, "J": 4, "K": 5, "M": 6,
    "N": 7, "Q": 8, "U": 9, "V": 10, "X": 11, "Z": 12,
}
_CODE_FOR_MONTH = {m: c for c, m in MONTH_CODES.items()}
_MONTH_NAMES = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
    "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}

CALENDAR_ROOTS = frozenset({"NQ", "ES", "MNQ", "MES", "YM", "RTY"})
ROLL_DAYS_BEFORE_EXPIRY = 8
_QUARTERLY = (3, 6, 9, 12)

# Contracts whose last trading day is not the plain third Friday (an exchange holiday
# moved it), and contracts whose CME roll date is not expiry minus 8 days. Keyed by the
# canonical code. Filled from the 2026-11-06 fact-check (CME pages, see the vault note).
_EXPIRY_EXCEPTIONS: dict[str, date] = {}
_ROLL_DAY_EXCEPTIONS: dict[str, date] = {}

_SHORT = re.compile(r"^([A-Z]+)([FGHJKMNQUVXZ])(\d{1,2}|\d{4})$")
_NUMERIC = re.compile(r"^([A-Z]+)\s+(\d{1,2})-(\d{2}|\d{4})$")
_NAMED = re.compile(r"^([A-Z]+)\s+([A-Z]{3})(\d{2}|\d{4})$")


@dataclass(frozen=True, order=True)
class Contract:
    """One futures contract. Orders by (year, month), then root."""

    year: int = field(init=False, repr=False)
    month: int = field(init=False, repr=False)
    root: str = field(init=False, repr=False)

    def __init__(self, root: str, month: int, year: int) -> None:
        object.__setattr__(self, "root", root)
        object.__setattr__(self, "month", int(month))
        object.__setattr__(self, "year", int(year))

    def __repr__(self) -> str:
        return f"Contract({self.root!r}, {self.month}, {self.year})"

    @property
    def code(self) -> str:
        """Canonical code, e.g. ``NQZ2025``."""
        return f"{self.root}{_CODE_FOR_MONTH[self.month]}{self.year:04d}"


def _bad(code: str, why: str) -> ValueError:
    return ValueError(
        f"not a futures contract code: {code!r} ({why}). Accepted forms: NQZ5, NQZ25, "
        "NQZ2025, 'NQ 12-25', 'NQ DEC25'"
    )


def _year(digits: str, code: str, near: date | None) -> int:
    if len(digits) == 4:
        return int(digits)
    if len(digits) == 2:
        return 2000 + int(digits)
    if near is None:
        raise ValueError(
            f"contract code {code!r} has a one-digit year, which needs near=<a date in "
            "the data> to tell 2015 from 2025"
        )
    d = int(digits)
    base = near.year - near.year % 10
    candidates = [base - 10 + d, base + d, base + 10 + d]
    dist = sorted((abs(y - near.year), y) for y in candidates)
    if dist[0][0] == dist[1][0]:
        raise ValueError(
            f"contract code {code!r}: the one-digit year is equally near {dist[0][1]} and "
            f"{dist[1][1]} for near={near}; pass a code with a two- or four-digit year"
        )
    return dist[0][1]


def parse_contract(code: str, *, near: date | None = None) -> Contract:
    """Parse a vendor contract code into a ``Contract``.

    ``near`` (a date inside the contract's data) resolves a one-digit year to the
    nearest matching year. Anything that is not a contract code raises ValueError.
    """
    text = code.strip().upper() if isinstance(code, str) else ""
    m = _SHORT.match(text)
    if m:
        root, month = m.group(1), MONTH_CODES[m.group(2)]
        return Contract(root, month, _year(m.group(3), code, near))
    m = _NUMERIC.match(text) or _NAMED.match(text)
    if m:
        root, mon = m.group(1), m.group(2)
        month = int(mon) if mon.isdigit() else _MONTH_NAMES.get(mon, 0)
        if not 1 <= month <= 12:
            raise _bad(code, f"no month {mon!r}")
        return Contract(root, month, _year(m.group(3), code, near))
    raise _bad(code, "unrecognised form")


def _third_friday(year: int, month: int) -> date:
    first = date(year, month, 1)
    offset = (4 - first.weekday()) % 7          # Friday is weekday 4
    return first + timedelta(days=offset + 14)


def expiry(contract: Contract) -> date:
    """Last trading day of an equity-index quarterly contract.

    The third Friday of the contract month, unless an exchange holiday moved it. Only
    ``CALENDAR_ROOTS`` are covered; any other root raises ValueError.
    """
    if contract.root not in CALENDAR_ROOTS:
        raise ValueError(
            f"no expiry calendar for {contract.code}: the calendar roll covers the "
            f"equity-index roots {sorted(CALENDAR_ROOTS)} only. Roll {contract.root} with "
            "trigger='volume' or pass roll_dates={'<contract>': 'YYYY-MM-DD'}"
        )
    if contract.month not in _QUARTERLY:
        raise ValueError(
            f"{contract.code}: equity-index futures list quarterly months only "
            "(H, M, U, Z)"
        )
    return _EXPIRY_EXCEPTIONS.get(contract.code, _third_friday(contract.year,
                                                                contract.month))


def calendar_roll_day(contract: Contract) -> date:
    """CME's roll day for ``contract``: ``ROLL_DAYS_BEFORE_EXPIRY`` calendar days before
    its expiry, unless CME published a different date (``_ROLL_DAY_EXCEPTIONS``)."""
    exp = expiry(contract)
    return _ROLL_DAY_EXCEPTIONS.get(contract.code,
                                    exp - timedelta(days=ROLL_DAYS_BEFORE_EXPIRY))
