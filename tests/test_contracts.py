"""Contract codes, the equity-index expiry calendar and the calendar roll day (spec §5)."""

from datetime import date

import pytest

from plutobt.contracts import (
    CALENDAR_ROOTS,
    MONTH_CODES,
    Contract,
    calendar_roll_day,
    expiry,
    parse_contract,
)


def test_month_codes_and_calendar_roots():
    assert MONTH_CODES == {"F": 1, "G": 2, "H": 3, "J": 4, "K": 5, "M": 6,
                           "N": 7, "Q": 8, "U": 9, "V": 10, "X": 11, "Z": 12}
    assert CALENDAR_ROOTS == frozenset({"NQ", "ES", "MNQ", "MES", "YM", "RTY"})


@pytest.mark.parametrize("code", ["NQZ5", "NQZ25", "NQZ2025", "NQ 12-25", "NQ DEC25"])
def test_parse_forms(code):
    assert parse_contract(code, near=date(2025, 9, 1)) == Contract("NQ", 12, 2025)


def test_parse_micro_root():
    c = parse_contract("MNQH6", near=date(2026, 1, 5))
    assert c == Contract("MNQ", 3, 2026)
    assert c.code == "MNQH2026"


def test_contract_orders_by_expiry_month():
    assert sorted([Contract("NQ", 3, 2026), Contract("NQ", 12, 2025),
                   Contract("NQ", 6, 2026)]) == [
        Contract("NQ", 12, 2025), Contract("NQ", 3, 2026), Contract("NQ", 6, 2026)]


def test_single_digit_year_resolves_near():
    assert parse_contract("NQZ5", near=date(2016, 1, 4)).year == 2015
    assert parse_contract("NQZ5", near=date(2025, 1, 2)).year == 2025


def test_single_digit_year_without_near_refused():
    with pytest.raises(ValueError, match="near"):
        parse_contract("NQZ5")


@pytest.mark.parametrize("code", ["NQ", "NQA5", "ZZZZ", "", "NQ 13-25"])
def test_bad_codes_refused(code):
    with pytest.raises(ValueError, match="contract code"):
        parse_contract(code, near=date(2025, 1, 2))


def test_expiry_non_calendar_root_refused():
    with pytest.raises(ValueError) as err:
        expiry(Contract("CL", 12, 2025))
    assert "volume" in str(err.value) and "roll_dates" in str(err.value)


def test_expiry_non_quarterly_month_refused():
    with pytest.raises(ValueError, match="quarterly"):
        expiry(Contract("NQ", 1, 2026))


def test_expiry_plain_third_friday():
    assert expiry(Contract("ES", 12, 2025)) == date(2025, 12, 19)
    assert calendar_roll_day(Contract("ES", 12, 2025)) == date(2025, 12, 15)


# CME Group "Equity Index Roll Dates" table, U.S. indexes, read 2026-11-06
# (cmegroup.com/trading/equity-index/rolldates.html): (year, month, expiration, roll).
CME_TABLE = [
    (2025, 3, date(2025, 3, 21), date(2025, 3, 17)),
    (2025, 6, date(2025, 6, 20), date(2025, 6, 16)),
    (2025, 9, date(2025, 9, 19), date(2025, 9, 15)),
    (2025, 12, date(2025, 12, 19), date(2025, 12, 15)),
    (2026, 3, date(2026, 3, 20), date(2026, 3, 16)),
    (2026, 6, date(2026, 6, 18), date(2026, 6, 15)),
    (2026, 9, date(2026, 9, 18), date(2026, 9, 14)),
    (2026, 12, date(2026, 12, 18), date(2026, 12, 14)),
    (2027, 3, date(2027, 3, 19), date(2027, 3, 15)),
    (2027, 6, date(2027, 6, 18), date(2027, 6, 14)),
    (2027, 9, date(2027, 9, 17), date(2027, 9, 13)),
    (2027, 12, date(2027, 12, 17), date(2027, 12, 13)),
    (2028, 3, date(2028, 3, 17), date(2028, 3, 13)),
    (2028, 6, date(2028, 6, 16), date(2028, 6, 12)),
]


@pytest.mark.parametrize("root", ["NQ", "ES"])
@pytest.mark.parametrize("year, month, exp, roll", CME_TABLE)
def test_expiry_matches_cme(root, year, month, exp, roll):
    assert expiry(Contract(root, month, year)) == exp


@pytest.mark.parametrize("root", ["NQ", "ES"])
@pytest.mark.parametrize("year, month, exp, roll", CME_TABLE)
def test_calendar_roll_day_matches_cme(root, year, month, exp, roll):
    assert calendar_roll_day(Contract(root, month, year)) == roll
