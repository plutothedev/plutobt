import math

import pytest

from plutobt.portfolio import Position, Trade


def test_position_defaults_flat():
    p = Position()
    assert p.qty == 0
    assert p.avg_price is None
    assert p.initial_stop is None


def test_position_is_mutable_dataclass():
    p = Position(qty=2, avg_price=100.0, initial_stop=95.0)
    p.qty = -1
    p.avg_price = 101.5
    assert p.qty == -1
    assert p.avg_price == 101.5
    assert p.initial_stop == 95.0


def test_trade_value_type_fields():
    t = Trade(
        entry_ts=1000,
        exit_ts=2000,
        side="long",
        contracts=2,
        entry_price=100.0,
        exit_price=108.0,
        pnl=160.0,
    )
    assert t.entry_ts == 1000
    assert t.exit_ts == 2000
    assert t.side == "long"
    assert t.contracts == 2
    assert t.entry_price == 100.0
    assert t.exit_price == 108.0
    assert t.pnl == 160.0
    assert t.r_multiple is None
    assert t.open_at_end is False


def test_trade_open_at_end_flag():
    t = Trade(
        entry_ts=1000,
        exit_ts=None,
        side="short",
        contracts=1,
        entry_price=110.0,
        exit_price=None,
        pnl=-50.0,
        open_at_end=True,
    )
    assert t.exit_ts is None
    assert t.exit_price is None
    assert t.open_at_end is True
    assert math.isclose(t.pnl, -50.0)


from plutobt.instruments import get_instrument
from plutobt.portfolio import Fill, Portfolio


def test_fill_value_type_fields():
    f = Fill(ts=1000, side="buy", contracts=2, price=100.25, commission=4.0)
    assert f.ts == 1000
    assert f.side == "buy"
    assert f.contracts == 2
    assert f.price == 100.25
    assert f.commission == 4.0


def test_fill_sell_side():
    f = Fill(ts=2000, side="sell", contracts=1, price=108.0, commission=2.0)
    assert f.side == "sell"
    assert f.contracts == 1


def test_portfolio_starts_flat_with_cash():
    nq = get_instrument("NQ")
    pf = Portfolio(starting_cash=50_000.0, instrument=nq)
    assert pf.cash == 50_000.0
    assert pf.position.qty == 0
    assert pf.position.avg_price is None
    assert pf.trades == []


def test_portfolio_equity_when_flat_equals_cash():
    nq = get_instrument("NQ")
    pf = Portfolio(starting_cash=50_000.0, instrument=nq)
    # Flat => no open MTM regardless of mark price.
    assert pf.equity(mark_price=20_000.0) == 50_000.0
    assert pf.equity(mark_price=15_000.0) == 50_000.0


def test_apply_fill_open_long_sets_vwap_and_only_commission_off_cash():
    nq = get_instrument("NQ")
    pf = Portfolio(starting_cash=50_000.0, instrument=nq)
    pf.apply_fill(Fill(ts=1000, side="buy", contracts=2, price=20_000.0, commission=5.0))
    assert pf.position.qty == 2
    assert pf.position.avg_price == 20_000.0
    # Performance-bond: notional (2 * 20000 * $20) is NOT deducted; only commission.
    assert pf.cash == 50_000.0 - 5.0


def test_apply_fill_scale_in_recomputes_vwap():
    nq = get_instrument("NQ")
    pf = Portfolio(starting_cash=50_000.0, instrument=nq)
    pf.apply_fill(Fill(ts=1000, side="buy", contracts=2, price=20_000.0, commission=4.0))
    pf.apply_fill(Fill(ts=1100, side="buy", contracts=1, price=20_030.0, commission=2.0))
    assert pf.position.qty == 3
    # VWAP = (2*20000 + 1*20030) / 3 = 20010.0
    assert pf.position.avg_price == pytest.approx(20_010.0)
    assert pf.cash == 50_000.0 - 4.0 - 2.0


def test_apply_fill_open_short_sets_negative_qty():
    nq = get_instrument("NQ")
    pf = Portfolio(starting_cash=50_000.0, instrument=nq)
    pf.apply_fill(Fill(ts=1000, side="sell", contracts=3, price=20_000.0, commission=6.0))
    assert pf.position.qty == -3
    assert pf.position.avg_price == 20_000.0
    assert pf.cash == 50_000.0 - 6.0


def test_apply_fill_full_close_long_realizes_pnl_and_books_trade():
    nq = get_instrument("NQ")
    pf = Portfolio(starting_cash=50_000.0, instrument=nq)
    pf.apply_fill(Fill(ts=1000, side="buy", contracts=1, price=20_000.0, commission=0.0))
    pf.apply_fill(Fill(ts=2000, side="sell", contracts=1, price=20_010.0, commission=0.0))
    # +10 pts * $20/pt * 1 = +$200 realized into cash.
    assert pf.cash == pytest.approx(50_200.0)
    assert pf.position.qty == 0
    assert pf.position.avg_price is None
    assert len(pf.trades) == 1
    tr = pf.trades[0]
    assert tr.side == "long"
    assert tr.entry_ts == 1000
    assert tr.exit_ts == 2000
    assert tr.contracts == 1
    assert tr.entry_price == pytest.approx(20_000.0)
    assert tr.exit_price == pytest.approx(20_010.0)
    assert tr.pnl == pytest.approx(200.0)
    assert tr.open_at_end is False


def test_apply_fill_partial_close_pro_rata_keeps_avg_price():
    nq = get_instrument("NQ")
    pf = Portfolio(starting_cash=50_000.0, instrument=nq)
    pf.apply_fill(Fill(ts=1000, side="buy", contracts=3, price=20_000.0, commission=0.0))
    pf.apply_fill(Fill(ts=2000, side="sell", contracts=1, price=20_020.0, commission=0.0))
    # Close 1 of 3 at +20 pts => +$400 realized; 2 remain at unchanged VWAP 20000.
    assert pf.position.qty == 2
    assert pf.position.avg_price == pytest.approx(20_000.0)
    assert pf.cash == pytest.approx(50_400.0)
    assert len(pf.trades) == 1
    assert pf.trades[0].contracts == 1
    assert pf.trades[0].pnl == pytest.approx(400.0)


def test_apply_fill_close_short_realizes_gain_when_price_falls():
    nq = get_instrument("NQ")
    pf = Portfolio(starting_cash=50_000.0, instrument=nq)
    pf.apply_fill(Fill(ts=1000, side="sell", contracts=2, price=20_000.0, commission=0.0))
    pf.apply_fill(Fill(ts=2000, side="buy", contracts=2, price=19_990.0, commission=0.0))
    # Short covered 10 pts lower * $20 * 2 = +$400.
    assert pf.position.qty == 0
    assert pf.cash == pytest.approx(50_400.0)
    assert pf.trades[0].side == "short"
    assert pf.trades[0].pnl == pytest.approx(400.0)


def test_apply_fill_flip_closes_then_opens_remainder():
    nq = get_instrument("NQ")
    pf = Portfolio(starting_cash=50_000.0, instrument=nq)
    pf.apply_fill(Fill(ts=1000, side="buy", contracts=2, price=20_000.0, commission=0.0))
    # Sell 3 while long 2: close 2 (+ realize), open short 1 at fill price.
    pf.apply_fill(Fill(ts=2000, side="sell", contracts=3, price=20_005.0, commission=0.0))
    assert pf.position.qty == -1
    assert pf.position.avg_price == pytest.approx(20_005.0)
    # Realized: close 2 long at +5 pts * $20 = +$200.
    assert pf.cash == pytest.approx(50_200.0)
    assert len(pf.trades) == 1
    assert pf.trades[0].contracts == 2
    assert pf.trades[0].side == "long"
    assert pf.trades[0].pnl == pytest.approx(200.0)
