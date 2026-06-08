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


def test_mark_updates_unrealized_via_equity():
    nq = get_instrument("NQ")
    pf = Portfolio(starting_cash=50_000.0, instrument=nq)
    pf.apply_fill(Fill(ts=1000, side="buy", contracts=1, price=20_000.0, commission=0.0))
    pf.mark(20_005.0)
    # Unrealized +5 pts * $20 = +$100; cash unchanged (no realization on mark).
    assert pf.cash == pytest.approx(50_000.0)
    assert pf.equity(mark_price=20_005.0) == pytest.approx(50_100.0)


def test_mark_does_not_move_cash():
    nq = get_instrument("NQ")
    pf = Portfolio(starting_cash=50_000.0, instrument=nq)
    pf.apply_fill(Fill(ts=1000, side="sell", contracts=2, price=20_000.0, commission=0.0))
    pf.mark(20_050.0)   # adverse for short
    assert pf.cash == pytest.approx(50_000.0)
    # Short down 50 pts * $20 * 2 = -$2000 unrealized.
    assert pf.equity(mark_price=20_050.0) == pytest.approx(48_000.0)


def test_margin_ok_gate_on_initial_margin():
    cl = get_instrument("CL")  # representative initial_margin from registry
    pf = Portfolio(starting_cash=10_000.0, instrument=cl)
    im = cl.initial_margin
    n_ok = int(10_000.0 // im)
    assert pf.margin_ok(n_ok) is True
    assert pf.margin_ok(n_ok + 1) is False


def test_margin_ok_uses_current_cash_not_equity():
    cl = get_instrument("CL")
    pf = Portfolio(starting_cash=cl.initial_margin, instrument=cl)
    # Exactly enough cash for 1 contract.
    assert pf.margin_ok(1) is True
    assert pf.margin_ok(2) is False


def test_margin_call_zero_when_above_maintenance():
    cl = get_instrument("CL")
    pf = Portfolio(starting_cash=10_000.0, instrument=cl)
    pf.apply_fill(Fill(ts=1000, side="buy", contracts=1, price=70.00, commission=0.0))
    # Mark equal to entry: equity == cash == 10000 > maintenance => no call.
    pf.mark(70.00)
    assert pf.margin_call_amount() == pytest.approx(0.0)


def test_margin_call_restore_to_initial_crude_3400():
    """Spec §10.5 / §5.2: crude initial 10,000, maint 7,000.

    Drive equity to 6,600 (below maintenance) on a 1-lot and assert the call
    restores to INITIAL: 10,000 - 6,600 = $3,400.
    """
    cl = get_instrument("CL")
    # Pin the reference margins so the test is independent of registry drift.
    assert cl.initial_margin == pytest.approx(10_000.0)
    assert cl.maintenance_margin == pytest.approx(7_000.0)

    pf = Portfolio(starting_cash=10_000.0, instrument=cl)
    pf.apply_fill(Fill(ts=1000, side="buy", contracts=1, price=70.00, commission=0.0))
    # CL = $1,000/point. Need equity 6,600 => loss of 3,400 => 3.40 points down.
    mark = 70.00 - 3.40
    pf.mark(mark)
    assert pf.equity(mark_price=mark) == pytest.approx(6_600.0)
    # equity 6,600 < maintenance 7,000 => breach.
    # call = initial*qty - equity = 10,000 - 6,600 = 3,400.
    assert pf.margin_call_amount() == pytest.approx(3_400.0)


def test_margin_call_uses_last_mark():
    cl = get_instrument("CL")
    pf = Portfolio(starting_cash=10_000.0, instrument=cl)
    pf.apply_fill(Fill(ts=1000, side="buy", contracts=2, price=70.00, commission=0.0))
    # 2 lots: maintenance breach when equity < 14,000; initial target 20,000.
    # But starting cash is only 10,000 — drive a small loss to confirm formula.
    pf.mark(69.50)  # -0.50 pt * $1000 * 2 = -$1000 => equity 9,000.
    # equity 9,000 < maint 14,000 => call = 20,000 - 9,000 = 11,000.
    assert pf.margin_call_amount() == pytest.approx(11_000.0)


def test_margin_call_zero_when_flat():
    cl = get_instrument("CL")
    pf = Portfolio(starting_cash=10_000.0, instrument=cl)
    assert pf.margin_call_amount() == pytest.approx(0.0)


def test_mtm_gap_open_entry_stop_exit_to_the_cent():
    """Spec §10.4: fill-basis MTM to the cent.

    NQ ($20/pt). Enter long at a GAP OPEN fill of 20,012.00 (prior close was
    20,000.00 — the prior-close-to-fill move is NOT ours). Exit via a STOP
    fill at 20,007.00. True P&L = (20,007 - 20,012) * $20 = -$100.00, exactly.
    A naive close-to-close ledger would mis-book the unowned +12pt gap.
    """
    nq = get_instrument("NQ")
    assert nq.point_value == 20.0
    pf = Portfolio(starting_cash=100_000.0, instrument=nq)

    # Entry at gap-open fill (NOT prior close). Commission 0 to isolate basis.
    pf.apply_fill(Fill(ts=1000, side="buy", contracts=1, price=20_012.00, commission=0.0))
    assert pf.position.avg_price == pytest.approx(20_012.00)

    # Interim mark at the entry bar's close (20,020) — unrealized only.
    pf.mark(20_020.00)
    assert pf.equity(mark_price=20_020.00) == pytest.approx(100_000.0 + (20_020.00 - 20_012.00) * 20.0)

    # Exit via stop fill at 20,007.00 (a non-close price).
    pf.apply_fill(Fill(ts=2000, side="sell", contracts=1, price=20_007.00, commission=0.0))

    # Realized fill-to-fill: (20,007 - 20,012) * $20 = -$100.00, to the cent.
    assert pf.cash == pytest.approx(99_900.00)
    assert pf.position.qty == 0
    assert len(pf.trades) == 1
    tr = pf.trades[0]
    assert tr.entry_price == pytest.approx(20_012.00)
    assert tr.exit_price == pytest.approx(20_007.00)
    assert tr.pnl == pytest.approx(-100.00)


def test_mtm_gap_open_with_commission_to_the_cent():
    """Same seam, now with commission on both sides, exact to the cent."""
    nq = get_instrument("NQ")
    pf = Portfolio(starting_cash=100_000.0, instrument=nq)
    pf.apply_fill(Fill(ts=1000, side="buy", contracts=1, price=20_012.00, commission=2.10))
    pf.apply_fill(Fill(ts=2000, side="sell", contracts=1, price=20_007.00, commission=2.10))
    # -100.00 gross - 2.10 - 2.10 commission = -104.20.
    assert pf.cash == pytest.approx(100_000.0 - 100.00 - 4.20)
    # Trade.pnl is gross of commission (price-basis only).
    assert pf.trades[0].pnl == pytest.approx(-100.00)


def test_trade_boundary_vwap_scale_in_out_pro_rata():
    """Spec §10.16: scale-in to VWAP, partial scale-out pro-rata."""
    nq = get_instrument("NQ")
    pf = Portfolio(starting_cash=100_000.0, instrument=nq)
    # Scale in: 2 @ 20,000 then 2 @ 20,040 => VWAP 20,020, qty 4.
    pf.apply_fill(Fill(ts=1000, side="buy", contracts=2, price=20_000.0, commission=0.0))
    pf.apply_fill(Fill(ts=1100, side="buy", contracts=2, price=20_040.0, commission=0.0))
    assert pf.position.qty == 4
    assert pf.position.avg_price == pytest.approx(20_020.0)

    # Scale out 3 @ 20,070 => realized (20,070 - 20,020) * $20 * 3 = +$3,000.
    pf.apply_fill(Fill(ts=2000, side="sell", contracts=3, price=20_070.0, commission=0.0))
    assert pf.position.qty == 1
    assert pf.position.avg_price == pytest.approx(20_020.0)  # VWAP unchanged on scale-out
    assert pf.cash == pytest.approx(103_000.0)
    assert len(pf.trades) == 1
    assert pf.trades[0].contracts == 3
    assert pf.trades[0].pnl == pytest.approx(3_000.0)
    assert pf.trades[0].open_at_end is False


def test_open_at_end_marked_and_flagged():
    """Spec §10.16 / §5.9: residual open position marked-to-last, flagged."""
    nq = get_instrument("NQ")
    pf = Portfolio(starting_cash=100_000.0, instrument=nq)
    pf.apply_fill(Fill(ts=1000, side="buy", contracts=2, price=20_000.0, commission=0.0))
    pf.apply_fill(Fill(ts=1100, side="buy", contracts=2, price=20_040.0, commission=0.0))
    pf.apply_fill(Fill(ts=2000, side="sell", contracts=3, price=20_070.0, commission=0.0))
    # 1 contract remains open at VWAP 20,020. Mark to last close 20,090 and flag.
    pf.mark(20_090.0)
    pf.mark_open_at_end(ts=2000)
    assert pf.position.qty == 1            # position stays open (mark, not close)
    assert len(pf.trades) == 2
    closed, open_trade = pf.trades
    assert closed.open_at_end is False
    assert open_trade.open_at_end is True
    assert open_trade.exit_ts is None
    assert open_trade.exit_price is None
    assert open_trade.contracts == 1
    # Unrealized at last mark: (20,090 - 20,020) * $20 * 1 = +$1,400.
    assert open_trade.pnl == pytest.approx(1_400.0)
    # Equity includes the open MTM; cash does NOT (unrealized).
    assert pf.cash == pytest.approx(103_000.0)
    assert pf.equity(mark_price=20_090.0) == pytest.approx(103_000.0 + 1_400.0)


def test_open_at_end_idempotent_no_double_book():
    nq = get_instrument("NQ")
    pf = Portfolio(starting_cash=100_000.0, instrument=nq)
    pf.apply_fill(Fill(ts=1000, side="buy", contracts=1, price=20_000.0, commission=0.0))
    pf.mark(20_010.0)
    pf.mark_open_at_end(ts=1000)
    pf.mark_open_at_end(ts=1000)
    open_trades = [t for t in pf.trades if t.open_at_end]
    assert len(open_trades) == 1


def test_open_at_end_noop_when_flat():
    nq = get_instrument("NQ")
    pf = Portfolio(starting_cash=100_000.0, instrument=nq)
    pf.mark_open_at_end(ts=1000)
    assert pf.trades == []
