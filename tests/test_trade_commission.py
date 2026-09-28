"""Trades carry the fees attributable to them; trade stats are net of fees."""
import pytest

from plutobt.instruments import get_instrument
from plutobt.metrics import expectancy, outlier_sensitivity, profit_factor
from plutobt.portfolio import Fill, Portfolio, Trade

NQ = get_instrument("NQ")  # point_value 20


def _pf():
    return Portfolio(starting_cash=100_000.0, instrument=NQ)


def test_round_trip_books_entry_and_exit_fees():
    p = _pf()
    p.apply_fill(Fill(ts=1, side="buy", contracts=2, price=100.0, commission=5.0))
    p.apply_fill(Fill(ts=2, side="sell", contracts=2, price=101.0, commission=5.0))
    (t,) = p.trades
    assert t.pnl == pytest.approx(40.0)
    assert t.commission == pytest.approx(10.0)
    assert t.net_pnl == pytest.approx(30.0)
    assert p.cash == pytest.approx(100_000.0 + 30.0)


def test_scale_ins_pool_their_entry_fees():
    p = _pf()
    p.apply_fill(Fill(ts=1, side="buy", contracts=1, price=100.0, commission=2.5))
    p.apply_fill(Fill(ts=2, side="buy", contracts=1, price=102.0, commission=2.5))
    p.apply_fill(Fill(ts=3, side="sell", contracts=2, price=103.0, commission=5.0))
    (t,) = p.trades
    assert t.commission == pytest.approx(10.0)


def test_partial_closes_split_the_entry_fee_pro_rata():
    p = _pf()
    p.apply_fill(Fill(ts=1, side="buy", contracts=2, price=100.0, commission=5.0))
    p.apply_fill(Fill(ts=2, side="sell", contracts=1, price=102.0, commission=2.5))
    p.apply_fill(Fill(ts=3, side="sell", contracts=1, price=99.0, commission=2.5))
    a, b = p.trades
    assert (a.commission, b.commission) == (pytest.approx(5.0), pytest.approx(5.0))
    assert (a.net_pnl, b.net_pnl) == (pytest.approx(35.0), pytest.approx(-25.0))


def test_a_flip_splits_the_fill_between_the_closed_and_the_new_position():
    p = _pf()
    p.apply_fill(Fill(ts=1, side="buy", contracts=1, price=100.0, commission=2.5))
    p.apply_fill(Fill(ts=2, side="sell", contracts=3, price=101.0, commission=7.5))
    p.apply_fill(Fill(ts=3, side="buy", contracts=2, price=100.0, commission=5.0))
    first, second = p.trades
    assert first.commission == pytest.approx(5.0)    # 2.5 entry + 7.5 x 1/3 exit
    assert second.side == "short"
    assert second.commission == pytest.approx(10.0)  # 7.5 x 2/3 entry + 5.0 exit
    assert second.net_pnl == pytest.approx(40.0 - 10.0)


def test_open_at_end_carries_its_entry_fee():
    p = _pf()
    p.apply_fill(Fill(ts=1, side="buy", contracts=1, price=100.0, commission=2.5))
    p.mark(105.0)
    p.mark_open_at_end(ts=9)
    (t,) = p.trades
    assert t.open_at_end and t.commission == pytest.approx(2.5) and t.net_pnl == pytest.approx(97.5)


def test_every_fee_paid_lands_on_exactly_one_trade():
    p = _pf()
    fills = [Fill(1, "buy", 1, 100.0, 2.5), Fill(2, "buy", 2, 101.0, 5.0),
             Fill(3, "sell", 1, 101.0, 2.5), Fill(4, "sell", 4, 99.0, 10.0),
             Fill(5, "buy", 1, 98.0, 2.5)]
    for f in fills:
        p.apply_fill(f)
    p.mark(97.0)
    p.mark_open_at_end(ts=6)
    assert sum(t.commission for t in p.trades) == pytest.approx(sum(f.commission for f in fills))


def test_r_multiple_is_measured_after_fees():
    p = _pf()
    p.apply_fill(Fill(ts=1, side="buy", contracts=1, price=100.0, commission=2.5))
    p.position.initial_stop = 99.0          # risk: 1 point = $20 per contract
    p.apply_fill(Fill(ts=2, side="sell", contracts=1, price=102.0, commission=2.5))
    (t,) = p.trades
    assert t.r_multiple == pytest.approx((40.0 - 5.0) / 20.0)


def test_open_at_end_r_multiple_is_after_the_entry_fee():
    p = _pf()
    p.apply_fill(Fill(ts=1, side="buy", contracts=1, price=100.0, commission=2.5))
    p.position.initial_stop = 99.0          # risk: 1 point = $20 per contract
    p.mark(102.0)
    p.mark_open_at_end(ts=9)
    (t,) = p.trades
    assert t.r_multiple == pytest.approx((40.0 - 2.5) / 20.0)


def _t(pnl, commission):
    return Trade(entry_ts=1, exit_ts=2, side="long", contracts=1, entry_price=1.0,
                 exit_price=2.0, pnl=pnl, commission=commission)


def test_trade_stats_are_net_of_fees():
    trades = [_t(30.0, 10.0), _t(5.0, 10.0)]   # net +20 and -5: the second wins gross, loses net
    assert profit_factor(trades) == pytest.approx(20.0 / 5.0)
    assert expectancy(trades) == pytest.approx(0.5 * 20.0 - 0.5 * 5.0)


def test_outlier_sensitivity_reads_net_pnl():
    import pandas as pd
    trades = [_t(30.0, 10.0), _t(5.0, 10.0), _t(12.0, 2.0)]   # net 20, -5, 10
    out = outlier_sensitivity(trades, pd.Series([100.0, 125.0]))
    assert out["pct_pnl_top1"] == pytest.approx(20.0 / 25.0)


def test_zero_commission_changes_nothing():
    t = Trade(entry_ts=1, exit_ts=2, side="long", contracts=1, entry_price=1.0,
              exit_price=2.0, pnl=12.5)
    assert t.commission == 0.0 and t.net_pnl == 12.5


def test_win_rate_is_net_of_fees():
    import pandas as pd

    from plutobt.engine import BacktestResult
    from plutobt.metrics import compute_metrics
    trades = [_t(30.0, 10.0), _t(5.0, 10.0)]   # net +20 and -5: one net winner of two
    result = BacktestResult(equity_curve=pd.Series([100.0, 101.0, 102.0]), trades=trades,
                            metrics={}, manifest={})
    assert compute_metrics(result, periods_per_year=1.0)["win_rate"] == pytest.approx(0.5)


def test_outlier_best_and_worst_are_ranked_net_of_fees():
    import pandas as pd
    # gross 30, 15, -5, -2 (best A, worst C); net 10, 15, -5, -12 (best B, worst D)
    trades = [_t(30.0, 20.0), _t(15.0, 0.0), _t(-5.0, 0.0), _t(-2.0, 10.0)]
    out = outlier_sensitivity(trades, pd.Series([100.0, 108.0]))
    assert out["total_pnl_ex_best"] == pytest.approx(10.0 - 5.0 - 12.0)
    assert out["profit_factor_ex_worst"] == pytest.approx((10.0 + 15.0) / 5.0)
