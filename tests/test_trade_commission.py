"""Trades carry the fees attributable to them; trade stats are net of fees."""
import math
import random

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
    assert t.open_at_end
    assert t.commission == pytest.approx(2.5)
    assert t.net_pnl == pytest.approx(97.5)


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


def _trade_stats(trades):
    """compute_metrics and outlier_sensitivity over these trades (a flat equity curve)."""
    import pandas as pd

    from plutobt.engine import BacktestResult
    from plutobt.metrics import compute_metrics
    equity = pd.Series([100_000.0, 100_000.0, 100_000.0])
    result = BacktestResult(equity_curve=equity, trades=trades, metrics={}, manifest={})
    return compute_metrics(result, periods_per_year=252.0), outlier_sensitivity(trades, equity)


# (opening side, entry, exit): each RTY round trip moves 1 tick = $5.00, the fees exactly
_RTY_NET_ZERO = [("buy", 2000.0, 2000.1), ("buy", 2000.1, 2000.2),
                 ("sell", 2000.2, 2000.1), ("sell", 2000.8, 2000.7)]


def test_trades_that_net_exactly_zero_are_scratches_not_wins_or_losses():
    """Four RTY round trips at $2.50/side each net $0, but float noise in the price
    differences leaves a few 1e-12 on them, two above zero and two below. Without the
    scratch band that noise booked two wins and two losses."""
    p = Portfolio(starting_cash=100_000.0, instrument=get_instrument("RTY"))
    for i, (side, entry, exit_) in enumerate(_RTY_NET_ZERO):
        back = "sell" if side == "buy" else "buy"
        p.apply_fill(Fill(ts=2 * i, side=side, contracts=1, price=entry, commission=2.5))
        p.apply_fill(Fill(ts=2 * i + 1, side=back, contracts=1, price=exit_, commission=2.5))
    nets = [t.net_pnl for t in p.trades]
    assert all(0.0 < abs(x) < 1e-9 for x in nets)      # $0 up to float noise...
    assert sum(x > 0.0 for x in nets) == 2              # ...two just above zero, two below
    m, out = _trade_stats(p.trades)
    assert m["win_rate"] == 0.0
    assert m["expectancy"] == 0.0
    assert math.isnan(m["profit_factor"])               # 0/0: no winner and no loser
    assert out["total_pnl"] == 0.0
    assert out["total_pnl_ex_best"] == 0.0


def test_a_real_sub_cent_net_win_is_still_a_win():
    """CHARACTERIZATION PIN (Task 8 review): the scratch band absorbs float noise only.
    1 AAPL share up 1 cent at $0.0035/side nets +$0.003: real money, a win.
    Mutation-proved: a one-cent band goes red."""
    p = Portfolio(starting_cash=100_000.0, instrument=get_instrument("AAPL"))
    p.apply_fill(Fill(ts=1, side="buy", contracts=1, price=100.0, commission=0.0035))
    p.apply_fill(Fill(ts=2, side="sell", contracts=1, price=100.01, commission=0.0035))
    (t,) = p.trades
    assert t.net_pnl == pytest.approx(0.003)
    m, _ = _trade_stats(p.trades)
    assert m["win_rate"] == 1.0
    assert m["expectancy"] == pytest.approx(0.003)


def test_profit_factor_of_scratches_only_is_nan():
    """0/0 (no winner and no loser) is NaN, not +inf."""
    assert math.isnan(profit_factor([_t(5.0, 5.0), _t(2.5, 2.5)]))


def test_profit_factor_with_no_closed_trades_is_nan():
    """CHARACTERIZATION PIN (Task 8 review): no closed trade is 0/0 as well.
    Mutation-proved: the empty guard returning +inf goes red."""
    held = Trade(entry_ts=1, exit_ts=None, side="long", contracts=1, entry_price=1.0,
                 exit_price=None, pnl=50.0, open_at_end=True, commission=2.5)
    assert math.isnan(profit_factor([]))
    assert math.isnan(profit_factor([held]))


def test_profit_factor_is_inf_only_with_net_profits_and_no_net_losers():
    """CHARACTERIZATION PIN (Task 8 review): a net winner beside a scratch, no net loser.
    Mutation-proved: a zero gross loss always answering NaN goes red."""
    assert profit_factor([_t(30.0, 10.0), _t(5.0, 5.0)]) == math.inf


def test_a_partial_close_r_is_measured_on_the_closed_contracts_after_fees():
    """CHARACTERIZATION PIN (Task 8 review): a partial close's R is over the contracts it
    closed, net of their fees. Mutation-proved: R over the whole position goes red."""
    p = _pf()
    p.apply_fill(Fill(ts=1, side="buy", contracts=2, price=100.0, commission=5.0))
    p.position.initial_stop = 99.0          # risk: 1 point = $20 per contract
    p.apply_fill(Fill(ts=2, side="sell", contracts=1, price=102.0, commission=2.5))
    (t,) = p.trades
    assert t.r_multiple == pytest.approx((40.0 - 2.5 - 2.5) / 20.0)   # 1 contract, not 2


def test_a_multi_contract_open_at_end_carries_all_its_remaining_entry_fees():
    """CHARACTERIZATION PIN (Task 8 review): 2 contracts held at the end carry both
    contracts' remaining entry fees, and their R is net of both. Mutation-proved: the fee
    booked per contract goes red, and so does R with a per-contract fee."""
    p = _pf()
    p.apply_fill(Fill(ts=1, side="buy", contracts=3, price=100.0, commission=7.5))
    p.position.initial_stop = 99.0
    p.apply_fill(Fill(ts=2, side="sell", contracts=1, price=101.0, commission=2.5))
    p.mark(102.0)
    p.mark_open_at_end(ts=9)
    closed, held = p.trades
    assert held.open_at_end
    assert held.contracts == 2
    assert held.commission == pytest.approx(5.0)                       # 2 x 2.5 entry
    assert held.r_multiple == pytest.approx((80.0 - 5.0) / (20.0 * 2))
    assert closed.commission + held.commission == pytest.approx(7.5 + 2.5)


def test_fees_are_conserved_and_net_pnl_sums_to_the_equity_change():
    """PROPERTY PIN (Task 8 review), seeded and fast: 500 random fill sequences (scale-ins,
    partial and full closes, flips, a position still open at the end) with ticket fees NOT
    proportional to contracts. Every fee paid lands on the trades, and the trades' net P&L
    adds up to the change in equity. Mutation-proved against every mutant that loses or
    adds a fee; one that only moves a fee between trades keeps both totals, and the pro-rata
    pins above catch that."""
    rng = random.Random(8)
    for _ in range(500):
        p = _pf()
        paid = 0.0
        n_fills = rng.randint(1, 12)
        for ts in range(1, n_fills + 1):
            fee = round(rng.uniform(0.0, 9.0), 2)
            price = 100.0 + 0.25 * rng.randint(-40, 40)
            side = rng.choice(("buy", "sell"))
            p.apply_fill(Fill(ts, side, rng.randint(1, 5), price, fee))
            paid += fee
        mark = 100.0 + 0.25 * rng.randint(-40, 40)
        p.mark(mark)
        p.mark_open_at_end(ts=n_fills + 1)
        assert sum(t.commission for t in p.trades) == pytest.approx(paid, abs=1e-9)
        change = p.equity(mark) - 100_000.0
        assert sum(t.net_pnl for t in p.trades) == pytest.approx(change, abs=1e-9)
