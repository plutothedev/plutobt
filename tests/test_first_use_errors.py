"""3.0.1, audit findings 11 and 13: the mistakes a newcomer makes first get a message they
can act on, not an AttributeError from inside the engine."""
import pandas as pd
import pytest

from plutobt import (
    BacktestConfig,
    Engine,
    compute_metrics,
    format_report,
    get_instrument,
    load_bars,
)
from plutobt.broker import FrameError
from plutobt.strategies.ema_cross import EMACross
from plutobt.strategy import Strategy
from tests.study_fixtures import CFG, walk

NQ = get_instrument("NQ")


class ForgetsReturn(Strategy):
    def precompute(self, df):
        df["x"] = df.close.rolling(5).mean()

    def on_bar(self, ctx):
        pass


class ReturnsSeries(ForgetsReturn):
    def precompute(self, df):
        return df.close.rolling(5).mean()


@pytest.mark.parametrize("check", [True, False])
@pytest.mark.parametrize("cls, got", [(ForgetsReturn, "None"), (ReturnsSeries, "a Series")])
def test_precompute_that_does_not_return_the_frame(cls, got, check):
    from dataclasses import replace

    with pytest.raises(FrameError) as err:
        Engine(NQ, replace(CFG, check_causality=check)).run(walk(200), cls())
    text = str(err.value)
    assert f"{cls.__name__}.precompute returned {got}" in text
    assert "return df" in text


def test_a_strategy_class_instead_of_an_instance():
    with pytest.raises(TypeError, match=r"pass a strategy instance, not the class: "
                                        r"EMACross\(\)"):
        Engine(NQ, CFG).run(walk(200), EMACross)


def test_capitalised_columns_name_the_fix():
    df = walk(200).rename(columns=str.capitalize)
    with pytest.raises(ValueError) as err:
        Engine(NQ, CFG).run(df, EMACross())
    text = str(err.value)
    assert "the frame has no 'ts' column" in text
    assert "it has: Ts, Open, High, Low, Close, Volume" in text
    assert "'Ts' looks like 'ts'" in text and "lower case" in text


def test_a_datetime_index_names_the_fix():
    df = walk(200)
    df.index = pd.to_datetime(df.pop("ts"), unit="s")
    with pytest.raises(ValueError) as err:
        Engine(NQ, CFG).run(df, EMACross())
    text = str(err.value)
    assert "the frame has no 'ts' column" in text
    assert "its index holds datetimes" in text and "unix seconds" in text


def test_a_missing_price_column_lists_what_the_frame_has():
    df = walk(200).rename(columns={"close": "Close"})
    with pytest.raises(ValueError) as err:
        Engine(NQ, CFG).run(df, EMACross())
    assert "no 'close' column" in str(err.value)
    assert "'Close' looks like 'close'" in str(err.value)


@pytest.mark.parametrize("bad", ["1767277800", True, None.__class__, [1767277800]])
def test_trade_from_ts_must_be_a_number(bad):
    df = walk(200)
    with pytest.raises(ValueError, match="trade_from_ts must be the ts of a bar: a whole "
                                         "number of unix seconds"):
        Engine(NQ, CFG).run(df, EMACross(), trade_from_ts=bad)


def test_trade_from_ts_still_takes_numpy_and_float_whole_numbers():
    import numpy as np

    df = walk(200)
    ts = df.ts.iloc[100]
    a = Engine(NQ, CFG).run(df, EMACross(), trade_from_ts=int(ts))
    b = Engine(NQ, CFG).run(df, EMACross(), trade_from_ts=np.int64(ts))
    c = Engine(NQ, CFG).run(df, EMACross(), trade_from_ts=float(ts))
    assert a.equity_curve.equals(b.equity_curve) and a.equity_curve.equals(c.equity_curve)


def test_format_report_before_compute_metrics_says_what_to_call():
    res = Engine(NQ, CFG).run(walk(200), EMACross())
    with pytest.raises(ValueError, match=r"result\.metrics is empty: call "
                                         r"compute_metrics\(result, periods_per_year=\.\.\.\)"):
        format_report(res, symbol="NQ")


def test_compute_metrics_without_periods_per_year_says_what_it_is():
    res = Engine(NQ, BacktestConfig(check_causality=False)).run(walk(200), EMACross())
    with pytest.raises(ValueError) as err:
        compute_metrics(res, periods_per_year=None)
    text = str(err.value)
    assert "periods_per_year" in text and "bars a year" in text
    assert "20412" in text and "252" in text


def test_load_bars_on_a_csv_points_at_load_csv(tmp_path):
    p = tmp_path / "bars.csv"
    p.write_text("ts,open,high,low,close,volume\n1,2,3,1,2,10\n", encoding="utf-8")
    with pytest.raises(ValueError, match="load_csv"):
        load_bars(p)
