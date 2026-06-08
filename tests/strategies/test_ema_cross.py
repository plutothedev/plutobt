import numpy as np
import pandas as pd

from plutobt.strategies.ema_cross import EMACross


def _ramp_then_fall(n_up=120, n_down=120):
    # Strictly rising then strictly falling closes -> a long signal then a flat signal.
    up = np.linspace(100.0, 220.0, n_up)
    down = np.linspace(220.0, 100.0, n_down)
    close = np.concatenate([up, down])
    n = close.size
    ts = np.arange(n, dtype=np.int64) * 300  # 5m bars, unix seconds
    return pd.DataFrame(
        {
            "ts": ts,
            "open": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": np.full(n, 1000.0),
        }
    )


def test_precompute_adds_expected_columns_and_is_causal_warmup():
    strat = EMACross()
    df = _ramp_then_fall()
    out = strat.precompute(df.copy())
    # signal columns exist
    for col in ("ema_f", "ema_s", "long"):
        assert col in out.columns
    # adjust=False recursive EMA with min_periods warm-up: first 19 fast / 49 slow are NaN
    assert out["ema_f"].iloc[:19].isna().all()
    assert not np.isnan(out["ema_f"].iloc[19])
    assert out["ema_s"].iloc[:49].isna().all()
    assert not np.isnan(out["ema_s"].iloc[49])
    # 'long' is a clean boolean (NaN comparisons resolve False), never NaN
    assert out["long"].dtype == bool
    # reference EMA matches a hand-rolled recursive ewm(adjust=False)
    ref_f = df["close"].ewm(span=20, adjust=False, min_periods=20).mean()
    np.testing.assert_allclose(out["ema_f"].to_numpy(), ref_f.to_numpy(), equal_nan=True)


def test_precompute_long_flag_tracks_fast_over_slow():
    strat = EMACross()
    df = _ramp_then_fall()
    out = strat.precompute(df.copy())
    warm = out.iloc[50:]  # past slow warm-up
    expected = warm["ema_f"] > warm["ema_s"]
    pd.testing.assert_series_equal(warm["long"], expected, check_names=False)


def test_on_bar_buys_flat_long_and_attaches_stop():
    from plutobt.instruments import get_instrument
    from plutobt.portfolio import Position

    strat = EMACross()
    nq = get_instrument("NQ")

    class _SpyBroker:
        def __init__(self):
            self.calls = []
            self.pending = None

        def buy(self, contracts, stop=None, limit=None):
            self.calls.append(("buy", contracts, stop, limit))

        def close(self):
            self.calls.append(("close",))

    broker = _SpyBroker()
    bar = type("Bar", (), {"long": True, "close": 200.0})()
    ctx = Context_for_test(bar=bar, position=Position(qty=0), broker=broker,
                           instrument=nq, tick=nq.tick_size, equity=100000.0, state={})
    strat.on_bar(ctx)
    assert broker.calls == [("buy", 1, 200.0 - 20 * nq.tick_size, None)]


def test_on_bar_closes_when_long_false_and_in_position():
    from plutobt.instruments import get_instrument
    from plutobt.portfolio import Position

    strat = EMACross()
    nq = get_instrument("NQ")

    class _SpyBroker:
        def __init__(self):
            self.calls = []
            self.pending = None

        def buy(self, contracts, stop=None, limit=None):
            self.calls.append(("buy", contracts, stop, limit))

        def close(self):
            self.calls.append(("close",))

    broker = _SpyBroker()
    bar = type("Bar", (), {"long": False, "close": 200.0})()
    ctx = Context_for_test(bar=bar, position=Position(qty=1, avg_price=199.0),
                           broker=broker, instrument=nq, tick=nq.tick_size,
                           equity=100000.0, state={})
    strat.on_bar(ctx)
    assert broker.calls == [("close",)]


def Context_for_test(**kw):
    from plutobt.strategy import Context
    return Context(**kw)
