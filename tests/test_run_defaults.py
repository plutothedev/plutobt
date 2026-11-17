import pandas as pd
import pytest

from plutobt.data import load_bars
from plutobt.run_defaults import DEFAULT_COSTS, bars_per_year, default_config


def test_defaults_cover_buyable_roots_and_never_zero():
    assert set(DEFAULT_COSTS) == {"NQ", "MNQ", "ES", "MES", "YM", "RTY", "CL", "GC", "ZN"}
    assert all(c > 0 and s > 0 for c, s in DEFAULT_COSTS.values())


def test_defaults_match_shipped_configs():
    assert DEFAULT_COSTS["NQ"] == (2.50, 1.0)
    assert DEFAULT_COSTS["ES"] == (2.50, 1.0)
    assert DEFAULT_COSTS["CL"] == (3.50, 3.0)


def test_slippage_never_below_registry():
    from plutobt.instruments import REGISTRY
    for sym, (_, slip) in DEFAULT_COSTS.items():
        assert slip >= REGISTRY[sym].slippage_ticks


def test_default_config_fields():
    cfg = default_config("NQ", periods_per_year=20412.0)
    assert (cfg.commission_per_side, cfg.slippage_ticks, cfg.periods_per_year,
            cfg.allow_short) == (2.50, 1.0, 20412.0, False)


def test_default_config_unknown_symbol_message():
    with pytest.raises(KeyError, match="pass --config FILE"):
        default_config("AAPL", periods_per_year=252.0)


def test_bars_per_year_rth_5m_sample():
    assert bars_per_year(load_bars("data/sample/nq_5m.json")) == 81 * 252


def test_bars_per_year_daily():
    # 10 weekdays, one bar each at 15:00 CT (21:00 UTC in winter) -> one bar per day
    ts = pd.date_range("2026-01-05 21:00", periods=10, freq="B", tz="UTC")
    df = pd.DataFrame({"ts": (ts.astype("int64") // 10**9).astype("int64")})
    assert bars_per_year(df) == 252.0


def test_bars_per_year_empty_frame_refused():
    with pytest.raises(ValueError, match="at least one bar"):
        bars_per_year(pd.DataFrame({"ts": pd.Series([], dtype="int64")}))


def test_cost_label_text():
    from plutobt.run_defaults import COST_LABEL
    assert COST_LABEL == "representative public retail costs, not a live account"
