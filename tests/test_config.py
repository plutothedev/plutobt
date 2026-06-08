from dataclasses import fields, is_dataclass

from plutobt.config import BacktestConfig


def test_backtestconfig_is_dataclass_with_exact_fields():
    assert is_dataclass(BacktestConfig)
    names = {f.name for f in fields(BacktestConfig)}
    assert names == {
        "commission_per_side",
        "slippage_ticks",
        "risk_free",
        "periods_per_year",
        "carry_overnight",
        "allow_short",
    }


def test_backtestconfig_defaults():
    cfg = BacktestConfig()
    assert cfg.commission_per_side == 0.0
    assert cfg.slippage_ticks is None
    assert cfg.risk_free == 0.0
    assert cfg.periods_per_year is None
    assert cfg.carry_overnight is False
    assert cfg.allow_short is True


def test_backtestconfig_overrides():
    cfg = BacktestConfig(commission_per_side=2.5, slippage_ticks=1.0, allow_short=False)
    assert cfg.commission_per_side == 2.5
    assert cfg.slippage_ticks == 1.0
    assert cfg.allow_short is False
