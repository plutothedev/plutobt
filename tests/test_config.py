import tomllib
from dataclasses import fields, is_dataclass
from pathlib import Path

import pytest

from plutobt.config import BacktestConfig, load_config


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


# ---------------------------------------------------------------------------
# M8.1: load_config tests
# ---------------------------------------------------------------------------

def _write(tmp_path: Path, text: str) -> str:
    p = tmp_path / "cfg.toml"
    p.write_text(text, encoding="utf-8")
    return str(p)


def test_load_config_returns_backtestconfig_with_all_fields(tmp_path):
    path = _write(
        tmp_path,
        """
        commission_per_side = 2.5
        slippage_ticks = 2.0
        risk_free = 0.0
        periods_per_year = 20412.0
        carry_overnight = false
        allow_short = true
        """,
    )
    cfg = load_config(path)
    assert isinstance(cfg, BacktestConfig)
    assert cfg.commission_per_side == 2.5
    assert cfg.slippage_ticks == 2.0
    assert cfg.risk_free == 0.0
    assert cfg.periods_per_year == 20412.0
    assert cfg.carry_overnight is False
    assert cfg.allow_short is True


def test_load_config_applies_dataclass_defaults_for_omitted_keys(tmp_path):
    path = _write(tmp_path, "commission_per_side = 1.0\n")
    cfg = load_config(path)
    assert cfg.commission_per_side == 1.0
    assert cfg.slippage_ticks is None        # None => use instrument default
    assert cfg.risk_free == 0.0
    assert cfg.periods_per_year is None       # None => derived from data/session
    assert cfg.carry_overnight is False
    assert cfg.allow_short is True


def test_load_config_rejects_unknown_keys(tmp_path):
    path = _write(tmp_path, "commission_per_side = 1.0\nbogus_key = 7\n")
    with pytest.raises(ValueError, match="unknown config key: bogus_key"):
        load_config(path)


def test_load_config_empty_file_is_all_defaults(tmp_path):
    path = _write(tmp_path, "")
    cfg = load_config(path)
    assert cfg == BacktestConfig()
