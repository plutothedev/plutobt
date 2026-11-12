"""3.0.1 (audit minors): values that cannot be meant are refused where they are given, by
name, before any run starts."""
import math
from dataclasses import replace

import pytest

from plutobt.config import BacktestConfig, load_config
from plutobt.instruments import get_instrument
from plutobt.pbo import pbo
from plutobt.splits import cscv_blocks, purged_folds, walk_forward_windows
from plutobt.study import run_study
from tests.study_fixtures import CFG, Coin, Hold, walk

NQ = get_instrument("NQ")


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_a_non_finite_setting_is_refused_before_any_run(tmp_path, bad):
    led = tmp_path / "t.jsonl"
    with pytest.raises(ValueError, match=r"setting 'bias'.*finite"):
        run_study(Coin, {"seed": [0, 1], "bias": [0.5, bad]}, walk(600), NQ, CFG, ledger=led)
    assert not led.exists()                       # nothing ran, nothing was recorded


@pytest.mark.parametrize("name, bad", [("blocks", 16.5), ("wf_splits", 2.5), ("cv_folds", 2.5),
                                       ("embargo_bars", 2.7), ("blocks", True),
                                       ("wf_splits", "2"), ("max_configs", 10.5)])
def test_counts_must_be_whole_numbers(name, bad):
    with pytest.raises(ValueError, match=rf"{name} must be a whole number"):
        run_study(Hold, {"side": [1, -1]}, walk(600, drift=0.3), NQ, CFG, ledger=False,
                  **{name: bad})


def test_a_whole_float_count_is_taken_as_that_number():
    kwargs = dict(ledger=False, blocks=4, wf_splits=2, cv_folds=2, embargo_bars=3)
    a = run_study(Hold, {"side": [1, -1]}, walk(600, drift=0.3), NQ, CFG, **kwargs)
    floats = {k: (float(v) if k != "ledger" else v) for k, v in kwargs.items()}
    b = run_study(Hold, {"side": [1, -1]}, walk(600, drift=0.3), NQ, CFG, **floats)
    assert a.pbo.pbo == b.pbo.pbo and a.cv.embargo == b.cv.embargo == 3
    assert len(a.walk_forward.windows) == len(b.walk_forward.windows) == 2


def test_the_split_functions_refuse_fractions_themselves():
    for call in (lambda: cscv_blocks(960, 16.5), lambda: walk_forward_windows(960, 2.5, False),
                 lambda: purged_folds(960, 2.5, 0), lambda: purged_folds(960, 2, 1.5)):
        with pytest.raises(ValueError, match="whole number"):
            call()
    import numpy as np

    with pytest.raises(ValueError, match="whole number"):
        pbo(np.random.default_rng(0).standard_normal((960, 3)), blocks=16.5)


# --- config ---------------------------------------------------------------------------------

@pytest.mark.parametrize("field, bad, why", [
    ("carry_overnight", "false", "true or false"),
    ("allow_short", 0, "true or false"),
    ("check_causality", "no", "true or false"),
    ("commission_per_side", "2.5", "a number"),
    ("commission_per_side", float("nan"), "finite"),
    ("slippage_ticks", True, "a number"),
    ("periods_per_year", 0, "above 0"),
    ("periods_per_year", "20412", "a number"),
    ("break_on", 5, "text"),
])
def test_a_config_value_of_the_wrong_kind_is_refused_by_name(field, bad, why):
    with pytest.raises(ValueError) as err:
        BacktestConfig(**{field: bad})
    assert f"config {field}" in str(err.value) and why in str(err.value)
    with pytest.raises(ValueError):
        replace(CFG, **{field: bad})


def test_a_negative_cost_is_still_a_rebate():
    # deliberate since 2026-09-28 (test_a_negative_commission_is_a_rebate_and_runs)
    assert BacktestConfig(commission_per_side=-0.5, slippage_ticks=-1.0)


def test_ordinary_configs_still_build():
    import numpy as np

    cfg = BacktestConfig(commission_per_side=2, slippage_ticks=np.float64(1.0),
                         periods_per_year=np.int64(252), carry_overnight=np.bool_(True))
    assert cfg.commission_per_side == 2 and math.isclose(cfg.periods_per_year, 252)
    assert BacktestConfig(slippage_ticks=None, periods_per_year=None).slippage_ticks is None


def test_a_quoted_false_in_a_config_file_is_refused(tmp_path):
    p = tmp_path / "c.toml"
    p.write_text('carry_overnight = "false"\n', encoding="utf-8")
    with pytest.raises(ValueError, match=r"config carry_overnight.*true or false"):
        load_config(str(p))


def test_an_unknown_config_key_lists_the_valid_ones(tmp_path):
    p = tmp_path / "c.toml"
    p.write_text("comission_per_side = 2.5\n", encoding="utf-8")
    with pytest.raises(ValueError) as err:
        load_config(str(p))
    text = str(err.value)
    assert "unknown config key: comission_per_side" in text
    assert "commission_per_side" in text and "slippage_ticks" in text
    assert "did you mean 'commission_per_side'" in text
