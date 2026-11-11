"""The run ledger (3.0 spec §5): every completed setting run is one line, a try is the
distinct (strategy, data, params), and nothing is ever skipped silently."""
import json
import math

import numpy as np
import pytest

from plutobt.ledger import (
    DEFAULT_PATH,
    Ledger,
    LedgerError,
    params_key,
    strategy_id,
    try_key,
)

DATA = "a" * 64
OTHER = "b" * 64
STRAT = "tests.fake:Cross"


def test_default_path():
    assert DEFAULT_PATH == ".plutobt/trials.jsonl"


def test_record_then_tries_round_trip(tmp_path):
    led = Ledger(tmp_path / "t.jsonl")
    led.record(STRAT, {"fast": 10, "slow": 50}, DATA, 1000, 0.05)
    led.record(STRAT, {"fast": 20, "slow": 50}, DATA, 1000, -0.01)
    tries = Ledger(tmp_path / "t.jsonl").tries(STRAT, DATA)
    assert tries == {try_key({"fast": 10, "slow": 50}, ""): 0.05,
                     try_key({"fast": 20, "slow": 50}, ""): -0.01}


def test_rerun_is_one_try_latest_sharpe_wins(tmp_path):
    led = Ledger(tmp_path / "t.jsonl")
    led.record(STRAT, {"fast": 10}, DATA, 1000, 0.05)
    led.record(STRAT, {"fast": 10}, DATA, 1000, 0.07)
    assert led.tries(STRAT, DATA) == {try_key({"fast": 10}, ""): 0.07}


def test_different_data_counts_separately(tmp_path):
    led = Ledger(tmp_path / "t.jsonl")
    led.record(STRAT, {"fast": 10}, DATA, 1000, 0.05)
    led.record(STRAT, {"fast": 10}, OTHER, 900, 0.02)
    led.record(STRAT, {"fast": 20}, OTHER, 900, 0.03)
    assert len(led.tries(STRAT, DATA)) == 1
    assert len(led.tries(STRAT, OTHER)) == 2


def test_renamed_class_is_a_fresh_count(tmp_path):
    class Cross:
        pass

    class CrossRenamed:
        pass

    assert strategy_id(Cross) != strategy_id(CrossRenamed)
    assert strategy_id(Cross).endswith(":test_renamed_class_is_a_fresh_count.<locals>.Cross")
    led = Ledger(tmp_path / "t.jsonl")
    led.record(strategy_id(Cross), {"fast": 10}, DATA, 1000, 0.05)
    assert led.tries(strategy_id(CrossRenamed), DATA) == {}


def test_params_key_normalizes_numpy_and_integral_floats():
    assert params_key({"fast": np.int64(20)}) == params_key({"fast": 20})
    assert params_key({"fast": 20.0}) == params_key({"fast": 20})
    assert params_key({"x": np.float64(0.5)}) == params_key({"x": 0.5})
    assert params_key({"b": 1, "a": 2}) == params_key({"a": 2, "b": 1})
    assert params_key({"fast": 20.5}) != params_key({"fast": 20})


def test_damaged_line_refused(tmp_path):
    p = tmp_path / "t.jsonl"
    led = Ledger(p)
    led.record(STRAT, {"fast": 10}, DATA, 1000, 0.05)
    with p.open("a", encoding="utf-8") as fh:
        fh.write("{not json\n")
    with pytest.raises(LedgerError, match=r"t\.jsonl.*line 2"):
        Ledger(p).tries(STRAT, DATA)


def test_unknown_schema_version_refused(tmp_path):
    p = tmp_path / "t.jsonl"
    led = Ledger(p)
    led.record(STRAT, {"fast": 10}, DATA, 1000, 0.05)
    line = json.loads(p.read_text(encoding="utf-8").splitlines()[0])
    line["v"] = 2
    with p.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(line) + "\n")
    with pytest.raises(LedgerError, match=r"line 2.*version 2"):
        Ledger(p).tries(STRAT, DATA)


def test_missing_folder_is_created(tmp_path):
    p = tmp_path / "deep" / "er" / "t.jsonl"
    led = Ledger(p)
    assert led.is_new
    led.record(STRAT, {"fast": 10}, DATA, 1000, 0.05)
    assert p.exists()


def test_is_new(tmp_path):
    p = tmp_path / "t.jsonl"
    assert Ledger(p).is_new
    Ledger(p).record(STRAT, {"fast": 10}, DATA, 1000, 0.05)
    assert not Ledger(p).is_new


def test_nan_sharpe_round_trips(tmp_path):
    p = tmp_path / "t.jsonl"
    Ledger(p).record(STRAT, {"fast": 10}, DATA, 1000, float("nan"))
    assert json.loads(p.read_text(encoding="utf-8"))["sharpe"] is None
    (val,) = Ledger(p).tries(STRAT, DATA).values()
    assert math.isnan(val)


def test_line_has_exactly_the_schema_keys(tmp_path):
    import plutobt

    p = tmp_path / "t.jsonl"
    Ledger(p).record(STRAT, {"fast": np.int64(10)}, DATA, 1000, 0.05)
    line = json.loads(p.read_text(encoding="utf-8"))
    assert set(line) == {"v", "time", "strategy", "code_sha256", "params", "data_sha256",
                         "bars", "sharpe", "plutobt"}
    assert line["v"] == 1 and line["params"] == {"fast": 10}
    assert line["plutobt"] == plutobt.__version__
    assert line["time"].endswith("+00:00")


def test_non_json_value_refused(tmp_path):
    with pytest.raises(LedgerError, match="fast"):
        Ledger(tmp_path / "t.jsonl").record(STRAT, {"fast": object()}, DATA, 1000, 0.1)


def test_code_sha256_is_stable_and_tells_code_apart():
    from plutobt.ledger import code_sha256
    from tests.study_fixtures import Coin, Hold

    assert code_sha256(Coin) == code_sha256(Coin)
    assert code_sha256(Coin) != code_sha256(Hold)
    assert len(code_sha256(Coin)) == 64


def test_code_sha256_without_source_falls_back_to_bytecode():
    from plutobt.ledger import code_sha256

    ns = {}
    exec("from plutobt.strategy import Strategy\n"            # noqa: S102 (no source file)
         "class A(Strategy):\n    def on_bar(self, ctx):\n        return 1\n", ns)
    exec("from plutobt.strategy import Strategy\n"            # noqa: S102
         "class A(Strategy):\n    def on_bar(self, ctx):\n        return 2\n", ns2 := {})
    assert code_sha256(ns["A"]) != code_sha256(ns2["A"])


def test_resolved_settings_include_class_defaults():
    from plutobt.ledger import resolved_settings
    from tests.study_fixtures import Coin

    assert resolved_settings(Coin, {"seed": 4}) == {"seed": 4, "bias": 0.5}
