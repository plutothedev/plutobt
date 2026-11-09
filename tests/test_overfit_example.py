"""examples/overfit_study.py runs end to end on the synthetic sample (3.0 Task 10)."""
import os
import subprocess
import sys

from tests.conftest import REPO_ROOT


def _legacy_encoding_env():
    # CI on Windows: no UTF-8 mode, so a piped stdout is cp1252 and cannot encode the
    # report's warning sign. Reproduce that here whatever this machine's settings are.
    env = {k: v for k, v in os.environ.items()
           if k not in ("PYTHONUTF8", "PYTHONIOENCODING")}
    env["PYTHONUTF8"] = "0"
    env["PYTHONIOENCODING"] = "cp1252"
    return env


def test_overfit_example_runs(tmp_path):
    done = subprocess.run(
        [sys.executable, str(REPO_ROOT / "examples" / "overfit_study.py"),
         "--ledger", str(tmp_path / "trials.jsonl")],
        capture_output=True, encoding="utf-8", check=False, cwd=REPO_ROOT,
        env=_legacy_encoding_env())
    assert done.returncode == 0, done.stderr
    assert done.stdout.startswith("=== PlutoBT study: EMACross on NQ, 6 settings ===")
    assert "N = 6 (new ledger)" in done.stdout
    assert (tmp_path / "trials.jsonl").exists()
    assert "⚠ winner changed" in done.stdout     # the warning sign survives, in UTF-8
