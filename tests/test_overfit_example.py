"""examples/overfit_study.py runs end to end on the synthetic sample (3.0 Task 10)."""
import subprocess
import sys

from tests.conftest import REPO_ROOT


def test_overfit_example_runs(tmp_path):
    done = subprocess.run(
        [sys.executable, str(REPO_ROOT / "examples" / "overfit_study.py"),
         "--ledger", str(tmp_path / "trials.jsonl")],
        capture_output=True, text=True, check=False, cwd=REPO_ROOT)
    assert done.returncode == 0, done.stderr
    assert done.stdout.startswith("=== PlutoBT study: EMACross on NQ, 6 settings ===")
    assert "N = 6 (new ledger)" in done.stdout
    assert (tmp_path / "trials.jsonl").exists()
