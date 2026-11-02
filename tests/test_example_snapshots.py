"""Snapshot pins: the FULL printed output of every on-camera example.

Each script's stdout is captured and compared, line for line, with a frozen copy in
tests/expected_output/. Any change to a number, label or line a viewer sees fails here
before it reaches the camera, including drift no targeted test thought to check.

When a change to the printed output is deliberate, regenerate with
``PYTHONPATH=src python scripts/freeze_examples.py`` and review the diff of
tests/expected_output/ in the same commit, with the reason in the commit message.
Never regenerate to make this test pass without explaining why the output moved.
"""
from __future__ import annotations

import contextlib
import difflib
import importlib
import io
import os
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
EXPECTED_DIR = ROOT / "tests" / "expected_output"
# Script name (examples/<name>.py) -> its frozen output (tests/expected_output/<name>.txt).
EXAMPLES = ("run_demo", "causality_demo", "cost_sweep", "cold_open_lookahead")
_REPO = "<repo>"


def normalise(text: str) -> str:
    """LF line endings, and the repo's absolute path (either slash) as <repo>."""
    text = text.replace("\r\n", "\n")
    for root in (str(ROOT), ROOT.as_posix()):
        text = text.replace(root, _REPO)
    return text


def render(name: str) -> str:
    """Run examples/<name>.py's main() from the repo root and return its stdout,
    normalised."""
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    module = importlib.import_module(f"examples.{name}")
    buf = io.StringIO()
    cwd = os.getcwd()
    os.chdir(ROOT)                       # run_demo's paths are repo-relative
    try:
        with contextlib.redirect_stdout(buf):
            module.main()
    finally:
        os.chdir(cwd)
    return normalise(buf.getvalue())


def expected_path(name: str) -> pathlib.Path:
    return EXPECTED_DIR / f"{name}.txt"


@pytest.mark.parametrize("name", EXAMPLES)
def test_example_output_matches_its_frozen_snapshot(name):
    path = expected_path(name)
    assert path.is_file(), f"missing {path}; see scripts/freeze_examples.py"
    want = normalise(path.read_text(encoding="utf-8"))
    got = render(name)
    if got != want:
        diff = "".join(difflib.unified_diff(
            want.splitlines(keepends=True), got.splitlines(keepends=True),
            fromfile=f"expected_output/{name}.txt", tofile=f"examples/{name}.py now"))
        pytest.fail(
            f"examples/{name}.py printed something different from its snapshot. If the "
            f"change is deliberate, regenerate with scripts/freeze_examples.py and say "
            f"why in the commit.\n{diff}")


def test_no_snapshot_carries_a_local_path():
    for name in EXAMPLES:
        text = expected_path(name).read_text(encoding="utf-8")
        assert "C:\\Users" not in text and "/Users/" not in text, name
