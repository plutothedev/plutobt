"""Freeze the printed output of every on-camera example into tests/expected_output/.

tests/test_example_snapshots.py compares each example's full stdout with these files,
so any drift in an on-camera number fails a test.

Run this ONLY after a deliberate, reviewed change to what an example prints, then read
the diff of tests/expected_output/ and commit it with the reason. NEVER run it to make
a failing snapshot test pass without explaining why the output moved: a snapshot
re-frozen to silence a failure has stopped being a measurement, and an unexplained
drift in a printed number is exactly what the test exists to catch.

Run:  PYTHONPATH=src python scripts/freeze_examples.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    sys.path.insert(0, str(ROOT))
    from tests.test_example_snapshots import EXAMPLES, EXPECTED_DIR, expected_path, render

    EXPECTED_DIR.mkdir(parents=True, exist_ok=True)
    for name in EXAMPLES:
        text = render(name)
        expected_path(name).write_text(text, encoding="utf-8", newline="\n")
        print(f"froze examples/{name}.py -> {expected_path(name).relative_to(ROOT).as_posix()}"
              f" ({len(text.splitlines())} lines)")


if __name__ == "__main__":
    main()
