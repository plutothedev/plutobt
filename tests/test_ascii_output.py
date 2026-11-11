"""3.0.1, audit finding 9: nothing the library prints may need more than ASCII. A Windows
console that is redirected to a file or a pipe encodes as cp1252 unless Python's UTF-8 mode
is on, and one warning sign crashed print() there."""
import ast
import os
import subprocess
import sys

import pytest

from tests.conftest import REPO_ROOT


def _non_docstring_strings(tree):
    docs = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef) \
                and node.body and isinstance(node.body[0], ast.Expr) \
                and isinstance(node.body[0].value, ast.Constant):
            docs.add(id(node.body[0].value))
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) \
                and id(node) not in docs:
            yield node


def test_every_string_the_library_and_examples_can_print_is_ascii():
    bad = []
    files = sorted((REPO_ROOT / "src" / "plutobt").rglob("*.py")) \
        + sorted((REPO_ROOT / "examples").glob("*.py"))
    assert len(files) > 20
    for path in files:
        for node in _non_docstring_strings(ast.parse(path.read_text(encoding="utf-8"))):
            if not node.value.isascii():
                bad.append(f"{path.relative_to(REPO_ROOT).as_posix()}:{node.lineno}")
    assert bad == []


def test_every_pinned_output_is_ascii():
    files = sorted((REPO_ROOT / "tests" / "expected_output").glob("*.txt"))
    assert len(files) >= 6
    assert [p.name for p in files if not p.read_text(encoding="utf-8").isascii()] == []


@pytest.mark.parametrize("name", ["run_demo", "causality_demo", "cost_sweep",
                                  "cold_open_lookahead"])
def test_every_example_runs_on_a_redirected_windows_console(name):
    """run_demo and causality_demo crashed here in 3.0.0 (UnicodeEncodeError)."""
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONUTF8", "PYTHONIOENCODING")}
    env.update(PYTHONUTF8="0", PYTHONIOENCODING="cp1252")
    done = subprocess.run([sys.executable, str(REPO_ROOT / "examples" / f"{name}.py")],
                          capture_output=True, encoding="cp1252", check=False, cwd=REPO_ROOT,
                          env=env)
    assert done.returncode == 0, done.stderr[-600:]
    assert done.stdout.isascii() and len(done.stdout) > 100
