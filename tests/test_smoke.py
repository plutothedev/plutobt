"""Smoke tests: package installs, imports, tooling is pinned and lint-clean (M1)."""
import importlib
import pathlib
import subprocess
import sys
import tomllib

import plutobt
from tests.conftest import REPO_ROOT

CONTRACT_MODULES = [
    "plutobt.instruments",
    "plutobt.data",
    "plutobt.portfolio",
    "plutobt.broker",
    "plutobt.strategy",
    "plutobt.engine",
    "plutobt.lookahead",
    "plutobt.metrics",
    "plutobt.config",
    "plutobt.plotting",
]

PINNED = {
    "pandas": "2.2.2",
    "numpy": "1.26.4",
    "matplotlib": "3.8.4",
    "pytest": "8.2.0",
    "ruff": "0.4.4",
}


def test_package_imports():
    assert importlib.import_module("plutobt") is plutobt


def test_version_is_pep440_string():
    assert isinstance(plutobt.__version__, str)
    assert plutobt.__version__ == "3.0.3"


def test_pyproject_declares_package_and_src_layout():
    with (REPO_ROOT / "pyproject.toml").open("rb") as fh:
        cfg = tomllib.load(fh)
    assert cfg["project"]["name"] == "plutobt"
    assert cfg["project"]["version"] == "3.0.3"
    # 3.0.1: the pinned numpy, pandas and matplotlib have no 3.13 builds, so the
    # project says so instead of failing in a source build
    assert cfg["project"]["requires-python"] == ">=3.11,<3.13"
    assert cfg["tool"]["setuptools"]["packages"]["find"]["where"] == ["src"]
    assert "tests" in cfg["tool"]["pytest"]["ini_options"]["testpaths"]
    assert "--import-mode=importlib" in cfg["tool"]["pytest"]["ini_options"]["addopts"]


def test_all_contract_modules_import():
    for name in CONTRACT_MODULES:
        assert importlib.import_module(name) is not None


def _parse_lock(text):
    pins = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name, _, version = line.partition("==")
        if version:
            pins[name.lower()] = version.split()[0]
    return pins


def test_requirements_lock_pins_exact_versions():
    lock = (REPO_ROOT / "requirements.lock").read_text(encoding="utf-8")
    pins = _parse_lock(lock)
    for pkg, ver in PINNED.items():
        assert pins.get(pkg) == ver, f"{pkg} must be pinned to =={ver}"


def test_ruff_check_passes():
    proc = subprocess.run(
        [sys.executable, "-m", "ruff", "check", "src", "tests"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_ci_workflow_runs_matrix_on_pinned_stack():
    wf = (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    # matrix must cover the supported interpreters (§5.10 CI matrix)
    assert "3.11" in wf
    assert "3.12" in wf
    # CI must install the FROZEN lockfile, not loose deps (§5.10 reproducibility)
    assert "requirements.lock" in wf
    # CI must run lint and the test suite
    assert "ruff check" in wf
    assert "pytest" in wf


def test_pathlib_root_exists():
    assert isinstance(REPO_ROOT, pathlib.Path)
    assert REPO_ROOT.is_dir()


def test_public_api_names_resolve():
    """import plutobt; key public names must resolve without AttributeError."""
    for name in (
        "Engine", "BacktestResult", "BacktestConfig", "load_config",
        "Strategy", "Context", "get_instrument", "REGISTRY",
        "load_bars", "compute_metrics", "format_report",
    ):
        assert hasattr(plutobt, name), f"plutobt.{name} not found in __init__"
    # Spot-check that they are the real objects, not None.
    assert plutobt.Engine is not None
    assert plutobt.BacktestResult is not None


def test_roll_api_is_exported():
    for name in ("load_csv", "load_csv_contracts", "ContractSet", "build_continuous",
                 "load_continuous", "save_continuous", "slice_continuous", "RollMeta"):
        assert name in plutobt.__all__, name
        assert getattr(plutobt, name) is not None


def test_overfit_api_is_exported():
    for name in ("run_study", "StudyResult", "format_study", "deflated_sharpe",
                 "expected_max_sharpe", "Ledger"):
        assert name in plutobt.__all__, name
        assert getattr(plutobt, name) is not None


def test_cli_import_does_not_shadow_build_continuous():
    """Review #6: a CLI module named build_continuous replaced the public function on the
    package once anything imported it."""
    import types

    importlib.import_module("plutobt.continuous")
    from plutobt import build_continuous

    assert isinstance(build_continuous, types.FunctionType)
    assert not hasattr(plutobt, "build_continuous") or callable(plutobt.build_continuous)
    assert not isinstance(plutobt.build_continuous, types.ModuleType)


def test_readme_states_the_python_versions_the_pins_support():
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    assert "Requires Python 3.11 or 3.12" in readme
    assert "Python 3.11+" not in readme
    lock = (REPO_ROOT / "requirements.lock").read_text(encoding="utf-8")
    assert "# Python: 3.11 or 3.12" in lock


def test_the_readme_roll_example_runs_as_written(tmp_path, monkeypatch, capsys):
    """3.0.1, audit finding 11: the block used an undefined `strategy` and stopped before
    compute_metrics, so a reader could not get a report from it."""
    from plutobt.continuous import main

    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    start = readme.index("```python\nfrom plutobt import (BacktestConfig, Engine, compute_metrics")
    block = readme[start + len("```python\n"):readme.index("```", start + 10)]
    assert 'load_continuous("nq_cont.csv")' in block and "format_report" in block
    monkeypatch.chdir(tmp_path)
    assert main([str(REPO_ROOT / "data" / "sample" / "nq_contracts"), "--preset", "generic",
                 "--root", "NQ", "--ts-is", "open", "-o", "nq_cont.csv"]) == 0
    capsys.readouterr()
    exec(compile(block, "README.md", "exec"), {})        # noqa: S102
    out = capsys.readouterr().out
    assert "Sharpe" in out and "IN-SAMPLE" in out and "engine version = 3.0.3" in out


def test_the_repo_ignores_what_the_readme_commands_write():
    ignore = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert "/nq_cont.csv" in ignore and "/nq_cont.roll.json" in ignore


def test_onboarding_matches_3_0():
    text = (REPO_ROOT / "ONBOARDING.md").read_text(encoding="utf-8")
    assert "No out-of-sample split in v1" not in text
    assert "12 modules" not in text
    assert "3.0" in text and "run_study" in text
