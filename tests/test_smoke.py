"""Smoke test: the package installs, imports, and exposes a version string."""
import importlib
import pathlib
import tomllib

from tests.conftest import REPO_ROOT

import plutobt

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
    assert plutobt.__version__ == "0.1.0"


def test_pyproject_declares_package_and_src_layout():
    root = pathlib.Path(__file__).resolve().parents[1]
    with (root / "pyproject.toml").open("rb") as fh:
        cfg = tomllib.load(fh)
    assert cfg["project"]["name"] == "plutobt"
    assert cfg["project"]["version"] == "0.1.0"
    assert cfg["project"]["requires-python"] == ">=3.11"
    # src layout: setuptools must find packages under src/
    assert cfg["tool"]["setuptools"]["packages"]["find"]["where"] == ["src"]
    # pytest must look in tests/ with importmode=importlib (src layout best practice)
    assert "tests" in cfg["tool"]["pytest"]["ini_options"]["testpaths"]
    assert cfg["tool"]["pytest"]["ini_options"]["importmode"] == "importlib"


def test_all_contract_modules_import():
    for name in CONTRACT_MODULES:
        mod = importlib.import_module(name)
        assert mod is not None


def _parse_lock(text):
    pins = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # format: name==version  (uv pip compile / pip freeze style)
        name, _, version = line.partition("==")
        if version:
            pins[name.lower()] = version.split()[0]
    return pins


def test_requirements_lock_pins_exact_versions():
    lock = (REPO_ROOT / "requirements.lock").read_text(encoding="utf-8")
    pins = _parse_lock(lock)
    for pkg, ver in PINNED.items():
        assert pins.get(pkg) == ver, f"{pkg} must be pinned to =={ver}"
