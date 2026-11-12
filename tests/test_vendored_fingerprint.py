"""3.0.2: the code fingerprint when plutobt is vendored inside another package.

A project that ships its own copy (``host/_vendor/plutobt``) and runs from a source checkout
imports plutobt as ``host._vendor.plutobt``. Importing that name runs ``host/__init__.py``
and ``host/_vendor/__init__.py``, and 3.0.1 counted both as the user's code: a version bump
of the host made every repeated setting a new try.

The rule: a package above plutobt is part of the code when a file names it (``import host``,
``from host import NAME``, ``import host.tools`` with no ``as``), when a loaded module holds
it, or when the class has no file. It is not part of the code merely for being on the way to
something else: to plutobt, to another host module, or to a strategy that lives inside the
host.

What that gives up against 3.0.1: code in those packages' ``__init__.py`` that the strategy
depends on with no import statement naming the package. That is ``__import__("host")``
inside a function (which a plain install misses for any module), and a side effect of the
``__init__`` itself, such as it patching a value in another module. A plain install does see
the second kind in the user's own packages, so a research project that vendors plutobt
inside its own top-level package loses that for that package's ``__init__.py`` (and for
``_vendor/__init__.py``), and for nothing below them.

Each case builds that layout from this package's own source and asks a fresh Python, so the
copy under test is the code in this checkout whatever name it is imported by here."""
import json
import os
import re
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from plutobt import ledger

PKG = ledger.__name__.rpartition(".")[0]         # this package's import name, as loaded here
SRC = Path(ledger.__file__).resolve().parent
VENDORED = "host._vendor.plutobt"

HOST_INIT = "src/host/__init__.py"
VENDOR_INIT = "src/host/_vendor/__init__.py"
STRATS_INIT = "src/host/strategies/__init__.py"
MINE = "src/host/strategies/mine.py"
CUTS = "src/host/strategies/cuts.py"

# Prints the files hashed for a strategy, then for each path named on the command line
# whether adding a line of code to that file changes the hash (the file is put back).
ASKER = '''
import json, sys
from pathlib import Path

from host._vendor.plutobt.ledger import code_files, code_sha256


def report(S):
    root = Path(sys.argv[1]).resolve()
    files = []
    for f in code_files(S):
        p = Path(f).resolve()
        files.append(p.relative_to(root).as_posix() if root in p.parents else p.as_posix())
    before = code_sha256(S)
    moved = {}
    for rel in json.loads(sys.argv[2]):
        path = root / rel
        old = path.read_bytes()
        path.write_bytes(old + b"\\nEDITED = 1\\n")
        moved[rel] = code_sha256(S) != before
        path.write_bytes(old)
    print(json.dumps({"files": sorted(files), "moved": moved}))
'''
BY_NAME = "import importlib, sys, asker; asker.report(importlib.import_module(sys.argv[3]).S)"

BODY = '''


class S({base}):
    def on_bar(self, ctx):
        if {test} and ctx.position.qty == 0:
            ctx.broker.buy(contracts=1)
'''
AS_MAIN = '''

if __name__ == "__main__":
    import asker
    asker.report(S)
'''


def strat(imports: str, test: str = "True", base: str = "Strategy") -> str:
    return textwrap.dedent(imports).strip() + "\n" + BODY.format(base=base, test=test)


PLUTOBT = "from host._vendor.plutobt.strategy import Strategy"
INSIDE = "from .._vendor.plutobt.strategy import Strategy"
HELPER = "def ok(ctx):\n    return ctx.bar.close > 0\n"


def write(path: Path, source: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")


@pytest.fixture(scope="module")
def checkout(tmp_path_factory):
    """A source checkout of a package called ``host`` with this plutobt vendored inside."""
    root = tmp_path_factory.mktemp("vendored")
    pkg = root / "src" / "host" / "_vendor" / "plutobt"
    shutil.copytree(SRC, pkg, ignore=shutil.ignore_patterns("__pycache__"))
    name = re.compile(rf"\b(from|import) {re.escape(PKG)}\b")
    for path in pkg.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        path.write_text(name.sub(rf"\1 {VENDORED}", text), encoding="utf-8")
    write(root / "tools" / "asker.py", ASKER)
    return root


@pytest.fixture
def host(checkout):
    """The checkout with fresh host files and an empty ``work`` folder for the user's own.
    Returns ask(files, module, edits, main): write the files, load the strategy `S` from
    the module (or run the module with ``python -m`` when ``main``), and report."""
    for folder in ("work", "src/host/strategies"):
        shutil.rmtree(checkout / folder, ignore_errors=True)
    for path in (checkout / "src" / "host").glob("*.py"):
        path.unlink()
    write(checkout / HOST_INIT, '__version__ = "1.0.0"\nTHRESH = 0\n')
    write(checkout / VENDOR_INIT, 'NAMES = ("plutobt",)\nLEVEL = 0\n')
    (checkout / "work").mkdir()
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(
        str(checkout / folder) for folder in ("src", "work", "tools")))

    def ask(files: dict, module: str = "strat", edits=(), main: bool = False) -> dict:
        for rel, source in files.items():
            write(checkout / rel, source)
        how = ["-m", module] if main else ["-c", BY_NAME]
        out = subprocess.run([sys.executable, *how, str(checkout), json.dumps(list(edits)),
                              module], capture_output=True, text=True, cwd=checkout,
                             env=env, check=False)
        assert out.returncode == 0, out.stderr
        return json.loads(out.stdout.strip().splitlines()[-1])

    return ask


# Ways of importing plutobt and nothing else from the host: the two files above it are not
# the user's code, and an edit to either is not a new try.
ONLY_PLUTOBT = {
    "from the module": strat(PLUTOBT),
    "from the package": strat("from host._vendor.plutobt import Strategy"),
    "the package by name": strat("from host._vendor import plutobt",
                                 base="plutobt.Strategy"),
    "a module under another name": strat("import host._vendor.plutobt.strategy as st",
                                         base="st.Strategy"),
    "a star import": strat("from host._vendor.plutobt.strategy import *"),
    "inside a function": strat(PLUTOBT + "\n\n\ndef late():\n"
                               "    from host._vendor.plutobt.config import BacktestConfig\n"
                               "    return BacktestConfig\n"),
}


@pytest.mark.parametrize("form", ONLY_PLUTOBT)
def test_importing_plutobt_does_not_make_the_packages_above_it_user_code(host, form):
    got = host({"work/strat.py": ONLY_PLUTOBT[form]}, edits=[HOST_INIT, VENDOR_INIT])
    assert got["files"] == ["work/strat.py"], got["files"]
    assert got["moved"] == {HOST_INIT: False, VENDOR_INIT: False}, got["moved"]


def test_the_users_own_files_still_count_when_plutobt_is_vendored(host):
    got = host({"work/helper.py": HELPER,
                "work/strat.py": strat(PLUTOBT + "\nimport helper", "helper.ok(ctx)")},
               edits=["work/helper.py", "work/strat.py", HOST_INIT, VENDOR_INIT])
    assert got["files"] == ["work/helper.py", "work/strat.py"], got["files"]
    assert got["moved"] == {"work/helper.py": True, "work/strat.py": True,
                            HOST_INIT: False, VENDOR_INIT: False}, got["moved"]


def test_a_library_strategy_hashes_its_own_file_only_when_vendored(host):
    got = host({"work/strat.py":
                "from host._vendor.plutobt.strategies.ema_cross import EMACross as S\n"},
               edits=[HOST_INIT, VENDOR_INIT])
    assert got["files"] == ["src/host/_vendor/plutobt/strategies/ema_cross.py"], got["files"]
    assert got["moved"] == {HOST_INIT: False, VENDOR_INIT: False}, got["moved"]


def test_a_strategy_inside_the_host_counts_its_own_package_not_the_ones_above_plutobt(host):
    """The host's own built-in strategies: a version bump of the host is not a new try for
    them either, while their own package and the helper next to them are their code."""
    got = host({STRATS_INIT: "", CUTS: HELPER,
                MINE: strat(INSIDE + "\nfrom .cuts import ok", "ok(ctx)")},
               "host.strategies.mine", edits=[CUTS, STRATS_INIT, HOST_INIT, VENDOR_INIT])
    assert got["files"] == [STRATS_INIT, CUTS, MINE], got["files"]
    assert got["moved"] == {CUTS: True, STRATS_INIT: True,
                            HOST_INIT: False, VENDOR_INIT: False}, got["moved"]


def test_a_host_module_imported_under_another_name_does_not_bring_the_host_package(host):
    got = host({"src/host/tools.py": HELPER,
                "work/strat.py": strat(PLUTOBT + "\nimport host.tools as tools",
                                       "tools.ok(ctx)")},
               edits=["src/host/tools.py", HOST_INIT])
    assert got["moved"] == {"src/host/tools.py": True, HOST_INIT: False}, got["moved"]


# Ways a strategy reaches code by choice. Each names files whose edit must be a new try:
# missing one of these would let a real edit pass as a repeat.
# name: (files, module, run it with python -m, the files whose edit must change the hash)
BY_CHOICE = {
    "a name from the host package": (
        {"work/strat.py": strat(PLUTOBT + "\nfrom host import THRESH", "THRESH < 1")},
        "strat", False, [HOST_INIT]),
    "the host package itself": (
        {"work/strat.py": strat(PLUTOBT + "\nimport host", "host.THRESH < 1")},
        "strat", False, [HOST_INIT]),
    "a name the host package re-exports": (
        {HOST_INIT: "from host.params import THRESH\n", "src/host/params.py": "THRESH = 0\n",
         "work/strat.py": strat(PLUTOBT + "\nfrom host import THRESH", "THRESH < 1")},
        "strat", False, [HOST_INIT, "src/host/params.py"]),
    "a name from the vendor package": (
        {"work/strat.py": strat(PLUTOBT + "\nfrom host._vendor import LEVEL", "LEVEL < 1")},
        "strat", False, [VENDOR_INIT]),
    "a host module reached through the package": (
        {HOST_INIT: "from host import tools\n", "src/host/tools.py": HELPER,
         "work/strat.py": strat(PLUTOBT + "\nimport host", "host.tools.ok(ctx)")},
        "strat", False, [HOST_INIT, "src/host/tools.py"]),
    "a host module taken from the package": (
        {"src/host/tools.py": HELPER,
         "work/strat.py": strat(PLUTOBT + "\nfrom host import tools", "tools.ok(ctx)")},
        "strat", False, [HOST_INIT, "src/host/tools.py"]),
    "plutobt by its full dotted name, which binds the host": (
        {"work/strat.py": strat("import host._vendor.plutobt.strategy",
                                "host.THRESH < 1 and host._vendor.LEVEL < 1",
                                base="host._vendor.plutobt.strategy.Strategy")},
        "strat", False, [HOST_INIT, VENDOR_INIT]),
    "a strategy inside the host that reads its package": (
        {STRATS_INIT: "",
         MINE: strat(INSIDE + "\nfrom .. import THRESH", "THRESH < 1")},
        "host.strategies.mine", False, [MINE, HOST_INIT]),
    "a strategy inside the host run with python -m": (
        # only numbers are taken, so nothing the loaded module holds points at the files
        {STRATS_INIT: "", "src/host/strategies/params.py": "LIMIT = 1\n",
         MINE: strat(INSIDE + "\nfrom .. import THRESH\nfrom .params import LIMIT",
                     "THRESH < LIMIT") + AS_MAIN},
        "host.strategies.mine", True,
        [MINE, "src/host/strategies/params.py", STRATS_INIT, HOST_INIT]),
    "a host file reached through a host module the strategy imports": (
        {"src/host/limits.py": "CUT = 0\n",
         "src/host/tools.py": "from host.limits import CUT\n\n\ndef ok(ctx):\n"
                              "    return ctx.bar.close > CUT\n",
         "work/strat.py": strat(PLUTOBT + "\nimport host.tools as tools", "tools.ok(ctx)")},
        "strat", False, ["src/host/tools.py", "src/host/limits.py"]),
    "the user's own package run with python -m": (
        # nothing to do with the host: `python -m pkg.mod` names the module "__main__",
        # and its relative imports are followed from the package Python recorded for it
        {"work/pkg/__init__.py": "", "work/pkg/params.py": "LIMIT = 1\n",
         "work/pkg/mine.py": strat(PLUTOBT + "\nfrom .params import LIMIT",
                                   "ctx.bar.close > LIMIT") + AS_MAIN},
        "pkg.mine", True, ["work/pkg/mine.py", "work/pkg/params.py", "work/pkg/__init__.py"]),
    "a user module under plutobt's own name, in another folder": (
        {"work/plug/cuts.py": HELPER,
         "work/plug/mine.py": strat(PLUTOBT + "\nfrom .cuts import ok", "ok(ctx)"),
         "work/strat.py": textwrap.dedent('''\
            from pathlib import Path

            import host._vendor.plutobt.strategies as library

            library.__path__.append(str(Path(__file__).resolve().parent / "plug"))
            from host._vendor.plutobt.strategies.mine import S
            ''')},
        "strat", False, ["work/plug/mine.py", "work/plug/cuts.py"]),
    "a class with no file that reads the host": (
        {"work/strat.py": textwrap.dedent('''\
            from host import THRESH
            from host._vendor.plutobt.strategy import Strategy

            cell = {"Strategy": Strategy, "THRESH": THRESH, "__name__": "__main__"}
            exec("class S(Strategy):\\n"
                 "    def on_bar(self, ctx):\\n"
                 "        if THRESH < 1:\\n"
                 "            ctx.broker.buy(contracts=1)\\n", cell)
            S = cell["S"]
            ''')},
        "strat", False, [HOST_INIT]),
}


@pytest.mark.parametrize("way", BY_CHOICE)
def test_code_the_strategy_reaches_by_choice_is_still_its_code(host, way):
    files, module, main, must_move = BY_CHOICE[way]
    got = host(files, module, edits=must_move, main=main)
    assert got["moved"] == {rel: True for rel in must_move}, got
