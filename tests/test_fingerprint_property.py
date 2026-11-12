"""The code fingerprint as a property, over random project layouts (3.0.1).

Three review rounds each found one more way a strategy's code can reach it that the
fingerprint missed, so the exit is a property, not a reviewer's ok. For a random project
(plain modules, packages, namespace packages; ``import x``, ``from x import NAME``,
``from pkg import module``; at the top of a file or inside a function), the fingerprint
must:

  - read exactly the strategy's file and every file its imports can reach;
  - import nothing while doing so;
  - be the same before and after the function-level imports have run;
  - change when any reachable file is edited, and not when an unreachable one is.
"""
import importlib
import sys

import numpy as np
import pytest

from plutobt.ledger import code_files, code_sha256

STRATEGY = '''
from plutobt.strategy import Strategy


class S(Strategy):
    def on_bar(self, ctx):
        pass
'''


def _plan(rng, tag):
    """Random modules and import edges. Module k may import only later modules, so the
    graph has no import-time cycle; index -1 is the strategy file."""
    count = int(rng.integers(3, 7))
    mods = []
    for i in range(count):
        kind = str(rng.choice(["plain", "package", "namespace"]))
        if kind == "plain":
            mods.append({"name": f"m{i}_{tag}", "parent": None, "value_file": f"m{i}_{tag}.py",
                         "files": [f"m{i}_{tag}.py"]})
        elif kind == "package":
            mods.append({"name": f"pk{i}_{tag}.inner", "parent": f"pk{i}_{tag}",
                         "value_file": f"pk{i}_{tag}/inner.py",
                         "files": [f"pk{i}_{tag}/__init__.py", f"pk{i}_{tag}/inner.py"]})
        else:
            mods.append({"name": f"ns{i}_{tag}.inner", "parent": f"ns{i}_{tag}",
                         "value_file": f"ns{i}_{tag}/inner.py",
                         "files": [f"ns{i}_{tag}/inner.py"]})
    edges = {}
    for src in range(-1, count):
        later = list(range(src + 1, count))
        if not later:
            continue
        most = min(3, len(later))
        take = int(rng.integers(1 if src == -1 else 0, most + 1))
        for dst in rng.choice(later, size=take, replace=False):
            styles = ["import", "from_value"] + (["from_parent"] if mods[dst]["parent"] else [])
            edges[(src, int(dst))] = (str(rng.choice(styles)), bool(rng.integers(0, 2)))
    return mods, edges


def _source(index, mods, edges, values):
    lines = [] if index == -1 else [f"VALUE = {values[index]}"]
    for (src, dst), (style, in_function) in sorted(edges.items()):
        if src != index:
            continue
        target = mods[dst]
        if style == "import":
            line = f"import {target['name']}"
        elif style == "from_value":
            line = f"from {target['name']} import VALUE as V{dst}"
        else:
            line = f"from {target['parent']} import inner as I{dst}"
        if in_function:
            lines += [f"def use_{dst}():", f"    {line}", "    return 0"]
        else:
            lines.append(line)
    return "\n".join(lines) + "\n" + (STRATEGY if index == -1 else "")


def _reachable(count, edges, only_top_level=False):
    seen, stack = set(), [-1]
    while stack:
        src = stack.pop()
        for (a, b), (_, in_function) in edges.items():
            if a == src and b not in seen and not (only_top_level and in_function):
                seen.add(b)
                stack.append(b)
    return seen


def _write_and_load(root, tag, mods, edges, values):
    for i, mod in enumerate(mods):
        for rel in mod["files"]:
            path = root / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(_source(i, mods, edges, values) if rel == mod["value_file"] else "",
                            encoding="utf-8")
    (root / f"strat_{tag}.py").write_text(_source(-1, mods, edges, values), encoding="utf-8")
    for name in [m for m in sys.modules if tag in m]:
        del sys.modules[name]
    importlib.invalidate_caches()
    return importlib.import_module(f"strat_{tag}").S


def _run_function_imports(tag):
    """Call every use_*() until nothing new loads: every function-level import has run."""
    done = set()
    while True:
        todo = [(n, m) for n, m in list(sys.modules.items()) if tag in n and n not in done]
        if not todo:
            return
        for name, module in todo:
            done.add(name)
            for attr in [a for a in vars(module) if a.startswith("use_")]:
                getattr(module, attr)()


def _names(files, root):
    return sorted(str(f).replace("\\", "/").split(root.name + "/", 1)[1] for f in files)


@pytest.mark.parametrize("seed", range(40))
def test_the_fingerprint_reads_every_reachable_file_and_nothing_else(seed, tmp_path, monkeypatch):
    rng = np.random.default_rng(seed)
    tag = f"fp{seed}x{abs(hash(tmp_path.name)) % 10 ** 6}"
    monkeypatch.syspath_prepend(str(tmp_path))
    mods, edges = _plan(rng, tag)
    values = [10 + i for i in range(len(mods))]
    try:
        cls = _write_and_load(tmp_path, tag, mods, edges, values)
        reach = _reachable(len(mods), edges)
        expected = sorted([f"strat_{tag}.py"] + [f for i in reach for f in mods[i]["files"]])

        loaded_before = {n for n in sys.modules if tag in n}
        first = code_sha256(cls)
        assert _names(code_files(cls), tmp_path) == expected
        assert {n for n in sys.modules if tag in n} == loaded_before      # imported nothing
        assert code_sha256(cls) == first

        _run_function_imports(tag)                    # now every function-level import ran
        assert code_sha256(cls) == first
        assert _names(code_files(cls), tmp_path) == expected

        for i in range(len(mods)):                    # edit each module's value in turn
            edited = list(values)
            edited[i] += 1000
            moved = code_sha256(_write_and_load(tmp_path, tag, mods, edges, edited)) != first
            assert moved == (i in reach), (i, mods[i]["name"], sorted(reach))
    finally:
        for name in [m for m in sys.modules if tag in m]:
            del sys.modules[name]
