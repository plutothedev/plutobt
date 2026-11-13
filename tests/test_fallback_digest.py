"""A class with no file (a notebook cell) is hashed from its own methods (ledger._class_fallback).
Until 3.0.3 that was each method's bytecode plus repr(co_consts). The repr of a nested code
object (a lambda, a generator expression, a nested function, a comprehension on 3.11)
carries a memory address, and a set's order follows the hash seed, so the hash differed in
every process: every rerun was a new try. It also left out the names a method uses and its
default values, so calling another helper, or changing a default, was not a new try.

Each case runs in its own process with a chosen hash seed, as a rerun would."""
import os
import pathlib
import subprocess
import sys

import pytest

SRC = str(pathlib.Path(__file__).resolve().parents[1] / "src")

# The cell is run into a registered module with no __file__, as a notebook's __main__ is.
_RUN = '''
import sys, types
from plutobt.ledger import code_sha256
cell = types.ModuleType("plutobt_test_cell")
sys.modules[cell.__name__] = cell
exec(compile(sys.stdin.read(), "<cell>", "exec"), vars(cell))
print(code_sha256(cell.S))
'''

HEAD = "from plutobt.strategy import Strategy\n\n"


def _cell(method_body: str, signature: str = "self, ctx", extra: str = "") -> str:
    return (HEAD + extra + "class S(Strategy):\n"
            f"    def on_bar({signature}):\n" + "".join(
                f"        {line}\n" for line in method_body.splitlines()))


def cell_hash(src: str, seed: int) -> str:
    env = dict(os.environ, PYTHONHASHSEED=str(seed))
    env["PYTHONPATH"] = os.pathsep.join(p for p in (SRC, env.get("PYTHONPATH", "")) if p)
    out = subprocess.run([sys.executable, "-c", _RUN], input=src, capture_output=True,
                         text=True, env=env, check=True)
    return out.stdout.strip()


PLAIN = _cell("if ctx.bar.close > 5:\n    ctx.broker.buy(1)")
SHAPES = {
    "lambda": _cell("pick = lambda bar: bar.close\nif pick(ctx.bar) > 5:\n    ctx.broker.buy(1)"),
    "genexpr": _cell("if sum(x for x in (1, 2)) > 5:\n    ctx.broker.buy(1)"),
    "nested_def": _cell("def pick(bar):\n    return bar.close\nif pick(ctx.bar) > 5:\n"
                        "    ctx.broker.buy(1)"),
    "listcomp": _cell("if len([x for x in (1, 2)]) > 5:\n    ctx.broker.buy(1)"),
    "set_literal": _cell("if ctx.state.get('side', 'a') in {'a', 'b', 'c', 'd', 'e', 'f', 'g',"
                         " 'h'}:\n    ctx.broker.buy(1)"),
}
HELPERS = ("def helper_a(bar):\n    return bar.close\n\n\n"
           "def helper_b(bar):\n    return bar.open\n\n\n")


@pytest.mark.parametrize("shape", sorted(SHAPES))
def test_the_hash_of_a_class_with_no_file_is_the_same_in_every_process(shape):
    src = SHAPES[shape]
    first = cell_hash(src, 1)
    assert first == cell_hash(src, 2) == cell_hash(src, 1)


def test_calling_another_helper_is_new_code():
    calls_a = _cell("if helper_a(ctx.bar) > 5:\n    ctx.broker.buy(1)", extra=HELPERS)
    calls_b = _cell("if helper_b(ctx.bar) > 5:\n    ctx.broker.buy(1)", extra=HELPERS)
    assert cell_hash(calls_a, 1) != cell_hash(calls_b, 1)


def test_a_changed_default_is_new_code():
    five = _cell("if ctx.bar.close > n:\n    ctx.broker.buy(1)", signature="self, ctx, n=5")
    six = _cell("if ctx.bar.close > n:\n    ctx.broker.buy(1)", signature="self, ctx, n=6")
    assert cell_hash(five, 1) != cell_hash(six, 1)


def test_a_changed_keyword_only_default_is_new_code():
    five = _cell("if ctx.bar.close > n:\n    ctx.broker.buy(1)", signature="self, ctx, *, n=5")
    six = _cell("if ctx.bar.close > n:\n    ctx.broker.buy(1)", signature="self, ctx, *, n=6")
    assert cell_hash(five, 1) != cell_hash(six, 1)


def test_the_same_cell_is_the_same_hash():
    assert cell_hash(PLAIN, 1) == cell_hash(PLAIN, 2)


def test_a_different_method_body_is_new_code():
    assert cell_hash(PLAIN, 1) != cell_hash(PLAIN.replace("> 5", "> 6"), 1)
