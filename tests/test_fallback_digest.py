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


# ---- the whole-branch review's findings (3.0.3)

UNSTABLE_DEFAULTS = {
    # a function default: its repr carries its address
    "function": ("import numpy as np\n\n", "self, ctx, agg=np.mean"),
    # a sentinel object: its repr carries its address
    "sentinel": ("_MISSING = object()\n\n", "self, ctx, flag=_MISSING"),
    # a dict holding a set of strings: the set's order follows the hash seed
    "set_in_dict": ("", "self, ctx, opts={'side': {'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'}}"),
}


@pytest.mark.parametrize("kind", sorted(UNSTABLE_DEFAULTS))
def test_a_default_whose_repr_moves_still_hashes_the_same_in_every_process(kind):
    # The defaults were added to the fallback in 3.0.3; hashed by repr, these differed in
    # every process (review I1), where 3.0.2 left defaults out and was stable.
    extra, signature = UNSTABLE_DEFAULTS[kind]
    src = _cell("pass", signature=signature, extra=extra)
    first = cell_hash(src, 1)
    assert first == cell_hash(src, 2) == cell_hash(src, 1)


def test_a_function_default_changed_for_another_is_new_code():
    mean = _cell("pass", signature="self, ctx, agg=np.mean", extra="import numpy as np\n\n")
    median = _cell("pass", signature="self, ctx, agg=np.median", extra="import numpy as np\n\n")
    assert cell_hash(mean, 1) != cell_hash(median, 1)


# ---- the fix-pass re-review's findings (3.0.3): a default edit must count

DEFAULT_EDITS = {
    "enum": ("import enum\n\n\nclass Side(enum.Enum):\n    LONG = 1\n    SHORT = 2\n\n\n",
             "side=Side.LONG", "side=Side.SHORT"),
    "time": ("import datetime as dt\n\n", "open_=dt.time(9, 30)", "open_=dt.time(9, 31)"),
    "np_scalar": ("import numpy as np\n\n", "k=np.float64(1.5)", "k=np.float64(2.5)"),
    "np_array": ("import numpy as np\n\n", "w=np.array([1, 2])", "w=np.array([1, 3])"),
    "partial": ("import functools\n\n", "f=functools.partial(max, 1)",
                "f=functools.partial(max, 2)"),
    "decimal": ("import decimal\n\n", "fee=decimal.Decimal('0.5')", "fee=decimal.Decimal('0.6')"),
    "lambda": ("", "key=lambda x: x", "key=lambda x: -x"),
    "bytearray": ("", "buf=bytearray(b'a')", "buf=bytearray(b'b')"),
}


@pytest.mark.parametrize("kind", sorted(DEFAULT_EDITS))
def test_a_default_edit_is_new_code_and_the_hash_is_stable(kind):
    extra, before, after = DEFAULT_EDITS[kind]
    one = _cell("pass", signature=f"self, ctx, {before}", extra=extra)
    two = _cell("pass", signature=f"self, ctx, {after}", extra=extra)
    assert cell_hash(one, 1) == cell_hash(one, 2)
    assert cell_hash(one, 1) != cell_hash(two, 1)


def test_a_decorated_method_body_edit_in_a_cell_is_new_code():
    deco = ("import functools\n\n\ndef logged(f):\n    @functools.wraps(f)\n"
            "    def wrapper(*args, **kwargs):\n        return f(*args, **kwargs)\n"
            "    return wrapper\n\n\n")
    src = (HEAD + deco + "class S(Strategy):\n    @logged\n    def on_bar(self, ctx):\n"
           "        if ctx.bar.close > 5:\n            ctx.broker.buy(1)\n")
    assert cell_hash(src, 1) != cell_hash(src.replace("> 5", "> 6"), 1)


def test_a_method_that_takes_wraps_from_its_base_has_its_body_in_the_fallback():
    src = (HEAD + "import functools\n\n\nclass S(Strategy):\n"
           "    @functools.wraps(Strategy.on_bar)\n    def on_bar(self, ctx):\n"
           "        if ctx.bar.close > 5:\n            ctx.broker.buy(1)\n")
    assert cell_hash(src, 1) != cell_hash(src.replace("> 5", "> 6"), 1)


def test_an_object_array_default_hashes_the_same_in_every_process():
    # An array of Python objects: its bytes are pointers (fix pass 2's re-review).
    src = _cell("pass", signature="self, ctx, w=np.array(['a', 1.5, None], dtype=object)",
                extra="import numpy as np\n\n")
    assert cell_hash(src, 1) == cell_hash(src, 2) == cell_hash(src, 1)


def test_a_default_whose_repr_raises_does_not_break_the_hash():
    extra = ("class Bad:\n    def __repr__(self):\n        raise RuntimeError('no repr')\n\n\n")
    src = _cell("pass", signature="self, ctx, b=Bad()", extra=extra)
    assert len(cell_hash(src, 1)) == 64
