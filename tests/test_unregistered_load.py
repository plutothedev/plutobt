"""A strategy whose module is not in sys.modules (a host that loads a file with
importlib.util.spec_from_file_location and never registers it, or runpy.run_path) hashed
no file until 3.0.3: its code was the methods' bytecode alone, so an edit outside them
(a module constant, a helper, an import) was not a new try. The class's file is now found
from the functions written in the class body (their code names the file), and hashed with
the user modules it imports, as for any other file."""
import importlib.util
import runpy
import sys

import pytest

from plutobt.ledger import code_files, code_sha256

MINE = '''from plutobt.strategy import Strategy

THRESHOLD = 5


def above(bar):
    return bar.close > THRESHOLD


class Mine(Strategy):
    def on_bar(self, ctx):
        if above(ctx.bar):
            ctx.broker.buy(1)
'''


def load(path):
    """The file loaded the way a host that does not register it loads it."""
    spec = importlib.util.spec_from_file_location("host_user_strategy", str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def tag(tmp_path, monkeypatch):
    """A name no other test uses, for helper modules on sys.path, removed afterwards."""
    name = "unreg_" + tmp_path.name.replace("-", "_")
    monkeypatch.syspath_prepend(str(tmp_path))
    yield name
    for key in [k for k in sys.modules if k.startswith(name)]:
        del sys.modules[key]


def test_an_unregistered_class_hashes_its_file(tmp_path):
    mine = tmp_path / "mine.py"
    mine.write_text(MINE, encoding="utf-8")
    assert code_files(load(mine).Mine) == [str(mine.resolve())]


def test_an_edit_outside_the_methods_is_new_code(tmp_path):
    mine = tmp_path / "mine.py"
    mine.write_text(MINE, encoding="utf-8")
    before = code_sha256(load(mine).Mine)
    mine.write_text(MINE.replace("THRESHOLD = 5", "THRESHOLD = 6"), encoding="utf-8")
    assert code_sha256(load(mine).Mine) != before


def test_a_user_module_the_unregistered_file_imports_is_code(tmp_path, tag):
    helper = tmp_path / f"{tag}_helper.py"
    helper.write_text("LIMIT = 3\n", encoding="utf-8")
    mine = tmp_path / "mine.py"
    mine.write_text(f"import {tag}_helper\n" + MINE, encoding="utf-8")
    cls = load(mine).Mine
    assert set(code_files(cls)) == {str(mine.resolve()), str(helper.resolve())}
    before = code_sha256(cls)
    helper.write_text("LIMIT = 4\n", encoding="utf-8")
    assert code_sha256(cls) != before


def test_a_class_run_with_runpy_hashes_its_file(tmp_path):
    mine = tmp_path / "mine.py"
    mine.write_text(MINE, encoding="utf-8")
    namespace = runpy.run_path(str(mine))
    assert code_files(namespace["Mine"]) == [str(mine.resolve())]


def test_a_class_made_inside_a_function_hashes_its_file(tmp_path):
    factory = tmp_path / "factory.py"
    factory.write_text('''from plutobt.strategy import Strategy


def make():
    class Mine(Strategy):
        def on_bar(self, ctx):
            pass
    return Mine
''', encoding="utf-8")
    assert code_files(load(factory).make()) == [str(factory.resolve())]


def test_a_class_whose_only_function_is_a_property_hashes_its_file(tmp_path):
    mine = tmp_path / "mine.py"
    mine.write_text('''from plutobt.strategy import Strategy


class Mine(Strategy):
    @property
    def size(self):
        return 1
''', encoding="utf-8")
    assert code_files(load(mine).Mine) == [str(mine.resolve())]


def test_a_method_written_in_another_file_does_not_make_that_its_file(tmp_path, tag):
    # The class body takes a function written elsewhere: its code names the other file,
    # but it is not written in the class, so the class has no file of its own.
    other = tmp_path / f"{tag}_other.py"
    other.write_text("def on_bar(self, ctx):\n    pass\n", encoding="utf-8")
    child = tmp_path / "child.py"
    child.write_text(f'''import {tag}_other
from plutobt.strategy import Strategy


class Child(Strategy):
    on_bar = {tag}_other.on_bar
''', encoding="utf-8")
    assert str(other.resolve()) not in code_files(load(child).Child)


def test_a_class_with_only_values_keeps_the_fallback(tmp_path):
    values = tmp_path / "values.py"
    values.write_text("from plutobt.strategy import Strategy\n\n\nclass V(Strategy):\n"
                      "    x = 1\n", encoding="utf-8")
    assert code_files(load(values).V) == []


def test_a_method_wrapped_by_a_decorator_from_another_file_is_not_that_files(tmp_path, tag):
    # functools.wraps copies the method's __qualname__ onto the wrapper, whose code is the
    # decorator's file: that file is not the class's (review I2). The method it wraps is,
    # and the decorator's file follows only as a module the class's file imports.
    deco = tmp_path / f"{tag}_deco.py"
    deco.write_text('''import functools


def logged(f):
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        return f(*args, **kwargs)
    return wrapper
''', encoding="utf-8")
    mine = tmp_path / "mine.py"
    mine.write_text(f'''from plutobt.strategy import Strategy
from {tag}_deco import logged


class Mine(Strategy):
    @logged
    def on_bar(self, ctx):
        pass
''', encoding="utf-8")
    assert code_files(load(mine).Mine) == [str(mine.resolve()), str(deco.resolve())]


WRAPPED_HERE = '''import functools

from plutobt.strategy import Strategy

THRESHOLD = 5


def logged(f):
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        return f(*args, **kwargs)
    return wrapper


class Mine(Strategy):
    @logged
    def on_bar(self, ctx):
        if ctx.bar.close > THRESHOLD:
            ctx.broker.buy(1)
'''


def test_methods_wrapped_by_a_decorator_in_the_same_file_still_hash_the_file(tmp_path):
    # The wrapper's own code is the decorator's; the method it wraps (__wrapped__) is the
    # class's, written in this file (the fix-pass re-review: going by the wrapper alone
    # found no file, and no edit was a new try).
    mine = tmp_path / "mine.py"
    mine.write_text(WRAPPED_HERE, encoding="utf-8")
    assert code_files(load(mine).Mine) == [str(mine.resolve())]
    before = code_sha256(load(mine).Mine)
    mine.write_text(WRAPPED_HERE.replace("THRESHOLD = 5", "THRESHOLD = 6"), encoding="utf-8")
    assert code_sha256(load(mine).Mine) != before


def test_methods_wrapped_by_a_decorator_from_another_file_hash_their_own_file(tmp_path, tag):
    deco = tmp_path / f"{tag}_deco2.py"
    deco.write_text("import functools\n\n\ndef logged(f):\n    @functools.wraps(f)\n"
                    "    def wrapper(*args, **kwargs):\n        return f(*args, **kwargs)\n"
                    "    return wrapper\n", encoding="utf-8")
    mine = tmp_path / "mine.py"
    mine.write_text(WRAPPED_HERE.replace("def logged(f):", f"from {tag}_deco2 import logged"
                                         "\n\n\ndef _unused(f):"), encoding="utf-8")
    files = code_files(load(mine).Mine)
    assert files[0] == str(mine.resolve())
