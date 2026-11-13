"""The class-attribute matrix for the look-ahead check (3.0.3): every attribute kind crossed
with every operation that applies, as module-level strategies (so their methods pickle by
name). tests/test_class_attribute_matrix.py runs it. Built as the exit gate of the 3.0.3
review loop, where three rounds of fixes each broke a case the round before had fixed; the
same table, run against 3.0.2, found 0 differences other than the three gaps 3.0.3 closes,
and it fails on both earlier 3.0.3 drafts.
Operations:
- read:   precompute only reads the attribute. Benign: must run wherever 3.0.2 runs it.
- same:   precompute sets the attribute to the object it holds. Benign, as read.
- mutate: precompute changes the attribute's own state in place with a value of the whole
          frame (an append, a key, a fit through a bound method). A leak: 3.0.3 must refuse
          wherever 3.0.2 refused, and should refuse it everywhere.
- swap:   precompute replaces the attribute with a new object under the same name carrying
          a value of the whole frame. A leak: 3.0.3 must refuse it."""
import collections
import dataclasses
import datetime as dt
import decimal
import enum
import fractions
import functools
import re
import threading
import types
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from plutobt.strategy import Strategy


class Side(enum.Enum):
    LONG = 1
    SHORT = 2


@dataclasses.dataclass
class Params:
    fast: int = 5
    tag: str = "x"


class Model:
    def __init__(self):
        self.mean = None

    def fit(self, value):
        self.mean = value


def helper(x):
    return x


Pair = collections.namedtuple("Pair", "a b")


def _kinds():
    """name -> (factory of the attribute value, mutate op or None)."""
    def append(v, x):
        v.append(x)

    def setkey(v, x):
        v["k"] = x

    return {
        "int": (lambda: 5, None),
        "float": (lambda: 1.5, None),
        "str": (lambda: "abc", None),
        "bytes": (lambda: b"abc", None),
        "tuple": (lambda: (1, 2.0, "x"), None),
        "frozenset": (lambda: frozenset({"a", "b"}), None),
        "list": (lambda: [1.0, 2.0], append),
        "dict": (lambda: {"a": 1.0}, setkey),
        "set": (lambda: {1.0}, lambda v, x: v.add(x)),
        "ordereddict": (lambda: collections.OrderedDict(a=1.0), setkey),
        "defaultdict": (lambda: collections.defaultdict(float, a=1.0), setkey),
        "deque": (lambda: collections.deque([1.0]), append),
        "np_array": (lambda: np.zeros(3), lambda v, x: v.__setitem__(0, x)),
        "series": (lambda: pd.Series([0.0, 0.0]), lambda v, x: v.__setitem__(0, x)),
        "frame": (lambda: pd.DataFrame({"a": [0.0, 0.0]}),
                  lambda v, x: v.__setitem__("a", [x, x])),
        "zoneinfo": (lambda: ZoneInfo("America/New_York"), None),
        "ts_tz": (lambda: pd.Timestamp("2024-01-02 09:30", tz=ZoneInfo("America/New_York")),
                  None),
        "datetime": (lambda: dt.datetime(2024, 1, 2, 9, 30), None),
        "enum": (lambda: Side.LONG, None),
        "decimal": (lambda: decimal.Decimal("0.5"), None),
        "fraction": (lambda: fractions.Fraction(1, 3), None),
        "regex": (lambda: re.compile("a+"), None),
        "rng": (lambda: np.random.default_rng(1), lambda v, x: v.random()),
        "partial": (lambda: functools.partial(max, 1), None),
        "namedtuple": (lambda: Pair(1, 2), None),
        "dataclass": (lambda: Params(), lambda v, x: setattr(v, "fast", int(x) % 97)),
        "lambda_in_list": (lambda: [lambda y: y, 0.0], lambda v, x: v.__setitem__(1, x)),
        "model_fit": (lambda: Model().fit, lambda v, x: v(x)),
        "list_append": (lambda: [].append, lambda v, x: v(x)),
        "lock_acquire_in_list": (lambda: [threading.Lock().acquire, 0.0],
                                 lambda v, x: v.__setitem__(1, x)),
        "function": (lambda: helper, None),
    }


KINDS = _kinds()


def _strategy(name, kind, op):
    make, mutate = KINDS[kind]
    value = make()

    def precompute(self, df, _op=op, _mutate=mutate, _name=name):
        cls = type(self)
        held = cls.attr
        if _op == "read":
            repr(held)
        elif _op == "same":
            cls.attr = held
        elif _op == "mutate":
            _mutate(held, float(df["close"].min()))
        elif _op == "swap":
            if getattr(cls.attr, "__name__", None) == "helper":
                top = float(df["close"].min())

                def helper(x, _t=top):        # same name, same qualname as the original
                    return _t
                helper.__qualname__ = "helper"
                helper.__module__ = __name__
                cls.attr = helper
        return df

    def on_bar(self, ctx):
        pass

    cls = type(name, (Strategy,), {"attr": value, "precompute": precompute,
                                   "on_bar": on_bar, "__module__": __name__})
    cls.__qualname__ = name
    return cls


CASES = {}
for _kind, (_make, _mutate) in KINDS.items():
    for _op in ["read", "same"] + (["mutate"] if _mutate else []) + (
            ["swap"] if _kind == "function" else []):
        _name = f"S_{_kind}_{_op}"
        CASES[_name] = (_kind, _op)
        globals()[_name] = _strategy(_name, _kind, _op)


class S_method_swap(Strategy):
    """Replaces its own on_bar with a same-named wrapper carrying the whole frame's low."""

    def precompute(self, df):
        cls = type(self)
        if not getattr(cls.on_bar, "__wrapped__", None):
            top = float(df["close"].min())
            inner = cls.on_bar

            @functools.wraps(inner)
            def on_bar(self, ctx, _t=top):
                return inner(self, ctx)
            cls.on_bar = on_bar
        return df

    def on_bar(self, ctx):
        pass


class S_method_same(Strategy):
    """Sets its own on_bar to the function it already is."""

    def precompute(self, df):
        type(self).on_bar = type(self).__dict__["on_bar"]
        return df

    def on_bar(self, ctx):
        pass


CASES["S_method_swap"] = ("method", "swap")
CASES["S_method_same"] = ("method", "same")


def _leaky_fit(self, value):
    self.mean = value


_SWAP_MODEL = Model()


class S_bound_method_swap(Strategy):
    """Swaps the bound fit it holds for a same-named bound method of the same model."""

    attr = _SWAP_MODEL.fit

    def precompute(self, df):
        cls = type(self)
        if cls.attr.__func__ is Model.fit:
            leaky = functools.wraps(Model.fit)(_leaky_fit)
            cls.attr = types.MethodType(leaky, _SWAP_MODEL)
        return df

    def on_bar(self, ctx):
        pass


class S_bound_method_in_list_swap(Strategy):
    """The same swap, inside a list the class holds, in place."""

    attr = [_SWAP_MODEL.fit, 0.0]

    def precompute(self, df):
        cls = type(self)
        if cls.attr[0].__func__ is Model.fit:
            leaky = functools.wraps(Model.fit)(_leaky_fit)
            cls.attr[0] = types.MethodType(leaky, _SWAP_MODEL)
        return df

    def on_bar(self, ctx):
        pass


class _Stage:
    def __init__(self, nxt=None):
        self.seen = 0.0
        self.out = nxt.update if nxt is not None else None

    def update(self, x):
        self.seen = x


def _chain(depth):
    stage = None
    for _ in range(depth):
        stage = _Stage(stage)
    return stage


class S_deep_chain_read(Strategy):
    """A pipeline of stages, each holding the next one's bound update, 22 deep, read."""

    attr = _chain(22)

    def precompute(self, df):
        repr(type(self).attr)
        return df

    def on_bar(self, ctx):
        pass


class S_deep_chain_mutate(Strategy):
    """The same pipeline, its last stage fed a value of the whole frame."""

    attr = _chain(22)

    def precompute(self, df):
        stage = type(self).attr
        while stage.out is not None:
            stage = stage.out.__self__
        stage.update(float(df["close"].min()))
        return df

    def on_bar(self, ctx):
        pass


class LossyModel:
    """A model whose pickle leaves out its fitted state (a cache), held with a lambda."""

    def __init__(self, loss):
        self.loss = loss
        self.coef = None

    def __getstate__(self):
        return {"loss": self.loss, "coef": None}

    def fit(self, x):
        self.coef = float(x.mean())


class S_lossy_model_swap(Strategy):
    """Replaces the model with one fitted on the whole frame: neither has an ordinary pickle
    and their identity-keyed pickles drop the fit, so only identity tells them apart."""

    attr = LossyModel(lambda r: r * r)

    def precompute(self, df):
        fitted = LossyModel(type(self).attr.loss)
        fitted.fit(df["close"])
        type(self).attr = fitted
        return df

    def on_bar(self, ctx):
        pass


class RefClient:
    """Reference data that connects on its first read (then it has no pickle)."""

    def __init__(self):
        self.conn = None

    def holidays(self):
        if self.conn is None:
            import sqlite3
            self.conn = sqlite3.connect(":memory:")
        return []


class S_lazy_client_read(Strategy):
    attr = RefClient()

    def precompute(self, df):
        type(self).attr.holidays()
        return df

    def on_bar(self, ctx):
        pass


class LazyTable:
    """A table loaded on first read by a lambda it then drops (then it has a pickle)."""

    def __init__(self, loader):
        self.loader = loader
        self.table = None

    def get(self):
        if self.table is None:
            self.table = self.loader()
            self.loader = None
        return self.table


class S_lazy_table_read(Strategy):
    attr = LazyTable(lambda: {"NQ": 0.25})

    def precompute(self, df):
        type(self).attr.get()
        return df

    def on_bar(self, ctx):
        pass


def scale(x):
    """Replaces itself in this module on its first call (a lazy initialiser)."""
    global scale

    def scale(x):  # noqa: F811
        return x
    return x


class S_self_replacing_helper_read(Strategy):
    attr = [scale]

    def precompute(self, df):
        type(self).attr[0](1.0)
        return df

    def on_bar(self, ctx):
        pass


class LambdaModel:
    def __init__(self):
        self.loss = lambda r: r
        self.coef = None

    def fit(self, x):
        self.coef = float(x.mean())


class S_lambda_model_mutate(Strategy):
    """Fits a model holding a lambda in place on the whole frame."""

    attr = LambdaModel()

    def precompute(self, df):
        if type(self).attr.coef is None:
            type(self).attr.fit(df["close"])
        return df

    def on_bar(self, ctx):
        pass


CASES["S_lossy_model_swap"] = ("lossy_model", "swap")
CASES["S_lazy_client_read"] = ("lazy_client", "read")
CASES["S_lazy_table_read"] = ("lazy_table", "read")
CASES["S_self_replacing_helper_read"] = ("self_replacing_helper", "read")
CASES["S_lambda_model_mutate"] = ("lambda_model", "mutate")

# Changed in place with a value of the whole frame and NOT refused, as on 3.0.2: the value has
# no ordinary pickle, so the check compares it by identity, and it is the same object. 3.0.3
# drafts that saw these did so with a second pickle that refused harmless cases and missed a
# swap 3.0.2 refused (the fifth review).
KNOWN_UNSEEN = {"S_lambda_in_list_mutate", "S_lock_acquire_in_list_mutate",
                "S_lambda_model_mutate"}
CASES["S_bound_method_swap"] = ("bound_method", "swap")
CASES["S_bound_method_in_list_swap"] = ("bound_method", "swap")
CASES["S_deep_chain_read"] = ("deep_chain", "read")
CASES["S_deep_chain_mutate"] = ("deep_chain", "mutate")
