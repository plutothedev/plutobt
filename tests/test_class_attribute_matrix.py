"""Every class-attribute kind in tests/class_attribute_cases.py, crossed with every operation:
a precompute that only reads the attribute, or sets it to the object it holds, runs with no
warning; one that changes it in place with a value of the whole frame, or replaces it with a
same-named object, is refused. Against 3.0.2 this matrix differs only where 3.0.3 closes a
gap (a same-named method swap; an in-place edit inside a list holding a lambda or a lock's
bound method), and it caught the regressions two earlier drafts of 3.0.3 had."""
import warnings

import numpy as np
import pandas as pd
import pytest

from plutobt.config import BacktestConfig
from plutobt.engine import Engine
from plutobt.instruments import get_instrument
from plutobt.lookahead import LookaheadError
from tests import class_attribute_cases as cases


def _bars(n=60):
    rng = np.random.default_rng(1)
    c = 20000.0 + np.cumsum(rng.normal(0, 5.0, size=n))
    return pd.DataFrame({"ts": np.arange(0, 300 * n, 300, dtype="int64"), "open": c,
                         "high": c + 2.0, "low": c - 2.0, "close": c,
                         "volume": np.ones(n, dtype="int64")})


def _run(name):
    engine = Engine(get_instrument("NQ"), BacktestConfig(commission_per_side=0.0,
                                                          slippage_ticks=0.0))
    return engine.run(_bars(), getattr(cases, name)())


BENIGN = sorted(n for n, (_, op) in cases.CASES.items() if op in ("read", "same"))
LEAKS = sorted(n for n, (_, op) in cases.CASES.items() if op in ("mutate", "swap"))


@pytest.mark.parametrize("name", BENIGN)
def test_reading_or_resetting_a_class_attribute_runs_unwarned(name):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert _run(name) is not None


@pytest.mark.parametrize("name", LEAKS)
def test_changing_a_class_attribute_with_the_whole_frame_is_refused(name):
    with pytest.raises(LookaheadError, match="changed class attribute"):
        _run(name)
