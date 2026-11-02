"""Engine.run checks the strategy's precompute for look-ahead by default (film-ready
audit, 2026-11-02): the README's "no look-ahead by construction" is now enforced."""
import pathlib
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from plutobt.broker import FrameError
from plutobt.config import BacktestConfig, load_config
from plutobt.data import load_bars
from plutobt.engine import Engine
from plutobt.instruments import get_instrument
from plutobt.lookahead import LookaheadError
from plutobt.strategies.ema_cross import EMACross
from plutobt.strategy import Strategy

NQ = get_instrument("NQ")
ROOT = pathlib.Path(__file__).resolve().parents[1]


def _bars(n=60):
    rng = np.random.default_rng(1)
    c = 20000.0 + np.cumsum(rng.normal(0, 5.0, size=n))
    return pd.DataFrame({"ts": np.arange(0, 300 * n, 300, dtype="int64"),
                         "open": c, "high": c + 2.0, "low": c - 2.0, "close": c,
                         "volume": np.ones(n, dtype="int64")})


class Fn(Strategy):
    """precompute = fn; on_bar goes long when 'go' is truthy, flat otherwise."""

    def __init__(self, fn):
        self.fn = fn

    def precompute(self, df):
        return self.fn(df)

    def on_bar(self, ctx):
        go = bool(getattr(ctx.bar, "go", False))
        if go and ctx.position.qty == 0:
            ctx.broker.buy(contracts=1)
        elif not go and ctx.position.qty != 0:
            ctx.broker.close()


def _peek(df):
    df = df.copy()
    df["go"] = (df["close"].shift(-1) > df["close"]).astype(float)   # next bar: a leak
    return df


def _honest(df):
    df = df.copy()
    df["go"] = (df["close"] > df["close"].shift(1)).astype(float)
    return df


def _sparse_peek(df):
    df = _honest(df)
    if len(df) > 38:
        df.loc[37, "go"] = float(df.loc[38, "close"] > df.loc[37, "close"])
    return df


def _eng(**cfg):
    return Engine(NQ, BacktestConfig(commission_per_side=0.0, slippage_ticks=0.0, **cfg))


def test_engine_refuses_a_shift_minus_one_signal_by_default():
    # Positive control: unchecked, this leak traded to a fat equity curve (probe p03).
    with pytest.raises(LookaheadError, match="non-causal column 'go'") as e:
        _eng().run(_bars(), Fn(_peek))
    assert "check_causality" in str(e.value)       # the message says how to opt out


def test_engine_runs_an_honest_strategy_with_the_check_on():
    # Negative control: the same shape of signal, built causally, runs.
    result = _eng().run(_bars(), Fn(_honest))
    assert len(result.equity_curve) == 60 and result.trades


def test_check_causality_false_skips_the_check():
    result = _eng(check_causality=False).run(_bars(), Fn(_peek))
    assert len(result.equity_curve) == 60


def test_causality_sample_picks_the_grid():
    # The default ("auto") is the full grid on a frame this short, which catches a leak
    # on one bar; a sampled grid can miss it (documented on assert_causal).
    assert BacktestConfig().causality_sample == "auto"
    with pytest.raises(LookaheadError, match="non-causal column 'go'"):
        _eng().run(_bars(), Fn(_sparse_peek))
    assert len(_eng(causality_sample=5).run(_bars(), Fn(_sparse_peek)).equity_curve) == 60


@pytest.mark.parametrize("bad", [0, -3, 2.5, True, "all"])
def test_a_bad_causality_sample_is_refused_before_the_strategy_runs(bad):
    def never(df):
        raise AssertionError("precompute ran")

    with pytest.raises(ValueError, match="causality_sample"):
        _eng(causality_sample=bad).run(_bars(), Fn(never))


@pytest.mark.parametrize("col", ["open", "high", "low", "close", "volume", "ts"])
def test_a_precompute_that_rewrites_a_base_column_is_refused(col):
    # Even a same-bar rewrite (buy at every bar's low, probe p04) changes the fills, so
    # the engine refuses any change to a base column the caller's frame holds.
    def rewrite(df):
        df = df.copy()
        df[col] = df[col] - 1 if col != "ts" else df[col] + 1
        return df

    with pytest.raises(FrameError,
                       match=f"the strategy's precompute changed column '{col}' at row 0"):
        _eng(check_causality=False).run(_bars(), Fn(rewrite))


def test_a_precompute_that_changes_the_session_column_is_refused():
    df = _bars(6)
    df["session"] = ["ETH", "RTH", "RTH", "RTH", "RTH", "ETH"]

    def relabel(d):
        return d.assign(session="RTH")

    with pytest.raises(FrameError, match="changed column 'session' at row 0"):
        _eng().run(df, Fn(relabel))


def test_a_precompute_that_drops_rows_is_refused():
    # Unchecked, a dropna precompute ran on fewer bars without a word (probe p12).
    def dropna(df):
        df = df.copy()
        df["sma"] = df["close"].rolling(5).mean()
        return df.dropna().reset_index(drop=True)

    with pytest.raises(FrameError, match="returned 56 rows for 60 input bars"):
        _eng().run(_bars(), Fn(dropna))


def test_a_precompute_that_adds_a_session_column_is_checked_not_refused():
    # A base column the caller's frame lacks may be added; the causality check covers it.
    def add_session(df):
        return df.assign(session=1)

    assert len(_eng().run(_bars(), Fn(add_session)).equity_curve) == 60


@pytest.mark.parametrize("sym", ["nq", "es", "cl"])
def test_demo_runs_identically_with_the_check_on_and_off(sym):
    cfg = load_config(str(ROOT / "configs" / f"{sym}.toml"))
    assert cfg.check_causality is True             # the demo configs leave it on
    df = load_bars(str(ROOT / "data" / "sample" / f"{sym}_5m.json"))
    inst = get_instrument(sym.upper())
    on = Engine(inst, cfg).run(df, EMACross())
    off = Engine(inst, replace(cfg, check_causality=False)).run(df, EMACross())
    assert list(on.equity_curve) == list(off.equity_curve)
    assert [(t.entry_ts, t.exit_ts, t.pnl) for t in on.trades] == [
        (t.entry_ts, t.exit_ts, t.pnl) for t in off.trades]


def test_the_engine_trades_the_frame_it_checked():
    # The traded frame comes from a copy of the strategy on the full frame; each prefix
    # runs on its own copy; the strategy passed in runs the full frame last, so the
    # state on_bar sees is the traded call's.
    seen = []

    def record(df):
        seen.append(len(df))
        return _honest(df)

    _eng().run(_bars(), Fn(record))
    assert seen == [60, *range(1, 60), 60]


# ---------------------------------------------------------------------------
# Review fix (2026-11-02, second round): state saved by precompute cannot leak into the
# check. Every call starts from a deepcopy of the strategy as it was passed in.
# ---------------------------------------------------------------------------

class FitOnce(Strategy):
    """Fits a z-score's mean and std on the first frame it sees, then reuses them: the
    engine's first call is the full frame, so the fit is the whole sample's."""

    def __init__(self):
        self.mu = self.sd = None

    def precompute(self, df):
        if self.mu is None:
            self.mu, self.sd = df["close"].mean(), df["close"].std()
        return df.assign(z=(df["close"] - self.mu) / self.sd)


def test_a_fit_once_whole_sample_state_is_refused():
    # Unchecked (probe p_state), every check reused the full-sample fit and ran clean.
    with pytest.raises(LookaheadError, match="non-causal column 'z'"):
        _eng().run(_bars(), FitOnce())


def test_a_refused_strategy_is_left_as_passed_and_refused_again():
    # Only copies ran, so the strategy keeps no whole-sample fit for a second run to
    # reuse; the same object is refused again.
    s = FitOnce()
    for _ in range(2):
        with pytest.raises(LookaheadError, match="non-causal column 'z'"):
            _eng().run(_bars(), s)
        assert s.mu is None


def test_a_leak_only_in_the_first_call_is_refused():
    # Probe p_state's Counting: it leaked on the engine's first (traded) call only.
    class FirstCall(Strategy):
        def __init__(self):
            self.k = 0

        def precompute(self, df):
            self.k += 1
            x = df["close"].shift(-1).ffill() if self.k == 1 else df["close"]
            return df.assign(x=x)

    with pytest.raises(LookaheadError, match="non-causal column 'x'"):
        _eng().run(_bars(), FirstCall())


def test_a_leak_in_the_traded_call_through_class_state_is_refused():
    # Class attributes are shared by the copies, so this leaks on the first call ever
    # and never again; the check compares the prefixes with the frame that is traded.
    class Shared(Strategy):
        calls = 0

        def precompute(self, df):
            type(self).calls += 1
            x = df["close"].shift(-1).ffill() if type(self).calls == 1 else df["close"]
            return df.assign(x=x)

    with pytest.raises(LookaheadError, match="non-causal column 'x'"):
        _eng().run(_bars(), Shared())


def test_a_seeded_rng_made_in_init_runs():
    # Probe p_state: each call drew fresh numbers from one RNG and was refused. Every
    # copy now starts from the same RNG state, as the traded call did.
    class Coin(Fn):
        def __init__(self):
            super().__init__(None)
            self.rng = np.random.default_rng(7)

        def precompute(self, df):
            return df.assign(go=self.rng.random(len(df)) < 0.3)

    on = _eng().run(_bars(), Coin())
    off = _eng(check_causality=False).run(_bars(), Coin())
    assert list(on.equity_curve) == list(off.equity_curve) and on.trades


def test_on_bar_runs_on_the_strategy_passed_in_with_the_traded_calls_state():
    class Remembers(Strategy):
        def __init__(self):
            self.fitted_on, self.seen = None, []

        def precompute(self, df):
            self.fitted_on = len(df)
            return df

        def on_bar(self, ctx):
            self.seen.append(self.fitted_on)

    s = Remembers()
    _eng().run(_bars(), s)
    assert s.seen == [60] * 60 and s.fitted_on == 60


def test_a_precompute_that_differs_on_the_strategy_passed_in_is_refused():
    # The traded frame came from a copy; on_bar runs on the strategy passed in, so its
    # own full-frame call must give the same frame, or on_bar's state is not the
    # traded one.
    class Drifts(Strategy):
        calls = 0

        def precompute(self, df):
            type(self).calls += 1
            return df.assign(x=float(type(self).calls > 60))   # call 61 = the last one

    with pytest.raises(LookaheadError, match="column 'x'.*same starting state"):
        _eng().run(_bars(), Drifts())


def test_a_strategy_that_cannot_be_copied_is_refused_with_the_switch_named():
    import threading

    class Locked(Fn):
        def __init__(self):
            super().__init__(_honest)
            self.lock = threading.Lock()

    with pytest.raises(TypeError, match="deepcopy.*check_causality = false"):
        _eng().run(_bars(), Locked())
    assert len(_eng(check_causality=False).run(_bars(), Locked()).equity_curve) == 60


def test_engine_refuses_a_backfilled_first_return():
    # Review fix: row 0 = close1/close0 - 1 leaked into on_bar(0) past the old grid.
    def bfill(df):
        return _honest(df).assign(r=df["close"].pct_change().bfill())

    with pytest.raises(LookaheadError, match="non-causal column 'r' at t=0"):
        _eng().run(_bars(), Fn(bfill))


# ---------------------------------------------------------------------------
# Review fix (2026-11-02): object columns holding arrays are compared, not crashed on.
# ---------------------------------------------------------------------------

def _windows(lookahead):
    def fn(df):
        df = df.copy()
        c = df["close"].to_numpy()
        df["win"] = [c[max(0, i - 2):i + 1 + lookahead] for i in range(len(df))]
        return df
    return fn


def test_an_honest_column_of_array_windows_runs():
    # Unchecked: "The truth value of an array with more than one element is ambiguous".
    assert len(_eng().run(_bars(), Fn(_windows(0))).equity_curve) == 60


def test_a_column_of_array_windows_that_reach_ahead_is_caught():
    with pytest.raises(LookaheadError, match="non-causal column 'win'"):
        _eng().run(_bars(), Fn(_windows(1)))


def test_an_object_column_that_cannot_be_compared_is_a_frame_error():
    class Vague:
        def __eq__(self, other):
            return np.array([True, False])   # neither true nor false

    def vague(df):
        return df.assign(v=[Vague() for _ in range(len(df))])

    with pytest.raises(FrameError, match="column 'v' holds Vague values"):
        _eng().run(_bars(), Fn(vague))


# ---------------------------------------------------------------------------
# Review decision (2026-11-02, second round): the kept-frame refusal is removed. It was
# bypassable nine ways (a list, a deque, a dict keyed by ts, a closure, a transposed
# array...) and refused honest strategies, so the boundary is stated instead: the check
# covers precompute, on_bar must decide from ctx, and cheating code is not sandboxed.
# ---------------------------------------------------------------------------

class Keeps(Strategy):
    """Stores the bar frame on self in precompute, as the refused check looked for."""

    def precompute(self, df):
        self.df = df.reset_index(drop=True)
        return df


def test_holding_the_bar_frame_is_not_refused_the_check_is_not_a_sandbox():
    # The documented boundary: storing the frame is against the contract, but no
    # Python check can stop code written to read ahead from on_bar.
    assert len(_eng().run(_bars(), Keeps()).equity_curve) == 60
    assert "not a sandbox" in Strategy.__doc__


def test_former_false_positives_of_the_kept_frame_check_run():
    # A lookup table as long as the bars, and an equity curve kept for plotting then
    # re-run (probe p_hoard), were refused by the removed check.
    class Table(Fn):
        def __init__(self, n):
            super().__init__(_honest)
            self.weights = np.linspace(0.0, 1.0, n)

    assert len(_eng().run(_bars(), Table(60)).equity_curve) == 60
    s = EMACross()
    df = load_bars(str(ROOT / "data" / "sample" / "nq_5m.json"))
    first = _eng().run(df, s)
    s.curve = first.equity_curve
    assert list(_eng().run(df, s).equity_curve) == list(first.equity_curve)


def test_honest_strategies_with_attributes_still_run():
    # Parameters, a short lookup table and the demo strategy are not the bar frame.
    class Params(Fn):
        def __init__(self):
            super().__init__(_honest)
            self.spans = [20, 50]
            self.table = pd.Series([1.0, 2.0, 3.0])
            self.weights = np.ones(5)
            self.opts = {"stop": 20, "levels": np.arange(3)}

    assert len(_eng().run(_bars(), Params()).equity_curve) == 60
    df = load_bars(str(ROOT / "data" / "sample" / "nq_5m.json"))
    assert len(_eng().run(df, EMACross()).equity_curve) == len(df)


# ---------------------------------------------------------------------------
# Review decision (2026-11-02): causality_sample = "auto" scales to long histories.
# ---------------------------------------------------------------------------

def test_auto_is_the_full_grid_up_to_2000_bars():
    from plutobt.lookahead import _grid

    assert list(_grid(2000, "auto")) == list(range(1999))
    assert list(_grid(2000, None)) == list(range(1999))


def test_auto_samples_a_longer_frame_keeping_both_ends():
    from plutobt.lookahead import _grid

    grid = list(_grid(8000, "auto"))
    assert grid[:20] == list(range(20)) and grid[-20:] == list(range(7979, 7999))
    assert grid == sorted(set(grid)) and 250 <= len(grid) <= 290
    assert list(_grid(8000, None)) == list(range(7999))   # explicit None: every prefix


def test_auto_warns_when_it_samples_and_still_catches_an_edge_leak():
    n = 2500
    with pytest.warns(UserWarning, match="causality check sampled"):
        result = _eng().run(_bars(n), Fn(_honest))
    assert len(result.equity_curve) == n
    leak = Fn(lambda d: _honest(d).assign(r=d["close"].diff().bfill()))
    with pytest.raises(LookaheadError, match="non-causal column 'r' at t=0"), pytest.warns(
            UserWarning, match="causality check sampled"):
        _eng().run(_bars(n), leak)


def test_auto_does_not_warn_on_a_short_frame():
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _eng().run(_bars(), Fn(_honest))


# ---------------------------------------------------------------------------
# Review fix (2026-11-02, second round): the sampled-grid warning points at the user's
# own call and says how many prefixes were checked and how to check them all.
# ---------------------------------------------------------------------------

def test_the_sampled_warning_points_at_the_engine_run_call():
    import warnings

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _eng().run(_bars(3000), Fn(_honest))
    (w,) = (w for w in caught if "causality check sampled" in str(w.message))
    assert pathlib.Path(w.filename).resolve() == pathlib.Path(__file__).resolve()
    text = str(w.message)
    assert "checked 286 of the 2999 prefixes of 3000 bars" in text
    assert "causality_sample = None" in text and "every prefix" in text


def test_the_sampled_warning_points_at_a_direct_assert_causal_call():
    import warnings

    from plutobt.lookahead import assert_causal

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert_causal(_honest, _bars(2500), sample="auto")
    (w,) = caught
    assert pathlib.Path(w.filename).resolve() == pathlib.Path(__file__).resolve()
    assert "sample=None" in str(w.message)
