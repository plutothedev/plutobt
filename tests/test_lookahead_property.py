"""Randomized property test of the look-ahead check: the exit criterion of the 2026-11-02
redesign (snapshot + deepcopy per call, scale-aware tolerance).

Synthetic OHLCV series at price scales from 1e-3 to 1e5, per-bar volatility from 1e-5 to
2e-2 and 50..400 bars. Every HONEST feature must pass the full truncation grid on every
series; every PLANTED LEAK must be caught on every series, and so must the stateful leak
(a z-score fitted once, on the first precompute call) through Engine.run.

The default suite runs a short, seeded slice (about 3 s). ``pytest --slow`` runs the
full sweep (60 series up to 400 bars) and prints the pass/catch counts.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from plutobt.config import BacktestConfig
from plutobt.engine import Engine
from plutobt.instruments import get_instrument
from plutobt.lookahead import LookaheadError, assert_causal
from plutobt.strategy import Strategy

SEED = 20261102
NQ = get_instrument("NQ")


# ---------------------------------------------------------------------------
# Synthetic bars
# ---------------------------------------------------------------------------

def _series(rng: np.random.Generator, n_lo: int, n_hi: int) -> tuple[pd.DataFrame, dict]:
    """One OHLCV frame: log-normal closes from a random price scale and volatility."""
    n = int(rng.integers(n_lo, n_hi + 1))
    p0 = float(10 ** rng.uniform(-3, 5))
    sigma = float(10 ** rng.uniform(-5, np.log10(2e-2)))
    drift = float(rng.choice([0.0, 0.5, -0.5, 2.0])) * sigma
    close = p0 * np.exp(np.cumsum(rng.normal(drift, sigma, n)))
    open_ = close * np.exp(rng.normal(0.0, sigma / 2, n))
    wick = np.abs(rng.normal(0.0, sigma, (2, n)))
    df = pd.DataFrame({
        "ts": 1_700_000_000 + 60 * np.arange(n, dtype="int64"),
        "open": open_,
        "high": np.maximum(open_, close) * np.exp(wick[0]),
        "low": np.minimum(open_, close) * np.exp(-wick[1]),
        "close": close,
        "volume": rng.integers(1, 1000, n).astype("int64"),
        # A trading-session ID that never decreases: a new session every 20..80 bars.
        "session": 20240101 + np.cumsum(rng.random(n) < 1 / rng.integers(20, 81)),
    })
    return df, {"n": n, "p0": p0, "sigma": sigma, "drift": drift}


def _lr(d):
    return np.log(d["close"]).diff()


def _rsi(c, span=14):
    delta = c.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / span, adjust=False).mean()
    down = (-delta).clip(lower=0).ewm(alpha=1 / span, adjust=False).mean()
    return 100 - 100 / (1 + up / down)


def _honest_features() -> dict:
    """name -> f(d) returning one causal Series: row t uses rows 0..t only."""
    f = {}
    for w in (3, 10, 30):
        for stat in ("mean", "std", "var", "median", "min", "max"):
            f[f"roll{w}_{stat}"] = (lambda w, stat: lambda d: getattr(
                d["close"].rolling(w), stat)())(w, stat)
        f[f"roll{w}_q25"] = (lambda w: lambda d: d["close"].rolling(w).quantile(0.25))(w)
    # Skew and kurtosis on returns, at every window: pandas' result is bit-stable there.
    for w in (5, 10, 20, 30):
        for stat in ("skew", "kurt"):
            f[f"lr_roll{w}_{stat}"] = (lambda w, stat: lambda d: getattr(
                _lr(d).rolling(w), stat)())(w, stat)
    # Not swept: skew and kurtosis of raw price LEVELS. pandas centres them on the whole
    # input (it subtracts round(mean)), so on a series that wanders far from its mean, or
    # sits near a price of 0.5 where that rounding flips, honest values move between a
    # prefix and the full frame by up to 1% of the column on short windows and still by
    # more than a part per million at 10 and 50 bars (measured 2026-11-02, 3 of 3,240
    # verdicts in the full sweep). No tolerance tells that from a leak; it is a
    # documented limit (assert_causal, README). Realistic cases (the NQ sample, 1,500
    # one-minute-like NQ bars at rolling(20)) pass in test_lookahead.py.
    f["ewm10_adjust"] = lambda d: d["close"].ewm(span=10, adjust=True).mean()
    f["ewm10_recursive"] = lambda d: d["close"].ewm(span=10, adjust=False).mean()
    f["ewm20_std"] = lambda d: d["close"].ewm(span=20, adjust=False).std()
    f["exp_mean"] = lambda d: d["close"].expanding().mean()
    f["exp_std"] = lambda d: d["close"].expanding().std()
    f["exp_max"] = lambda d: d["close"].expanding().max()
    f["pct_change"] = lambda d: d["close"].pct_change()
    f["diff"] = lambda d: d["close"].diff()
    f["log_return"] = _lr
    f["roll20_corr"] = lambda d: d["close"].rolling(20).corr(d["volume"])
    f["roll20_cov"] = lambda d: d["close"].rolling(20).cov(d["high"] - d["low"])
    f["rsi14"] = lambda d: _rsi(d["close"])
    f["session_cumvol"] = lambda d: d.groupby("session")["volume"].cumsum()
    f["session_vwap"] = lambda d: ((d["close"] * d["volume"]).groupby(d["session"]).cumsum()
                                   / d.groupby("session")["volume"].cumsum())
    f["cummax"] = lambda d: d["close"].cummax()
    f["ffill_sparse"] = lambda d: d["close"].where(np.arange(len(d)) % 7 == 0).ffill()
    f["realized_var"] = lambda d: (_lr(d) ** 2).rolling(10).sum()
    # Returns of 1e-5 size: values near 1e-10, where an absolute atol would see nothing.
    f["realized_var_tiny"] = lambda d: ((1e-5 * np.sign(_lr(d))) ** 2
                                        + (1e-3 * _lr(d)) ** 2).rolling(10).sum()
    f["expanding_z"] = lambda d: ((d["close"] - d["close"].expanding().mean())
                                  / d["close"].expanding().std())
    f["above_ema"] = lambda d: d["close"] > d["close"].ewm(span=10, adjust=False).mean()
    f["side_text"] = lambda d: pd.Series(np.where(d["close"].diff() > 0, "up", "down"),
                                         index=d.index)
    return f


def _leaks(n_full: int) -> dict:
    """name -> f(d) whose value at some row t depends on a row after t."""
    f = {}
    for k in range(1, 6):
        f[f"shift_minus_{k}"] = (lambda k: lambda d: d["close"].shift(-k))(k)
    f["feature_shift_minus_1"] = lambda d: d["close"].rolling(10).mean().shift(-1)
    f["bfill_return"] = lambda d: d["close"].pct_change().bfill()
    f["bfill_rolling"] = lambda d: d["close"].rolling(10).mean().bfill()
    f["centered_rolling5"] = lambda d: d["close"].rolling(5, center=True).mean()
    f["centered_rolling3_minp"] = lambda d: _lr(d).rolling(3, center=True,
                                                          min_periods=1).mean()
    f["whole_sample_z"] = lambda d: (d["close"] - d["close"].mean()) / d["close"].std()
    f["whole_sample_minmax"] = lambda d: ((d["close"] - d["close"].min())
                                          / (d["close"].max() - d["close"].min()))
    f["whole_sample_rank"] = lambda d: d["close"].rank(pct=True)
    f["future_max"] = lambda d: d["close"][::-1].cummax()[::-1]
    f["tiny_minus_sample_mean"] = lambda d: _tiny(d) - _tiny(d).mean()
    f["lr_minus_sample_mean"] = lambda d: _lr(d) - _lr(d).mean()

    def second_to_last_reads_last(d):
        # A leak in one row only, the last row the check can test (t = n-2).
        x = d["close"].rolling(3, min_periods=1).mean().to_numpy().copy()
        if len(d) == n_full:
            x[-2] = d["close"].iloc[-1]
        return pd.Series(x, index=d.index)

    def last_row_whole_sample(d):
        # The traded frame's row n-2 holds the whole sample's mean return; every other
        # row is the honest expanding mean, and so is every prefix. (On a price LEVEL the
        # same swap moves a quiet series by about one part per million, the documented
        # resolution of the float compare, so it is planted on returns.)
        x = _lr(d).expanding().mean().to_numpy().copy()
        if len(d) == n_full:
            x[-2] = _lr(d).mean()
        return pd.Series(x, index=d.index)

    f["only_row_n_minus_2"] = second_to_last_reads_last
    f["only_row_n_minus_2_sample_mean"] = last_row_whole_sample
    f["side_text_next_bar"] = lambda d: pd.Series(
        np.where(d["close"].shift(-1) > d["close"], "up", "down"), index=d.index)
    return f


def _tiny(d):
    return ((1e-5 * _lr(d)) ** 2).rolling(5).mean()


# ---------------------------------------------------------------------------
# Running the cases
# ---------------------------------------------------------------------------

def _honest_failures(df: pd.DataFrame, features: dict) -> list[tuple[str, str]]:
    """Run assert_causal (full grid) on all honest features at once; a refused column is
    recorded and dropped and the rest re-run, so every feature gets a verdict."""
    names, failed = list(features), []
    while names:
        def precompute(d, names=tuple(names)):
            return d.assign(**{name: features[name](d) for name in names})
        try:
            assert_causal(precompute, df, sample=None)
            return failed
        except LookaheadError as e:
            col = str(e).split("'")[1]
            failed.append((col, str(e)[:160]))
            names.remove(col)
    return failed


def _leak_caught(df: pd.DataFrame, fn) -> bool:
    try:
        assert_causal(lambda d: d.assign(x=fn(d)), df, sample=None)
    except LookaheadError as e:
        return "'x'" in str(e)
    return False


class _FitOnce(Strategy):
    """Fits a z-score's mean and std on the FIRST frame it sees (the whole sample when
    the engine's first call is the full frame), then reuses them: a stateful leak."""

    def __init__(self):
        self.mu = self.sd = None

    def precompute(self, d):
        if self.mu is None:
            self.mu, self.sd = d["close"].mean(), d["close"].std()
        return d.assign(z=(d["close"] - self.mu) / self.sd)

    def on_bar(self, ctx):
        if ctx.position.qty == 0 and ctx.bar.z < -1:
            ctx.broker.buy(1)
        elif ctx.position.qty > 0 and ctx.bar.z > 0:
            ctx.broker.close()


class _SeededCoin(Strategy):
    """An honest random-entry benchmark: a seeded RNG made in __init__."""

    def __init__(self):
        self.rng = np.random.default_rng(7)

    def precompute(self, d):
        return d.assign(coin=self.rng.random(len(d)) < 0.05)


def _engine_verdict(df, strategy) -> str:
    cfg = BacktestConfig(slippage_ticks=0.0, causality_sample=None)
    try:
        Engine(NQ, cfg).run(df, strategy)
    except LookaheadError:
        return "refused"
    return "ran"


def sweep(n_series: int, n_lo: int, n_hi: int, seed: int = SEED) -> dict:
    rng = np.random.default_rng(seed)
    honest = _honest_features()
    out = {"series": 0, "honest_passed": 0, "honest_total": 0, "honest_failures": [],
           "leaks_caught": 0, "leaks_total": 0, "leaks_missed": []}
    for _ in range(n_series):
        df, meta = _series(rng, n_lo, n_hi)
        out["series"] += 1
        failures = _honest_failures(df, honest)
        out["honest_total"] += len(honest) + 1          # + the seeded-RNG strategy
        out["honest_passed"] += len(honest) - len(failures)
        out["honest_failures"] += [(meta, *f) for f in failures]
        if _engine_verdict(df, _SeededCoin()) == "ran":
            out["honest_passed"] += 1
        else:
            out["honest_failures"].append((meta, "seeded_coin", "refused"))
        leaks = _leaks(len(df))
        for name, fn in leaks.items():
            out["leaks_total"] += 1
            if _leak_caught(df, fn):
                out["leaks_caught"] += 1
            else:
                out["leaks_missed"].append((meta, name))
        out["leaks_total"] += 1                         # + the fit-once strategy
        if _engine_verdict(df, _FitOnce()) == "refused":
            out["leaks_caught"] += 1
        else:
            out["leaks_missed"].append((meta, "fit_once_state"))
    return out


def _report(out: dict) -> str:
    return (f"{out['series']} series: honest {out['honest_passed']}/{out['honest_total']} "
            f"passed, leaks {out['leaks_caught']}/{out['leaks_total']} caught")


def _assert_clean(out: dict) -> None:
    assert not out["honest_failures"], (_report(out), out["honest_failures"][:5])
    assert not out["leaks_missed"], (_report(out), out["leaks_missed"][:5])
    assert out["honest_passed"] == out["honest_total"]
    assert out["leaks_caught"] == out["leaks_total"]


@pytest.mark.filterwarnings("ignore::RuntimeWarning")   # log/divide on 1-bar prefixes
def test_property_quick_slice():
    # Short seeded slice for the default suite: the scales and leak list are the same.
    out = sweep(n_series=6, n_lo=50, n_hi=90)
    print(_report(out))
    _assert_clean(out)


@pytest.mark.slow
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_property_full_sweep():
    out = sweep(n_series=60, n_lo=50, n_hi=400)
    print(_report(out))
    _assert_clean(out)
