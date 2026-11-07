"""Bar loop: decision@t / fill@t+1 open; warm-up skip; discontinuity carry;
manifest build (§2, §5.1, §5.5, §5.9, §5.10)."""
from __future__ import annotations

import collections
import copy
import functools
import inspect
import keyword
import math
import numbers
import pickle
import warnings
from dataclasses import asdict, dataclass, field, replace

import numpy as np
import pandas as pd

from plutobt import __version__
from plutobt.broker import Broker, FrameError
from plutobt.config import BacktestConfig
from plutobt.data import (
    assert_positive_prices,
    frame_sha256,
    ohlcv_problem,
    ts_unit_problem,
)
from plutobt.instruments import Instrument
from plutobt.lookahead import (
    _ROW_RULE,
    LookaheadError,
    LookaheadWarning,
    _first_mismatch,
    assert_causal,
    first_difference,
)
from plutobt.portfolio import MarginCall, Portfolio, Trade
from plutobt.strategy import Context, Strategy

# ---------------------------------------------------------------------------
# §0.1: BacktestResult defined ONCE at module scope (consumers import from here)
# ---------------------------------------------------------------------------

@dataclass
class BacktestResult:
    """Backtest output: equity curve, closed/open-at-end trades, metrics, manifest, and
    every margin call the run hit (spec §5.2), in bar order.

    ``margin_refusals`` counts the orders the margin check refused outright or cut
    down to their closing half, each order once: a thin run can then say why it is
    thin instead of reading as "the signal never fired"."""
    equity_curve: pd.Series
    trades: list[Trade]
    metrics: dict
    manifest: dict
    margin_calls: list[MarginCall] = field(default_factory=list)
    margin_refusals: int = 0


# ---------------------------------------------------------------------------
# Internal constants
# ---------------------------------------------------------------------------

# Columns that are always present in the raw frame; never treated as signal cols.
_BASE_COLS = frozenset(("ts", "open", "high", "low", "close", "volume", "session"))
_STARTING_CASH = 100_000.0

# v1 default roll/sample vintage; NOT the run date (back-adjustment construction
# is out of v1, §5.6). Stamped into the manifest's roll metadata only for a run on a
# shipped sample file, recognized by its data_sha256 (load_bars of data/sample/*).
_ROLL_AS_OF_DEFAULT = "2026-11-02"   # the date the samples were regenerated
_SAMPLE_SHA256 = {
    "123af852f8fb1364e02b5761d048a397adb5b87a53301567c85410b139fc8ca7": "nq_5m.json",
    "24f31941d385678f0a4f8acd12e16b602b777263ea726222f1a948a6b8716daf": "es_5m.json",
    "fb019e9fd374e5c12e0ff173627fa21a2fb83831a8ecd05eec7e8cdce9e887c7": "cl_5m.json",
}
_ROLL_UNKNOWN = "unknown/user-supplied"

# The time rule, quoted by every refusal of a frame's bar timestamps.
_TIME_RULE = ("bar timestamps (the 'ts' column, in whole unix seconds) must strictly "
              "increase, since the engine runs the bars in row order")

# The unit rule, quoted by every refusal of a ts that is not whole unix seconds.
_TS_UNIT_RULE = "bar timestamps (the 'ts' column) must be whole unix seconds"

# The column-name rule, quoted by every refusal of a frame's column names.
_COLUMN_RULE = ("every column name must be a Python identifier that does not start with "
                "an underscore and is not a keyword (e.g. 'ema_20', not 'ema-20'), since "
                "on_bar reads each one as ctx.bar.<name>")

# The session-ID rule for break_on="session", quoted by every refusal of the column.
_SESSION_RULE = (
    "break_on='session' needs a 'session' column holding a trading-session ID on every "
    "bar: a number that never decreases (e.g. the CME trading day as an int YYYYMMDD), "
    "not RTH/ETH labels"
)


def _time_problem(frame: pd.DataFrame) -> str | None:
    """The first bar whose ts is not after the one before it, naming both; None if the
    timestamps strictly increase."""
    if "ts" not in frame.columns:
        return "the frame has no 'ts' column"
    if pd.api.types.is_datetime64_any_dtype(frame["ts"]):
        return (f"the 'ts' column holds datetimes ({frame['ts'].dtype}), not a number of "
                "unix seconds; convert it to int unix seconds first")
    ts = frame["ts"].tolist()
    for i, value in enumerate(ts):
        if isinstance(value, bool) or not isinstance(value, numbers.Real):
            return (f"row {i} holds ts {value!r} ({type(value).__name__}), not a number "
                    "of unix seconds")
        if i and not value > ts[i - 1]:      # written so that a NaN ts is refused too
            return f"row {i} (ts {value}) is not after row {i - 1} (ts {ts[i - 1]})"
    return None


def _ts_unit_problem(frame: pd.DataFrame) -> str | None:
    """The first ts that is not whole unix seconds (a fraction, or another unit such as
    milliseconds), naming its row; None if every one is. Runs after _time_problem, so
    every ts is a number."""
    for i, value in enumerate(frame["ts"].tolist()):
        # Whole seconds, so a Fill's int ts is exact; and seconds, not milliseconds.
        if not (math.isfinite(value) and float(value).is_integer()):
            return f"row {i} holds ts {value!r}, not a whole number of unix seconds"
        unit = ts_unit_problem(value)
        if unit is not None:
            return f"row {i}: {unit}"
    return None


def _column_problem(frame: pd.DataFrame) -> str | None:
    """The first column name a namedtuple field (ctx.bar.<name>) cannot be; None if none."""
    seen = set()
    for name in frame.columns:
        if name in seen:
            return f"column {name!r} appears twice"
        seen.add(name)
        if (not isinstance(name, str) or not name.isidentifier()
                or name.startswith("_") or keyword.iskeyword(name)):
            return f"column {name!r} is not a name on_bar can read as ctx.bar.<name>"
    return None


def _session_problem(frame: pd.DataFrame) -> str | None:
    """The first break of the session-ID rule in ``frame``, naming its row; None if none.

    A missing ID (NaN, None, pd.NA) is refused, as is anything but a number (RTH/ETH
    labels need not decrease, across a weekend say, so their type gives them away) and
    any ID below the one before it.
    """
    if "session" not in frame.columns:
        return "the frame has no 'session' column"
    ts = frame["ts"].tolist() if "ts" in frame.columns else None
    prev = None
    for i, value in enumerate(frame["session"].tolist()):
        row = f"row {i}" if ts is None else f"row {i} (ts {ts[i]})"
        if value is None or value is pd.NA or (isinstance(value, float) and math.isnan(value)):
            return f"{row} has no session ID ({value!r})"
        if isinstance(value, bool) or not isinstance(value, numbers.Real):
            return f"{row} holds {value!r} ({type(value).__name__}), not a number"
        if prev is not None and value < prev:
            return f"{row} holds {value!r}, below the {prev!r} before it"
        prev = value
    return None


def _snapshot(strategy: Strategy) -> Strategy:
    """A deepcopy of the strategy as passed in, which every precompute call of the
    look-ahead check starts from; a strategy that cannot be copied is refused, naming
    the switch that skips the check."""
    try:
        return copy.deepcopy(strategy)
    except Exception as e:  # deepcopy surfaces TypeError, PicklingError, RecursionError...
        raise TypeError(
            "the look-ahead check runs precompute on copies of the strategy "
            f"(copy.deepcopy), and this one cannot be copied: {type(e).__name__}: {e}. "
            "Keep plain data on the strategy (numbers, arrays, frames, a seeded RNG) and "
            "open connections, locks or files inside on_bar, or set check_causality = "
            "false to skip the check"
        ) from e


def _fingerprint(value) -> tuple:
    """(value, its pickle, or None when it has none): what _changed compares."""
    try:
        return value, pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception:  # a lock, a local function, a property: compared by identity
        return value, None


def _changed(before: tuple, after: tuple) -> bool:
    """Whether an attribute changed: by its pickle when both have one (an in-place
    edit keeps the object but not its pickle), else by identity."""
    (u, pu), (v, pv) = before, after
    if pu is not None and pv is not None:
        return pu != pv
    return u is not v


def _state(namespace) -> dict:
    """The fingerprint of each attribute in ``namespace`` but the dunders."""
    return {k: _fingerprint(v) for k, v in dict(namespace).items()
            if not (k.startswith("__") and k.endswith("__"))}


def _set_since(before: dict, after: dict) -> list:
    """The attributes in ``after`` that are new or changed since ``before``."""
    return [k for k in after if k not in before or _changed(before[k], after[k])]


def _class_state(strategy: Strategy) -> list:
    """The attributes of each class the strategy is built from (its MRO, Strategy
    included, but object): shared by every copy the look-ahead check runs, so none may
    change."""
    return [(cls, _state(vars(cls))) for cls in type(strategy).__mro__
            if cls is not object]


def _restore_in_place(value, old) -> bool:
    """Put ``old``'s contents back into ``value`` itself when its type allows (a list,
    dict or set, exactly), so every alias of it sees the restore. False otherwise."""
    if type(value) is not type(old) or type(value) not in (list, dict, set):
        return False
    if isinstance(value, list):
        value[:] = old
    else:
        value.clear()
        value.update(old)
    return True


def _refuse_class_changes(saved: list) -> None:
    """Refuse a precompute that changed a class attribute since ``saved`` (set, added,
    removed or edited in place), after putting every one back as it was.

    The restore is best-effort for an object edited in place: a list, dict or set gets
    its old contents back in place (so aliases of it are restored too); any other
    object is replaced by a copy rebuilt from its pickle, which leaves other references
    to it holding the edit. The run is refused either way."""
    changed = []
    for cls, before in saved:
        after = _state(vars(cls))
        for name in _set_since(before, after) + [k for k in before if k not in after]:
            changed.append(f"{cls.__name__}.{name}")
            if name not in before:
                delattr(cls, name)
                continue
            value, raw = before[name]
            if name in after and after[name][0] is value and raw is not None:
                old = pickle.loads(raw)        # edited in place: its old contents
                if _restore_in_place(value, old):
                    continue
                value = old
            setattr(cls, name, value)
    if changed:
        raise LookaheadError(
            f"the strategy's precompute changed class attribute {', '.join(changed)}. "
            "A class attribute is shared by every copy the look-ahead check runs, so a "
            "value fitted on the full frame (the first call) would reach every prefix "
            "and pass unseen. Keep fitted state on self (each call gets its own copy) "
            "or compute it as a column. Restored to its value before the run (best-effort "
            "for an object edited in place: a list, dict or set is restored in place; "
            "other objects are replaced, and other references to them keep the edit). "
            "Module-level globals are out of the check's reach: keep them out of "
            "precompute")


def _bar_free_cached_property(strategy: Strategy, snapshot: Strategy, name: str,
                              value: tuple) -> bool:
    """True when ``name`` is a functools.cached_property that a fresh copy of the
    strategy, which never saw the bars, computes to the same value: a function of the
    parameters alone, not of the frame."""
    if not isinstance(inspect.getattr_static(type(strategy), name, None),
                      functools.cached_property):
        return False
    try:
        fresh = getattr(copy.deepcopy(snapshot), name)
    except Exception:       # it needs something precompute set: it may read the bars
        return False
    return not _changed(_fingerprint(fresh), value)


def _warn_set_attributes(before: dict, strategy: Strategy, snapshot: Strategy) -> None:
    """Warn, at the caller of Engine.run, about each attribute the traded precompute
    set on ``strategy`` (``before`` is its state just before that call), but a
    cached_property computed from the parameters alone."""
    after = _state(getattr(strategy, "__dict__", {}))
    for name in _set_since(before, after):
        if _bar_free_cached_property(strategy, snapshot, name, after[name]):
            continue
        warnings.warn(
            f"precompute set self.{name}; if it was computed from the bars, on_bar using "
            "it is look-ahead (the check compares precompute's columns, never a value "
            "kept on self): compute per-bar values as columns instead",
            LookaheadWarning, stacklevel=3)


def _warm_up_end(df: pd.DataFrame, data: pd.DataFrame, signal_cols: list) -> int:
    """The row of the first bar with every signal column set (len(data) if none is):
    the bars before it are the warm-up, where on_bar is skipped. A NaN after it is
    refused, naming the column and ts: a plain ValueError when the caller's frame
    already held it, a FrameError when the strategy's precompute made it."""
    if not signal_cols:
        return 0
    nan = data[signal_cols].isna().to_numpy()
    valid = ~nan.any(axis=1)
    if not valid.any():
        return len(data)
    start = int(np.argmax(valid))
    late = nan[start:].any(axis=1)
    if not late.any():
        return start
    row = start + int(np.argmax(late))
    col = signal_cols[int(np.argmax(nan[row]))]
    ts = data["ts"].tolist()
    problem = (
        f"column {col!r} is NaN at row {row} (ts {ts[row]}), after the warm-up ended at "
        f"row {start} (ts {ts[start]}). NaN is allowed only in the leading warm-up, where "
        "on_bar is skipped; skipping it later would silently block that bar's exits. "
        "Fill the gap (say, ffill) or fix the column"
    )
    if col in df.columns and pd.isna(df[col].iloc[row]):
        raise ValueError(problem)
    raise FrameError(f"the strategy's precompute left a NaN: {problem}")


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class Engine:
    """Wires instrument + config + portfolio + broker into the bar loop."""

    def __init__(self, instrument: Instrument, config: BacktestConfig) -> None:
        self.instrument = instrument
        self.config = config

    def run(self, df: pd.DataFrame, strategy: Strategy) -> BacktestResult:
        """Execute the backtest loop over ``df``.

        Checks first: a bad config (``break_on``; a commission, slippage or
            periods_per_year that is not finite) and a bad ``df`` (bar timestamps
            that are not whole unix seconds (a ts of 1e11 or more is named as the
            milliseconds, microseconds or nanoseconds it looks like) or do not
            strictly increase; a column name that is not a Python identifier; in
            session mode, a bad ``session`` column; an open, high, low, close or
            volume that is not a finite number, or a bar whose high or low does not
            hold its open and close; any price <= 0 unless
            ``allow_nonpositive_prices``) are refused with a ValueError, naming the
            row and column, before the strategy's ``precompute`` runs.
            The same frame checks run on what ``precompute`` returns, and a frame
            it broke raises FrameError, naming it; so does a
            precompute that drops, adds or reorders rows, or changes a base column
            (ts, open, high, low, close, volume, session) the caller's frame holds,
            since the fills read those columns.
        Look-ahead guard (``check_causality``, on by default): the strategy is
            snapshotted with ``copy.deepcopy`` before any precompute runs. The traded
            frame is precompute on a fresh copy, on the full frame; each prefix
            ``df[:t+1]`` (``causality_sample="auto"`` samples above 2,000 bars, with a
            warning; see BacktestConfig) runs on its own fresh copy and must match the
            traded frame's first t+1 rows, so what is checked is what is traded and no
            state one call saves (a fit-once mean) reaches another. A leak (a
            ``.shift(-1)``, a whole-sample mean) raises LookaheadError naming the
            column, before any bar runs. ``on_bar`` runs on the strategy passed in,
            which then runs the full frame itself so its state is the traded call's
            (its frame must match the traded one); a leak found in the prefixes
            leaves it as it was passed in. A strategy that cannot be deep-copied is refused with a
            TypeError naming the switch. Class attributes are shared by every copy,
            so a precompute that changes one (a fit-once mean stored on the class) is
            refused with a LookaheadError naming it, and it is put back as it was;
            module-level globals are out of reach. Each attribute the traded
            precompute sets on ``self`` (a full-sample quantile kept for ``on_bar``)
            draws a LookaheadWarning naming it, at the line that called Engine.run: the
            check compares columns, never a value on self. A precompute that raises
            on a short prefix is re-raised as a FrameError naming the prefix and the
            way past it (a warm-up guard, or the switch). The guard checks precompute
            only: ``on_bar`` must decide from ``ctx`` alone, and no check can stop
            strategy code written to read ahead some other way (see strategy.py).
        Decision-at-t: ``on_bar`` is called with bar t's data. ``ctx.position`` is
            a copy of the position for the bar: a strategy's writes to it never
            reach the book (stops move with ``ctx.broker.amend_stop``).
        Fill-at-t+1: any order placed in ``on_bar(t)`` is filled at bar t+1's open.
        Equity at t: the book after bar t's own fills (the open fill, then stops
            against bar t's range), marked at close t; a fill at bar t+1's open
            first shows at t+1, never on the bar that decided it.
        Warm-up: ``on_bar`` is skipped on the leading bars where any signal column
            (any column but the base ones) is NaN, up to the first bar with all of
            them set. A NaN after that bar is refused before any bar runs, naming the
            column and ts (a FrameError when the precompute made it): skipping
            ``on_bar`` there would silently block that bar's exits.
        Breaks: what is a break depends on ``break_on``. ``"gap"`` (default): a gap
            longer than the modal bar interval, or any change of a ``session`` column
            (RTH/ETH labels work here). ``"session"``: only a change of ``session``,
            which must then hold a trading-session ID on every bar, a number that
            never decreases (e.g. the CME trading day as an int YYYYMMDD), not RTH/ETH
            labels. Both modes treat a break the same way.
            ``carry_overnight=False`` (default) means ENTRIES never carry across a
            break: a pending order that would open or add to a position is cancelled.
            It does NOT mean flat overnight: a pending EXIT (an order that reduces
            the position) survives the break and fills at the next session's open,
            as a real exit order would; a flip keeps only its closing contracts (the
            new position's half is an entry); a margin liquidation always survives.
            An open position is never force-flattened at a break: it is held across
            it unless the strategy exits, and an exit decided on a session's last
            bar fills at the next session's open, so that position is held overnight.
            ``carry_overnight=True`` carries every pending order to the next bar. A
            resting protective stop is never touched either way.
        Margin calls (§5.2): after bar t's mark, equity below the maintenance margin of
            the held contracts records a MarginCall in ``margin_calls`` (the call
            restores to INITIAL: amount = initial - equity; no deposit is invented).
            One call per breach: it is logged when equity first drops under, and the
            breach ends when equity is back at or above maintenance or the position
            goes flat or changes (a changed position still under is a new call).
            ``on_margin_call="liquidate"`` (default) then queues a forced market exit
            of the whole position, replacing any order placed on bar t; it fills at
            bar t+1's open (no break cancels it) and its Trade has
            ``margin_liquidation=True``. ``"record"`` keeps the position.
        Open-at-end: a position still open on the last bar is marked-to-last via
            ``portfolio.mark_open_at_end`` (§0.6 — single source of truth).
        """
        self._check_config()
        self._check_frame(df, precomputed=False)       # the caller's own frame
        problem = ohlcv_problem(df)
        if problem is not None:
            raise ValueError(
                f"{problem}. The fills read these prices, so every bar needs finite "
                "open, high, low, close (and volume), with low <= min(open, close) and "
                "high >= max(open, close): a NaN compares false against every stop and "
                "limit, so it would silently fill nothing. validate_bars(df) lists every "
                "bad row")
        if not self.config.allow_nonpositive_prices:
            try:
                assert_positive_prices(df)
            except ValueError as refused:
                raise ValueError(
                    f"{refused} Engine.run refuses such a frame by default: it is usually "
                    "a back-adjusted series pushed through zero. Set "
                    "allow_nonpositive_prices = true only for an instrument that really "
                    "traded there (CL settled at -37.63 on 2020-04-20)"
                ) from refused
        if self.config.check_causality:
            # Snapshot first: the traded call and every prefix call start from their own
            # deepcopy of the strategy as passed in, so state one call saves (a fit-once
            # mean, an RNG's position) never reaches another. Class attributes are
            # shared by the copies, so they must not change at all.
            snapshot = _snapshot(strategy)
            classes = _class_state(strategy)
            traded = copy.deepcopy(snapshot)
            own = _state(getattr(traded, "__dict__", {}))
            data = traded.precompute(df.copy()).reset_index(drop=True)
            _refuse_class_changes(classes)
            _warn_set_attributes(own, traded, snapshot)
        else:
            data = strategy.precompute(df.copy()).reset_index(drop=True)
        self._check_frame(data, precomputed=True)      # what the strategy made of it
        self._check_precompute_kept_the_bars(df, data)
        if self.config.check_causality:
            try:
                self._check_causality(df, strategy, snapshot, data)
            finally:
                _refuse_class_changes(classes)  # by a prefix call, or the last one
        session_breaks = self.config.break_on == "session"
        signal_cols = [c for c in data.columns if c not in _BASE_COLS]
        warm_up_end = _warm_up_end(df, data, signal_cols)

        portfolio = Portfolio(starting_cash=_STARTING_CASH, instrument=self.instrument)
        broker = Broker(self.instrument, self.config, portfolio)

        # Pre-compute the modal in-session bar interval (seconds) for gap detection.
        ts_arr = data["ts"].to_numpy()
        n = len(data)
        if n >= 2:
            diffs = pd.Series(ts_arr[1:] - ts_arr[:-1])
            interval = int(diffs.mode().iloc[0])
        else:
            interval = 0
        has_session = "session" in data.columns

        # Build a namedtuple Row type from the (potentially augmented) column list.
        Row = collections.namedtuple("Row", list(data.columns))
        ts_at = list(data.columns).index("ts")

        equity_index: list[int] = []
        equity_vals: list[float] = []
        margin_calls: list[MarginCall] = []
        liquidate = self.config.on_margin_call == "liquidate"
        breach_qty: int | None = None    # the position the open breach was called on
        state: dict = {}

        prev = None
        for t in range(n):
            row = data.iloc[t]
            values = [row[c] for c in data.columns]
            # ts as a Python int (a whole number, checked above): iloc reads an
            # all-numeric row as float64, which made every Fill and Trade ts a float.
            values[ts_at] = int(values[ts_at])
            bar = Row(*values)

            # Fill the order placed on bar t-1 against THIS bar's open (and stops
            # against this bar's range) before anything reads the book for bar t, so
            # equity[t] holds bar t's own fills and never the next bar's. At a break
            # (see the docstring) cancel_pending_entry runs first and cancels an entry
            # only, in either break_on mode.
            if prev is not None:
                if session_breaks:
                    # Only a change of trading session is a break: a stretch of
                    # bars nobody traded in (omitted by the data vendor) is not.
                    discontinuity = prev.session != bar.session
                else:
                    gap = int(bar.ts) - int(prev.ts)
                    discontinuity = interval > 0 and gap > interval
                    if has_session:
                        discontinuity = discontinuity or (prev.session != bar.session)
                if discontinuity and not self.config.carry_overnight:
                    # §0: an ENTRY never crosses a break; an EXIT does, and fills at
                    # the next session's open as a real exit order would (gap mode
                    # used to drop it too, holding the position overnight and on;
                    # fixed 2026-11-02). A resting protective stop is untouched.
                    broker.cancel_pending_entry(keep_exits=True)
                broker.fill_pending(bar)
            prev = bar

            # Warm-up: skip on_bar until every signal col is set (§5.9); a NaN after
            # that was refused above.
            if t >= warm_up_end:
                ctx = Context(
                    bar=bar,
                    # A copy for this bar: a strategy writing it (qty = 0 to "exit", or
                    # initial_stop to "trail") must never rewrite the book or its R.
                    position=replace(portfolio.position),
                    broker=broker,
                    instrument=self.instrument,
                    tick=self.instrument.tick_size,
                    equity=portfolio.equity(float(bar.close)),
                    state=state,
                )
                strategy.on_bar(ctx)

            # equity[t]: the book after bar t's fills (above), marked at close t. An
            # order placed by on_bar(t) fills at the top of t+1, so it shows from t+1.
            portfolio.mark(float(bar.close))
            equity_index.append(int(bar.ts))
            equity_vals.append(portfolio.equity(float(bar.close)))

            # §5.2: maintenance is the floor, checked on the book marked at close t.
            call = portfolio.margin_call(ts=int(bar.ts))
            if call is None:
                breach_qty = None
            elif breach_qty != portfolio.position.qty:
                # A new breach (not the one already called on this same position): log
                # it once, and queue the forced exit once.
                breach_qty = portfolio.position.qty
                margin_calls.append(call)
                if liquidate:
                    broker.liquidate()

        # §0.6: mark any open position to last close; no broker call — portfolio only.
        if portfolio.position.qty != 0:
            portfolio.mark_open_at_end(ts=int(data.iloc[-1].ts))

        equity_curve = pd.Series(
            equity_vals, index=pd.Index(equity_index, name="ts")
        )
        manifest = self._build_manifest(df)
        return BacktestResult(
            equity_curve=equity_curve,
            trades=portfolio.trades,
            metrics={},
            manifest=manifest,
            margin_calls=margin_calls,
            margin_refusals=len(broker._margin_refused),
        )

    def _check_causality(self, df: pd.DataFrame, strategy: Strategy, snapshot: Strategy,
                         data: pd.DataFrame) -> None:
        """The look-ahead guard. ``data`` is the frame the loop will trade, made by a copy
        of ``snapshot`` (the strategy as passed in). Each prefix df[:t+1] runs on its own
        fresh copy of ``snapshot`` and must match ``data``'s first t+1 rows, so what is
        checked is exactly what is traded. Then the strategy passed in, still in the
        snapshot's state (only copies have run), runs the full frame itself: on_bar runs
        on it, so its state must be the traded call's, and its frame must match ``data``.
        A leak found in the prefixes leaves the strategy as it was passed in."""
        hint = ("Each bar's values must come from that bar and the ones before it. Fix "
                "the precompute (check_causality = false skips this guard; use it only to "
                "study a known leak)")
        def prefix_call(d: pd.DataFrame) -> pd.DataFrame:
            try:
                return copy.deepcopy(snapshot).precompute(d)
            except Exception as e:
                k = len(d)
                raise FrameError(
                    f"the strategy's precompute raised {type(e).__name__}: {e} on the "
                    f"prefix df[:{k}] (its first {k} bar{'' if k == 1 else 's'}), while "
                    "the look-ahead check was running: it reruns precompute on prefixes "
                    "of the bars, the shortest 1 bar long, to prove no bar's values "
                    "depend on later bars. Make the indicator safe on a short frame (a "
                    "warm-up guard: NaN until it has enough bars, say by checking "
                    "len(df) first), or set check_causality = false to skip the check"
                ) from e

        try:
            # stacklevel 4: the sampled-grid warning names the caller of Engine.run
            # (assert_causal <- _check_causality <- run <- caller).
            assert_causal(prefix_call, df,
                          sample=self.config.causality_sample, full=data, stacklevel=4)
        except LookaheadError as leak:
            raise LookaheadError(
                f"the strategy's precompute looks ahead: {leak}. {hint}") from leak
        again = strategy.precompute(df.copy()).reset_index(drop=True)
        if len(again) != len(data):
            raise FrameError(f"the strategy's precompute returned {len(again)} rows for "
                             f"{len(df)} input bars on its second full-frame call: "
                             f"{_ROW_RULE}")
        differs = first_difference(again, data)
        if differs is not None:
            col, row = differs
            where = "is missing" if row is None else f"differs at row {row}"
            raise LookaheadError(
                f"the strategy's precompute gave a different frame on the strategy passed "
                f"in than on its copy (column {col!r} {where}), though both ran the full "
                "frame from the same starting state: it depends on something outside the "
                "strategy and the bars (a class attribute, a global random state, the "
                "clock), so the frame on_bar's state comes from is not the one checked. "
                f"{hint}")

    def _check_frame(self, frame: pd.DataFrame, *, precomputed: bool) -> None:
        """Refuse a frame the loop cannot run: bar timestamps that do not strictly
        increase (checked first: out-of-order bars also scramble the session IDs) or
        are not whole unix seconds (each under its own rule) and,
        in session mode, a broken session column. A problem in the caller's frame is a
        plain ValueError; one in what the strategy's precompute returned is a FrameError
        naming it."""
        rules = [(_time_problem, _TIME_RULE, "removed or broke the bar timestamps"),
                 (_ts_unit_problem, _TS_UNIT_RULE, "removed or broke the bar timestamps")]
        if self.config.break_on == "session":
            rules.append((_session_problem, _SESSION_RULE,
                          "removed or broke the session column"))
        rules.append((_column_problem, _COLUMN_RULE,
                      "made a column name the bar loop cannot use"))
        for find, rule, what in rules:
            problem = find(frame)
            if problem is None:
                continue
            if precomputed:
                raise FrameError(f"the strategy's precompute {what}: {problem}. {rule}")
            raise ValueError(f"{rule}; {problem}")

    @staticmethod
    def _check_precompute_kept_the_bars(df: pd.DataFrame, data: pd.DataFrame) -> None:
        """Refuse a precompute that dropped or added rows, or changed a base column the
        caller's frame holds: the loop fills orders against these bars as given."""
        if len(data) != len(df):
            raise FrameError(f"the strategy's precompute returned {len(data)} rows for "
                             f"{len(df)} input bars: {_ROW_RULE}")
        given = df.reset_index(drop=True)
        for col in sorted(_BASE_COLS & set(given.columns)):
            if col not in data.columns:
                raise FrameError(f"the strategy's precompute removed column {col!r}")
            bad = _first_mismatch(data[col], given[col])
            if bad is not None:
                raise FrameError(
                    f"the strategy's precompute changed column {col!r} at row {bad} "
                    f"(ts {given['ts'].iloc[bad]}): {given[col].iloc[bad]!r} became "
                    f"{data[col].iloc[bad]!r}. Orders fill against the bars as given; "
                    "put a derived value in a new column"
                )

    def _check_config(self) -> None:
        """Refuse a config the engine cannot honour, before any strategy code runs."""
        cfg = self.config
        if cfg.break_on not in ("gap", "session"):
            raise ValueError(f"break_on must be 'gap' or 'session', got {cfg.break_on!r}")
        if cfg.on_margin_call not in ("liquidate", "record"):
            raise ValueError("on_margin_call must be 'liquidate' or 'record', "
                             f"got {cfg.on_margin_call!r}")
        if not math.isfinite(cfg.commission_per_side):
            # NaN or inf would poison every statistic without an error (TOML can hold
            # both); a negative fee is a rebate and stays allowed.
            raise ValueError(
                "commission_per_side must be a finite number (a negative fee is a rebate), "
                f"got {cfg.commission_per_side!r}"
            )
        for name in ("slippage_ticks", "periods_per_year"):
            value = getattr(cfg, name)
            if value is not None and not math.isfinite(value):
                # None is each one's default: the instrument's slippage, and a
                # periods_per_year derived from the data (TOML can hold nan and inf).
                raise ValueError(f"{name} must be a finite number or None, got {value!r}")
        sample = cfg.causality_sample
        if sample is not None and sample != "auto" and (
                isinstance(sample, bool) or not isinstance(sample, int) or sample < 1):
            raise ValueError("causality_sample must be 'auto', None (check every bar) or "
                             f"a whole number of bars >= 1, got {sample!r}")

    def _build_manifest(self, df: pd.DataFrame) -> dict:
        """Build the run manifest; data_sha256 is the sha256 hex digest of the
        canonical CSV bytes of the input frame (§5.10, §0 key name).

        ``periods_per_year`` lives ONLY inside ``manifest["config"]`` (single
        source = config); consumers read it from there, never a top-level dup.
        """
        sha = frame_sha256(df)
        if sha in _SAMPLE_SHA256:
            # §5.6 roll metadata: the shipped sample is a frozen, single-expiry
            # synthetic snapshot. These document the continuous-series policy;
            # full back-adjustment construction is out of v1 scope.
            roll = ("none_single_expiry", "none_single_expiry", "none (single expiry, no roll)",
                    _ROLL_AS_OF_DEFAULT)
        else:
            # Any other frame: the engine cannot know how it was rolled or adjusted.
            roll = (_ROLL_UNKNOWN,) * 4
        return {
            "data_sha256": sha,
            "version": __version__,
            "config": asdict(self.config),
            "instrument": self.instrument.symbol,
            **dict(zip(("roll_trigger", "splice_method", "adjustment_method",
                        "as_of_date"), roll, strict=True)),
        }
