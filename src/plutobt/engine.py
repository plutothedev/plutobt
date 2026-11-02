"""Bar loop: decision@t / fill@t+1 open; warm-up skip; discontinuity carry;
manifest build (§2, §5.1, §5.5, §5.9, §5.10)."""
from __future__ import annotations

import collections
import hashlib
import math
import numbers
from dataclasses import asdict, dataclass, field, replace

import pandas as pd

from plutobt import __version__
from plutobt.broker import Broker, FrameError
from plutobt.config import BacktestConfig
from plutobt.instruments import Instrument
from plutobt.portfolio import MarginCall, Portfolio, Trade
from plutobt.strategy import Context, Strategy

# ---------------------------------------------------------------------------
# §0.1: BacktestResult defined ONCE at module scope (consumers import from here)
# ---------------------------------------------------------------------------

@dataclass
class BacktestResult:
    """Backtest output: equity curve, closed/open-at-end trades, metrics, manifest, and
    every margin call the run hit (spec §5.2), in bar order."""
    equity_curve: pd.Series
    trades: list[Trade]
    metrics: dict
    manifest: dict
    margin_calls: list[MarginCall] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Internal constants
# ---------------------------------------------------------------------------

# Columns that are always present in the raw frame; never treated as signal cols.
_BASE_COLS = frozenset(("ts", "open", "high", "low", "close", "volume", "session"))
_STARTING_CASH = 100_000.0

# v1 default roll/sample vintage; NOT the run date (back-adjustment construction
# is out of v1, §5.6). Stamped into every manifest's roll metadata.
_ROLL_AS_OF_DEFAULT = "2026-06-07"

# The time-order rule, quoted by every refusal of a frame's bar order.
_TIME_RULE = ("bar timestamps (the 'ts' column) must strictly increase, since the engine "
              "runs the bars in row order")

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
    ts = frame["ts"].tolist()
    for i in range(1, len(ts)):
        if not ts[i] > ts[i - 1]:      # written so that a NaN ts is refused too
            return f"row {i} (ts {ts[i]}) is not after row {i - 1} (ts {ts[i - 1]})"
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
            that do not strictly increase; in session mode, a bad ``session``
            column) are refused with a ValueError before the strategy's
            ``precompute`` runs. The same frame checks run on what ``precompute``
            returns, and a frame it broke raises FrameError, naming it.
        Decision-at-t: ``on_bar`` is called with bar t's data. ``ctx.position`` is
            a copy of the position for the bar: a strategy's writes to it never
            reach the book (stops move with ``ctx.broker.amend_stop``).
        Fill-at-t+1: any order placed in ``on_bar(t)`` is filled at bar t+1's open.
        Equity at t: the book after bar t's own fills (the open fill, then stops
            against bar t's range), marked at close t; a fill at bar t+1's open
            first shows at t+1, never on the bar that decided it.
        Warm-up: ``on_bar`` is skipped while any derived signal column is NaN.
        Breaks: at a break the pending order is cancelled (``carry_overnight=False``,
            the default) or carried to the next bar (``carry_overnight=True``); a
            resting protective stop is never touched. ``break_on="gap"`` (default):
            a gap longer than the modal bar interval, or any change of a ``session``
            column (RTH/ETH labels work here), is a break, and the pending order is
            cancelled even when it is an exit (known; the golden run depends on it).
            ``break_on="session"``: only a change of ``session`` is a break; an entry
            is cancelled, an exit (an order that reduces the position) survives and
            fills at the next session's open, and a flip keeps only its closing
            contracts. ``session`` must then hold a trading-session ID on every bar,
            a number that never decreases (e.g. the CME trading day as an int
            YYYYMMDD), not RTH/ETH labels.
        Margin calls (§5.2): after bar t's mark, equity below the maintenance margin of
            the held contracts records a MarginCall in ``margin_calls`` (the call
            restores to INITIAL: amount = initial - equity; no deposit is invented).
            ``on_margin_call="liquidate"`` (default) then queues a forced market exit
            of the whole position, replacing any order placed on bar t; it fills at
            bar t+1's open (no break cancels it) and its Trade has
            ``margin_liquidation=True``. ``"record"`` keeps the position.
        Open-at-end: a position still open on the last bar is marked-to-last via
            ``portfolio.mark_open_at_end`` (§0.6 — single source of truth).
        """
        self._check_config()
        self._check_frame(df, precomputed=False)       # the caller's own frame
        data = strategy.precompute(df.copy()).reset_index(drop=True)
        self._check_frame(data, precomputed=True)      # what the strategy made of it
        session_breaks = self.config.break_on == "session"
        signal_cols = [c for c in data.columns if c not in _BASE_COLS]

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

        equity_index: list[int] = []
        equity_vals: list[float] = []
        margin_calls: list[MarginCall] = []
        liquidate = self.config.on_margin_call == "liquidate"
        state: dict = {}

        prev = None
        for t in range(n):
            row = data.iloc[t]
            bar = Row(*[row[c] for c in data.columns])

            # Fill the order placed on bar t-1 against THIS bar's open (and stops
            # against this bar's range) before anything reads the book for bar t, so
            # equity[t] holds bar t's own fills and never the next bar's. At a break
            # (see the docstring) cancel_pending_entry runs first: gap mode cancels
            # the pending order, exits included; session mode cancels an entry only.
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
                    # §0: gap mode drops the pending order, exits too (known, kept for
                    # the golden run); session mode keeps an EXIT, which fills at the
                    # next session's open as a real exit order would. A resting
                    # protective stop on a held position is untouched either way.
                    broker.cancel_pending_entry(keep_exits=session_breaks)
                broker.fill_pending(bar)
            prev = bar

            # Warm-up: skip on_bar while ANY required signal col is NaN (§5.9).
            warm = any(pd.isna(getattr(bar, c)) for c in signal_cols)

            if not warm:
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
            if call is not None:
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
        )

    def _check_frame(self, frame: pd.DataFrame, *, precomputed: bool) -> None:
        """Refuse a frame the loop cannot run: bar timestamps that do not strictly
        increase (checked first: out-of-order bars also scramble the session IDs) and,
        in session mode, a broken session column. A problem in the caller's frame is a
        plain ValueError; one in what the strategy's precompute returned is a FrameError
        naming it."""
        rules = [(_time_problem, _TIME_RULE, "the bar timestamps")]
        if self.config.break_on == "session":
            rules.append((_session_problem, _SESSION_RULE, "the session column"))
        for find, rule, what in rules:
            problem = find(frame)
            if problem is None:
                continue
            if precomputed:
                raise FrameError(
                    f"the strategy's precompute removed or broke {what}: {problem}. {rule}"
                )
            raise ValueError(f"{rule}; {problem}")

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

    def _build_manifest(self, df: pd.DataFrame) -> dict:
        """Build the run manifest; data_sha256 is the sha256 hex digest of the
        canonical CSV bytes of the input frame (§5.10, §0 key name).

        ``periods_per_year`` lives ONLY inside ``manifest["config"]`` (single
        source = config); consumers read it from there, never a top-level dup.
        """
        raw = df.to_csv(index=False).encode("utf-8")
        return {
            "data_sha256": hashlib.sha256(raw).hexdigest(),
            "version": __version__,
            "config": asdict(self.config),
            "instrument": self.instrument.symbol,
            # §5.6 roll metadata: the shipped sample is a frozen, single-expiry
            # synthetic snapshot. These document the continuous-series policy;
            # full back-adjustment construction is out of v1 scope.
            "roll_trigger": "none_single_expiry",
            "splice_method": "none_single_expiry",
            "adjustment_method": "backwards_difference",
            "as_of_date": _ROLL_AS_OF_DEFAULT,
        }
