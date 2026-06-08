"""Bar loop: decision@t / fill@t+1 open; warm-up skip; discontinuity carry;
manifest build (§2, §5.1, §5.5, §5.9, §5.10)."""
from __future__ import annotations

import collections
import hashlib
from dataclasses import asdict, dataclass

import pandas as pd

from plutobt import __version__
from plutobt.broker import Broker
from plutobt.config import BacktestConfig
from plutobt.instruments import Instrument
from plutobt.portfolio import Portfolio, Trade
from plutobt.strategy import Context, Strategy

# ---------------------------------------------------------------------------
# §0.1: BacktestResult defined ONCE at module scope (consumers import from here)
# ---------------------------------------------------------------------------

@dataclass
class BacktestResult:
    """Backtest output: equity curve, closed/open-at-end trades, metrics, manifest."""
    equity_curve: pd.Series
    trades: list[Trade]
    metrics: dict
    manifest: dict


# ---------------------------------------------------------------------------
# Internal constants
# ---------------------------------------------------------------------------

# Columns that are always present in the raw frame; never treated as signal cols.
_BASE_COLS = frozenset(("ts", "open", "high", "low", "close", "volume", "session"))
_STARTING_CASH = 100_000.0

# v1 default roll/sample vintage; NOT the run date (back-adjustment construction
# is out of v1, §5.6). Stamped into every manifest's roll metadata.
_ROLL_AS_OF_DEFAULT = "2026-06-07"


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

        Decision-at-t: ``on_bar`` is called with bar t's data.
        Fill-at-t+1: any order placed in ``on_bar(t)`` is filled at bar t+1's open.
        Warm-up: ``on_bar`` is skipped while any derived signal column is NaN.
        Discontinuity: a gap > the modal bar interval triggers cancel-by-default
            (``carry_overnight=False``) or carry (``carry_overnight=True``).
        Open-at-end: a position still open on the last bar is marked-to-last via
            ``portfolio.mark_open_at_end`` (§0.6 — single source of truth).
        """
        data = strategy.precompute(df.copy()).reset_index(drop=True)
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
        state: dict = {}

        for t in range(n):
            row = data.iloc[t]
            bar = Row(*[row[c] for c in data.columns])

            # Warm-up: skip on_bar while ANY required signal col is NaN (§5.9).
            warm = any(pd.isna(getattr(bar, c)) for c in signal_cols)

            if not warm:
                ctx = Context(
                    bar=bar,
                    position=portfolio.position,
                    broker=broker,
                    instrument=self.instrument,
                    tick=self.instrument.tick_size,
                    equity=portfolio.equity(float(bar.close)),
                    state=state,
                )
                strategy.on_bar(ctx)

            # Fill pending order against next bar's OPEN (t+1).
            # Detect a session discontinuity; on gap+default policy, cancel the
            # unfilled entry order but leave any resting protective stop intact.
            if t + 1 < n:
                nxt = data.iloc[t + 1]
                gap = int(nxt.ts) - int(bar.ts)
                discontinuity = interval > 0 and gap > interval
                if has_session:
                    discontinuity = discontinuity or (bar.session != nxt.session)
                if discontinuity and not self.config.carry_overnight:
                    # §0: cancel_pending_entry drops the unfilled entry; resting
                    # protective stop on a held position is deliberately untouched.
                    broker.cancel_pending_entry()
                broker.fill_pending(nxt)

            portfolio.mark(float(bar.close))
            equity_index.append(int(bar.ts))
            equity_vals.append(portfolio.equity(float(bar.close)))

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
        )

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
