"""Strategy base class and the per-bar read-only Context (§4, §5.1)."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from plutobt.broker import Broker
from plutobt.instruments import Instrument
from plutobt.portfolio import Position


@dataclass
class Context:
    """Rebuilt each bar; bar data is read-only (§5.10).

    ``bar`` is an immutable namedtuple row view of bar t. ``position`` is a
    read-only snapshot of the position for this bar, always a typed ``Position``
    (never None): it is a copy, so writing to it changes nothing. Move a stop
    with ``ctx.broker.amend_stop(price)`` and exit with ``ctx.broker.close()``.
    ``state`` is a free-form dict persisted across bars for sanctioned mutable
    strategy state.
    """

    bar: object
    position: Position
    broker: Broker
    instrument: Instrument
    tick: float
    equity: float
    state: dict


class Strategy:
    """Base strategy: vectorized causal ``precompute`` + stateful ``on_bar``.

    The look-ahead contract, which "no look-ahead by construction" rests on:

    - ``precompute(df)`` returns ``df`` with derived columns added: one row per bar, in
      order, base columns (ts, open, high, low, close, volume, session) unchanged. Row
      t's values may use bars 0..t only. ``Engine.run`` enforces all of this before
      the first bar (``check_causality``).
    - ``on_bar(ctx)`` may use only what ``ctx`` holds: bar t's row (``ctx.bar``), the
      position, equity and the ``ctx.state`` it wrote on earlier bars.
    - Neither keeps the frame: ``precompute`` is STATELESS, so it never stores ``df``
      (or anything built from it) on ``self`` for ``on_bar`` to read later: it could
      index past bar t. ``Engine.run`` refuses a strategy left holding a DataFrame,
      Series or array as long as the bars (directly, or in a list, tuple or dict
      attribute); a frame hidden deeper is still outside what any check can see.
    """

    def precompute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Vectorized, CAUSAL, STATELESS indicator/signal computation.

        Add columns to ``df`` and return it (see the class docstring for the rules).
        The engine calls it more than once: on the full frame, then on prefixes
        ``df[:t+1]`` for the causality check. Base implementation is identity
        passthrough; subclasses override.
        """
        return df

    def on_bar(self, ctx: Context) -> None:
        """Per-bar event hook; place orders via ``ctx.broker``. No-op by default.

        Decide from ``ctx`` only (bar t's row, position, equity, ``ctx.state``); never
        from a frame kept from ``precompute``, which holds the bars after t.
        ``ctx.position`` is a read-only snapshot: stops move with
        ``ctx.broker.amend_stop(...)`` and exits go through ``ctx.broker.close()``
        (or an order), never through writes to the position.
        Protect an entry by attaching its stop to the order (``buy(1, stop=...)``),
        above all a limit entry: a stop set with ``amend_stop`` while flat lasts only
        for the next bar, so a limit that fills later would be unprotected.
        """
        return None
