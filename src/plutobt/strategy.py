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
    - Neither keeps the frame: ``precompute`` never stores ``df`` (or anything built
      from it) on ``self`` for ``on_bar`` to read later, since it holds the bars after t.

    What the automatic check covers, and what it does not: it catches look-ahead in
    ``precompute`` (a ``.shift(-1)``, a backfill, a centred window, a whole-sample
    mean), which is the common honest mistake. It is not a sandbox: Python cannot stop
    code written to cheat, so a strategy that hides the bars somewhere and reads ahead
    from ``on_bar`` is not caught. Keeping ``on_bar`` to ``ctx`` is the author's side
    of the contract.
    """

    def precompute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Vectorized, CAUSAL indicator/signal computation.

        Add columns to ``df`` and return it (see the class docstring for the rules).
        With ``check_causality`` on, the engine calls it more than once, each check
        call on its own deepcopy of the strategy as it was passed to ``Engine.run``:
        on the full frame (the frame that is traded), on each prefix ``df[:t+1]``, then
        once more on the strategy itself, so ``on_bar`` sees the traded call's state.
        So nothing one call saves on ``self`` reaches another: a mean fitted "once" on the
        first frame seen is refitted on each prefix, and is caught if it used the whole
        sample. State the strategy already holds when ``Engine.run`` is called is a
        parameter (fit a model on training data before the run, never inside
        ``precompute`` on the bars being tested). Class attributes are shared by every
        copy, so a precompute that changes one is refused (and it is put back); an
        attribute it sets on ``self`` is warned about by name, since the check
        compares columns and ``on_bar`` reading a full-frame value from ``self`` is
        look-ahead: compute per-bar values as columns. A function held on ``self`` is
        shared by the copies, not copied, and module-level globals are not copied
        either, so state either one holds is outside the snapshot and out of the
        check's reach. Base implementation is identity passthrough; subclasses
        override.
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
