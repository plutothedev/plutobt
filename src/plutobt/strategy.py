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

    ``bar`` is an immutable namedtuple row view of bar t. ``position`` is
    always a typed ``Position`` (never None). ``state`` is a free-form dict
    persisted across bars for sanctioned mutable strategy state.
    """

    bar: object
    position: Position
    broker: Broker
    instrument: Instrument
    tick: float
    equity: float
    state: dict


class Strategy:
    """Base strategy: vectorized causal ``precompute`` + stateful ``on_bar``."""

    def precompute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Vectorized, CAUSAL, STATELESS indicator/signal computation.

        Base implementation is identity passthrough; subclasses override.
        """
        return df

    def on_bar(self, ctx: Context) -> None:
        """Per-bar event hook; place orders via ``ctx.broker``. No-op by default."""
        return None
