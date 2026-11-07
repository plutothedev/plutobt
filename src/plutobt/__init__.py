"""PlutoBT: a hybrid futures backtesting engine.

Vectorized signals + event-driven execution, realistic futures fills/margin,
verified CME contract specs, and (2.0) back-adjusted continuous contracts.

Public API
----------
    Engine             : run a backtest over a bar DataFrame
    BacktestResult     : the result returned by Engine.run
    BacktestConfig     : cost/config parameters for a backtest
    load_config        : load a BacktestConfig from a TOML file
    Strategy           : base class for user strategies
    Context            : per-bar context passed to Strategy.on_bar
    get_instrument     : look up a contract spec by symbol
    REGISTRY           : all registered Instrument specs
    load_bars          : load OHLCV bars from a JSON file
    load_csv           : load one contract's bars from a vendor CSV (presets)
    load_csv_contracts : load a file holding one or more contracts
    ContractSet        : a validated set of one root's contracts
    build_continuous   : build a back-adjusted continuous series and its RollMeta
    save_continuous    : write a series and its .roll.json metadata
    load_continuous    : read them back, refusing an edited series
    slice_continuous   : a date range of a series, metadata re-pinned
    RollMeta           : how a continuous series was built
    compute_metrics    : assemble the full metrics bundle
    format_report      : render a human-readable IN-SAMPLE report
"""

__version__ = "2.0.0"

# Import individual names to populate the package namespace.
# Order matters for readability; keep lowest-level modules first.
from plutobt.config import BacktestConfig, load_config
from plutobt.contracts import ContractSet
from plutobt.data import load_bars
from plutobt.engine import BacktestResult, Engine
from plutobt.importers import load_csv, load_csv_contracts
from plutobt.instruments import REGISTRY, get_instrument
from plutobt.metrics import compute_metrics
from plutobt.report import format_report
from plutobt.roll import (
    RollMeta,
    build_continuous,
    load_continuous,
    save_continuous,
    slice_continuous,
)
from plutobt.strategy import Context, Strategy

__all__ = [
    "__version__",
    "Engine",
    "BacktestResult",
    "BacktestConfig",
    "load_config",
    "Strategy",
    "Context",
    "get_instrument",
    "REGISTRY",
    "load_bars",
    "load_csv",
    "load_csv_contracts",
    "ContractSet",
    "build_continuous",
    "save_continuous",
    "load_continuous",
    "slice_continuous",
    "RollMeta",
    "compute_metrics",
    "format_report",
]
