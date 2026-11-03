"""PlutoBT — a hybrid futures backtesting engine.

Vectorized signals + event-driven execution, realistic futures fills/margin,
verified CME contract specs.

Public API
----------
    Engine          — run a backtest over a bar DataFrame
    BacktestResult  — the result returned by Engine.run
    BacktestConfig  — cost/config parameters for a backtest
    load_config     — load a BacktestConfig from a TOML file
    Strategy        — base class for user strategies
    Context         — per-bar context passed to Strategy.on_bar
    get_instrument  — look up a contract spec by symbol
    REGISTRY        — all registered Instrument specs
    load_bars       — load OHLCV bars from a JSON file
    compute_metrics — assemble the full metrics bundle
    format_report   — render a human-readable IN-SAMPLE report
"""

__version__ = "0.3.0"

# Import individual names to populate the package namespace.
# Order matters for readability; keep lowest-level modules first.
from plutobt.config import BacktestConfig, load_config
from plutobt.data import load_bars
from plutobt.engine import BacktestResult, Engine
from plutobt.instruments import REGISTRY, get_instrument
from plutobt.metrics import compute_metrics
from plutobt.report import format_report
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
    "compute_metrics",
    "format_report",
]
