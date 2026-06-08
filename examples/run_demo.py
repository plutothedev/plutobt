"""End-to-end PlutoBT demo: load -> run EMACross -> metrics report.

This is the video payoff. One engine, swap symbol + per-instrument cost config to
show the engine generalizes while COSTS DO NOT auto-carry.
"""
from __future__ import annotations

from plutobt.config import load_config
from plutobt.data import load_bars
from plutobt.engine import BacktestResult, Engine
from plutobt.instruments import get_instrument
from plutobt.metrics import compute_metrics
from plutobt.report import format_report
from plutobt.strategies.ema_cross import EMACross

# Symbol -> (sample data file, per-instrument cost config).
# Each symbol uses its own realistic synthetic price file (distinct seeds/price levels).
DEMO_TARGETS: dict[str, tuple[str, str]] = {
    "NQ": ("data/sample/nq_5m.json", "configs/nq.toml"),
    "ES": ("data/sample/es_5m.json", "configs/es.toml"),
    "CL": ("data/sample/cl_5m.json", "configs/cl.toml"),
}


def run_symbol(symbol: str, data_path: str, config_path: str) -> BacktestResult:
    instrument = get_instrument(symbol)
    config = load_config(config_path)
    df = load_bars(data_path)
    engine = Engine(instrument, config)
    result = engine.run(df, EMACross())
    # Ensure metrics are assembled + labeled IN-SAMPLE with the run's annualization basis.
    # periods_per_year lives in the config (single source), not a top-level manifest dup.
    ppy = result.manifest["config"]["periods_per_year"]
    result.metrics = compute_metrics(result, periods_per_year=ppy)
    return result


def run_demo_all() -> dict[str, BacktestResult]:
    """Run the EMACross demo across every DEMO_TARGETS symbol.

    Demonstrates GENERALIZATION: identical engine + strategy, swapped symbol and
    per-instrument cost config. The engine carries; the costs do not.
    """
    results: dict[str, BacktestResult] = {}
    for symbol, (data_path, config_path) in DEMO_TARGETS.items():
        results[symbol] = run_symbol(symbol, data_path=data_path, config_path=config_path)
    return results


def main() -> None:
    for symbol, result in run_demo_all().items():
        print(format_report(result, symbol=symbol))
        print()


if __name__ == "__main__":
    main()
