"""Did you overfit? A study of the demo strategy over a small grid (PlutoBT 3.0).

The data is the repo's SYNTHETIC NQ sample (a seeded random walk, see data/sample/NOTICE.md),
so this is not a real-data parameter sweep: it shows what the tools print, on bars with no
edge to find. Six EMA settings run once each, every run lands in the ledger, the best is
deflated against every try, and the pick is tested out-of-sample.

Run:  PYTHONPATH=src python examples/overfit_study.py [--ledger PATH]
"""
from __future__ import annotations

import argparse
import pathlib

from plutobt import format_study, get_instrument, load_bars, load_config, run_study
from plutobt.ledger import DEFAULT_PATH
from plutobt.strategies.ema_cross import EMACross

ROOT = pathlib.Path(__file__).resolve().parents[1]
GRID = {"fast_span": [10, 20, 30], "slow_span": [50, 100]}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--ledger", default=DEFAULT_PATH,
                   help=f"the trials ledger (default {DEFAULT_PATH})")
    a = p.parse_args(argv)
    df = load_bars(ROOT / "data" / "sample" / "nq_5m.json")
    cfg = load_config(str(ROOT / "configs" / "nq.toml"))
    result = run_study(EMACross, GRID, df, get_instrument("NQ"), cfg, ledger=a.ledger)
    print(format_study(result, "NQ"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
