"""Freeze the golden expectations from the current engine output (§5.10/§10.8).

Run ONCE on the pinned stack (numpy==1.26.4, pandas==2.2.2). Re-run only after
a deliberate, reviewed change to the sample, config, or engine — never to mask
an unexplained drift. Emits configs/golden_expected.json.
"""
import json
from pathlib import Path

from plutobt.config import load_config
from plutobt.data import load_bars
from plutobt.engine import Engine
from plutobt.instruments import get_instrument
from plutobt.metrics import compute_metrics

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "data" / "sample" / "nq_5m.json"
GOLDEN_CFG = ROOT / "configs" / "golden.toml"
OUT = ROOT / "configs" / "golden_expected.json"


def _import_golden_strategy():
    import sys
    sys.path.insert(0, str(ROOT))
    from tests.conftest import GoldenStrategy
    return GoldenStrategy


def main() -> None:
    GoldenStrategy = _import_golden_strategy()
    df = load_bars(str(SAMPLE))
    inst = get_instrument("NQ")
    cfg = load_config(str(GOLDEN_CFG))
    result = Engine(inst, cfg).run(df, GoldenStrategy())
    metrics = compute_metrics(result, periods_per_year=cfg.periods_per_year)
    numeric = {
        k: v for k, v in metrics.items()
        if isinstance(v, (int, float)) and not isinstance(v, bool)
    }
    payload = {
        "equity_ts": [int(x) for x in result.equity_curve.index],
        "equity_values": [float(x) for x in result.equity_curve.to_numpy()],
        "metrics": numeric,
        "manifest": {
            "data_sha256": result.manifest["data_sha256"],
            "adjustment_method": result.manifest["adjustment_method"],
            "splice_method": result.manifest["splice_method"],
            "roll_trigger": result.manifest["roll_trigger"],
            "as_of_date": result.manifest["as_of_date"],
        },
    }
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                   encoding="utf-8")
    print(f"froze {len(payload['equity_ts'])} equity points + "
          f"{len(numeric)} metrics -> {OUT}")


if __name__ == "__main__":
    main()
