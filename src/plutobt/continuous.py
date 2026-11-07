"""Build a back-adjusted continuous series from a folder of contract files (spec §6.5).

    python -m plutobt.continuous data/sample/nq_contracts --preset generic \\
        --root NQ --ts-is open -o nq_cont.csv

Prints the roll table (day, from, to, gap in points and dollars) and writes the series
plus its ``.roll.json`` metadata. A refusal prints its reason and exits with status 2.
"""
from __future__ import annotations

import argparse
import json
import sys

from plutobt.contracts import ContractSet
from plutobt.importers import PRESETS
from plutobt.roll import TRIGGERS, build_continuous, save_continuous, trading_day


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m plutobt.continuous",
        description="Build a back-adjusted continuous futures series from one file per "
                    "contract.")
    p.add_argument("folder", help="folder holding one .csv/.txt file per contract")
    p.add_argument("--preset", choices=PRESETS, required=True)
    p.add_argument("--root", required=True, help="contract root, e.g. NQ")
    p.add_argument("--tz", default=None,
                   help="IANA zone for timestamps written without one, e.g. "
                        "America/Chicago")
    p.add_argument("--ts-is", choices=("open", "close"), default=None,
                   help="whether each timestamp is the bar's open or close (required "
                        "for --preset generic)")
    p.add_argument("--trigger", choices=TRIGGERS, default="calendar")
    p.add_argument("--roll-dates", default=None,
                   help='JSON file mapping each outgoing contract to its roll day, e.g. '
                        '{"CLZ5": "2025-11-14"}')
    p.add_argument("-o", "--out", required=True, help="output CSV path")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        load_kwargs = {"tz": args.tz, "ts_is": args.ts_is}
        cs = ContractSet.from_folder(args.folder, args.preset, args.root, **load_kwargs)
        roll_dates = None
        if args.roll_dates:
            with open(args.roll_dates, encoding="utf-8") as fh:
                roll_dates = json.load(fh)
        df, meta = build_continuous(cs, args.trigger, roll_dates)
        csv_path, json_path = save_continuous(df, meta, args.out)
    except (ValueError, OSError) as refused:
        print(f"continuous: {refused}", file=sys.stderr)
        return 2
    print(f"{meta.root}: {len(cs.contracts)} contracts, {len(meta.rolls)} rolls "
          f"({meta.roll_trigger} trigger, backward additive)")
    print(f"{'roll day':<12}{'from':<10}{'to':<10}{'gap pts':>10}{'gap $':>12}")
    for r in meta.rolls:
        print(f"{trading_day(r['roll_ts']).isoformat():<12}{r['from']:<10}{r['to']:<10}"
              f"{r['gap_points']:>10}{r['gap_dollars']:>12}")
    print(f"wrote {len(df)} bars -> {csv_path}")
    print(f"wrote metadata -> {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
