# PlutoBT Sample Data — NOTICE & DATA LICENSE

**This NOTICE governs the data files in `data/sample/`. It is separate from
the repository's MIT code license: the MIT license covers PlutoBT source code,
NOT this data directory.**

## Provenance

- **Source:** SYNTHETIC. These bars are computer-generated, not derived from
  any exchange (CME), vendor feed, or live broker account. No real market data
  is redistributed here, so no exchange/vendor data-licensing terms apply.
- **Instrument modeled:** NQ (E-mini Nasdaq-100) 5-minute RTH bars, as a
  mechanics demonstration only — NOT real prices and not predictive of NQ.
- **Generator:** `scripts/gen_sample.py` (seeded `numpy` random walk,
  `SEED = 20260607`, deterministic and regenerable).
- **As-of date:** 2026-06-07.
- **Frozen:** `nq_5m.json` is a frozen, versioned snapshot. The golden run
  (§5.10) pins it by SHA-256 (manifest key `data_sha256`). Do not edit the file
  by hand; regenerate via the script if the format ever changes, then re-freeze
  the golden expectations (`scripts/freeze_golden.py`).

## Redistribution

You may freely use, copy, modify, and redistribute the files in `data/sample/`,
with or without attribution. Because the data is synthetic, redistribution is
unrestricted.

## Format

`nq_5m.json` is a JSON array of `[ts, open, high, low, close, volume]` rows;
`ts` is unix seconds marking the bar OPEN (§5.5). Prices are snapped to the
NQ 0.25 tick grid and are strictly positive.
