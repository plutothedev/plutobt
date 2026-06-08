# PlutoBT Sample Data — NOTICE & DATA LICENSE

**This NOTICE governs the data files in `data/sample/`. It is separate from
the repository's MIT code license: the MIT license covers PlutoBT source code,
NOT this data directory.**

## Provenance

All files in this directory are **SYNTHETIC**. The bars are computer-generated,
not derived from any exchange (CME), vendor feed, or live broker account. No
real market data is redistributed here, so no exchange/vendor data-licensing
terms apply.

### nq_5m.json — NQ E-mini Nasdaq-100 5-minute bars

- **Instrument modeled:** NQ, 5-minute RTH bars (mechanics demonstration only)
- **Base price:** ~18 000, tick 0.25
- **Generator:** `scripts/gen_sample.py`, seed `20260607`, deterministic
- **As-of date:** 2026-06-07
- **Frozen:** `nq_5m.json` is the primary golden-run input. The golden test
  (§5.10) pins it by SHA-256 (`manifest["data_sha256"]` is the SHA-256 of the
  CSV-encoded parsed frame, not the raw JSON bytes). Do not edit by hand;
  regenerate via the script if the format changes, then re-freeze golden
  expectations (`scripts/freeze_golden.py`).

### es_5m.json — ES E-mini S&P 500 5-minute bars

- **Instrument modeled:** ES, 5-minute RTH bars (mechanics demonstration only)
- **Base price:** ~5 000, tick 0.25
- **Generator:** `scripts/gen_sample.py`, seed `20260608`, deterministic
- **As-of date:** 2026-06-07
- **Purpose:** Generalization demo — realistic ES price levels (~$50/pt) vs NQ
  (~$20/pt). Running the same strategy on ES vs NQ with distinct data files
  demonstrates that costs and price levels are correctly decoupled.

### cl_5m.json — CL WTI Crude Oil 5-minute bars

- **Instrument modeled:** CL WTI Crude, 5-minute bars (mechanics demonstration only)
- **Base price:** ~70, tick 0.01
- **Generator:** `scripts/gen_sample.py`, seed `20260609`, deterministic
- **As-of date:** 2026-06-07
- **Purpose:** Generalization demo — realistic CL price levels (~$1 000/pt) and
  volatility vs the equity index instruments. Distinct data avoids the nonsense
  of running ES/CL through NQ price bars.

## Redistribution

You may freely use, copy, modify, and redistribute the files in `data/sample/`,
with or without attribution. Because all data is synthetic, redistribution is
unrestricted.

## Format

All files are JSON arrays of `[ts, open, high, low, close, volume]` rows.
`ts` is unix seconds marking the bar OPEN (§5.5). Prices are snapped to each
instrument's tick grid and are strictly positive. 324 bars (4 RTH sessions,
81 bars/day at 5-minute intervals).
