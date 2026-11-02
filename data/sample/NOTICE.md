# PlutoBT Sample Data: NOTICE & DATA LICENSE

**This NOTICE governs the data files in `data/sample/`. It is separate from
the repository's MIT code license: the MIT license covers PlutoBT source code,
NOT this data directory.**

## Provenance

All files in this directory are **SYNTHETIC**. The bars are computer-generated
by a seeded random walk, not derived from any exchange (CME), vendor feed, or
live broker account. No real market data is redistributed here, so no
exchange/vendor data-licensing terms apply. The dates and session times are
real calendar dates and real exchange hours; the prices and volumes are not.

### Session grid (all three files)

- **Bars:** 5-minute, REGULAR TRADING HOURS only, on each instrument's session
  from the registry in `src/plutobt/instruments.py` (America/Chicago clock).
- **Trading days:** 20, Monday 2026-03-02 to Friday 2026-03-27. The range
  crosses four weekends and the US DST change on Sunday 2026-03-08 (CT moves
  from UTC-6 to UTC-5, so the same 08:30 CT open is an hour earlier in UTC).
- **No bars outside RTH:** each session stops at its close and the next one
  opens the following trading day, with an overnight price gap.
- **Generator:** `scripts/gen_sample.py`, deterministic (one seed per file).
  Regenerating reproduces every byte; `tests/test_sample_data.py` pins that.
- **As-of date:** 2026-11-02 (the sample vintage stamped in run manifests: the date
  the three files were regenerated on the RTH session grid).

### nq_5m.json: NQ E-mini Nasdaq-100

- **Session:** 08:30-15:15 CT, 81 bars/day, **1,620 bars**
- **Base price:** 18 000, tick 0.25; per-bar sigma 6 pt, overnight sigma 36 pt
- **Seed:** `20260607`
- **Frozen:** `nq_5m.json` is the golden-run input. The golden test (§5.10)
  pins it by SHA-256 (`manifest["data_sha256"]` is the SHA-256 of the
  CSV-encoded parsed frame, not the raw JSON bytes). Do not edit by hand;
  regenerate via the script, then re-freeze the golden expectations
  (`scripts/freeze_golden.py`).

### es_5m.json: ES E-mini S&P 500

- **Session:** 08:30-15:15 CT, 81 bars/day, **1,620 bars**
- **Base price:** 5 000, tick 0.25; per-bar sigma 2 pt, overnight sigma 12 pt
- **Seed:** `20260608`
- **Purpose:** generalization demo: ES price levels (~$50/pt) vs NQ (~$20/pt),
  so the same strategy runs on distinct data with its own costs.

### cl_5m.json: CL WTI Crude Oil

- **Session:** 08:00-13:30 CT, 66 bars/day, **1,320 bars**
- **Base price:** 70, tick 0.01; per-bar sigma 0.25, overnight sigma 1.5
- **Seed:** `20260609`
- **Purpose:** generalization demo: CL price levels (~$1 000/pt), its own
  session hours and its own costs. This seed's path happens to fall from 70 to
  about 44 over the month; it is a random walk, not a forecast.

## Redistribution

You may freely use, copy, modify, and redistribute the files in `data/sample/`,
with or without attribution. Because all data is synthetic, redistribution is
unrestricted.

## Format

All files are JSON arrays of `[ts, open, high, low, close, volume]` rows, one
bar per line, LF line endings. `ts` is unix seconds marking the bar OPEN (§5.5).
Prices are snapped to each instrument's tick grid and are strictly positive.
