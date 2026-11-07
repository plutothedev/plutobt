"""ContractSet: a validated set of one root's contracts (spec §5)."""

import numpy as np
import pandas as pd
import pytest

from plutobt.contracts import Contract, ContractSet

T0 = 1_767_277_800          # 2026-01-01 14:30 UTC, arbitrary anchor


def bars(start: int, n: int = 20, interval: int = 300, price: float = 18000.0,
         tick: float = 0.25) -> pd.DataFrame:
    """n on-grid bars from ``start``, rising one tick a bar."""
    close = price + np.arange(n) * tick
    return pd.DataFrame({
        "ts": start + np.arange(n, dtype=np.int64) * interval,
        "open": close - tick, "high": close + tick, "low": close - 2 * tick,
        "close": close, "volume": np.full(n, 100, dtype=np.int64),
    })


def quarter_frames(*codes, interval=300):
    out, start = {}, T0
    for code in codes:
        out[code] = bars(start, interval=interval)
        start += 10 * interval          # each contract overlaps the next by 10 bars
    return out


def test_from_frames_sorts_by_expiry():
    cs = ContractSet.from_frames(quarter_frames("NQZ2026", "NQH2026", "NQU2026",
                                                "NQM2026"))
    assert cs.root == "NQ"
    assert cs.contracts == [Contract("NQ", 3, 2026), Contract("NQ", 6, 2026),
                            Contract("NQ", 9, 2026), Contract("NQ", 12, 2026)]
    assert list(cs.frames) == ["NQH2026", "NQM2026", "NQU2026", "NQZ2026"]
    assert cs.interval == 300
    assert cs.file_sha256 == {}


def test_short_codes_are_canonicalised():
    cs = ContractSet.from_frames(quarter_frames("NQH26", "NQM26"))
    assert list(cs.frames) == ["NQH2026", "NQM2026"]


def test_mixed_roots_refused():
    frames = quarter_frames("NQH2026", "ESM2026")
    with pytest.raises(ValueError, match="ES") as err:
        ContractSet.from_frames(frames)
    assert "NQ" in str(err.value)


def test_duplicate_contract_refused():
    frames = quarter_frames("NQZ2025", "NQZ25")
    with pytest.raises(ValueError, match="NQZ2025"):
        ContractSet.from_frames(frames)


def test_calendar_chain_gap_refused():
    frames = quarter_frames("NQH2025", "NQM2025", "NQZ2025")
    with pytest.raises(ValueError, match="NQU2025"):
        ContractSet.from_frames(frames)


def test_non_calendar_root_needs_overlap():
    frames = {"CLF2026": bars(T0), "CLG2026": bars(T0 + 100 * 300)}
    with pytest.raises(ValueError, match="overlap") as err:
        ContractSet.from_frames(frames)
    assert "CLF2026" in str(err.value) and "CLG2026" in str(err.value)


def test_empty_contract_refused():
    frames = quarter_frames("NQH2026", "NQM2026")
    frames["NQM2026"] = frames["NQM2026"].iloc[:0]
    with pytest.raises(ValueError, match="NQM2026"):
        ContractSet.from_frames(frames)


def test_single_contract_refused():
    with pytest.raises(ValueError, match="two"):
        ContractSet.from_frames(quarter_frames("NQH2026"))


def test_mixed_intervals_refused():
    frames = quarter_frames("NQH2026", "NQM2026")
    frames["NQM2026"] = bars(T0 + 3000, n=100, interval=60)
    with pytest.raises(ValueError, match="NQM2026") as err:
        ContractSet.from_frames(frames)
    assert "60" in str(err.value) and "300" in str(err.value)


def test_off_tick_price_refused():
    frames = quarter_frames("NQH2026", "NQM2026")
    frames["NQM2026"].loc[3, "close"] = 18000.6
    frames["NQM2026"].loc[3, "high"] = 18001.0
    with pytest.raises(ValueError, match="tick") as err:
        ContractSet.from_frames(frames)
    msg = str(err.value)
    assert "NQM2026" in msg and "close" in msg and "row 3" in msg


def test_bad_bar_refused_naming_contract():
    frames = quarter_frames("NQH2026", "NQM2026")
    frames["NQH2026"].loc[2, "high"] = 1.0
    with pytest.raises(ValueError, match="NQH2026"):
        ContractSet.from_frames(frames)


def test_contract_column_must_match_key():
    frames = quarter_frames("NQH2026", "NQM2026")
    frames["NQM2026"]["contract"] = "NQU2026"
    with pytest.raises(ValueError, match="NQU2026"):
        ContractSet.from_frames(frames)


def test_from_folder_hashes_files(tmp_path):
    import hashlib

    for code, df in quarter_frames("NQH2026", "NQM2026").items():
        df.to_csv(tmp_path / f"{code}.csv", index=False)
    cs = ContractSet.from_folder(str(tmp_path), "generic", "NQ", ts_is="open")
    assert list(cs.frames) == ["NQH2026", "NQM2026"]
    for code in cs.frames:
        raw = (tmp_path / f"{code}.csv").read_bytes()
        assert cs.file_sha256[code] == hashlib.sha256(raw).hexdigest()


def test_from_folder_wrong_root_refused(tmp_path):
    for code, df in quarter_frames("NQH2026", "NQM2026").items():
        df.to_csv(tmp_path / f"{code}.csv", index=False)
    with pytest.raises(ValueError, match="ES"):
        ContractSet.from_folder(str(tmp_path), "generic", "ES", ts_is="open")


def test_from_folder_empty_refused(tmp_path):
    with pytest.raises(ValueError, match="no .csv or .txt"):
        ContractSet.from_folder(str(tmp_path), "generic", "NQ", ts_is="open")
