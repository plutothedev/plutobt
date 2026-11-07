"""python -m plutobt.build_continuous (spec §6.5)."""
from pathlib import Path

from plutobt.build_continuous import main
from plutobt.roll import load_continuous

DIR = Path(__file__).resolve().parents[1] / "data" / "sample" / "nq_contracts"


def test_cli_on_sample(tmp_path, capsys):
    out = tmp_path / "nq_cont.csv"
    code = main([str(DIR), "--preset", "generic", "--root", "NQ", "--ts-is", "open",
                 "-o", str(out)])
    assert code == 0
    printed = capsys.readouterr().out
    assert out.exists() and (tmp_path / "nq_cont.roll.json").exists()
    for row in ("2026-03-16", "NQH2026", "NQM2026", "25.0", "500.0",
                "2026-06-15", "-15.0", "-300.0", "2026-09-14", "30.0", "600.0"):
        assert row in printed, row
    assert str(out) in printed and "nq_cont.roll.json" in printed
    df, meta = load_continuous(str(out))
    assert len(meta.rolls) == 3


def test_cli_roll_dates_file(tmp_path, capsys):
    dates = tmp_path / "dates.json"
    dates.write_text('{"NQH6": "2026-03-17", "NQM6": "2026-06-16", "NQU6": "2026-09-15"}')
    code = main([str(DIR), "--preset", "generic", "--root", "NQ", "--ts-is", "open",
                 "--trigger", "dates", "--roll-dates", str(dates),
                 "-o", str(tmp_path / "x.csv")])
    assert code == 0
    assert "2026-03-17" in capsys.readouterr().out


def test_cli_refusal_exit_code(tmp_path, capsys):
    code = main([str(DIR), "--preset", "generic", "--root", "CL", "--ts-is", "open",
                 "-o", str(tmp_path / "x.csv")])
    assert code == 2
    err = capsys.readouterr().err
    assert "CL" in err and "Traceback" not in err
