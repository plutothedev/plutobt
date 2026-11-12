"""python -m plutobt.continuous (spec §6.5; renamed from build_continuous so the CLI
module never shadows the build_continuous function)."""
from pathlib import Path

from plutobt.continuous import main
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


# --- 3.0.1 (audit minors): plain messages for the first mistakes -----------------------------

def _refused(capsys, argv):
    code = main(argv)
    out, err = capsys.readouterr()
    assert code == 2 and out == "" and err.startswith("continuous: ")
    assert "WinError" not in err and "Errno" not in err and "Traceback" not in err
    return err


def test_a_folder_that_does_not_exist(tmp_path, capsys):
    err = _refused(capsys, [str(tmp_path / "nope"), "--preset", "generic", "--root", "NQ",
                            "--ts-is", "open", "-o", str(tmp_path / "o.csv")])
    assert "does not exist" in err and "nope" in err


def test_a_file_given_as_the_folder(tmp_path, capsys):
    one = next(DIR.glob("*.csv"))
    err = _refused(capsys, [str(one), "--preset", "generic", "--root", "NQ", "--ts-is", "open",
                            "-o", str(tmp_path / "o.csv")])
    assert "is a file, not a folder" in err and "one CSV per contract" in err


def test_trigger_dates_without_roll_dates_names_the_flag(tmp_path, capsys):
    err = _refused(capsys, [str(DIR), "--preset", "generic", "--root", "NQ", "--ts-is", "open",
                            "--trigger", "dates", "-o", str(tmp_path / "o.csv")])
    assert "--trigger dates needs --roll-dates" in err


def test_a_roll_dates_file_that_does_not_exist(tmp_path, capsys):
    err = _refused(capsys, [str(DIR), "--preset", "generic", "--root", "NQ", "--ts-is", "open",
                            "--trigger", "dates", "--roll-dates", str(tmp_path / "no.json"),
                            "-o", str(tmp_path / "o.csv")])
    assert "--roll-dates" in err and "does not exist" in err
