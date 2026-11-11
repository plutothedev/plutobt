"""python -m plutobt.deflate, the no-code deflator (3.0 spec §4.1)."""
import pathlib
import subprocess
import sys

import pytest

from plutobt.deflate import main

SNAP = pathlib.Path(__file__).parent / "expected_output" / "deflate_cli.txt"
ARGS = ["--sharpe", "2.1", "--periods-per-year", "12", "--returns", "60", "--trials", "40"]


def run(capsys, argv):
    code = main(argv)
    out, err = capsys.readouterr()
    return code, out, err


def test_cli_snapshot(capsys):
    code, out, err = run(capsys, ARGS)
    assert code == 0 and err == ""
    assert out == SNAP.read_text(encoding="utf-8")


def test_cli_numbers_are_the_library_numbers(capsys):
    import math

    from plutobt.deflate import expected_max_sharpe, null_var_sr
    from plutobt.metrics import psr_from_moments

    _, out, _ = run(capsys, ARGS)
    sr = 2.1 / math.sqrt(12)
    bar = expected_max_sharpe(40, null_var_sr(60))
    assert f"about {bar * math.sqrt(12):.2f} by luck" in out
    assert f"Deflated Sharpe: {psr_from_moments(sr, 60, 0.0, 3.0, bar):.3f}" in out


def test_cli_skew_kurtosis_drops_the_normal_line(capsys):
    code, out, _ = run(capsys, ARGS + ["--skew", "-0.5", "--kurtosis", "5"])
    assert code == 0
    assert "normal returns" not in out
    assert "skew -0.5, kurtosis 5" in out


def test_cli_one_of_skew_kurtosis_refused(capsys):
    code, _, err = run(capsys, ARGS + ["--skew", "-0.5"])
    assert code == 2 and "both" in err


def test_cli_warns_under_30_returns(capsys):
    code, out, _ = run(capsys, ["--sharpe", "1.5", "--periods-per-year", "12",
                                "--returns", "24", "--trials", "5"])
    assert code == 0
    assert "Warning: under 30 returns, every Sharpe statistic here is unreliable" in out


def test_cli_one_try(capsys):
    code, out, _ = run(capsys, ["--sharpe", "1.5", "--periods-per-year", "252",
                                "--returns", "500", "--trials", "1"])
    assert code == 0
    assert "best of 1 try" in out and "nothing to deflate" in out


@pytest.mark.parametrize("extra, reason", [
    (["--sharpe", "nan"], "finite"),
    (["--trials", "0"], "trials"),
    (["--returns", "1"], "returns"),
    (["--periods-per-year", "0"], "periods"),
    (["--skew", "2", "--kurtosis", "3"], "impossible"),
])
def test_cli_refusals(capsys, extra, reason):
    argv = list(ARGS)
    for flag, value in zip(extra[::2], extra[1::2], strict=True):
        if flag in argv:
            argv[argv.index(flag) + 1] = value
        else:
            argv += [flag, value]
    code, out, err = run(capsys, argv)
    assert code == 2 and out == ""
    assert err.startswith("deflate: ") and reason in err


def test_cli_help_names_tradingview(capsys):
    with pytest.raises(SystemExit) as stop:
        main(["--help"])
    assert stop.value.code == 0
    out = capsys.readouterr().out
    assert "TradingView" in out and "risk-free" in out and "3.46" in out


def test_cli_runs_as_module():
    done = subprocess.run([sys.executable, "-m", "plutobt.deflate", *ARGS],
                          capture_output=True, text=True, check=False)
    assert done.returncode == 0, done.stderr
    assert done.stdout.startswith("PlutoBT deflate:")


def test_cli_as_module_prints_no_python_warning():
    """3.0.1, audit finding 8: every run began with a runpy RuntimeWarning on stderr,
    because the package imported plutobt.deflate before python -m ran it."""
    done = subprocess.run([sys.executable, "-m", "plutobt.deflate", *ARGS],
                          capture_output=True, text=True, check=False)
    assert done.returncode == 0
    assert done.stderr == ""


def test_importing_the_package_does_not_load_the_command():
    done = subprocess.run(
        [sys.executable, "-c",
         "import sys, plutobt; print('plutobt.deflate' in sys.modules)"],
        capture_output=True, text=True, check=False)
    assert done.stdout.strip() == "False", done.stderr


def test_the_3_0_0_import_path_still_works():
    import plutobt.deflation as deflation
    from plutobt.deflate import (  # noqa: F401
        EULER_GAMMA,
        SURVIVES,
        deflated_sharpe,
        expected_max_sharpe,
        null_var_sr,
        verdict,
    )

    assert expected_max_sharpe is deflation.expected_max_sharpe


@pytest.mark.parametrize("extra, reason", [
    (["--trials", "10000000000000000"], "--trials"),
    (["--trials", "100000000000000000000"], "--trials"),
    (["--sharpe", "1e308"], "--sharpe"),
    (["--sharpe", "1e155"], "--sharpe"),
])
def test_cli_refuses_absurd_values_without_a_traceback(capsys, extra, reason):
    argv = list(ARGS)
    argv[argv.index(extra[0]) + 1] = extra[1]
    code, out, err = run(capsys, argv)
    assert code == 2 and out == ""
    assert err.startswith("deflate: ") and reason in err and "Traceback" not in err


def test_cli_names_the_real_reason_when_the_moments_leave_no_standard_error(capsys):
    # skew 2, kurtosis 5 is possible, but with a per-period Sharpe of 1 the Sharpe's
    # variance term 1 - skew*SR + (kurtosis - 1)/4 * SR^2 is 0: no standard error exists
    argv = ["--sharpe", "1", "--periods-per-year", "1", "--returns", "60", "--trials", "5",
            "--skew", "2", "--kurtosis", "5"]
    code, out, _ = run(capsys, argv)
    assert code == 0
    assert "no return volatility" not in out
    assert "Verdict: not computable: this skew and kurtosis leave" in out


def test_cli_one_try_is_not_labelled_several(capsys):
    argv = list(ARGS)
    argv[argv.index("--trials") + 1] = "1"
    _, out, _ = run(capsys, argv)
    assert "best of several tries" not in out
    assert "IN-SAMPLE, a single try; not predictive of live results." in out
