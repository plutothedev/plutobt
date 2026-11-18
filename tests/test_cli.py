"""The ``plutobt`` command: quote, fetch, run and strategies. Never touches the network:
every Databento client here is ``tests/fake_databento`` through ``buy.make_client``."""
from __future__ import annotations

import io
import json
import os
import subprocess
import sys
from datetime import date
from pathlib import Path

import pytest

from plutobt import buy as B
from plutobt import cli
from plutobt.store import COLUMNS, Store
from tests import fake_databento as fake

KEY = "db-test-not-a-key"
REPO = Path(__file__).resolve().parents[1]
SAMPLE = REPO / "data" / "sample" / "nq_contracts" / "NQH2026.csv"
NQ_TOML = REPO / "configs" / "nq.toml"
QUOTE_ARGS = ["NQ", "--from", "2024-02-01", "--to", "2024-04-01", "--bars", "1h"]
PROMPT = "Buy on your Databento account? [y/N] "


@pytest.fixture
def market(tmp_path, monkeypatch):
    """A fake NQ market behind buy.make_client, a per-test store and working folder."""
    monkeypatch.setenv("PLUTOBT_DATA_DIR", str(tmp_path / "store"))
    monkeypatch.setenv("DATABENTO_API_KEY", KEY)
    monkeypatch.chdir(tmp_path)
    m = fake.nq_market()
    monkeypatch.setattr(B, "make_client", lambda: fake.Historical(key=KEY, market=m))
    return m


@pytest.fixture
def answer(monkeypatch):
    """Set what the trader types at the prompt; records every prompt shown."""
    prompts: list[str] = []

    def set_reply(reply):
        def fake_input(prompt=""):
            prompts.append(prompt)
            if isinstance(reply, BaseException) or (
                    isinstance(reply, type) and issubclass(reply, BaseException)):
                raise reply
            return reply(prompt) if callable(reply) else reply
        monkeypatch.setattr("builtins.input", fake_input)
        return prompts

    return set_reply


def expected_price(market, store_dir=None, start=date(2024, 2, 1), end=date(2024, 4, 1),
                   bars="1h"):
    """What buy.quote says for the request, on a fresh client and a scratch store."""
    calls = list(market.calls)
    q = B.quote(fake.Historical(key=KEY, market=market), "NQ", bars, start, end,
                Store(store_dir))
    market.calls[:] = calls
    return q.price_usd


def one_line(text: str) -> str:
    lines = text.strip("\n").splitlines()
    assert len(lines) == 1, text
    assert "Traceback" not in text
    return lines[0]


# -- quote -----------------------------------------------------------------------

def test_cli_quote_prints_price_and_nothing_charged(market, tmp_path, capsys):
    price = expected_price(market, tmp_path / "scratch")
    assert cli.main(["quote", *QUOTE_ARGS]) == 0
    out = capsys.readouterr().out
    assert (f"Databento price: ${price:.2f} for NQ 1h bars, 2024-02-01 to 2024-04-01 "
            "(Databento's own estimate).") in out
    assert "Nothing was charged." in out
    assert "Already on this computer" not in out and "the range ends there" not in out
    assert market.count("get_range") == 0
    assert not (tmp_path / "store").exists() or not list((tmp_path / "store").rglob("*.csv"))


def test_cli_quote_names_what_is_held(market, answer, capsys):
    answer("y")
    assert cli.main(["fetch", *QUOTE_ARGS]) == 0
    capsys.readouterr()
    assert cli.main(["quote", *QUOTE_ARGS]) == 0
    out = capsys.readouterr().out
    assert ("NQ 1h bars, 2024-02-01 to 2024-04-01: already on this computer, no new charge "
            "(NQH2024, NQM2024).") in out
    assert "Nothing was charged." in out


def test_cli_quote_says_when_the_range_is_clamped(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("PLUTOBT_DATA_DIR", str(tmp_path / "store"))
    monkeypatch.chdir(tmp_path)
    m = fake.nq_market(available_end="2024-03-20T00:00:00.000000000Z")
    monkeypatch.setattr(B, "make_client", lambda: fake.Historical(key=KEY, market=m))
    assert cli.main(["quote", *QUOTE_ARGS]) == 0
    out = capsys.readouterr().out
    assert "for NQ 1h bars, 2024-02-01 to 2024-03-20 (Databento's own estimate)." in out
    assert "Databento has data up to 2024-03-20; the range ends there." in out


def test_cli_quote_without_databento(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("PLUTOBT_DATA_DIR", str(tmp_path / "store"))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(sys.modules, "databento", None)
    monkeypatch.setenv("DATABENTO_API_KEY", KEY)
    assert cli.main(["quote", *QUOTE_ARGS]) == 2
    err = one_line(capsys.readouterr().err)
    assert 'pip install -e ".[databento]"' in err and "not installed" in err
    assert "plutobt[databento]" not in err


def test_cli_quote_without_key(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("PLUTOBT_DATA_DIR", str(tmp_path / "store"))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(sys.modules, "databento", fake)
    monkeypatch.delenv("DATABENTO_API_KEY", raising=False)
    assert cli.main(["quote", *QUOTE_ARGS]) == 2
    assert "Set DATABENTO_API_KEY" in one_line(capsys.readouterr().err)


def test_cli_error_never_shows_the_key(market, capsys):
    market.fail_get_cost = RuntimeError(f"bad key {KEY}")
    assert cli.main(["quote", *QUOTE_ARGS]) == 2
    captured = capsys.readouterr()
    assert KEY not in captured.out + captured.err
    assert "***" in one_line(captured.err)


def test_cli_unbuyable_root_refused(market, capsys):
    assert cli.main(["quote", "AAPL", "--from", "2024-02-01", "--to", "2024-04-01",
                     "--bars", "1h"]) == 2
    assert "AAPL" in one_line(capsys.readouterr().err)


@pytest.mark.parametrize("args", [
    ["NQ", "--from", "2024-02-30", "--to", "2024-04-01", "--bars", "1h"],
    ["NQ", "--from", "2024-02-01", "--to", "2024-04-01", "--bars", "5m"],
    ["NQ", "--from", "2024-02-01", "--bars", "1h"],
])
def test_cli_bad_arguments_one_line_exit_2(market, capsys, args):
    assert cli.main(["quote", *args]) == 2
    one_line(capsys.readouterr().err)
    assert market.calls == []


# -- fetch -----------------------------------------------------------------------

@pytest.mark.parametrize("reply", ["y", "Y", "yes", " YES "])
def test_cli_fetch_yes_buys(market, answer, tmp_path, capsys, reply):
    prompts = answer(reply)
    assert cli.main(["fetch", *QUOTE_ARGS]) == 0
    out = capsys.readouterr().out
    assert prompts == [PROMPT]
    assert "Databento price: $" in out and "Nothing was charged." in out
    assert f"Bought. Saved on this computer in {tmp_path / 'store'}" in out
    assert market.count("get_range") == 2


@pytest.mark.parametrize("reply", ["n", "", "no", "yep", "y es"])
def test_cli_fetch_no_buys_nothing(market, answer, capsys, reply):
    answer(reply)
    assert cli.main(["fetch", *QUOTE_ARGS]) == 1
    out = capsys.readouterr().out
    assert "Nothing was bought." in out and "Bought." not in out
    assert market.count("get_range") == 0


@pytest.mark.parametrize("exc", [EOFError, KeyboardInterrupt])
def test_cli_fetch_eof_is_no(market, answer, capsys, exc):
    answer(exc)
    assert cli.main(["fetch", *QUOTE_ARGS]) == 1
    captured = capsys.readouterr()
    assert "Nothing was bought." in captured.out
    assert "Traceback" not in captured.err
    assert market.count("get_range") == 0


def test_cli_fetch_twice_second_free(market, answer, capsys):
    prompts = answer("y")
    assert cli.main(["fetch", *QUOTE_ARGS]) == 0
    capsys.readouterr()
    assert cli.main(["fetch", *QUOTE_ARGS]) == 0
    out = capsys.readouterr().out
    assert "Nothing to buy: all of it is already on this computer." in out
    assert len(prompts) == 1
    assert market.count("get_range") == 2


def test_cli_price_changed_exit_1(market, answer, capsys):
    def raise_price(prompt):
        market.rate_per_day = 0.02
        return "y"

    answer(raise_price)
    assert cli.main(["fetch", *QUOTE_ARGS]) == 1
    new = expected_price(market)
    out = capsys.readouterr().out
    assert (f"The price changed to ${new:.2f} since the quote. Nothing was bought. "
            "Run the command again to see the new price.") in out
    assert market.count("get_range") == 0


def test_cli_buys_at_the_unrounded_price(market, answer, monkeypatch, capsys):
    market.cost_fn = lambda kw: 0.1234
    seen = []
    real = B.buy

    def spy(client, q, approved_usd, store, **kw):
        seen.append((approved_usd, q.price_usd))
        return real(client, q, approved_usd, store, **kw)

    monkeypatch.setattr(B, "buy", spy)
    answer("y")
    assert cli.main(["fetch", *QUOTE_ARGS]) == 0
    assert "$0.25" in capsys.readouterr().out
    assert seen == [(pytest.approx(0.2468, abs=1e-12), pytest.approx(0.2468, abs=1e-12))]


def test_cli_fetch_yes_flag_skips_the_prompt(market, answer, capsys):
    prompts = answer(AssertionError("no prompt with --yes"))
    assert cli.main(["fetch", *QUOTE_ARGS, "--yes"]) == 0
    assert prompts == []
    assert "Bought." in capsys.readouterr().out


@pytest.mark.parametrize("yes", [[], ["--yes"]])
def test_cli_max_cost_refuses_before_the_prompt(market, answer, capsys, yes):
    prompts = answer("y")
    assert cli.main(["fetch", *QUOTE_ARGS, "--max-cost", "0.10", *yes]) == 2
    captured = capsys.readouterr()
    assert "above --max-cost $0.10. Nothing was bought." in one_line(captured.err)
    assert prompts == []
    assert market.count("get_range") == 0


def test_cli_max_cost_checked_on_the_fresh_price(market, answer, capsys):
    def raise_price(prompt):
        market.rate_per_day = 0.05
        return "y"

    answer(raise_price)
    assert cli.main(["fetch", *QUOTE_ARGS, "--max-cost", "1.00"]) == 2
    assert "above --max-cost $1.00" in one_line(capsys.readouterr().err)
    assert market.count("get_range") == 0


@pytest.mark.parametrize("bad", ["-1", "nan", "inf", "five"])
def test_cli_max_cost_must_be_a_price(market, capsys, bad):
    assert cli.main(["fetch", *QUOTE_ARGS, "--max-cost", bad]) == 2
    one_line(capsys.readouterr().err)
    assert market.calls == []


def test_cli_buy_failure_is_one_line(market, answer, capsys):
    market.fail_get_range[2] = RuntimeError(f"boom {KEY}")
    answer("y")
    assert cli.main(["fetch", *QUOTE_ARGS]) == 2
    captured = capsys.readouterr()
    err = one_line(captured.err)
    assert "Nothing was retried" in err and KEY not in captured.out + captured.err
    assert market.count("get_range") == 2


# -- run -------------------------------------------------------------------------

def csv_args(*extra):
    return ["run", "--data", str(SAMPLE), "--preset", "generic", "--ts-is", "open",
            "--symbol", "NQ", *extra]


def test_cli_run_csv_builtin(market, tmp_path, capsys):
    assert cli.main([*csv_args(), "--strategy", "ema_cross"]) == 0
    out = capsys.readouterr().out
    assert "IN-SAMPLE" in out and str(SAMPLE) in out
    chart = [ln for ln in out.splitlines() if ln.startswith("Chart: ")]
    assert len(chart) == 1 and Path(chart[0][len("Chart: "):]).is_file()
    assert (tmp_path / ".plutobt" / "trials.jsonl").is_file()
    assert market.calls == []


def test_cli_run_no_ledger_and_chart_dir(market, tmp_path, capsys):
    assert cli.main([*csv_args(), "--strategy", "ema_cross", "--no-ledger",
                     "--chart-dir", str(tmp_path / "charts")]) == 0
    out = capsys.readouterr().out
    assert "Ledger: off" in out
    assert not (tmp_path / ".plutobt").exists()
    assert list((tmp_path / "charts").glob("*.png"))


def test_cli_run_generic_needs_ts_is(market, capsys):
    args = ["run", "--data", str(SAMPLE), "--symbol", "NQ", "--strategy", "ema_cross"]
    assert cli.main(args) == 2
    assert "--ts-is" in one_line(capsys.readouterr().err)


STRATEGY_FILE = """
from plutobt.strategies.ema_cross import EMACross


class MyCross(EMACross):
    fast_span: int = 10
"""


def test_cli_run_file_strategy_with_set(market, tmp_path, capsys):
    path = tmp_path / "my_cross.py"
    path.write_text(STRATEGY_FILE, encoding="utf-8")
    assert cli.main([*csv_args(), "--strategy", f"{path}:MyCross",
                     "--set", "fast_span=5", "--set", "stop_ticks=12.5"]) == 0
    assert "IN-SAMPLE" in capsys.readouterr().out
    lines = (tmp_path / ".plutobt" / "trials.jsonl").read_text().splitlines()
    params = json.loads(lines[-1])["params"]
    assert params["fast_span"] == 5 and params["stop_ticks"] == 12.5


def test_cli_run_bad_setting_refused(market, capsys):
    assert cli.main([*csv_args(), "--strategy", "ema_cross", "--set", "fast=5"]) == 2
    assert "fast is not a setting of EMACross" in one_line(capsys.readouterr().err)


LEAKY_FILE = """
from plutobt.strategy import Strategy


class Leaky(Strategy):
    def precompute(self, df):
        df["peek"] = df["close"].shift(-1)
        return df

    def on_bar(self, ctx):
        pass
"""


def test_cli_run_lookahead_strategy_refused(market, tmp_path, capsys):
    path = tmp_path / "leaky_cli.py"
    path.write_text(LEAKY_FILE, encoding="utf-8")
    assert cli.main([*csv_args(), "--strategy", str(path)]) == 2
    assert "peek" in one_line(capsys.readouterr().err)


def test_cli_run_with_config(market, capsys):
    assert cli.main([*csv_args(), "--strategy", "ema_cross", "--config", str(NQ_TOML)]) == 0
    out = capsys.readouterr().out
    assert f"Costs: from your config {NQ_TOML}" in out


def test_cli_allow_short_with_config_refused(market, capsys):
    assert cli.main([*csv_args(), "--strategy", "ema_cross", "--config", str(NQ_TOML),
                     "--allow-short"]) == 2
    assert one_line(capsys.readouterr().err).endswith(
        "--allow-short has no effect with --config; set allow_short in the config file")


def test_cli_run_session_all(market, capsys):
    assert cli.main([*csv_args(), "--strategy", "ema_cross", "--session", "all"]) == 0
    assert "Hours: full day (--session all)" in capsys.readouterr().out


def test_cli_run_unknown_symbol_refused(market, capsys):
    args = ["run", "--data", str(SAMPLE), "--preset", "generic", "--ts-is", "open",
            "--symbol", "ZZ", "--strategy", "ema_cross"]
    assert cli.main(args) == 2
    assert "unknown symbol" in one_line(capsys.readouterr().err)


def test_cli_run_missing_file_refused(market, tmp_path, capsys):
    args = ["run", "--data", str(tmp_path / "nope.csv"), "--preset", "generic",
            "--ts-is", "open", "--symbol", "NQ", "--strategy", "ema_cross"]
    assert cli.main(args) == 2
    one_line(capsys.readouterr().err)


def test_cli_run_continuous(market, tmp_path, capsys):
    from plutobt.continuous import main as build

    out_csv = tmp_path / "nq_cont.csv"
    assert build([str(SAMPLE.parent), "--preset", "generic", "--root", "NQ", "--ts-is",
                  "open", "-o", str(out_csv)]) == 0
    capsys.readouterr()
    assert cli.main(["run", "--continuous", str(out_csv), "--symbol", "NQ",
                     "--strategy", "ema_cross"]) == 0
    out = capsys.readouterr().out
    assert "Hours: continuous file run as built" in out and str(out_csv) in out
    assert cli.main(["run", "--continuous", str(out_csv), "--symbol", "ES",
                     "--strategy", "ema_cross"]) == 2
    assert "NQ" in one_line(capsys.readouterr().err)
    assert cli.main(["run", "--continuous", str(out_csv), "--symbol", "NQ",
                     "--strategy", "ema_cross", "--session", "all"]) == 2
    one_line(capsys.readouterr().err)


BUY_ARGS = ["--buy", "NQ", "--from", "2024-02-01", "--to", "2024-04-01", "--bars", "1h"]


def test_cli_run_buy_path_end_to_end(market, answer, tmp_path, capsys):
    answer(AssertionError("no prompt with --yes"))
    price = expected_price(market, tmp_path / "scratch")
    assert cli.main(["run", "--strategy", "ema_cross", *BUY_ARGS, "--yes"]) == 0
    out = capsys.readouterr().out
    assert (f"Data: bought from Databento, NQ 1h 2024-02-01 to 2024-04-01, Databento's "
            f"estimate at purchase: ${price:.2f} (exact charge: your Databento billing "
            "page)") in out
    assert "Bought. Saved on this computer in" in out
    assert "IN-SAMPLE" in out and "Chart: " in out
    assert "Hours: regular trading hours only" in out
    assert market.count("get_range") == 2


def test_cli_run_buy_second_time_is_free(market, answer, capsys):
    prompts = answer("y")
    assert cli.main(["run", "--strategy", "ema_cross", *BUY_ARGS]) == 0
    capsys.readouterr()
    assert cli.main(["run", "--strategy", "ema_cross", *BUY_ARGS]) == 0
    out = capsys.readouterr().out
    assert ("bought from Databento earlier, NQ 1h 2024-02-01 to 2024-04-01: already on this "
            "computer, no new charge") in out
    assert "Nothing to buy: all of it is already on this computer." in out
    assert len(prompts) == 1 and market.count("get_range") == 2


def test_cli_run_buy_no_runs_nothing(market, answer, tmp_path, capsys):
    answer("n")
    assert cli.main(["run", "--strategy", "ema_cross", *BUY_ARGS]) == 1
    out = capsys.readouterr().out
    assert "Nothing was bought." in out and "Chart: " not in out
    assert market.count("get_range") == 0
    assert not (tmp_path / "plutobt_runs").exists()


def test_cli_run_buy_one_contract(market, answer, capsys):
    answer("y")
    args = ["--buy", "NQ", "--from", "2024-02-01", "--to", "2024-03-01", "--bars", "1h"]
    assert cli.main(["run", "--strategy", "ema_cross", *args]) == 0
    assert "IN-SAMPLE" in capsys.readouterr().out


def test_cli_run_buy_checks_the_strategy_before_any_databento_call(market, capsys):
    assert cli.main(["run", "--strategy", "no_such", *BUY_ARGS, "--yes"]) == 2
    one_line(capsys.readouterr().err)
    assert market.calls == []
    assert cli.main(["run", "--strategy", "ema_cross", "--set", "nope=1", *BUY_ARGS,
                     "--yes"]) == 2
    one_line(capsys.readouterr().err)
    assert market.calls == []


@pytest.mark.parametrize("args", [
    ["--data", str(SAMPLE), "--symbol", "NQ", "--ts-is", "open", *BUY_ARGS],
    ["--data", str(SAMPLE), "--continuous", "x.csv", "--symbol", "NQ"],
    [],
    ["--data", str(SAMPLE), "--ts-is", "open"],
    ["--buy", "NQ", "--from", "2024-02-01", "--bars", "1h"],
    ["--data", str(SAMPLE), "--symbol", "NQ", "--ts-is", "open", "--bars", "1h"],
    [*BUY_ARGS, "--preset", "tradingview"],
    [*BUY_ARGS, "--symbol", "ES"],
])
def test_cli_run_two_data_sources_refused(market, capsys, args):
    assert cli.main(["run", "--strategy", "ema_cross", *args]) == 2
    one_line(capsys.readouterr().err)
    assert market.calls == []


# -- strategies, entry points, output ----------------------------------------------

def test_cli_strategies_lists_three(capsys):
    assert cli.main(["strategies"]) == 0
    out = capsys.readouterr().out
    for name in ("ema_cross", "opening_range", "donchian"):
        assert f"{name}:" in out
    assert "Fast/slow EMA crossover." in out
    assert "fast_span = 20" in out and "range_minutes = 30" in out
    assert "entry_bars = 20" in out and "exit_bars = 10" in out


def test_cli_help_exit_0(capsys):
    assert cli.main(["--help"]) == 0
    out = capsys.readouterr().out
    for name in ("quote", "fetch", "run", "strategies"):
        assert name in out


def test_cli_no_command_exit_2(capsys):
    assert cli.main([]) == 2
    one_line(capsys.readouterr().err)


def _env(tmp_path):
    env = dict(os.environ)
    env.update(PYTHONIOENCODING="cp1252", PLUTOBT_DATA_DIR=str(tmp_path / "store"))
    env.pop("DATABENTO_API_KEY", None)
    return env


def test_module_entry_point(tmp_path):
    r = subprocess.run([sys.executable, "-m", "plutobt", "--help"], capture_output=True,
                       cwd=tmp_path, env=_env(tmp_path))
    assert r.returncode == 0, r.stderr
    assert b"strategies" in r.stdout


def test_cli_output_is_ascii_cp1252(tmp_path):
    for args in (["strategies"],
                 ["run", "--data", str(SAMPLE), "--preset", "generic", "--ts-is", "open",
                  "--symbol", "NQ", "--strategy", "ema_cross"]):
        r = subprocess.run([sys.executable, "-m", "plutobt", *args], capture_output=True,
                           cwd=tmp_path, env=_env(tmp_path))
        assert r.returncode == 0, r.stderr
        r.stdout.decode("ascii")
        r.stderr.decode("ascii")


def test_cli_quote_and_fetch_output_is_ascii_cp1252(market, answer, monkeypatch):
    # The fake client cannot reach a subprocess, so the quote runs in process with
    # stdout and stderr as cp1252 text streams, the way a Windows console has them.
    answer("y")
    for args in (["quote", *QUOTE_ARGS], ["fetch", *QUOTE_ARGS], ["quote", *QUOTE_ARGS],
                 ["quote", "AAPL", *QUOTE_ARGS[1:]]):
        out, err = io.BytesIO(), io.BytesIO()
        monkeypatch.setattr(sys, "stdout", io.TextIOWrapper(out, encoding="cp1252"))
        monkeypatch.setattr(sys, "stderr", io.TextIOWrapper(err, encoding="cp1252"))
        cli.main(args)
        sys.stdout.flush()
        sys.stderr.flush()
        out.getvalue().decode("ascii")
        err.getvalue().decode("ascii")
        assert out.getvalue() or err.getvalue()


# -- whole-branch review fixes (3.1) -----------------------------------------------

def test_cli_run_refuses_a_file_of_another_root(market, capsys):
    args = ["run", "--data", str(SAMPLE), "--preset", "generic", "--ts-is", "open",
            "--symbol", "ES", "--strategy", "ema_cross"]
    assert cli.main(args) == 2
    assert one_line(capsys.readouterr().err).endswith(
        "This file holds NQ contract NQH2026 but --symbol is ES; use --symbol NQ.")


@pytest.mark.parametrize("preset, flags, named", [
    ("databento", ["--tz", "America/Chicago"], "--tz"),
    ("databento", ["--ts-is", "close"], "--ts-is"),
    ("ninjatrader", ["--tz", "America/Chicago"], "--tz"),
])
def test_cli_run_refuses_a_flag_the_preset_never_uses(market, tmp_path, capsys, preset,
                                                      flags, named):
    args = ["run", "--data", str(tmp_path / "missing.csv"), "--preset", preset,
            "--symbol", "NQ", "--strategy", "ema_cross", *flags]
    assert cli.main(args) == 2
    err = one_line(capsys.readouterr().err)
    assert named in err and f"--preset {preset}" in err and "missing.csv" not in err


def _tv_unix(tmp_path):
    import pandas as pd
    df = pd.read_csv(SAMPLE)
    path = tmp_path / "tv_export.csv"
    df.rename(columns={"ts": "time", "volume": "Volume"}).to_csv(path, index=False)
    return path


def _generic_text(tmp_path, zoned: bool):
    import pandas as pd
    df = pd.read_csv(SAMPLE)
    when = pd.to_datetime(df["ts"], unit="s", utc=True)
    if zoned:
        df["ts"] = when.dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    else:
        df["ts"] = when.dt.tz_convert("America/Chicago").dt.strftime("%Y-%m-%d %H:%M:%S")
    path = tmp_path / ("zoned.csv" if zoned else "naive.csv")
    df.to_csv(path, index=False)
    return path


def _data_line(out: str) -> str:
    lines = [ln for ln in out.splitlines() if ln.startswith("Data: ")]
    assert len(lines) == 1, out
    return lines[0]


def test_cli_run_data_note_says_the_zone_actually_used(market, tmp_path, capsys):
    tv = _tv_unix(tmp_path)
    assert cli.main(["run", "--data", str(tv), "--preset", "tradingview", "--symbol", "NQ",
                     "--strategy", "ema_cross", "--tz", "America/Chicago"]) == 0
    line = _data_line(capsys.readouterr().out)
    assert "America/Chicago" not in line and "zone: none needed (unix time)" in line
    assert "timestamps read as the bar open" in line

    zoned = _generic_text(tmp_path, zoned=True)
    assert cli.main(["run", "--data", str(zoned), "--preset", "generic", "--ts-is", "open",
                     "--symbol", "NQ", "--strategy", "ema_cross",
                     "--tz", "America/Chicago"]) == 0
    line = _data_line(capsys.readouterr().out)
    assert "America/Chicago" not in line and "zone: from the file" in line

    naive = _generic_text(tmp_path, zoned=False)
    assert cli.main(["run", "--data", str(naive), "--preset", "generic", "--ts-is", "open",
                     "--symbol", "NQ", "--strategy", "ema_cross",
                     "--tz", "America/Chicago"]) == 0
    assert "zone: America/Chicago (given with --tz)" in _data_line(capsys.readouterr().out)


def test_cli_generic_without_ts_is_refused_before_loading(market, tmp_path, capsys):
    args = ["run", "--data", str(tmp_path / "missing.csv"), "--symbol", "NQ",
            "--strategy", "no_such_strategy"]
    assert cli.main(args) == 2
    err = one_line(capsys.readouterr().err)
    assert "--ts-is" in err and "missing.csv" not in err and "no_such" not in err


def _offline_command(out: str):
    lines = [ln for ln in out.splitlines() if ln.startswith("Series saved for offline runs: ")]
    assert len(lines) == 1, out
    path = Path(lines[0][len("Series saved for offline runs: "):])
    cmds = [ln.strip() for ln in out.splitlines() if ln.strip().startswith("plutobt run ")]
    assert len(cmds) == 1, out
    import shlex
    argv = [t.strip('"') for t in shlex.split(cmds[0], posix=False)][1:]
    return path, argv


def _no_databento(monkeypatch):
    def refuse():
        raise AssertionError("an offline run must not make a Databento client")
    monkeypatch.setattr(B, "make_client", refuse)
    monkeypatch.delenv("DATABENTO_API_KEY", raising=False)


def test_cli_fetch_saves_a_series_that_runs_offline(market, answer, tmp_path, capsys,
                                                    monkeypatch):
    answer("y")
    assert cli.main(["fetch", *QUOTE_ARGS]) == 0
    path, argv = _offline_command(capsys.readouterr().out)
    assert path.is_file() and path.with_suffix(".roll.json").is_file()
    assert path.parent == tmp_path / "store" / "series"
    assert path.name == "NQ_1h_2024-02-01_2024-04-01_rth.csv"
    assert argv[:2] == ["run", "--continuous"]
    calls = len(market.calls)
    _no_databento(monkeypatch)
    assert cli.main(argv) == 0
    out = capsys.readouterr().out
    assert "IN-SAMPLE" in out and str(path) in out
    assert len(market.calls) == calls


def test_cli_fetch_one_contract_saves_a_plain_csv_that_runs_offline(market, answer, tmp_path,
                                                                    capsys, monkeypatch):
    answer("y")
    assert cli.main(["fetch", "NQ", "--from", "2024-02-01", "--to", "2024-03-01",
                     "--bars", "1h", "--session", "all"]) == 0
    path, argv = _offline_command(capsys.readouterr().out)
    assert path.name == "NQ_1h_2024-02-01_2024-03-01_all.csv" and path.is_file()
    assert "--data" in argv and "--session" in argv
    _no_databento(monkeypatch)
    assert cli.main(argv) == 0
    out = capsys.readouterr().out
    assert "IN-SAMPLE" in out and "Hours: full day (--session all)" in out


def test_cli_run_buy_saves_the_series_it_ran(market, answer, tmp_path, capsys, monkeypatch):
    answer("y")
    assert cli.main(["run", "--strategy", "donchian", *BUY_ARGS]) == 0
    out = capsys.readouterr().out
    path, argv = _offline_command(out)
    assert "--strategy" in argv and argv[argv.index("--strategy") + 1] == "donchian"
    _no_databento(monkeypatch)
    assert cli.main(argv) == 0
    assert "IN-SAMPLE" in capsys.readouterr().out


def test_cli_note_is_redacted_and_ascii(market, capsys):
    cli._note(UserWarning(f"careful {KEY} \u56fe"), UserWarning, "x.py", 1)
    err = one_line(capsys.readouterr().err)
    assert KEY not in err and "***" in err and "\\u56fe" in err
    err.encode("ascii")


def test_cli_sampled_causality_note_is_for_a_trader(market, capsys):
    text = ("causality check sampled: checked 400 of the 9999 prefixes of 10000 bars (every "
            "prefix is checked up to 2000 bars), so a leak on rows between them can be "
            "missed. To check every prefix, set causality_sample = None in BacktestConfig "
            "(or pass sample=None to assert_causal); the cost grows with the square of the "
            "bar count")
    cli._note(UserWarning(text), UserWarning, "x.py", 1)
    err = one_line(capsys.readouterr().err)
    assert "BacktestConfig" not in err and "assert_causal" not in err
    assert "400" in err and "10000" in err and "standard" in err and "missed" in err


@pytest.mark.parametrize("bars", ["1h", "1d"])
def test_cli_opening_range_refuses_long_bars_before_any_call(market, capsys, bars):
    args = ["run", "--strategy", "opening_range", "--buy", "NQ", "--from", "2024-02-01",
            "--to", "2024-04-01", "--bars", bars, "--yes"]
    assert cli.main(args) == 2
    assert "opening_range needs bars of 30 minutes or less" in one_line(
        capsys.readouterr().err)
    assert market.calls == []


def test_cli_quote_all_held_says_no_new_charge_and_names_contracts(market, answer, capsys):
    answer("y")
    assert cli.main(["fetch", *QUOTE_ARGS]) == 0
    capsys.readouterr()
    assert cli.main(["quote", *QUOTE_ARGS]) == 0
    out = capsys.readouterr().out
    assert "$0.00" not in out
    assert "already on this computer, no new charge" in out
    assert "NQH2024" in out and "NQM2024" in out


def test_cli_run_buy_all_held_says_no_new_charge(market, answer, capsys):
    answer("y")
    assert cli.main(["run", "--strategy", "ema_cross", *BUY_ARGS]) == 0
    capsys.readouterr()
    assert cli.main(["run", "--strategy", "ema_cross", *BUY_ARGS]) == 0
    line = _data_line(capsys.readouterr().out)
    assert "$0.00" not in line and "already on this computer, no new charge" in line


def test_cli_fetch_help_on_cp1252_with_a_non_cp1252_store(tmp_path):
    env = _env(tmp_path)
    env["PLUTOBT_DATA_DIR"] = str(tmp_path / "store_\u56fe")
    r = subprocess.run([sys.executable, "-m", "plutobt", "fetch", "--help"],
                       capture_output=True, cwd=tmp_path, env=env)
    assert r.returncode == 0, r.stderr
    assert b"store_\\u56fe" in r.stdout and b"usage:" in r.stdout


# -- offline series save: guarded, atomic, reproducing ------------------------------

def _save_fails(monkeypatch):
    def boom(*args, **kwargs):
        raise PermissionError(13, "Permission denied")
    monkeypatch.setattr(cli, "_write_series_files", boom)


def test_cli_fetch_survives_a_failed_series_save(market, answer, capsys, monkeypatch):
    answer("y")
    _save_fails(monkeypatch)
    assert cli.main(["fetch", *QUOTE_ARGS]) == 0
    cap = capsys.readouterr()
    assert "Could not save the series for offline runs (" in cap.err
    assert "run the same command again (no new charge)" in cap.err
    assert "Series saved for offline runs" not in cap.out
    assert "plutobt run " not in cap.out
    cap.err.encode("ascii")


def test_cli_run_buy_survives_a_failed_series_save(market, answer, capsys, monkeypatch):
    answer("y")
    _save_fails(monkeypatch)
    assert cli.main(["run", "--strategy", "ema_cross", *BUY_ARGS]) == 0
    cap = capsys.readouterr()
    assert "Could not save the series for offline runs (" in cap.err
    assert "IN-SAMPLE" in cap.out and "Chart: " in cap.out
    assert "Series saved for offline runs" not in cap.out
    assert "plutobt run " not in cap.out


def test_cli_series_write_is_atomic(tmp_path, monkeypatch):
    import pandas as pd
    path = tmp_path / "s.csv"
    df = pd.DataFrame({"ts": [1, 2], "x": [1.0, 2.0]})

    class Meta:
        def to_json(self):
            return "{}"

    real = os.replace
    calls = []

    def flaky(src, dst):
        calls.append(dst)
        if len(calls) == 2:
            raise OSError("disk went away")
        return real(src, dst)

    monkeypatch.setattr(os, "replace", flaky)
    with pytest.raises(OSError):
        cli._write_series_files(df, Meta(), path)
    assert not path.exists()
    assert not path.with_name("s.roll.json").exists()
    assert list(tmp_path.iterdir()) == []


def test_cli_series_write_failure_midway_leaves_nothing(tmp_path, monkeypatch):
    import pandas as pd
    path = tmp_path / "s.csv"
    df = pd.DataFrame({c: [1] for c in COLUMNS})
    monkeypatch.setattr(pd.DataFrame, "to_csv",
                        lambda self, p, **kw: (Path(p).write_text("ts"), 1 / 0)[1])
    with pytest.raises(ZeroDivisionError):
        cli._write_series_files(df, None, path)
    assert list(tmp_path.iterdir()) == []


def test_cli_run_buy_offline_command_keeps_every_backtest_flag(market, answer, tmp_path,
                                                               capsys):
    answer("y")
    cfg = tmp_path / "my configs" / "nq.toml"
    cfg.parent.mkdir()
    cfg.write_text(NQ_TOML.read_text(encoding="utf-8"), encoding="utf-8")
    assert cli.main(["run", "--strategy", "ema_cross", *BUY_ARGS, "--set", "fast_span=5",
                     "--config", str(cfg)]) == 0
    out = capsys.readouterr().out
    cmd = [ln.strip() for ln in out.splitlines() if ln.strip().startswith("plutobt run ")]
    assert len(cmd) == 1, out
    assert "--set fast_span=5" in cmd[0]
    assert f'--config "{cfg}"' in cmd[0]


def test_cli_run_buy_offline_command_keeps_allow_short(market, answer, capsys):
    answer("y")
    assert cli.main(["run", "--strategy", "ema_cross", *BUY_ARGS, "--allow-short"]) == 0
    out = capsys.readouterr().out
    cmd = [ln.strip() for ln in out.splitlines() if ln.strip().startswith("plutobt run ")]
    assert len(cmd) == 1 and "--allow-short" in cmd[0], out


def test_cli_run_buy_offline_command_quotes_a_set_value_with_spaces():
    assert cli._run_flags(["a=b c", "d=1"], None, False) == ' --set "a=b c" --set d=1'
    assert cli._run_flags([], Path("x y.toml"), True) == ' --config "x y.toml" --allow-short'
