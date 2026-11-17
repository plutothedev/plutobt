"""The ``plutobt`` command (also ``python -m plutobt``), for a trader who does not code.

    plutobt quote NQ --from 2024-01-01 --to 2025-01-01 --bars 1m
    plutobt fetch NQ --from 2024-01-01 --to 2025-01-01 --bars 1m
    plutobt run --strategy ema_cross --data NQ_2024.csv --preset tradingview --symbol NQ
    plutobt run --strategy my_orb.py --buy NQ --from 2024-01-01 --to 2025-01-01 --bars 1m
    plutobt strategies

``fetch`` and ``run --buy`` are the only places a trader says yes to a Databento charge:
the price is shown, the trader types y or yes (or passes ``--yes``), Databento is asked
for the price again right before buying, and a different price buys nothing.

Exit status: 0 done, 1 the trader said no or the price changed (nothing bought), 2 refused
or an error (one plain line on stderr). Everything printed is plain ASCII.
"""
from __future__ import annotations

import argparse
import inspect
import math
import os
import re
import sys
import warnings
from datetime import date
from pathlib import Path

from plutobt import buy as _buy
from plutobt.config import load_config
from plutobt.contracts import parse_contract
from plutobt.importers import PRESETS, _load_csv
from plutobt.instruments import get_instrument
from plutobt.ledger import DEFAULT_PATH
from plutobt.roll import _meta_path, load_continuous
from plutobt.runner import run_backtest
from plutobt.store import COLUMNS, Store, data_dir
from plutobt.strategies.opening_range import OpeningRange
from plutobt.strategy_loader import BUILTINS, load_strategy, parse_settings, settings

PROG = "plutobt"
PROMPT = "Buy on your Databento account? [y/N] "
_DATES = ("Dates are YYYY-MM-DD UTC calendar days; the --from day is included and the "
          "--to day is not.")


class _Refused(Exception):
    """A request the command will not carry out; the text says what to change."""


class _Parser(argparse.ArgumentParser):
    def error(self, message):
        raise _Refused(f"{message} (see: {self.prog} --help)")

    def _print_message(self, message, file=None):
        # Help text can hold a path the console cannot show (the store folder): escape
        # it, never crash on --help.
        if message:
            stream = file or sys.stderr
            try:
                stream.write(message)
            except UnicodeEncodeError:
                stream.write(message.encode("ascii", "backslashreplace").decode("ascii"))
            except (AttributeError, OSError):
                pass


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def _out(text: str = "", *, err: bool = False) -> None:
    """Print one line. A character the console cannot show is escaped, never a crash."""
    stream = sys.stderr if err else sys.stdout
    try:
        print(text, file=stream)
    except UnicodeEncodeError:
        print(text.encode("ascii", "backslashreplace").decode("ascii"), file=stream)


def _plain(message: str) -> str:
    """One line, the key redacted, plain ASCII."""
    text = " ".join(_buy._redact(message, _buy._keys(None)).split())
    return text.encode("ascii", "backslashreplace").decode("ascii")


def _fail(message: str) -> int:
    _out(f"{PROG}: {_plain(message)}", err=True)
    return 2


# The engine's sampled look-ahead warning names Python settings (BacktestConfig,
# assert_causal) a trader at the command line never sees; the command says it plainly.
_SAMPLED = re.compile(r"causality check sampled: checked (\d+) of the (\d+) prefixes of "
                      r"(\d+) bars \(every prefix is checked up to (\d+) bars\)")


def _for_a_trader(text: str) -> str:
    m = _SAMPLED.match(text)
    if m is None:
        return text
    k, p, n, full = m.groups()
    return (f"look-ahead check sampled: the engine's standard look-ahead check, run on "
            f"every backtest, compared the strategy's signals at {k} of the {p} cut-off "
            f"points of these {n} bars instead of at every one (every one is checked on "
            f"data of up to {full} bars). A leak that shows only between the sampled "
            "points can be missed.")


def _note(message, category, filename, lineno, file=None, line=None) -> None:
    """A warning as one plain line, without the source file and code line."""
    _out(f"{PROG}: note: {_plain(_for_a_trader(' '.join(str(message).split())))}", err=True)


def _message(exc: BaseException) -> str:
    text = str(exc.args[0]) if isinstance(exc, KeyError) and exc.args else str(exc)
    if isinstance(exc, _buy.BuyError | _Refused | ValueError | KeyError | OSError):
        return text or type(exc).__name__
    return f"{type(exc).__name__}: {text}" if text else type(exc).__name__


# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------

def _day(text: str) -> date:
    try:
        return date.fromisoformat(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not a date like 2024-01-31") from None


def _usd(text: str) -> float:
    try:
        value = float(text)
    except ValueError:
        value = math.nan
    if not math.isfinite(value) or value < 0:
        raise argparse.ArgumentTypeError(f"{text!r} is not a dollar amount like 5 or 2.50")
    return value


def _buy_args(p: argparse.ArgumentParser, *, required: bool) -> None:
    p.add_argument("--from", dest="start", type=_day, required=required, metavar="YYYY-MM-DD",
                   help="first day (included)")
    p.add_argument("--to", dest="end", type=_day, required=required, metavar="YYYY-MM-DD",
                   help="end day (not included)")
    p.add_argument("--bars", choices=tuple(_buy.SCHEMAS), required=required,
                   help="bar size: 1m, 1h or 1d")


def _money_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--yes", action="store_true",
                   help="buy without asking (for scripts); --max-cost still applies")
    p.add_argument("--max-cost", type=_usd, default=None, metavar="USD",
                   help="refuse any price above this many dollars, even with --yes")


def _parser() -> argparse.ArgumentParser:
    roots = ", ".join(_buy._BUYABLE_ORDER)
    p = _Parser(prog=PROG, description="Backtest futures strategies on your own CSV or on "
                "Databento bars bought on your own account. Every result is in-sample.")
    sub = p.add_subparsers(dest="command", metavar="COMMAND")

    q = sub.add_parser("quote", help="show Databento's price for some bars; buys nothing",
                       description="Show Databento's own price estimate for a range of "
                       f"bars. Nothing is charged. Roots: {roots}. {_DATES}")
    q.add_argument("root", help=f"futures root: {roots}")
    _buy_args(q, required=True)
    q.set_defaults(func=_cmd_quote)

    f = sub.add_parser("fetch", help="buy bars from Databento after you say yes",
                       description="Show the price, ask before buying, then buy on your "
                       "Databento account (DATABENTO_API_KEY) and keep the bars on this "
                       f"computer in {data_dir()} so they are never bought twice. "
                       f"Roots: {roots}. {_DATES}")
    f.add_argument("root", help=f"futures root: {roots}")
    _buy_args(f, required=True)
    _money_args(f)
    f.add_argument("--session", choices=("rth", "all"), default="rth",
                   help="hours kept in the series saved for offline runs: rth (regular "
                        "trading hours, the default) or all (the full day)")
    f.set_defaults(func=_cmd_fetch)

    r = sub.add_parser("run", help="backtest a strategy on one data source",
                       description="Backtest a strategy on exactly one data source: "
                       "--data FILE, --continuous FILE or --buy ROOT. The report, a "
                       f"ledger line and an equity chart are written. {_DATES}")
    r.add_argument("--strategy", required=True, metavar="SPEC",
                   help="a built-in name (see: plutobt strategies), path.py, or "
                        "path.py:ClassName")
    r.add_argument("--set", action="append", default=[], metavar="NAME=VALUE",
                   help="change one of the strategy's settings (repeatable)")
    r.add_argument("--config", default=None, metavar="FILE",
                   help="cost config (.toml); without one, default costs are used")
    r.add_argument("--session", choices=("rth", "all"), default=None,
                   help="rth: regular trading hours only (default); all: the full day")
    r.add_argument("--allow-short", action="store_true",
                   help="allow short trades (without --config)")
    r.add_argument("--no-ledger", action="store_true", help="do not record this try")
    r.add_argument("--chart-dir", type=Path, default=Path("plutobt_runs"), metavar="DIR",
                   help="folder for the equity chart (default: plutobt_runs)")
    r.add_argument("--data", default=None, metavar="FILE", help="one contract's bars (CSV)")
    r.add_argument("--continuous", default=None, metavar="FILE",
                   help="a continuous series built by python -m plutobt.continuous")
    r.add_argument("--buy", default=None, metavar="ROOT",
                   help=f"buy (or reuse) Databento bars: {roots}")
    r.add_argument("--symbol", default=None, metavar="ROOT",
                   help="the instrument, e.g. NQ (with --data or --continuous)")
    r.add_argument("--preset", choices=PRESETS, default=None,
                   help="CSV layout for --data (default: generic)")
    r.add_argument("--tz", default=None, metavar="ZONE",
                   help="IANA zone for --data timestamps written without one")
    r.add_argument("--ts-is", choices=("open", "close"), default=None,
                   help="whether each --data timestamp is the bar's open or close "
                        "(required for --preset generic)")
    _buy_args(r, required=False)
    _money_args(r)
    r.set_defaults(func=_cmd_run)

    s = sub.add_parser("strategies", help="list the built-in strategies and their settings")
    s.set_defaults(func=_cmd_strategies)
    return p


# ---------------------------------------------------------------------------
# Quote and buy (the money path)
# ---------------------------------------------------------------------------

def _held_codes(q: _buy.Quote) -> str:
    return ", ".join(sorted({p.contract for p in q.held}, key=parse_contract))


def _print_quote(q: _buy.Quote) -> None:
    if not q.pieces:
        held = f" ({_held_codes(q)})" if q.held else ""
        _out(f"{q.root} {q.bars} bars, {q.start} to {q.end}: already on this computer, no "
             f"new charge{held}.")
    else:
        _out(f"Databento price: ${q.price_usd:.2f} for {q.root} {q.bars} bars, {q.start} "
             f"to {q.end} (Databento's own estimate).")
    _out("Nothing was charged.")
    if q.held and q.pieces:
        _out(f"Already on this computer, no charge: {_held_codes(q)}.")
    if q.data_ends is not None:
        _out(f"Databento has data up to {q.data_ends}; the range ends there.")


def _said_yes() -> bool:
    try:
        reply = input(PROMPT)
    except (EOFError, KeyboardInterrupt):
        _out()
        return False
    return reply.strip().lower() in ("y", "yes")


def _quote(root: str, bars: str, start: date, end: date):
    client = _buy.make_client()
    store = Store()
    q = _buy.quote(client, root, bars, start, end, store)
    _print_quote(q)
    return client, store, q


def _fetch(root: str, bars: str, start: date, end: date, *, yes: bool,
           max_cost: float | None):
    """Quote, ask, buy. Returns (exit status, store, quote); status 0 means the bars of
    the quote are all on this computer."""
    client, store, q = _quote(root, bars, start, end)
    if max_cost is not None and q.price_usd > max_cost:
        raise _Refused(f"Databento price ${q.price_usd:.2f} is above --max-cost "
                       f"${max_cost:.2f}. Nothing was bought.")
    if not q.pieces:
        _out("Nothing to buy: all of it is already on this computer.")
        return 0, store, q
    if not yes and not _said_yes():
        _out("Nothing was bought.")
        return 1, store, q
    try:
        # The unrounded quoted price: what the trader said yes to, to the cent and below.
        _buy.buy(client, q, q.price_usd, store, max_cost=max_cost)
    except _buy.PriceChanged as exc:
        _out(f"The price changed to ${exc.new_usd:.2f} since the quote. Nothing was bought. "
             "Run the command again to see the new price.")
        return 1, store, q
    except KeyboardInterrupt:
        raise _Refused("Stopped while buying. Bars saved before the stop stay on this "
                       "computer, and Databento may have billed a piece that was not saved "
                       "(your Databento billing page shows the exact charge). Run the "
                       "command again: it quotes only what is still missing.") from None
    _out(f"Bought. Saved on this computer in {data_dir()}")
    return 0, store, q


def _cmd_quote(a) -> int:
    _quote(a.root, a.bars, a.start, a.end)
    return 0


def _cmd_fetch(a) -> int:
    status, store, q = _fetch(a.root, a.bars, a.start, a.end, yes=a.yes,
                              max_cost=a.max_cost)
    if status != 0:
        return status
    problem = None
    try:
        df, meta = _buy.assemble(store, q, session=a.session)
    except _buy.BuyError as exc:
        problem = str(exc)
    if problem is not None:
        _out(f"The bars are on this computer, but they could not be joined into one series "
             f"for offline runs: {problem}")
        return 0
    _save_series(q, df, meta, a.session, "ema_cross", example=True)
    return 0


def _quoted(path: Path) -> str:
    return f'"{path}"'


def _run_flags(sets, config, allow_short: bool) -> str:
    """The flags of a run that shape its backtest, as text for the printed offline command."""
    text = ""
    for item in sets or []:
        text += f' --set "{item}"' if any(c.isspace() for c in item) else f" --set {item}"
    if config is not None:
        text += f" --config {_quoted(Path(config))}"
    if allow_short:
        text += " --allow-short"
    return text


def _write_series_files(df, meta, path: Path) -> None:
    """Write the CSV (and the .roll.json for a continuous series) through temp files in the
    same folder, then move them into place, so an interrupted save never leaves a short
    series behind that a later run would read without an error."""
    tmp_csv = path.with_name(path.name + ".tmp")
    tmp_json = None
    try:
        (df if meta is not None else df[COLUMNS]).to_csv(tmp_csv, index=False,
                                                         lineterminator=chr(10))
        if meta is not None:
            final_json = Path(_meta_path(str(path)))
            tmp_json = final_json.with_name(final_json.name + ".tmp")
            tmp_json.write_text(meta.to_json(), encoding="utf-8")
        os.replace(tmp_csv, path)
        if tmp_json is not None:
            try:
                os.replace(tmp_json, final_json)
            except BaseException:
                path.unlink(missing_ok=True)  # a new CSV without its metadata is unusable
                raise
    except BaseException:
        for tmp in (tmp_csv, tmp_json):
            if tmp is not None:
                tmp.unlink(missing_ok=True)
        raise


def _save_series(q: _buy.Quote, df, meta, session: str, strategy: str, *,
                 example: bool = False, run_flags: str = "") -> Path | None:
    """Save the assembled bars where ``plutobt run`` reads them without Databento or the
    key, and print the command that does. Two or more contracts: a continuous series
    and its .roll.json (``--continuous``). One contract: a plain CSV of bar opens in unix
    seconds (``--data ... --preset generic --ts-is open``). A failed save (a file open in
    another program, a full disk) is a note, not an error: the bars are already bought."""
    hours = "all" if q.bars == "1d" else session
    folder = data_dir() / "series"
    path = folder / f"{q.root}_{q.bars}_{q.start}_{q.end}_{hours}.csv"
    try:
        folder.mkdir(parents=True, exist_ok=True)
        _write_series_files(df, meta, path)
    except OSError as exc:
        _out(f"{PROG}: note: " + _plain(
            f"Could not save the series for offline runs ({_message(exc)}). The bought bars "
            "are still on this computer; close any program that has the file open and run "
            "the same command again (no new charge)."), err=True)
        return None
    if meta is not None:
        source = f"--continuous {_quoted(path)} --symbol {q.root}"
    else:
        source = (f"--data {_quoted(path)} --symbol {q.root} --preset generic --ts-is open"
                  + (" --session all" if hours == "all" and q.bars != "1d" else ""))
    _out(f"Series saved for offline runs: {path}")
    _out("Run a strategy on it again with no purchase and no key"
         + (" (any strategy; ema_cross shown):" if example else ":"))
    _out(f"  plutobt run {source} --strategy {strategy}{run_flags}")
    return path


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def _check_run(a) -> None:
    """Refuse a mixed-up request before anything is loaded, quoted or bought."""
    sources = [flag for flag, value in (("--data", a.data), ("--continuous", a.continuous),
                                        ("--buy", a.buy)) if value is not None]
    if not sources:
        raise _Refused("pick the data to run on: --data FILE, --continuous FILE or "
                       "--buy ROOT")
    if len(sources) > 1:
        raise _Refused(f"use one data source, not {' and '.join(sources)}")
    source = sources[0]
    if source != "--buy":
        buy_only = [flag for flag, given in (
            ("--from", a.start is not None), ("--to", a.end is not None),
            ("--bars", a.bars is not None), ("--yes", a.yes),
            ("--max-cost", a.max_cost is not None)) if given]
        if buy_only:
            raise _Refused(f"{', '.join(buy_only)} go with --buy, not {source}")
        if a.symbol is None:
            raise _Refused(f"{source} needs --symbol ROOT, for example --symbol NQ")
    else:
        missing = [flag for flag, value in (("--from", a.start), ("--to", a.end),
                                            ("--bars", a.bars)) if value is None]
        if missing:
            raise _Refused(f"--buy needs {', '.join(missing)}")
        if a.symbol is not None and a.symbol.strip().upper() != a.buy.strip().upper():
            raise _Refused(f"--buy {a.buy} names the symbol; drop --symbol {a.symbol}")
    if source != "--data":
        data_only = [flag for flag, value in (("--preset", a.preset), ("--tz", a.tz),
                                              ("--ts-is", a.ts_is)) if value is not None]
        if data_only:
            raise _Refused(f"{', '.join(data_only)} go with --data, not {source}")
    else:
        _check_preset_flags(a.preset or "generic", a.tz, a.ts_is)
    if source == "--continuous" and a.session is not None:
        raise _Refused("--continuous runs the file as built; drop --session")
    if a.allow_short and a.config is not None:
        raise _Refused("--allow-short has no effect with --config; set allow_short in the "
                       "config file")


def _check_preset_flags(preset: str, tz: str | None, ts_is: str | None) -> None:
    """Refuse a --tz or --ts-is the preset never reads, and a generic file without
    --ts-is, before anything is loaded."""
    if preset == "generic" and ts_is is None:
        raise _Refused("--preset generic (the default for --data) needs --ts-is open or "
                       "--ts-is close: whether each timestamp is the bar's open or its "
                       "close. Reading a close as an open is look-ahead.")
    unused: list[str] = []
    why = ""
    if preset == "databento":
        why = "Databento times are UTC bar opens"
        unused = [flag for flag, value in (("--tz", tz), ("--ts-is", ts_is))
                  if value is not None]
    elif preset == "ninjatrader":
        why = "NinjaTrader exports are in UTC"
        unused = ["--tz"] if tz is not None else []
    if unused:
        flags = " and ".join(unused)
        raise _Refused(f"--preset {preset} never uses {flags} ({why}); drop {flags}")


def _zone_text(info: dict, tz: str | None) -> str:
    zone = info["zone"]
    if zone == "unix":
        return "zone: none needed (unix time)"
    if zone == "zoned":
        return "zone: from the file"
    if zone == "naive":
        return f"zone: {tz} (given with --tz)"
    return "zone: UTC (the preset's own rule)"


def _check_bar_size(cls, params: dict, bars: str) -> None:
    """Refuse a bar size a built-in strategy cannot use, before any Databento call."""
    if isinstance(cls, type) and issubclass(cls, OpeningRange):
        minutes = {"1m": 1, "1h": 60, "1d": 1440}[bars]
        window = int(params.get("range_minutes", cls.range_minutes))
        if minutes > window:
            raise _Refused(f"opening_range needs bars of {window} minutes or less; "
                           f"--bars {bars} is too long (use --bars 1m)")


def _cmd_run(a) -> int:
    _check_run(a)
    # Everything that can be refused is checked before any Databento call.
    cls = load_strategy(a.strategy)
    params = parse_settings(cls, a.set)
    config = load_config(a.config) if a.config is not None else None
    cost_note = f"from your config {a.config}" if a.config is not None else None
    symbol = (a.buy if a.buy is not None else a.symbol).strip().upper()
    get_instrument(symbol)
    session = a.session or "rth"
    meta = None
    if a.buy is not None:
        _check_bar_size(cls, params, a.bars)

    if a.data is not None:
        preset = a.preset or "generic"
        # One file is run as one series: the contract tag the importer reads from the
        # file name would make the engine ask for a continuous build's RollMeta.
        df, info = _load_csv(a.data, preset, tz=a.tz, ts_is=a.ts_is)
        if "contract" in df.columns and len(df):
            code = str(df["contract"].iloc[0])
            root = parse_contract(code).root
            if root != symbol:
                raise _Refused(f"This file holds {root} contract {code} but --symbol is "
                               f"{symbol}; use --symbol {root}.")
        df = df.drop(columns="contract", errors="ignore")
        note = (f"{a.data} (preset {preset}, timestamps read as the bar {info['ts_is']}, "
                f"{_zone_text(info, a.tz)})")
    elif a.continuous is not None:
        df, meta = load_continuous(a.continuous)
        if meta.root != symbol:
            raise _Refused(f"{a.continuous} is a {meta.root} series, not {symbol}")
        session = "as_built"
        note = f"continuous file {a.continuous}"
    else:
        status, store, q = _fetch(symbol, a.bars, a.start, a.end, yes=a.yes,
                                  max_cost=a.max_cost)
        if status != 0:
            return status
        df, meta = _buy.assemble(store, q, session=session)
        _save_series(q, df, meta, session, a.strategy,
                     run_flags=_run_flags(a.set, a.config, a.allow_short))
        what = f"{q.root} {q.bars} {q.start} to {q.end}"
        if not q.pieces:
            note = (f"bought from Databento earlier, {what}: already on this computer, no "
                    "new charge")
        else:
            part = " for the part bought now" if q.held else ""
            note = (f"bought from Databento, {what}, Databento's estimate at purchase: "
                    f"${q.price_usd:.2f}{part} (exact charge: your Databento billing page)")

    outcome = run_backtest(df, symbol, cls, params, config=config, cost_note=cost_note,
                           session=session, allow_short=a.allow_short, roll_meta=meta,
                           data_note=note, ledger=False if a.no_ledger else DEFAULT_PATH,
                           chart_dir=a.chart_dir)
    _out(outcome.text)
    _out(f"Chart: {outcome.chart}")
    return 0


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

def _cmd_strategies(a) -> int:
    for name in BUILTINS:
        cls = load_strategy(name)
        doc = (inspect.getdoc(cls) or "").strip().splitlines()
        _out(f"{name}: {doc[0] if doc else cls.__name__}")
        found = settings(cls)
        for key, value in found.items():
            _out(f"  {key} = {value}")
        if not found:
            _out("  (no settings)")
    _out("Run one with: plutobt run --strategy NAME --set SETTING=VALUE ...")
    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    """Run the command; returns the exit status (0 done, 1 no or price changed, 2 refused
    or an error)."""
    with warnings.catch_warnings():
        warnings.showwarning = _note
        return _main(argv)


def _main(argv: list[str] | None) -> int:
    try:
        try:
            args = _parser().parse_args(argv)
        except SystemExit as exc:          # --help prints and exits 0
            return exc.code if isinstance(exc.code, int) else 0
        if args.command is None:
            raise _Refused("pick a command: quote, fetch, run or strategies (see: "
                           f"{PROG} --help)")
        return args.func(args)
    except KeyboardInterrupt:
        _out("Stopped.", err=True)
        return 1
    except Exception as exc:
        return _fail(_message(exc))
