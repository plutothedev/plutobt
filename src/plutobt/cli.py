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
import sys
import warnings
from datetime import date
from pathlib import Path

from plutobt import buy as _buy
from plutobt.config import load_config
from plutobt.importers import PRESETS, load_csv
from plutobt.instruments import get_instrument
from plutobt.ledger import DEFAULT_PATH
from plutobt.roll import load_continuous
from plutobt.runner import run_backtest
from plutobt.store import Store, data_dir
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


def _fail(message: str) -> int:
    text = " ".join(_buy._redact(message, _buy._keys(None)).split())
    _out(f"{PROG}: {text.encode('ascii', 'backslashreplace').decode('ascii')}", err=True)
    return 2


def _note(message, category, filename, lineno, file=None, line=None) -> None:
    """A warning as one plain line, without the source file and code line."""
    _out(f"{PROG}: note: {' '.join(str(message).split())}", err=True)


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

def _print_quote(q: _buy.Quote) -> None:
    _out(f"Databento price: ${q.price_usd:.2f} for {q.root} {q.bars} bars, {q.start} to "
         f"{q.end} (Databento's own estimate).")
    _out("Nothing was charged.")
    if q.held:
        n = len({p.contract for p in q.held})
        _out(f"Already on this computer: {n} contract month(s), no charge.")
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
    status, _, _ = _fetch(a.root, a.bars, a.start, a.end, yes=a.yes, max_cost=a.max_cost)
    return status


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
    if source == "--continuous" and a.session is not None:
        raise _Refused("--continuous runs the file as built; drop --session")
    if a.allow_short and a.config is not None:
        raise _Refused("--allow-short has no effect with --config; set allow_short in the "
                       "config file")


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

    if a.data is not None:
        preset = a.preset or "generic"
        # One file is run as one series: the contract tag the importer reads from the
        # file name would make the engine ask for a continuous build's RollMeta.
        df = load_csv(a.data, preset, tz=a.tz, ts_is=a.ts_is)
        df = df.drop(columns="contract", errors="ignore")
        note = f"{a.data} (preset {preset}"
        if a.ts_is is not None:
            note += f", timestamps read as the bar {a.ts_is}"
        note += f", zone {a.tz})" if a.tz is not None else ")"
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
        note = (f"bought from Databento, {q.root} {q.bars} {q.start} to {q.end}, "
                f"Databento's estimate at purchase: ${q.price_usd:.2f} (exact charge: your "
                "Databento billing page)")

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
