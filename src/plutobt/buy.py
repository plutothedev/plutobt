"""Quote, buy and assemble futures bars from Databento, on the trader's own account.

The money path, in the order the ``plutobt`` command runs it:

1. ``quote`` asks Databento (free calls only) which contract was the front month when
   (two symbology calls: continuous -> instrument id, then the ids -> raw symbols),
   works out each contract's window (its front-month days padded by ``PAD_DAYS`` around
   each roll so neighbouring contracts overlap there, never outside the requested
   range), takes away what the local
   store already holds, and prices the rest with Databento's free cost lookup. Nothing is
   charged.
2. The trader sees the price and says yes or no.
3. ``buy`` refuses a quote whose pieces the store already holds, then prices the same
   pieces again right before buying. A fresh price above ``max_cost`` buys nothing; a
   price different from the one the trader said yes to (by more than half a cent) buys
   nothing and raises ``PriceChanged``. Then each piece
   is bought with one ``get_range`` call and saved to the store at once. A failure stops
   the buy, names what was already saved, and is never retried.
4. ``assemble`` reads each contract back from the store and joins them into one
   back-adjusted continuous series with PlutoBT's own builder (calendar roll for the
   equity-index roots, volume roll for CL, GC and ZN).

The ``databento`` package is optional (``pip install -e ".[databento]"`` in the PlutoBT
folder) and is imported only inside ``make_client``. The API key is read from
``DATABENTO_API_KEY`` only, never saved, and replaced by ``***`` in any error text that
passes through this module.

Dates are ``datetime.date`` and every range is half-open [start, end) of UTC days.
"""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta

import pandas as pd

from plutobt.contracts import CALENDAR_ROOTS, ContractSet, calendar_roll_day, parse_contract
from plutobt.data import tag_sessions
from plutobt.instruments import get_instrument
from plutobt.roll import (
    RollMeta,
    _volume_roll_ts,
    build_continuous,
    slice_continuous,
    trading_day,
)
from plutobt.store import Store

DATASET = "GLBX.MDP3"
SCHEMAS = {"1m": "ohlcv-1m", "1h": "ohlcv-1h", "1d": "ohlcv-1d"}
_BUYABLE_ORDER = ("NQ", "MNQ", "ES", "MES", "YM", "RTY", "CL", "GC", "ZN")
BUYABLE = frozenset(_BUYABLE_ORDER)
PAD_DAYS = 10
PRICE_TOLERANCE_USD = 0.005
_KEY_ENV = "DATABENTO_API_KEY"
# A plain outright as GLBX names every front contract: root, month code, one-digit year.
_OUTRIGHT = re.compile(r"^[A-Z]{1,6}[FGHJKMNQUVXZ]\d$")


class BuyError(Exception):
    """A quote or buy that cannot go ahead. The text is plain and never holds the key."""


class PriceChanged(BuyError):
    """Databento's fresh price differs from the one the trader said yes to."""

    def __init__(self, new_usd: float, message: str | None = None):
        self.new_usd = float(new_usd)
        super().__init__(message or (
            f"Databento's price is now ${self.new_usd:.2f}. Nothing was bought; "
            "say yes to the new price to buy."))


@dataclass(frozen=True)
class Piece:
    """One contract's bars over [start, end): to buy (in ``Quote.pieces``) or already
    held (in ``Quote.held``). ``contract`` is the canonical code (``NQH2024``),
    ``raw_symbol`` Databento's (``NQH4``)."""

    contract: str
    raw_symbol: str
    start: date
    end: date


@dataclass(frozen=True)
class Quote:
    """What a request costs. ``start``/``end`` are the request, ``end`` clamped to the
    data Databento has (``data_ends`` is set to that end when it was clamped, else None).
    ``pieces`` are what would be bought, ``held`` what the store already has, and
    ``price_usd`` the sum of Databento's own estimate for each piece (0.0 when nothing is
    missing)."""

    root: str
    bars: str
    start: date
    end: date
    pieces: tuple[Piece, ...]
    held: tuple[Piece, ...]
    price_usd: float
    data_ends: date | None


# ---------------------------------------------------------------------------
# Key handling
# ---------------------------------------------------------------------------

def _redact(text: str, key) -> str:
    """``text`` with every occurrence of ``key`` (a string, or several) replaced by
    ``***``. A None or blank key changes nothing."""
    keys = [key] if isinstance(key, str) or key is None else list(key)
    out = str(text)
    for k in sorted((k for k in keys if isinstance(k, str) and k.strip()), key=len,
                    reverse=True):
        out = out.replace(k, "***")
    return out


def _keys(client) -> list[str]:
    """Every form of the key this process could leak: the environment value (raw and
    stripped) and whatever the client holds."""
    env = os.environ.get(_KEY_ENV)
    candidates = [env, env.strip() if isinstance(env, str) else None]
    for attr in ("key", "_key"):
        try:
            candidates.append(getattr(client, attr, None))
        except Exception:
            pass
    found: list[str] = []
    for k in candidates:
        if isinstance(k, str) and k.strip() and k not in found:
            found.append(k)
    return found


def _describe(exc: BaseException) -> str:
    try:
        text = str(exc)
    except Exception:
        text = ""
    name = type(exc).__name__
    return f"{name}: {text}" if text else name


def _call(what: str, keys: list[str], fn, *args, **kwargs):
    """Run a Databento call. Any exception becomes a BuyError whose text has the key
    redacted, raised outside the ``except`` so the original exception (which may hold
    the key) never rides along as its cause or context."""
    try:
        return fn(*args, **kwargs)
    except Exception as exc:
        message = _redact(f"{what} failed: {_describe(exc)}", keys)
    raise BuyError(message)


_INSTALL = 'in the PlutoBT folder run: pip install -e ".[databento]"'


def make_client() -> object:
    """A ``databento.Historical`` client on the trader's own key (``DATABENTO_API_KEY``).
    The only place PlutoBT imports ``databento``."""
    problem = None
    try:
        import databento
    except ModuleNotFoundError as exc:
        databento = None
        if exc.name != "databento":
            problem = _describe(exc)
    except ImportError as exc:
        databento = None
        problem = _describe(exc)
    if databento is None and problem is None:
        raise BuyError(f"Buying data needs the Databento add-on, which is not installed: "
                       f"{_INSTALL}")
    if databento is None:
        # Installed, but it (or a package it needs) failed to import: say so, with the
        # reason, on one plain ASCII line. Raised outside the except: no chained cause.
        reason = " ".join(_redact(problem, _keys(None)).split())
        reason = reason.encode("ascii", "backslashreplace").decode("ascii")
        raise BuyError(f"The Databento add-on is installed but failed to load ({reason}). "
                       f"To reinstall it, {_INSTALL}")
    raw = os.environ.get(_KEY_ENV)
    if raw is None or not raw.strip():
        raise BuyError("Set DATABENTO_API_KEY to your Databento API key (it is read from "
                       "the environment only and never saved).")
    return _call("Connecting to Databento", [raw, raw.strip()], databento.Historical,
                 key=raw.strip())


# ---------------------------------------------------------------------------
# Quote
# ---------------------------------------------------------------------------

def _utc(text) -> pd.Timestamp:
    t = pd.Timestamp(str(text))
    return t.tz_localize("UTC") if t.tz is None else t.tz_convert("UTC")


def _floor_date(text) -> date:
    return _utc(text).date()


def _ceil_date(text) -> date:
    t = _utc(text)
    return t.date() if t == t.normalize() else t.date() + timedelta(days=1)


def _day_ts(d: date) -> int:
    return int(pd.Timestamp(d, tz="UTC").timestamp())


def _schema_range(client, schema: str, keys: list[str]) -> tuple[date, date]:
    """(first full day, exclusive end day) of the schema's data. A partly available last
    day is excluded."""
    rng = _call("Asking Databento which dates it has", keys,
                client.metadata.get_dataset_range, DATASET)
    try:
        part = (rng.get("schema") or {}).get(schema) or rng
        return _ceil_date(part["start"]), _floor_date(part["end"])
    except Exception:
        pass
    raise BuyError(f"Databento's date range for {DATASET} {schema} came back in a form "
                   "PlutoBT cannot read; nothing was charged.")


def _check_price(value, raw: str, keys: list[str]) -> float:
    ok = (isinstance(value, int | float) and not isinstance(value, bool)
          and math.isfinite(float(value)) and float(value) >= 0)
    if not ok:
        try:
            shown = repr(value)
        except Exception:
            shown = type(value).__name__
        raise BuyError(_redact(f"Databento returned a price PlutoBT cannot use for {raw} "
                               f"({shown}); nothing was charged.", keys))
    return float(value)


def _price(client, pieces, schema: str, keys: list[str]) -> float:
    """Databento's free cost estimate, summed over ``pieces``, one call per piece."""
    total = 0.0
    for p in pieces:
        value = _call(f"Asking Databento the price of {p.raw_symbol}", keys,
                      client.metadata.get_cost, dataset=DATASET, symbols=[p.raw_symbol],
                      schema=schema, start=p.start.isoformat(), end=p.end.isoformat(),
                      stype_in="raw_symbol")
        total += _check_price(value, p.raw_symbol, keys)
    return round(total, 6)


def _check_request(root, bars, start, end) -> str:
    root = root.strip().upper() if isinstance(root, str) else ""
    if root not in BUYABLE:
        raise BuyError(f"PlutoBT buys futures bars for {', '.join(_BUYABLE_ORDER)} only, "
                       f"not {root or '(none)'}. Load other data (stocks such as AAPL) "
                       "from a CSV.")
    if bars not in SCHEMAS:
        raise BuyError(f"bars must be one of {', '.join(SCHEMAS)}, not {bars!r}")
    for name, d in (("start", start), ("end", end)):
        if not isinstance(d, date) or isinstance(d, datetime):
            raise BuyError(f"the {name} must be a date, not {d!r}")
    if start >= end:
        raise BuyError(f"the start date {start} must be before the end date {end}")
    return root


def _front_spans(answer, root: str, keys: list[str]) -> list[list]:
    """[Contract, raw symbol, first d0, last d1] per contract, in contract order, from a
    symbology.resolve answer (d1 exclusive).

    Grouped by the canonical contract, not the raw symbol: GLBX raw symbols carry a
    one-digit year, so ``NQH4`` is NQH2014 in one interval and NQH2024 in another. Each
    interval is read in its own decade (``parse_contract(raw, near=d0)``)."""
    spans: dict[str, list] = {}
    problem = None
    try:
        rows = (answer.get("result") or {}).get(f"{root}.v.0") or []
        for row in rows:
            raw = str(row["s"])
            d0 = date.fromisoformat(str(row["d0"])[:10])
            d1 = date.fromisoformat(str(row["d1"])[:10])
            if d0 >= d1:
                continue
            con = parse_contract(raw, near=d0)
            if con.code not in spans:
                spans[con.code] = [con, raw, d0, d1]
            else:
                spans[con.code][2] = min(spans[con.code][2], d0)
                spans[con.code][3] = max(spans[con.code][3], d1)
    except Exception as exc:
        problem = _redact(_describe(exc), keys)
    if problem is not None:
        raise BuyError(f"Databento's front-month list for {root} could not be read "
                       f"({problem}); nothing was charged.")
    for con, raw, _, _ in spans.values():
        if con.root != root:
            raise BuyError(f"Databento named {raw} as a {root} front contract; PlutoBT "
                           "will not buy a different root. Nothing was charged.")
    return sorted(spans.values(), key=lambda span: span[0])


def _front_ids(answer, root: str, keys: list[str]) -> list[tuple[date, date, str]]:
    """(d0, d1, instrument id) per front-month span, from a continuous -> instrument_id
    symbology.resolve answer (d1 exclusive; empty spans dropped)."""
    rows: list[tuple[date, date, str]] = []
    problem = None
    try:
        for row in (answer.get("result") or {}).get(f"{root}.v.0") or []:
            ident = str(row["s"]).strip()
            d0 = date.fromisoformat(str(row["d0"])[:10])
            d1 = date.fromisoformat(str(row["d1"])[:10])
            if not ident:
                raise ValueError("a front-month span has no instrument id")
            if d0 < d1:
                rows.append((d0, d1, ident))
    except Exception as exc:
        problem = _redact(_describe(exc), keys)
    if problem is not None:
        raise BuyError(f"Databento's front-month list for {root} could not be read "
                       f"({problem}); nothing was charged.")
    return rows


def _raw_rows(rows, answer, root: str, keys: list[str]) -> list[dict]:
    """The front-month rows with each instrument id replaced by its raw symbol, read from
    an instrument_id -> raw_symbol answer: the symbol of the id's interval that holds the
    row's whole span (x.d0 <= row.d0 and row.d1 <= x.d1).

    CME recycles instrument ids across unrelated instruments over the years, so one id
    can name options, spreads and other roots before and after its contract's life, with
    gaps between. An interval holding only part of the row is that case and is never
    guessed: no interval holding the whole row, or more than one naming different
    symbols, refuses. So does a symbol that is not a plain outright (root, month code,
    one-digit year, as GLBX names every front contract); a different root is refused
    later by ``_front_spans``."""
    out: list[dict] = []
    problem = None
    try:
        table = {str(k): v for k, v in (answer.get("result") or {}).items()}
        for d0, d1, ident in rows:
            names = set()
            for iv in table.get(ident) or []:
                lo = date.fromisoformat(str(iv["d0"])[:10])
                hi = date.fromisoformat(str(iv["d1"])[:10])
                if lo <= d0 and d1 <= hi:
                    names.add(str(iv["s"]))
            if len(names) != 1:
                what = "no contract" if not names else "more than one contract"
                problem = (f"Databento named {what} for {root} instrument id {ident} for "
                           f"the whole of {d0} to {d1} (end not included); nothing was "
                           "charged.")
                break
            raw = names.pop()
            if not _OUTRIGHT.match(raw):
                problem = (f"Databento named {raw!r} for {root} instrument id {ident} from "
                           f"{d0} to {d1}, which is not a plain futures contract; nothing "
                           "was charged.")
                break
            out.append({"d0": d0.isoformat(), "d1": d1.isoformat(), "s": raw})
    except Exception as exc:
        problem = (f"Databento's contract list for {root} could not be read "
                   f"({_redact(_describe(exc), keys)}); nothing was charged.")
    if problem is not None:
        raise BuyError(_redact(problem, keys))
    return out


def _unmapped(answer, symbol: str, lists=("not_found", "partial")) -> bool:
    """True when symbology.resolve lists ``symbol`` in any of ``lists``: ``not_found``
    (not mapped at all) or ``partial`` (mapped for only part of the range)."""
    try:
        listed = [x for name in lists for x in (answer.get(name) or [])]
    except Exception:
        return False
    return symbol in [str(x) for x in listed]


def quote(client, root: str, bars: str, start: date, end: date, store: Store) -> Quote:
    """Price ``root`` ``bars`` bars over [start, end) without buying anything."""
    root = _check_request(root, bars, start, end)
    schema = SCHEMAS[bars]
    keys = _keys(client)

    first_day, avail_end = _schema_range(client, schema, keys)
    if start < first_day:
        raise BuyError(f"Databento's {DATASET} {schema} data starts {first_day}; pick a "
                       "start date on or after that.")
    if start >= avail_end:
        raise BuyError(f"Databento's {DATASET} {schema} data ends {avail_end} (that day "
                       f"not included); the start date {start} is past it.")
    data_ends = None
    if end > avail_end:
        end, data_ends = avail_end, avail_end

    # GLBX refuses continuous -> raw_symbol (422), so two free calls: which instrument id
    # was front when, then ONE call naming the raw symbol of every id over the same range.
    cont = f"{root}.v.0"
    answer = _call("Asking Databento which contract was front when", keys,
                   client.symbology.resolve, dataset=DATASET, symbols=[cont],
                   stype_in="continuous", stype_out="instrument_id",
                   start_date=start.isoformat(), end_date=end.isoformat())
    rows = _front_ids(answer, root, keys)
    if rows and _unmapped(answer, cont):
        raise BuyError(f"Databento could not map {root} for part of that range; nothing "
                       "was priced or bought.")
    if not rows:
        raise BuyError(f"Databento lists no {root} front contract between {start} and {end} "
                       "(end not included); nothing was charged.")
    ids = list(dict.fromkeys(ident for _, _, ident in rows))
    named = _call("Asking Databento which contract each front-month id is", keys,
                  client.symbology.resolve, dataset=DATASET, symbols=ids,
                  stype_in="instrument_id", stype_out="raw_symbol",
                  start_date=start.isoformat(), end_date=end.isoformat())
    # Step 2's ``partial`` is normal on real GLBX: CME recycles ids, so an id's intervals
    # rarely cover a long window. Only ``not_found`` refuses here; ``_raw_rows`` then
    # demands one interval holding each front-month row whole.
    if any(_unmapped(named, ident, lists=("not_found",)) for ident in ids):
        raise BuyError(f"Databento could not map {root} for part of that range; nothing "
                       "was priced or bought.")
    spans = _front_spans({"result": {cont: _raw_rows(rows, named, root, keys)}}, root, keys)
    if not spans:
        raise BuyError(f"Databento lists no {root} front contract between {start} and {end} "
                       "(end not included); nothing was charged.")

    if root in CALENDAR_ROOTS:
        # A contract CME's calendar rolls into only after the end date never trades
        # inside the range: buy the outgoing contract through the end instead.
        while len(spans) >= 2 and calendar_roll_day(spans[-2][0]) > end:
            spans.pop()
        # Likewise a first contract CME's calendar rolled out of before the start date
        # (Databento's volume-ranked front can lag the calendar by a day or two).
        while len(spans) >= 2 and calendar_roll_day(spans[0][0]) < start:
            spans.pop(0)

    # Windows: the front-month days padded by PAD_DAYS around each roll so neighbours
    # overlap there, but never outside the request: the first contract starts at the
    # requested start and the last ends at the requested end.
    pieces: list[Piece] = []
    held: list[Piece] = []
    pad = timedelta(days=PAD_DAYS)
    for i, (con, raw, d0, d1) in enumerate(spans):
        w0 = start if i == 0 else max(d0 - pad, start)
        w1 = end if i == len(spans) - 1 else min(d1 + pad, end)
        if w0 >= w1:
            continue
        for s, e in store.missing(schema, con.code, w0, w1):
            pieces.append(Piece(con.code, raw, s, e))
        for s, e in store.held(schema, con.code):
            s, e = max(s, w0), min(e, w1)
            if s < e:
                held.append(Piece(con.code, raw, s, e))

    price = _price(client, pieces, schema, keys) if pieces else 0.0
    return Quote(root=root, bars=bars, start=start, end=end, pieces=tuple(pieces),
                 held=tuple(held), price_usd=price, data_ends=data_ends)


# ---------------------------------------------------------------------------
# Buy
# ---------------------------------------------------------------------------

def _label(p: Piece) -> str:
    return f"{p.contract} {p.start} to {p.end}"


def _engine_frame(raw: pd.DataFrame, p: Piece) -> pd.DataFrame:
    """Databento's ``to_df(price_type="float")`` frame as engine bars over [start, end):
    ts int64 unix seconds of the bar open (UTC), prices float64, volume int64. Spread rows
    (a ``symbol`` holding ``-``) are dropped."""
    df = raw
    if "symbol" in df.columns:
        df = df[~df["symbol"].astype(str).str.contains("-", regex=False)]
    idx = pd.DatetimeIndex(df["ts_event"] if "ts_event" in df.columns else df.index)
    idx = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
    out = pd.DataFrame({
        "ts": (idx.as_unit("ns").asi8 // 1_000_000_000).astype("int64"),
        "open": df["open"].to_numpy(dtype="float64"),
        "high": df["high"].to_numpy(dtype="float64"),
        "low": df["low"].to_numpy(dtype="float64"),
        "close": df["close"].to_numpy(dtype="float64"),
        "volume": df["volume"].to_numpy(dtype="int64"),
    })
    lo, hi = _day_ts(p.start), _day_ts(p.end)
    out = out[(out["ts"] >= lo) & (out["ts"] < hi)]
    return out.sort_values("ts", kind="stable").reset_index(drop=True)


def _finite(value) -> bool:
    return (isinstance(value, int | float) and not isinstance(value, bool)
            and math.isfinite(float(value)))


def buy(client, q: Quote, approved_usd: float, store: Store, *,
        max_cost: float | None = None) -> None:
    """Buy ``q.pieces`` on the trader's Databento account, for ``approved_usd`` only.

    Refuses (buying nothing) a quote whose pieces the store no longer misses in full.
    Then re-prices: a fresh price above ``max_cost`` raises BuyError, and a fresh price
    more than half a cent from ``approved_usd`` raises ``PriceChanged``; both buy
    nothing. Then one ``get_range`` per piece, each saved to ``store`` at once.
    A failure raises BuyError naming the pieces already saved; nothing is retried."""
    if not q.pieces:
        return
    if not _finite(approved_usd):
        raise BuyError(f"the approved price must be a number, not {approved_usd!r}. "
                       "Nothing was bought.")
    if max_cost is not None and not _finite(max_cost):
        raise BuyError(f"--max-cost must be a number, not {max_cost!r}. Nothing was bought.")
    schema = SCHEMAS[q.bars]
    keys = _keys(client)

    # Never buy what this computer already holds: a reused quote, or another run that
    # bought the same bars while the trader sat at the prompt.
    for p in q.pieces:
        if store.missing(schema, p.contract, p.start, p.end) != [(p.start, p.end)]:
            raise BuyError("The data on this computer changed since the quote; quote "
                           "again. Nothing was bought.")

    fresh = _price(client, q.pieces, schema, keys)
    # --max-cost first, so the trader is never asked to approve a price it would refuse.
    if max_cost is not None and fresh > float(max_cost):
        raise BuyError(f"Databento price ${fresh:.2f} is above --max-cost "
                       f"${float(max_cost):.2f}. Nothing was bought.")
    if abs(fresh - float(approved_usd)) > PRICE_TOLERANCE_USD:
        raise PriceChanged(fresh, (
            f"Databento's price is now ${fresh:.2f}, not the ${float(approved_usd):.2f} "
            "you said yes to. Nothing was bought; say yes to the new price to buy."))

    saved: list[Piece] = []
    for p in q.pieces:
        message = None
        # Set before the call: a call that raised (a timeout, a dropped connection) may
        # still have been billed, so the caveat goes with every failure from here on.
        called = False
        try:
            called = True
            data = client.timeseries.get_range(
                dataset=DATASET, symbols=[p.raw_symbol], schema=schema,
                start=p.start.isoformat(), end=p.end.isoformat(), stype_in="raw_symbol")
            frame = _engine_frame(data.to_df(price_type="float"), p)
            store.add(schema, p.contract, p.start, p.end, frame)
        except Exception as exc:
            message = _describe(exc)
        if message is not None:
            before = (f"Saved before the failure: {', '.join(_label(s) for s in saved)}."
                      if saved else "Before the failure nothing was saved.")
            billed = (f" Databento may have billed {_label(p)} although its bars were not "
                      "saved; your Databento billing page shows the exact charge."
                      if called else "")
            raise BuyError(_redact(
                f"Buying {_label(p)} failed: {message}. {before}{billed} Nothing was "
                "retried. Run the command again: it quotes only what is still missing.",
                keys))
        saved.append(p)


# ---------------------------------------------------------------------------
# Assemble
# ---------------------------------------------------------------------------

def assemble(store: Store, q: Quote, *, session: str) -> tuple[pd.DataFrame, RollMeta | None]:
    """The bought bars of ``q`` as one engine frame over [q.start, q.end).

    ``session="rth"`` keeps regular trading hours only, filtering each contract BEFORE
    the continuous build so every roll lands on a kept bar and the RollMeta pin holds;
    ``"all"`` keeps the full day. Daily bars are never filtered. One contract: its bars
    alone and no RollMeta. Two or more: a back-adjusted continuous series (calendar roll
    for the equity-index roots, volume roll otherwise), sliced to the range."""
    if session not in ("rth", "all"):
        raise BuyError(f"session must be 'rth' or 'all', not {session!r}")
    if q.bars not in SCHEMAS:
        raise BuyError(f"bars must be one of {', '.join(SCHEMAS)}, not {q.bars!r}")
    schema = SCHEMAS[q.bars]
    instrument = get_instrument(q.root)
    rth = session == "rth" and q.bars != "1d"

    windows: dict[str, tuple[date, date]] = {}
    for p in (*q.held, *q.pieces):
        s, e = windows.get(p.contract, (p.start, p.end))
        windows[p.contract] = (min(s, p.start), max(e, p.end))
    if not windows:
        raise BuyError(f"the quote for {q.root} holds no contracts")

    frames: dict[str, pd.DataFrame] = {}
    for code in sorted(windows, key=parse_contract):
        problem = None
        try:
            f = store.frame(schema, code, *windows[code])
        except ValueError as exc:
            problem = str(exc)
        if problem is not None:
            raise BuyError(f"{problem}. Buy the quoted data first.")
        if rth:
            f = tag_sessions(f, instrument)
            f = f[f["session"] == "RTH"].drop(columns="session").reset_index(drop=True)
        if len(f):
            frames[code] = f

    codes = list(frames)
    # A last contract that never takes over inside the bars held (the data ends before
    # its roll) would stop the build with no bar to roll at, and none of its bars would
    # be used. Leave it out only when the contract before it already covers the range
    # to the end; otherwise the series would silently stop early, so refuse.
    while len(codes) >= 2:
        prev, last = frames[codes[-2]], frames[codes[-1]]
        if q.root in CALENDAR_ROOTS:
            roll_day = calendar_roll_day(parse_contract(codes[-2]))
            unused = trading_day(int(last["ts"].iloc[-1])) < roll_day
            how = f"by CME's calendar roll on {roll_day}"
        else:
            unused = _volume_roll_ts(prev, last) is None
            how = "by volume"
        if not unused:
            break
        if windows[codes[-2]][1] < q.end:
            raise BuyError(
                f"{codes[-1]} never took over from {codes[-2]} {how} in the bars held, and "
                f"{codes[-2]}'s bars stop at {windows[codes[-2]][1]}, before the requested "
                f"end {q.end}. The bought data is kept; try an end date on or before "
                f"{windows[codes[-2]][1]}.")
        codes.pop()

    # The contracts used must cover the whole request: a series that starts late or
    # stops early would be a backtest over less than the trader asked for.
    if codes and (windows[codes[0]][0] > q.start or windows[codes[-1]][1] < q.end):
        raise BuyError(
            f"the {q.root} bars held cover {windows[codes[0]][0]} to "
            f"{windows[codes[-1]][1]}, not the requested {q.start} to {q.end} (end not "
            "included). Quote and buy the missing days first.")

    start_ts, end_ts = _day_ts(q.start), _day_ts(q.end) - 1
    if len(codes) <= 1:
        df = frames[codes[0]] if codes else None
        if df is not None:
            df = df[(df["ts"] >= start_ts) & (df["ts"] <= end_ts)].reset_index(drop=True)
        if df is None or df.empty:
            hours = " in regular trading hours" if rth else ""
            raise BuyError(f"no {q.root} bars{hours} between {q.start} and {q.end} (end "
                           "not included)")
        return df, None

    trigger = "calendar" if q.root in CALENDAR_ROOTS else "volume"
    problem = None
    try:
        cs = ContractSet.from_frames({c: frames[c] for c in codes})
        df, meta = build_continuous(cs, trigger=trigger)
        return slice_continuous(df, meta, start_ts, end_ts)
    except ValueError as exc:
        problem = str(exc)
    raise BuyError(f"could not join the {q.root} contracts into one series: {problem}")
