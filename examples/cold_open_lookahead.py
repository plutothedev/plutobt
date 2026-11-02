"""Cold-open demo: the lookahead lie, in one changed line.

Same data, same signal, ONE line flipped (WHEN the signal is allowed to act) and a
gorgeous equity curve collapses.

The signal is deliberately obvious: "did this bar close green?" (close > previous
close). You only know it once the bar has CLOSED. Each line below holds a position
through a bar, in at that bar's open and out at the next bar's open:

  the lie    position = signal            bar t is traded on bar t's OWN close, a
                                          close that does not exist yet when the bar
                                          opens. Impossible in real life.
  the truth  position = signal.shift(1)   the signal from bar t acts at bar t+1's
                                          open, the first price you could really get.

Both are frictionless pandas arithmetic: no commission, no slippage, unlevered
(growth of 1x notional). The "bar hit rate" is the share of bars held that went up,
NOT a trade win rate. The exact same off-by-one hides inside any indicator (EMA, RSI,
whatever); this one is just visible to the eye.

A third line runs the same signal through the real PlutoBT Engine with the NQ demo
costs (configs/nq.toml): decision at bar t, fill at bar t+1's open, slippage and
commission on every fill, one NQ contract, pending orders cancelled across the
overnight gap. Its win rate is the closed-trade win rate, net of costs.

This script is a teaching prop for the video's cold open, not part of the engine.

Run:  PYTHONPATH=src python examples/cold_open_lookahead.py
"""

from __future__ import annotations

import json
import pathlib

import pandas as pd

from plutobt.config import load_config
from plutobt.data import load_bars
from plutobt.engine import Engine
from plutobt.instruments import get_instrument
from plutobt.metrics import compute_metrics, sharpe
from plutobt.strategy import Strategy

_ROOT = pathlib.Path(__file__).resolve().parents[1]
SAMPLE = _ROOT / "data" / "sample" / "nq_5m.json"
NQ_CONFIG = _ROOT / "configs" / "nq.toml"
OUT_PNG = _ROOT / "examples" / "output" / "cold_open_lie_vs_truth.png"
# The sample is NQ 5-minute RTH bars, 08:30-15:15 CT: 81 bars/day x 252 = 20412. Read
# from the NQ config, the same basis the engine line uses (tests pin the two together).
PPY = load_config(str(NQ_CONFIG)).periods_per_year


def simulate(signal: pd.Series, bar_returns: pd.Series, *, cheat: bool):
    """Return (equity_curve, annualized_sharpe, per_bar_returns).

    ``bar_returns[t]`` is the move from bar t's open to bar t+1's open.

    THE ONE TOGGLE:
      cheat  -> position = signal           bar t traded on its own close   [the lie]
      honest -> position = signal.shift(1)  acts at the next bar's open     [the truth]
    """
    position = signal if cheat else signal.shift(1)
    strat_returns = (position.fillna(0.0) * bar_returns).fillna(0.0)
    equity = (1.0 + strat_returns).cumprod()
    return equity, sharpe(strat_returns, PPY), strat_returns


def bar_hit_rate(returns: pd.Series) -> float:
    """Share of the bars held (non-zero return) that went up. Not a trade win rate."""
    active = returns[returns != 0.0]
    return float((active > 0.0).mean()) if len(active) else 0.0


class GreenBar(Strategy):
    """The same "did this bar close green?" signal, through the real engine: long one
    contract while it holds, flat when it does not. Long only, no stop."""

    def precompute(self, df: pd.DataFrame) -> pd.DataFrame:
        df["green"] = df["close"] > df["close"].shift(1)   # bar t's own close: causal
        return df

    def on_bar(self, ctx) -> None:
        if ctx.bar.green and ctx.position.qty == 0:
            ctx.broker.buy(contracts=1)
        elif not ctx.bar.green and ctx.position.qty != 0:
            ctx.broker.close()


def run_engine(sample_path: str | None = None) -> dict:
    """The green-bar signal through Engine.run with the NQ demo costs."""
    cfg = load_config(str(NQ_CONFIG))
    inst = get_instrument("NQ")
    result = Engine(inst, cfg).run(load_bars(str(sample_path or SAMPLE)), GreenBar())
    m = compute_metrics(result, periods_per_year=cfg.periods_per_year)
    start = float(result.equity_curve.iloc[0])
    end = float(result.equity_curve.iloc[-1])
    # Where the money went: every fill paid slippage_ticks x tick_value per contract (a
    # position still open at the end paid its entry fill only), every trade commission.
    fills = sum((1 if t.open_at_end else 2) * t.contracts for t in result.trades)
    slippage = fills * cfg.slippage_ticks * inst.tick_value
    commission = sum(t.commission for t in result.trades)
    net = end - start
    return {
        "sharpe": m["sharpe"],
        "n_trades": m["n_trades"],
        "win_rate": m["win_rate"],
        "start_equity": start,
        "final_equity": end,
        "net_pnl": net,
        "slippage_usd": slippage,
        "commission_usd": commission,
        "gross_pnl": net + slippage + commission,
        "commission_per_side": cfg.commission_per_side,
        "slippage_ticks": cfg.slippage_ticks,
    }


def _save_plot(eq_cheat: pd.Series, eq_honest: pd.Series) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(eq_cheat.to_numpy(), label="THE LIE (bar traded on its own close)",
            linewidth=2.5)
    ax.plot(eq_honest.to_numpy(), label="THE TRUTH (next-bar-open fill, before costs)",
            linewidth=2.5)
    ax.set_title("Your backtest is lying to you (one line changed)")
    ax.set_xlabel("5-minute bar")
    ax.set_ylabel("growth of 1x notional (unlevered, no costs)")
    ax.legend()
    ax.grid(alpha=0.25)
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=130, bbox_inches="tight")
    plt.close(fig)


def run(sample_path: str | None = None):
    """Run both pandas simulations and the engine line; return the headline numbers."""
    rows = json.loads(pathlib.Path(sample_path or SAMPLE).read_text(encoding="utf-8"))
    opens = pd.Series([r[1] for r in rows], dtype="float64")
    close = pd.Series([r[4] for r in rows], dtype="float64")
    # Hold bar t from its open to the next bar's open (the last bar has no next open).
    bar_returns = (opens.shift(-1) / opens - 1.0).fillna(0.0)

    # The signal you only know AFTER the bar closes: "did this bar close green?"
    signal = (close > close.shift(1)).astype("float64")

    eq_cheat, sr_cheat, r_cheat = simulate(signal, bar_returns, cheat=True)
    eq_honest, sr_honest, r_honest = simulate(signal, bar_returns, cheat=False)

    return {
        "cheat_sharpe": sr_cheat,
        "cheat_hit": bar_hit_rate(r_cheat),
        "cheat_final": float(eq_cheat.iloc[-1]),
        "honest_sharpe": sr_honest,
        "honest_hit": bar_hit_rate(r_honest),
        "honest_final": float(eq_honest.iloc[-1]),
        "engine": run_engine(sample_path),
        "_eq_cheat": eq_cheat,
        "_eq_honest": eq_honest,
    }


def _usd(x: float) -> str:
    """Dollars with the sign in front of the $ (-$150.00, never $-150.00)."""
    return f"-${-x:,.2f}" if x < 0 else f"${x:,.2f}"


def main() -> None:
    res = run()
    eng = res["engine"]
    print("=== THE LIE: bar t traded on bar t's own close (impossible in real life) ===")
    print(
        f"  annualized Sharpe = {res['cheat_sharpe']:>8.2f}    "
        f"bar hit rate = {res['cheat_hit']:.0%}    "
        f"growth = x{res['cheat_final']:.2f} (unlevered, no costs)"
    )
    print("\n=== THE TRUTH: the signal acts at the next bar's open (before costs) ===")
    print(
        f"  annualized Sharpe = {res['honest_sharpe']:>8.2f}    "
        f"bar hit rate = {res['honest_hit']:.0%}    "
        f"growth = x{res['honest_final']:.2f} (unlevered, no costs)"
    )
    print("\n=== THE ENGINE: same signal through PlutoBT, NQ costs "
          f"(${eng['commission_per_side']:.2f}/side, {eng['slippage_ticks']:g} tick/side) ===")
    print(
        f"  annualized Sharpe = {eng['sharpe']:>8.2f}    "
        f"trades = {eng['n_trades']}    "
        f"win rate = {eng['win_rate']:.0%} (closed trades, net of costs)"
    )
    print(
        f"  1 NQ: gross {_usd(eng['gross_pnl'])}  - slippage {_usd(eng['slippage_usd'])}"
        f"  - commission {_usd(eng['commission_usd'])}  = net {_usd(eng['net_pnl'])}"
    )
    print("\nThe only thing I changed:")
    print("    position = signal           # the lie   (trade bar t on bar t's own close)")
    print("    position = signal.shift(1)  # the truth (act at the next bar's open)")
    _save_plot(res["_eq_cheat"], res["_eq_honest"])
    print(f"\nSaved before/after chart -> {OUT_PNG}")


if __name__ == "__main__":
    main()
