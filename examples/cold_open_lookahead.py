"""Cold-open demo: the lookahead lie, in one changed line.

Same data, same signal, ONE setting flipped — when you fill — and a gorgeous
equity curve collapses to nothing.

This is the canonical lookahead bug. A signal is only known AFTER a bar closes,
so the earliest you can actually trade is the NEXT bar's open. Filling at the
SAME bar's close lets a backtest "trade" on information it could not have had
in real life. Here the signal is deliberately obvious — "did this bar close
green?" — so the lie is visible to the eye; the exact same off-by-one hides
inside any indicator (EMA, RSI, whatever).

PlutoBT's engine forbids this by construction (decision@t / fill@t+1). This
script is a teaching prop for the video's cold open, not part of the engine.

Run:  python examples/cold_open_lookahead.py
"""

from __future__ import annotations

import pathlib

import pandas as pd

from plutobt.metrics import sharpe

PPY = 20412.0  # 5-min RTH bars/year (81 bars/day x 252); see spec section 5.7
_ROOT = pathlib.Path(__file__).resolve().parents[1]
SAMPLE = _ROOT / "data" / "sample" / "nq_5m.json"
OUT_PNG = _ROOT / "examples" / "output" / "cold_open_lie_vs_truth.png"


def simulate(signal: pd.Series, bar_returns: pd.Series, *, cheat: bool):
    """Return (equity_curve, annualized_sharpe, per_bar_returns).

    THE ONE TOGGLE:
      cheat  -> fill on the SAME bar's close (position = signal)        [the lie]
      honest -> fill on the NEXT bar's open  (position = signal.shift(1)) [the truth]
    """
    position = signal if cheat else signal.shift(1)
    strat_returns = (position.fillna(0.0) * bar_returns).fillna(0.0)
    equity = (1.0 + strat_returns).cumprod()
    return equity, sharpe(strat_returns, PPY), strat_returns


def _win_rate(returns: pd.Series) -> float:
    active = returns[returns != 0.0]
    return float((active > 0.0).mean()) if len(active) else 0.0


def _save_plot(eq_cheat: pd.Series, eq_honest: pd.Series) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(eq_cheat.to_numpy(), label="THE LIE (same-bar-close fill)", linewidth=2.5)
    ax.plot(eq_honest.to_numpy(), label="THE TRUTH (next-bar-open fill)", linewidth=2.5)
    ax.set_title("Your backtest is lying to you (one line changed)")
    ax.set_xlabel("5-minute bar")
    ax.set_ylabel("equity (x starting)")
    ax.legend()
    ax.grid(alpha=0.25)
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=130, bbox_inches="tight")
    plt.close(fig)


def run(sample_path: str | None = None):
    """Run both simulations; return a dict of the headline numbers."""
    import json

    rows = json.loads(pathlib.Path(sample_path or SAMPLE).read_text(encoding="utf-8"))
    close = pd.Series([r[4] for r in rows], dtype="float64")
    bar_returns = close.pct_change().fillna(0.0)

    # The signal you only know AFTER the bar closes: "did this bar close green?"
    signal = (close > close.shift(1)).astype("float64")

    eq_cheat, sr_cheat, r_cheat = simulate(signal, bar_returns, cheat=True)
    eq_honest, sr_honest, r_honest = simulate(signal, bar_returns, cheat=False)

    return {
        "cheat_sharpe": sr_cheat,
        "cheat_win": _win_rate(r_cheat),
        "cheat_final": float(eq_cheat.iloc[-1]),
        "honest_sharpe": sr_honest,
        "honest_win": _win_rate(r_honest),
        "honest_final": float(eq_honest.iloc[-1]),
        "_eq_cheat": eq_cheat,
        "_eq_honest": eq_honest,
    }


def main() -> None:
    res = run()
    print("=== THE LIE — fill on the SAME bar's close (impossible in real life) ===")
    print(
        f"  annualized Sharpe = {res['cheat_sharpe']:>8.2f}    "
        f"win rate = {res['cheat_win']:.0%}    final equity = x{res['cheat_final']:.2f}"
    )
    print("\n=== THE TRUTH — fill on the NEXT bar's open (what PlutoBT enforces) ===")
    print(
        f"  annualized Sharpe = {res['honest_sharpe']:>8.2f}    "
        f"win rate = {res['honest_win']:.0%}    final equity = x{res['honest_final']:.2f}"
    )
    print("\nThe only thing I changed:")
    print("    position = signal           # the lie  (fill this bar's close)")
    print("    position = signal.shift(1)  # the truth (fill next bar's open)")
    _save_plot(res["_eq_cheat"], res["_eq_honest"])
    print(f"\nSaved before/after chart -> {OUT_PNG}")


if __name__ == "__main__":
    main()
