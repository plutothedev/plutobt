"""Plotting smoke tests (headless Agg backend) for plot_equity/plot_trades (M8.9)."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless, before pyplot import
import matplotlib.pyplot as plt

from plutobt.config import load_config
from plutobt.data import load_bars
from plutobt.engine import Engine
from plutobt.instruments import get_instrument
from plutobt.plotting import plot_equity, plot_trades
from tests.conftest import GoldenStrategy

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "data" / "sample" / "nq_5m.json"
GOLDEN_CFG = ROOT / "configs" / "golden.toml"


def _run():
    df = load_bars(str(SAMPLE))
    cfg = load_config(str(GOLDEN_CFG))
    result = Engine(get_instrument("NQ"), cfg).run(df, GoldenStrategy())
    return df, result


def test_plot_equity_smoke_returns_axes_with_artists():
    _, result = _run()
    fig, ax = plt.subplots()
    out = plot_equity(result, ax=ax)
    assert out is not None
    assert len(ax.lines) >= 1  # equity curve drawn
    plt.close(fig)


def test_plot_equity_creates_its_own_axes_when_none():
    _, result = _run()
    out = plot_equity(result, ax=None)  # must not raise without an ax
    assert out is not None
    plt.close("all")


def test_plot_trades_smoke_draws_price_and_markers():
    df, result = _run()
    fig, ax = plt.subplots()
    out = plot_trades(df, result, ax=ax)
    assert out is not None
    assert len(ax.lines) + len(ax.collections) >= 1  # price line and/or markers
    plt.close(fig)


def test_plot_can_save_png(tmp_path):
    _, result = _run()
    fig, ax = plt.subplots()
    plot_equity(result, ax=ax)
    png = tmp_path / "equity.png"
    fig.savefig(png)
    assert png.is_file() and png.stat().st_size > 0
    plt.close(fig)


def _marker_of(collection):
    """The marker name ("^", "v", "x", ...) a scatter PathCollection was drawn with."""
    import numpy as np
    from matplotlib.markers import MarkerStyle

    verts = collection.get_paths()[0].vertices
    for name in ("^", "v", "x", "X", "o", "s", "D"):
        ms = MarkerStyle(name)
        ref = ms.get_path().transformed(ms.get_transform()).vertices
        if ref.shape == verts.shape and np.allclose(ref, verts):
            return name
    return None


def _colour_of(collection):
    from matplotlib.colors import to_hex

    fc = collection.get_facecolor()
    rgba = fc[0] if len(fc) else collection.get_edgecolor()[0]
    return to_hex(rgba)


def test_plot_trades_marks_each_side_and_each_exit_distinctly():
    # A short entry used to be drawn as the green up-triangle buy marker, the same as a
    # long entry. Inspect the marker and colour actually drawn at each fill.
    from types import SimpleNamespace

    import pandas as pd
    from matplotlib.colors import to_hex, to_rgb

    from plutobt.portfolio import Trade

    df = pd.DataFrame({"ts": [0, 1, 2, 3], "close": [10.0, 11.0, 12.0, 9.0]})
    trades = [
        Trade(entry_ts=0, exit_ts=1, side="long", contracts=1,
              entry_price=10.0, exit_price=11.0, pnl=1.0),
        Trade(entry_ts=2, exit_ts=3, side="short", contracts=1,
              entry_price=12.0, exit_price=9.0, pnl=3.0),
    ]
    fig, ax = plt.subplots()
    plot_trades(df, SimpleNamespace(trades=trades), ax=ax)
    drawn = {}
    for c in ax.collections:
        x, y = c.get_offsets()[0]
        drawn[(float(x), float(y))] = (_marker_of(c), _colour_of(c))
    plt.close(fig)

    long_entry, long_exit = drawn[(0.0, 10.0)], drawn[(1.0, 11.0)]
    short_entry, short_exit = drawn[(2.0, 12.0)], drawn[(3.0, 9.0)]
    # The exact legend the docstring promises (review, 2026-11-02: the test only checked
    # that the sides differed, so swapped or arbitrary colours would have passed):
    # colour = side (long green C2, short red C3), shape = what happened.
    green, red = to_hex("C2"), to_hex("C3")
    assert (green, red) == ("#2ca02c", "#d62728")     # matplotlib's default cycle
    assert long_entry == ("^", green)                 # a long buys: up triangle, green
    assert long_exit == ("x", green)
    assert short_entry == ("v", red)                  # a short sells: down triangle, red
    assert short_exit == ("x", red)
    for hexcol, hue in ((green, 1), (red, 0)):        # green really is green, red red
        rgb = to_rgb(hexcol)
        assert rgb[hue] == max(rgb) and rgb[hue] > 2 * min(rgb)
