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
