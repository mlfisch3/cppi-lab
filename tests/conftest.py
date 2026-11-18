"""Shared fixtures.

Qt is forced onto the ``offscreen`` platform and the software rasteriser
before any Qt module is imported, so the whole suite runs without a display.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterator
from dataclasses import replace
from datetime import UTC, datetime

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QTWEBENGINE_CHROMIUM_FLAGS", "--disable-gpu --no-sandbox")
os.environ.setdefault("CPPI_LAB_OPENGL", "0")

import numpy as np
import pytest

from cppi_lab.config.defaults import SimParams, default_params
from cppi_lab.data.loader import MINUTE_MS, MarketData, MarketWindow, load_bundled
from cppi_lab.engine.kernel_numba import SimResult, run_simulation
from cppi_lab.engine.signals import Signals, compute_signals


def pytest_configure(config: pytest.Config) -> None:
    """Register ``cppidoc://`` and share GL contexts before a QApplication exists."""
    from PySide6 import QtCore

    from cppi_lab.ui.doc_scheme import register_scheme

    QtCore.QCoreApplication.setAttribute(QtCore.Qt.ApplicationAttribute.AA_ShareOpenGLContexts)
    register_scheme()
    import pyqtgraph as pg

    pg.setConfigOptions(useOpenGL=False, antialias=True)


@pytest.fixture(scope="session", autouse=True)
def _release_web_profile() -> Iterator[None]:
    """Delete every web page before the shared QtWebEngine profile goes away."""
    yield
    from PySide6 import QtCore, QtWidgets

    from cppi_lab.ui.doc_workbench import release_profile

    app = QtWidgets.QApplication.instance()
    if app is not None:
        for _ in range(3):
            QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)
            app.processEvents()
    release_profile()
    if app is not None:
        QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)
        app.processEvents()


def ms(year: int, month: int, day: int, hour: int = 0, minute: int = 0) -> int:
    """Epoch milliseconds of a UTC wall-clock time."""
    return int(datetime(year, month, day, hour, minute, tzinfo=UTC).timestamp() * 1000)


@pytest.fixture(scope="session")
def bundled() -> MarketData:
    return load_bundled()


@pytest.fixture(scope="session")
def params() -> SimParams:
    return default_params()


def synthetic_window(
    n_bars: int,
    seed: int,
    start_price: float = 68_000.0,
    volatility: float = 0.0009,
    drift: float = 0.0,
    gap_free: bool = True,
) -> MarketWindow:
    """A random-walk market with a consistent OHLC envelope and taker volumes.

    With ``gap_free`` every bar opens exactly at the previous close, which is
    the "standard continuous-diffusion bar" of PRD section 6.
    """
    rng = np.random.default_rng(seed)
    returns = rng.normal(drift, volatility, n_bars)
    # Occasional fat-tailed bars so crashes and squeezes are exercised.
    shocks = rng.random(n_bars) < 0.01
    returns[shocks] *= rng.uniform(3.0, 9.0, int(shocks.sum()))
    close = start_price * np.exp(np.cumsum(returns))
    open_ = np.concatenate(([start_price], close[:-1]))
    if not gap_free:
        open_[1:] *= np.exp(rng.normal(0.0, volatility * 0.5, n_bars - 1))
    wick_up = np.abs(rng.normal(0.0, volatility * 0.6, n_bars))
    wick_down = np.abs(rng.normal(0.0, volatility * 0.6, n_bars))
    high = np.maximum(open_, close) * (1.0 + wick_up)
    low = np.minimum(open_, close) * (1.0 - wick_down)
    total = rng.lognormal(2.0, 0.9, n_bars)
    share = np.clip(0.5 + 0.35 * np.tanh(returns / volatility) + rng.normal(0, 0.1, n_bars), 0, 1)
    return MarketWindow(
        ts_ms=np.arange(n_bars, dtype=np.int64) * MINUTE_MS + ms(2026, 1, 1),
        open=np.ascontiguousarray(open_),
        high=np.ascontiguousarray(high),
        low=np.ascontiguousarray(low),
        close=np.ascontiguousarray(close),
        buy_vol=np.ascontiguousarray(total * share),
        sell_vol=np.ascontiguousarray(total * (1.0 - share)),
    )


def random_params(rng: np.random.Generator, base: SimParams | None = None) -> SimParams:
    """Draw every strategy parameter uniformly from its PRD section 5 range."""
    from cppi_lab.config.defaults import PARAM_SPECS, constrain

    p = base if base is not None else default_params()
    values: dict[str, float] = {}
    for spec in PARAM_SPECS:
        if spec.kind in ("datetime", "bool") or spec.key in ("duration_days", "anchor_price"):
            continue
        if spec.key in ("n0", "c0", "p_liq", "p_ref", "seed"):
            continue
        steps = spec.steps
        raw = spec.minimum + int(rng.integers(0, steps + 1)) * spec.step
        values[spec.key] = round(raw) if spec.kind == "int" else float(raw)
    # Discrete corner values of rho are the interesting ones; hit them often.
    values["rho"] = float(rng.choice([0.0, 0.05, 0.25, 0.5, 0.75, 1.0, values["rho"]]))
    values["seed"] = int(rng.integers(1, 9_999_999))
    values["n0"] = float(rng.choice([0.001, 0.05, 1.0, 2.5, 40.0, 1000.0]))
    values["c0"] = float(rng.choice([0.0, 0.0, 100.0, 25_000.0, 1_000_000.0]))
    return constrain(replace(p, **values))  # type: ignore[arg-type]


def anchored(p: SimParams, window: MarketWindow, rng: np.random.Generator) -> SimParams:
    """Place the floor and band around a window's first price."""
    from cppi_lab.config.defaults import constrain

    first = float(window.open[0])
    p_liq = first * float(rng.uniform(0.82, 0.985))
    p_ref = p_liq + max(1_000.0, first * float(rng.uniform(0.02, 0.22)))
    return constrain(replace(p, p_liq=p_liq, p_ref=p_ref))


Backtest = Callable[[MarketWindow, SimParams], tuple[Signals, SimResult]]


def backtest(window: MarketWindow, p: SimParams) -> tuple[Signals, SimResult]:
    signals = compute_signals(window, p)
    return signals, run_simulation(window, signals, p)


def real_window(data: MarketData, start: int, days: int, p: SimParams) -> MarketWindow:
    window = data.window(start, days)
    return window.rescaled(p.anchor_price / float(window.open[0])) if p.normalize else window
