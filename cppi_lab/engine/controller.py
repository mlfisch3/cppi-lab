"""Simulation controller: parameters in, telemetry out (TSD section 1.2).

The controller owns the active dataset and parameter struct and turns them
into a back-test.  It is deliberately free of any Qt dependency so the whole
recalculation pipeline can be tested and benchmarked headless.

Each stage is cached on exactly the inputs it depends on:

* the market window on ``(dataset, start, duration, normalisation)``;
* the signals on the window plus the ten signal parameters;
* the uniform variates on ``(seed, number of bars)``.

Dragging a fee or routing slider therefore only re-runs the kernel.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from cppi_lab.config.defaults import MC_SEEDS, ParamValue, SimParams, constrain, default_params
from cppi_lab.data.loader import MINUTE_MS, MarketData, MarketWindow
from cppi_lab.engine.kernel_numba import SimResult, run_simulation, uniform_draws
from cppi_lab.engine.monte_carlo import EnsembleResult, draw_block, run_ensemble
from cppi_lab.engine.signals import SIGNAL_PARAM_KEYS, Signals, compute_signals

FloatArray = NDArray[np.float64]

#: Largest uniform-variate block kept between ensemble runs (bytes).
_U_BLOCK_CACHE_BYTES = 96 * 1024 * 1024


@dataclass(frozen=True)
class RunTimings:
    """Wall-clock cost of the stages of one recalculation, in milliseconds."""

    window_ms: float
    signals_ms: float
    kernel_ms: float

    @property
    def total_ms(self) -> float:
        return self.window_ms + self.signals_ms + self.kernel_ms


@dataclass(frozen=True)
class RunOutput:
    """Everything the UI needs to render one back-test."""

    window: MarketWindow
    signals: Signals
    result: SimResult
    timings: RunTimings
    #: Monotonic counter; lets asynchronous consumers discard stale results.
    generation: int

    @property
    def params(self) -> SimParams:
        return self.result.params


class SimulationController:
    """Holds the dataset and parameters and runs the back-test pipeline."""

    def __init__(self, data: MarketData, params: SimParams | None = None) -> None:
        self._data = data
        self._params = self._fit(params if params is not None else default_params())
        self._generation = 0
        self._window_key: tuple[object, ...] | None = None
        self._window: MarketWindow | None = None
        self._signal_key: tuple[object, ...] | None = None
        self._signals: Signals | None = None
        self._draw_key: tuple[int, int] | None = None
        self._draws: FloatArray | None = None
        self._block_key: tuple[int, int, int] | None = None
        self._block: FloatArray | None = None

    # -- state ---------------------------------------------------------------
    @property
    def data(self) -> MarketData:
        return self._data

    @property
    def params(self) -> SimParams:
        return self._params

    @property
    def generation(self) -> int:
        return self._generation

    @property
    def start_bounds_ms(self) -> tuple[int, int]:
        """Earliest and latest allowed start time (PRD: dataset min to max)."""
        return self._data.start_ms, self._data.end_ms

    def set_data(self, data: MarketData) -> SimParams:
        """Switch dataset; the start time is clamped into the new range."""
        self._data = data
        self._window_key = self._signal_key = None
        self._window = None
        self._signals = None
        self._params = self._fit(self._params)
        return self._params

    def set_params(self, params: SimParams) -> SimParams:
        """Replace the parameter struct (ranges and couplings are enforced)."""
        self._params = self._fit(params)
        return self._params

    def update(self, key: str, value: ParamValue) -> SimParams:
        """Change one parameter and return the resulting (constrained) struct."""
        return self.set_params(self._params.with_value(key, value))

    def reset(self, key: str | None = None) -> SimParams:
        """Reset one parameter, or all of them, to the PRD defaults (FR-2.2)."""
        defaults = default_params()
        if key is None:
            return self.set_params(defaults)
        return self.update(key, getattr(defaults, key))

    def _fit(self, params: SimParams) -> SimParams:
        lo, hi = self._data.start_ms, self._data.end_ms
        start = min(max(int(params.start_ms), lo), hi)
        start -= start % MINUTE_MS
        if start < lo:
            start += MINUTE_MS
        fitted = constrain(params)
        if start != fitted.start_ms:
            fitted = constrain(_with_start(fitted, start))
        return fitted

    # -- pipeline ------------------------------------------------------------
    def _market_window(self) -> MarketWindow:
        p = self._params
        key = (id(self._data), p.start_ms, p.duration_days, p.normalize, p.anchor_price)
        if key != self._window_key or self._window is None:
            window = self._data.window(p.start_ms, p.duration_days)
            if p.normalize:
                window = window.rescaled(p.anchor_price / float(window.open[0]))
            self._window = window
            self._window_key = key
            self._signal_key = None
        return self._window

    def _window_signals(self, window: MarketWindow) -> Signals:
        p = self._params
        key = (self._window_key, *(getattr(p, name) for name in SIGNAL_PARAM_KEYS))
        if key != self._signal_key or self._signals is None:
            self._signals = compute_signals(window, p)
            self._signal_key = key
        return self._signals

    def _uniforms(self, n_bars: int) -> FloatArray:
        key = (int(self._params.seed), n_bars)
        if key != self._draw_key or self._draws is None:
            self._draws = uniform_draws(self._params.seed, n_bars)
            self._draw_key = key
        return self._draws

    def run(self) -> RunOutput:
        """Run the single-trajectory back-test for the current parameters."""
        t0 = time.perf_counter()
        window = self._market_window()
        t1 = time.perf_counter()
        signals = self._window_signals(window)
        t2 = time.perf_counter()
        result = run_simulation(window, signals, self._params, self._uniforms(window.n))
        t3 = time.perf_counter()
        self._generation += 1
        return RunOutput(
            window=window,
            signals=signals,
            result=result,
            timings=RunTimings((t1 - t0) * 1e3, (t2 - t1) * 1e3, (t3 - t2) * 1e3),
            generation=self._generation,
        )

    def ensemble_inputs(self) -> tuple[MarketWindow, Signals, SimParams, FloatArray | None]:
        """Snapshot the inputs of an ensemble run for use on a worker thread.

        The uniform block is returned when it is small enough to cache, so a
        parameter tweak that leaves the seed and window length unchanged does
        not regenerate 4.3 million variates.
        """
        window = self._market_window()
        signals = self._window_signals(window)
        params = self._params
        key = (int(params.seed), window.n, MC_SEEDS)
        if window.n * MC_SEEDS * 8 > _U_BLOCK_CACHE_BYTES:
            self._block = None
            self._block_key = None
            return window, signals, params, None
        if key != self._block_key or self._block is None:
            seeds = [int(params.seed) + m for m in range(MC_SEEDS)]
            self._block = draw_block(seeds, window.n)
            self._block_key = key
        return window, signals, params, self._block

    def run_ensemble(self) -> EnsembleResult:
        """Run the Monte Carlo ensemble for the current parameters (FR-3.3)."""
        window, signals, params, block = self.ensemble_inputs()
        return run_ensemble(window, signals, params, MC_SEEDS, block)


def _with_start(params: SimParams, start_ms: int) -> SimParams:
    from dataclasses import replace

    return replace(params, start_ms=start_ms)


def warmup() -> None:
    """Compile (or load from the on-disk cache) every Numba kernel.

    Called once at start-up so the first interactive recalculation is fast.
    """
    n = 64
    ts = np.arange(n, dtype=np.int64) * MINUTE_MS
    base = 100.0 + np.sin(np.arange(n, dtype=np.float64) / 5.0)
    window = MarketWindow(
        ts_ms=ts,
        open=base.copy(),
        high=base + 0.5,
        low=base - 0.5,
        close=base + 0.1,
        buy_vol=np.full(n, 3.0),
        sell_vol=np.full(n, 2.0),
    )
    params = default_params()
    signals = compute_signals(window, params)
    run_simulation(window, signals, params)
    run_ensemble(window, signals, params, members=2)
