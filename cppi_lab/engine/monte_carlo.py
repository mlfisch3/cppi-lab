"""Parallel Monte Carlo liquidity ensemble (PRD FR-3.3, TSD section 2.5).

The ensemble re-runs the *same* compiled state machine as the single
trajectory, once per seed, inside a ``prange`` loop.  Only the uniform
variates differ between members, so the spread of the resulting equity paths
isolates the effect of stochastic maker fills.

Seeding is counter based: member ``m`` uses ``PCG64(s_rng + m)``.  Member 0
is therefore exactly the active trajectory shown in the chart.
"""

from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
from numba import njit, prange
from numpy.typing import NDArray

from cppi_lab.config.defaults import MC_SEEDS, SimParams
from cppi_lab.data.loader import MarketWindow
from cppi_lab.engine.kernel_numba import (
    N_STATS,
    STAT_FINAL_EQUITY_USD,
    STAT_FIRST_LIQUIDATION_BAR,
    STAT_FLOOR_BREACHES,
    STAT_GAP_BAR,
    TELEMETRY_DTYPE,
    kernel_args,
    simulate_core,
)
from cppi_lab.engine.signals import Signals

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

#: The fan is sampled on at most this many bars per member, whatever the
#: window length, which bounds the equity matrix at about 35 MB.
MAX_FAN_SAMPLES = 43_200
#: Upper bound on the uniform-variate block generated per batch (bytes).
_BATCH_BYTES = 96 * 1024 * 1024
_DRAW_THREADS = 8


@njit(cache=True, nogil=True, parallel=True)
def ensemble_core(
    ts_ms: IntArray,
    open_: FloatArray,
    high: FloatArray,
    low: FloatArray,
    close: FloatArray,
    buy_vol: FloatArray,
    sell_vol: FloatArray,
    s_down: FloatArray,
    s_up: FloatArray,
    atr: FloatArray,
    u_block: FloatArray,
    n0: float,
    c0: float,
    p_liq: float,
    p_ref: float,
    dn_min: float,
    f_maker: float,
    f_taker: float,
    delta_slip: float,
    rho: float,
    lambda_liq: float,
    beta_vel: float,
    alpha_size: float,
    k_sell: float,
    scratch_rows: np.ndarray,
    scratch_flags: NDArray[np.uint8],
    equity: FloatArray,
    stride: int,
    stats: FloatArray,
) -> None:
    """Run one batch of ensemble members in parallel.

    Row ``m`` of ``u_block`` drives member ``m``; its sampled equity path and
    statistics land in row ``m`` of ``equity`` and ``stats``.
    """
    members = u_block.shape[0]
    for m in prange(members):
        simulate_core(
            ts_ms,
            open_,
            high,
            low,
            close,
            buy_vol,
            sell_vol,
            s_down,
            s_up,
            atr,
            u_block[m],
            n0,
            c0,
            p_liq,
            p_ref,
            dn_min,
            f_maker,
            f_taker,
            delta_slip,
            rho,
            lambda_liq,
            beta_vel,
            alpha_size,
            k_sell,
            scratch_rows,
            scratch_flags,
            False,
            equity[m],
            stride,
            stats[m],
        )


@njit(cache=True, nogil=True, parallel=True)
def percentile_band(
    equity: FloatArray, lower_pct: float, upper_pct: float
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Per-sample ``(lower, median, upper)`` percentiles across members.

    Uses NumPy's default linear interpolation between order statistics.
    """
    members, samples = equity.shape
    lower = np.empty(samples, dtype=np.float64)
    median = np.empty(samples, dtype=np.float64)
    upper = np.empty(samples, dtype=np.float64)
    quantiles = (lower_pct / 100.0, 0.5, upper_pct / 100.0)
    for j in prange(samples):
        column = np.sort(equity[:, j])
        for k in range(3):
            position = quantiles[k] * (members - 1)
            base = math.floor(position)
            if base >= members - 1:
                value = column[members - 1]
            else:
                frac = position - base
                value = column[base] + (column[base + 1] - column[base]) * frac
            if k == 0:
                lower[j] = value
            elif k == 1:
                median[j] = value
            else:
                upper[j] = value
    return lower, median, upper


def fan_stride(n_bars: int) -> int:
    """Sampling stride that keeps the fan at or below :data:`MAX_FAN_SAMPLES`."""
    return max(1, math.ceil(n_bars / MAX_FAN_SAMPLES))


def draw_block(seeds: list[int], n_bars: int, out: FloatArray | None = None) -> FloatArray:
    """Uniform variates for several seeds, one row each, generated in threads.

    Row ``i`` equals ``uniform_draws(seeds[i], n_bars)`` exactly.
    """
    block = np.empty((len(seeds), n_bars), dtype=np.float64) if out is None else out

    def fill(index: int) -> None:
        np.random.Generator(np.random.PCG64(seeds[index])).random(out=block[index])

    if len(seeds) == 1:
        fill(0)
        return block
    with ThreadPoolExecutor(max_workers=min(_DRAW_THREADS, len(seeds))) as pool:
        list(pool.map(fill, range(len(seeds))))
    return block


@dataclass(frozen=True)
class EnsembleResult:
    """Percentile band and aggregates of a Monte Carlo ensemble."""

    #: Bar index of every fan sample (``0, stride, 2 * stride, ...``).
    sample_idx: IntArray
    p05: FloatArray
    p50: FloatArray
    p95: FloatArray
    members: int
    #: Members that ended in a hard-buffer liquidation.
    liquidated_members: int
    #: Members whose open gapped through the trigger.
    gapped_members: int
    #: Conservation-assertion failures summed over all members.
    floor_breaches: int
    terminal_min: float
    terminal_max: float
    seed: int


def run_ensemble(
    window: MarketWindow,
    signals: Signals,
    params: SimParams,
    members: int = MC_SEEDS,
    u_block: FloatArray | None = None,
) -> EnsembleResult:
    """Simulate ``members`` seeds and reduce them to a 5-95 % equity band.

    ``u_block`` may supply pre-generated variates (shape ``members x T``) so a
    caller that re-runs the ensemble with unchanged seeds skips the draws.
    """
    n_bars = window.n
    stride = fan_stride(n_bars)
    samples = (n_bars + stride - 1) // stride
    equity = np.empty((members, samples), dtype=np.float64)
    stats = np.empty((members, N_STATS), dtype=np.float64)
    scratch_rows = np.zeros(1, dtype=TELEMETRY_DTYPE)
    scratch_flags = np.zeros(1, dtype=np.uint8)
    args = kernel_args(params)
    seeds = [int(params.seed) + m for m in range(members)]

    if u_block is not None:
        if u_block.shape != (members, n_bars):
            raise ValueError("u_block must have shape (members, n_bars)")
        batches = [(0, members, u_block)]
    else:
        per_batch = max(1, min(members, _BATCH_BYTES // max(1, n_bars * 8)))
        batches = []
        for first in range(0, members, per_batch):
            last = min(members, first + per_batch)
            batches.append((first, last, draw_block(seeds[first:last], n_bars)))

    for first, last, block in batches:
        ensemble_core(
            window.ts_ms,
            window.open,
            window.high,
            window.low,
            window.close,
            window.buy_vol,
            window.sell_vol,
            signals.s_down,
            signals.s_up,
            signals.atr,
            block,
            *args,
            scratch_rows,
            scratch_flags,
            equity[first:last],
            stride,
            stats[first:last],
        )

    p05, p50, p95 = percentile_band(equity, 5.0, 95.0)
    terminal = stats[:, STAT_FINAL_EQUITY_USD]
    return EnsembleResult(
        sample_idx=np.arange(samples, dtype=np.int64) * stride,
        p05=p05,
        p50=p50,
        p95=p95,
        members=members,
        liquidated_members=int((stats[:, STAT_FIRST_LIQUIDATION_BAR] >= 0).sum()),
        gapped_members=int((stats[:, STAT_GAP_BAR] >= 0).sum()),
        floor_breaches=int(stats[:, STAT_FLOOR_BREACHES].sum()),
        terminal_min=float(terminal.min()),
        terminal_max=float(terminal.max()),
        seed=int(params.seed),
    )
