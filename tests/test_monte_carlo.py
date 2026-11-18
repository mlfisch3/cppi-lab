"""Monte Carlo ensemble (PRD FR-3.3)."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from cppi_lab.config.defaults import MC_SEEDS, SimParams
from cppi_lab.data.loader import MarketData
from cppi_lab.engine import monte_carlo
from cppi_lab.engine.kernel_numba import uniform_draws
from cppi_lab.engine.monte_carlo import (
    MAX_FAN_SAMPLES,
    draw_block,
    fan_stride,
    percentile_band,
    run_ensemble,
)
from cppi_lab.engine.signals import compute_signals
from tests.conftest import backtest, ms, real_window


def test_draw_block_rows_are_the_per_seed_streams() -> None:
    seeds = [42, 43, 9_999_999, 1]
    block = draw_block(seeds, 2_000)
    assert block.shape == (4, 2_000)
    for row, seed in zip(block, seeds, strict=True):
        np.testing.assert_array_equal(row, uniform_draws(seed, 2_000))


def test_percentile_band_matches_numpy() -> None:
    rng = np.random.default_rng(3)
    equity = rng.normal(60_000, 900, (100, 257))
    low, mid, high = percentile_band(equity, 5.0, 95.0)
    np.testing.assert_allclose(low, np.percentile(equity, 5, axis=0), rtol=1e-13)
    np.testing.assert_allclose(mid, np.percentile(equity, 50, axis=0), rtol=1e-13)
    np.testing.assert_allclose(high, np.percentile(equity, 95, axis=0), rtol=1e-13)


def test_fan_stride_bounds_the_sample_count() -> None:
    assert fan_stride(43_200) == 1
    assert fan_stride(43_201) == 2
    assert fan_stride(525_600) == 13
    assert -(-525_600 // fan_stride(525_600)) <= MAX_FAN_SAMPLES


def test_member_zero_is_the_active_trajectory(bundled: MarketData, params: SimParams) -> None:
    p = replace(params, rho=0.0, normalize=True)
    window = real_window(bundled, ms(2026, 1, 1), 5, p)
    signals, single = backtest(window, p)
    one = run_ensemble(window, signals, p, members=1)
    np.testing.assert_array_equal(one.p05, single.telemetry["net_equity_usd"])
    np.testing.assert_array_equal(one.p95, single.telemetry["net_equity_usd"])
    assert one.terminal_min == single.stats.final_equity_usd


def test_ensemble_is_deterministic_and_brackets_its_members(
    bundled: MarketData, params: SimParams
) -> None:
    p = replace(params, rho=0.0, normalize=True)
    window = real_window(bundled, ms(2026, 1, 1), 5, p)
    signals = compute_signals(window, p)
    first = run_ensemble(window, signals, p)
    second = run_ensemble(window, signals, p)
    assert first.members == MC_SEEDS == 100
    np.testing.assert_array_equal(first.p05, second.p05)
    np.testing.assert_array_equal(first.p95, second.p95)
    assert (first.p05 <= first.p50).all()
    assert (first.p50 <= first.p95).all()
    assert (first.p95 - first.p05).max() > 0.0  # passive fills make the outcome uncertain
    assert first.floor_breaches == 0
    assert first.terminal_min <= first.terminal_max
    assert first.sample_idx[0] == 0
    assert first.sample_idx.shape == first.p05.shape


def test_precomputed_draws_give_the_same_result(bundled: MarketData, params: SimParams) -> None:
    p = replace(params, rho=0.25, normalize=True)
    window = real_window(bundled, ms(2026, 1, 1), 3, p)
    signals = compute_signals(window, p)
    block = draw_block([p.seed + m for m in range(MC_SEEDS)], window.n)
    a = run_ensemble(window, signals, p)
    b = run_ensemble(window, signals, p, u_block=block)
    np.testing.assert_array_equal(a.p05, b.p05)
    np.testing.assert_array_equal(a.p95, b.p95)
    with pytest.raises(ValueError):
        run_ensemble(window, signals, p, u_block=block[:, :-1])


def test_batched_draws_give_the_same_result(
    bundled: MarketData, params: SimParams, monkeypatch: pytest.MonkeyPatch
) -> None:
    p = replace(params, rho=0.0, normalize=True)
    window = real_window(bundled, ms(2026, 1, 1), 2, p)
    signals = compute_signals(window, p)
    whole = run_ensemble(window, signals, p)
    monkeypatch.setattr(monte_carlo, "_BATCH_BYTES", window.n * 8 * 7)  # 7 members per batch
    batched = run_ensemble(window, signals, p)
    np.testing.assert_array_equal(whole.p05, batched.p05)
    np.testing.assert_array_equal(whole.p50, batched.p50)
    np.testing.assert_array_equal(whole.p95, batched.p95)
    assert whole.liquidated_members == batched.liquidated_members


def test_all_taker_routing_collapses_the_fan(bundled: MarketData, params: SimParams) -> None:
    p = replace(params, rho=1.0, normalize=True)
    window = real_window(bundled, ms(2026, 1, 1), 5, p)
    result = run_ensemble(window, compute_signals(window, p), p)
    np.testing.assert_array_equal(result.p05, result.p95)


def test_long_window_is_subsampled(bundled: MarketData, params: SimParams) -> None:
    p = replace(params, normalize=True)
    window = real_window(bundled, ms(2025, 6, 1), 45, p)  # 64,800 bars -> stride 2
    result = run_ensemble(window, compute_signals(window, p), p, members=8)
    assert fan_stride(window.n) == 2
    assert result.p05.shape[0] == (window.n + 1) // 2
    assert result.sample_idx[1] == 2
    assert result.floor_breaches == 0
