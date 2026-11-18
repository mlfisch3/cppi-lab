"""PRNG determinism (TSD section 2.5): same seed, same trajectory, everywhere."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from cppi_lab.config.defaults import SimParams
from cppi_lab.data.loader import MarketData
from cppi_lab.engine.kernel_numba import uniform_draws
from tests.conftest import backtest, ms, real_window

# First variates of np.random.Generator(np.random.PCG64(42)).random(); PCG64 and the
# double conversion are specified bit for bit, so these hold on every platform.
SEED_42_HEAD = [
    0.7739560485559633,
    0.4388784397520523,
    0.8585979199113825,
    0.6973680290593639,
    0.09417734788764953,
]


def test_draws_are_the_documented_pcg64_stream() -> None:
    draws = uniform_draws(42, 1_000)
    assert draws[:5].tolist() == SEED_42_HEAD
    np.testing.assert_array_equal(draws, np.random.Generator(np.random.PCG64(42)).random(1_000))
    assert draws.dtype == np.float64
    assert draws.min() >= 0.0
    assert draws.max() < 1.0


def test_draws_are_indexed_by_bar_not_by_demand() -> None:
    """A longer run extends the stream; it does not reshuffle the common prefix."""
    np.testing.assert_array_equal(uniform_draws(7, 500), uniform_draws(7, 5_000)[:500])


def test_same_seed_reproduces_the_whole_trajectory(bundled: MarketData, params: SimParams) -> None:
    p = replace(params, rho=0.0, seed=1234)
    window = real_window(bundled, ms(2026, 2, 3), 6, p)
    _, first = backtest(window, p)
    _, second = backtest(window, p)
    assert first.telemetry.tobytes() == second.telemetry.tobytes()
    np.testing.assert_array_equal(first.flags, second.flags)
    assert first.stats == second.stats


def test_different_seeds_change_fills_but_not_the_market(
    bundled: MarketData, params: SimParams
) -> None:
    base = replace(params, rho=0.0)
    window = real_window(bundled, ms(2026, 2, 3), 6, base)
    _, a = backtest(window, replace(base, seed=1))
    _, b = backtest(window, replace(base, seed=2))
    assert a.stats.maker_attempts > 0
    assert not np.array_equal(a.telemetry["btc_held"], b.telemetry["btc_held"])
    for name in ("open", "high", "low", "close", "s_down", "s_up"):
        np.testing.assert_array_equal(a.telemetry[name], b.telemetry[name])


def test_strategy_parameters_do_not_disturb_the_variates(
    bundled: MarketData, params: SimParams
) -> None:
    """Common random numbers: U_t depends on the seed and the bar only."""
    window = real_window(bundled, ms(2026, 2, 3), 4, params)
    _, a = backtest(window, params)
    _, b = backtest(window, replace(params, rho=0.7, k_sell=1.6, lambda_liq=6.0, f_taker=0.006))
    np.testing.assert_array_equal(a.telemetry["u_rand_draw"], b.telemetry["u_rand_draw"])
    np.testing.assert_array_equal(a.telemetry["u_rand_draw"], uniform_draws(params.seed, window.n))
