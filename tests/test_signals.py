"""Signal math (TSD section 2.2) against a naive reference and Polars."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import polars as pl
import pytest

from cppi_lab.config.defaults import SimParams
from cppi_lab.data.loader import MarketData
from cppi_lab.engine.signals import (
    compute_signals,
    ema,
    rolling_max,
    rolling_min,
    rolling_sum,
    true_range,
)
from tests.conftest import ms, random_params, synthetic_window
from tests.reference import ref_signals

FIELDS = ("s_down", "s_up", "atr", "m_down", "i_down", "m_up", "b_up", "g_veto")


@pytest.mark.parametrize("seed", range(6))
def test_signals_equal_plain_python_reference(seed: int) -> None:
    rng = np.random.default_rng(100 + seed)
    window = synthetic_window(1_800, seed)
    p = random_params(rng)
    fast = compute_signals(window, p)
    slow = ref_signals(window, p)
    for name in FIELDS:
        np.testing.assert_array_equal(getattr(fast, name), np.array(getattr(slow, name)), name)


def test_signals_equal_reference_on_real_data(bundled: MarketData, params: SimParams) -> None:
    window = bundled.window(ms(2026, 2, 4), 2)
    fast = compute_signals(window, params)
    slow = ref_signals(window, params)
    for name in FIELDS:
        np.testing.assert_array_equal(getattr(fast, name), np.array(getattr(slow, name)), name)


def test_building_blocks_agree_with_polars() -> None:
    window = synthetic_window(3_000, 7)
    close = pl.Series(window.close)
    np.testing.assert_allclose(
        ema(window.close, 30), close.ewm_mean(span=30, adjust=False).to_numpy(), rtol=1e-12
    )
    np.testing.assert_allclose(
        rolling_sum(window.buy_vol, 10),
        pl.Series(window.buy_vol).rolling_sum(10, min_samples=1).to_numpy(),
        rtol=1e-11,
    )
    np.testing.assert_array_equal(
        rolling_max(window.high, 240),
        pl.Series(window.high).rolling_max(240, min_samples=1).to_numpy(),
    )
    np.testing.assert_array_equal(
        rolling_min(window.low, 240),
        pl.Series(window.low).rolling_min(240, min_samples=1).to_numpy(),
    )


def test_true_range_definition() -> None:
    high = np.array([10.0, 12.0, 11.0, 20.0])
    low = np.array([9.0, 11.5, 8.0, 19.0])
    close = np.array([9.5, 11.8, 9.0, 19.5])
    # TR_0 = H_0 - L_0; then max(H-L, |H-C_prev|, |L-C_prev|).
    np.testing.assert_allclose(true_range(high, low, close), [1.0, 2.5, 3.8, 11.0])


def test_ema_recursion_and_seed() -> None:
    x = np.array([1.0, 2.0, 4.0])
    alpha = 2.0 / (3 + 1.0)
    expected = [1.0, alpha * 2.0 + (1 - alpha) * 1.0]
    expected.append(alpha * 4.0 + (1 - alpha) * expected[1])
    np.testing.assert_allclose(ema(x, 3), expected)


def test_rolling_windows_are_partial_at_the_start() -> None:
    x = np.array([3.0, 1.0, 4.0, 1.0, 5.0])
    np.testing.assert_allclose(rolling_sum(x, 3), [3.0, 4.0, 8.0, 6.0, 10.0])
    np.testing.assert_allclose(rolling_max(x, 2), [3.0, 3.0, 4.0, 4.0, 5.0])
    np.testing.assert_allclose(rolling_min(x, 2), [3.0, 1.0, 1.0, 1.0, 1.0])


def test_signals_are_bounded(bundled: MarketData, params: SimParams) -> None:
    window = bundled.window(ms(2025, 10, 9), 4)
    s = compute_signals(window, params)
    for name in ("s_down", "s_up", "m_down", "i_down", "m_up", "b_up", "g_veto"):
        values = getattr(s, name)
        assert values.min() >= 0.0, name
        assert values.max() <= 1.0, name
    assert s.atr.min() >= 1.0


def test_crash_veto_gate_silences_growth(bundled: MarketData, params: SimParams) -> None:
    window = bundled.window(ms(2026, 2, 4), 3)
    s = compute_signals(window, params)
    vetoed = s.s_down >= params.tau_veto
    assert vetoed.any()
    assert np.all(s.s_up[vetoed] == 0.0)
    np.testing.assert_allclose(s.g_veto, np.maximum(0.0, 1.0 - s.s_down / params.tau_veto))


def test_decline_and_growth_respond_to_direction() -> None:
    n = 600
    falling = synthetic_window(n, 1, drift=-0.002, volatility=0.0003)
    rising = synthetic_window(n, 1, drift=+0.002, volatility=0.0003)
    p = SimParams()
    down, up = compute_signals(falling, p), compute_signals(rising, p)
    assert down.m_down[300:].mean() > 0.5
    assert down.m_up[300:].max() == 0.0
    assert up.m_up[300:].mean() > 0.5
    assert up.m_down[300:].max() == 0.0
    assert up.b_up[300:].mean() > 0.5


def test_imbalance_saturates_at_seventy_percent_sell_flow() -> None:
    window = synthetic_window(200, 3)
    total = window.buy_vol + window.sell_vol
    skewed = replace(window, buy_vol=total * 0.30, sell_vol=total * 0.70)
    s = compute_signals(skewed, SimParams())
    np.testing.assert_allclose(s.i_down, 1.0, atol=1e-12)
    balanced = replace(window, buy_vol=total * 0.5, sell_vol=total * 0.5)
    np.testing.assert_allclose(compute_signals(balanced, SimParams()).i_down, 0.0, atol=1e-12)
