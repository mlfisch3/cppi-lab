"""Simulation controller: caching, clamping, reset and rescaling (FR-1.3, FR-2)."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from cppi_lab.config.defaults import DEFAULT_START_MS, SimParams, default_params
from cppi_lab.data.loader import MINUTE_MS, MarketData
from cppi_lab.engine.controller import SimulationController
from tests.conftest import ms


@pytest.fixture
def controller(bundled: MarketData) -> SimulationController:
    return SimulationController(bundled)


def test_starts_at_the_prd_defaults(controller: SimulationController) -> None:
    assert controller.params == default_params()
    run = controller.run()
    assert run.window.n == 43_200
    assert run.window.ts_ms[0] == DEFAULT_START_MS
    assert run.result.n_bars == 43_200
    assert run.timings.total_ms > 0.0
    assert run.generation == 1


def test_identical_inputs_give_identical_output(controller: SimulationController) -> None:
    controller.update("normalize", True)
    a, b = controller.run(), controller.run()
    assert a.result.telemetry.tobytes() == b.result.telemetry.tobytes()
    assert b.generation == a.generation + 1


def test_stage_caches_are_reused_only_when_valid(controller: SimulationController) -> None:
    controller.update("normalize", True)
    base = controller.run()
    controller.update("rho", 0.5)  # routing only: window and signals reused
    routed = controller.run()
    assert routed.window is base.window
    assert routed.signals is base.signals
    controller.update("w_fast_d", 9)  # a signal parameter: signals recomputed
    resignalled = controller.run()
    assert resignalled.window is base.window
    assert resignalled.signals is not base.signals
    controller.update("start_ms", ms(2026, 1, 2))  # a new window: everything recomputed
    moved = controller.run()
    assert moved.window is not base.window
    controller.update("seed", 7)
    reseeded = controller.run()
    assert reseeded.signals is moved.signals
    assert not np.array_equal(
        reseeded.result.telemetry["u_rand_draw"], moved.result.telemetry["u_rand_draw"]
    )


def test_cached_stages_match_a_fresh_controller(
    bundled: MarketData, controller: SimulationController
) -> None:
    for key, value in (
        ("normalize", True),
        ("rho", 0.5),
        ("w_fast_d", 9),
        ("seed", 7),
        ("start_ms", ms(2026, 2, 1)),
        ("k_sell", 1.4),
    ):
        controller.update(key, value)
        controller.run()
    fresh = SimulationController(bundled, controller.params)
    assert controller.run().result.telemetry.tobytes() == fresh.run().result.telemetry.tobytes()


def test_start_time_is_clamped_to_the_dataset(controller: SimulationController) -> None:
    lo, hi = controller.start_bounds_ms
    assert controller.update("start_ms", lo - 10 * MINUTE_MS).start_ms == lo
    assert controller.update("start_ms", hi + 10 * MINUTE_MS).start_ms == hi
    assert controller.update("start_ms", ms(2026, 3, 1) + 12_345).start_ms == ms(2026, 3, 1)
    # A window that runs past the end of the data is simply shorter.
    controller.update("start_ms", hi - 99 * MINUTE_MS)
    assert controller.run().window.n == 100


def test_reset_one_and_reset_all(controller: SimulationController) -> None:
    controller.update("rho", 0.9)
    controller.update("k_sell", 1.7)
    assert controller.reset("rho").rho == 0.25
    assert controller.params.k_sell == 1.7
    assert controller.reset() == default_params()


def test_ranges_and_coupling_are_enforced(controller: SimulationController) -> None:
    assert controller.update("rho", 7.0).rho == 1.0
    assert controller.update("p_liq", 80_000.0).p_ref == 81_000.0
    with pytest.raises(KeyError):
        controller.update("bogus", 1.0)


def test_normalisation_anchors_the_first_open(controller: SimulationController) -> None:
    raw = controller.run()
    controller.set_params(replace(controller.params, normalize=True, anchor_price=68_000.0))
    scaled = controller.run()
    factor = 68_000.0 / float(raw.window.open[0])
    assert scaled.window.open[0] == pytest.approx(68_000.0)
    assert scaled.window.price_scale == pytest.approx(factor)
    np.testing.assert_allclose(scaled.window.close, raw.window.close * factor, rtol=1e-15)
    assert np.shares_memory(scaled.window.buy_vol, raw.window.buy_vol)  # volumes untouched
    np.testing.assert_array_equal(scaled.result.telemetry["close"], scaled.window.close)
    controller.update("anchor_price", 64_000.0)
    assert controller.run().window.open[0] == pytest.approx(64_000.0)


def test_default_window_has_nothing_to_trade(controller: SimulationController) -> None:
    """Jan 2026 stayed above P_ref, so the envelope never asks for an order."""
    run = controller.run()
    stats = run.result.stats
    assert run.window.low.min() > controller.params.p_ref
    assert stats.maker_attempts == stats.taker_fills == stats.liquidations == 0
    assert (run.result.telemetry["btc_held"] == 1.0).all()
    assert (run.result.telemetry["d_norm"] == 1.0).all()


def test_ensemble_matches_the_direct_call(controller: SimulationController) -> None:
    from cppi_lab.engine.monte_carlo import run_ensemble

    controller.set_params(replace(controller.params, normalize=True, duration_days=2, rho=0.0))
    run = controller.run()
    direct = run_ensemble(run.window, run.signals, controller.params)
    cached = controller.run_ensemble()
    again = controller.run_ensemble()  # second call reuses the cached uniform block
    np.testing.assert_array_equal(direct.p05, cached.p05)
    np.testing.assert_array_equal(cached.p95, again.p95)
    assert cached.seed == controller.params.seed


def test_switching_dataset_refits_the_start(bundled: MarketData) -> None:
    lo = bundled.index_at(ms(2024, 5, 1))
    hi = bundled.index_at(ms(2024, 5, 4))
    subset = MarketData(
        name="subset",
        source="test",
        ts_ms=bundled.ts_ms[lo:hi],
        open=bundled.open[lo:hi],
        high=bundled.high[lo:hi],
        low=bundled.low[lo:hi],
        close=bundled.close[lo:hi],
        buy_vol=bundled.buy_vol[lo:hi],
        sell_vol=bundled.sell_vol[lo:hi],
    )
    controller = SimulationController(bundled, SimParams())
    controller.run()
    params = controller.set_data(subset)
    assert params.start_ms == subset.end_ms  # 2026 lies after the subset: clamp to its end
    controller.update("start_ms", subset.start_ms)
    assert controller.run().window.n == subset.n
