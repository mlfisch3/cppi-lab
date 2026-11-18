"""The primary invariant: ``V_t >= F - 1e-6`` on every bar that does not gap.

Checked three ways: property-style over random continuous markets and random
parameters drawn from the full PRD ranges (Hypothesis), over many real
windows and seeds, and on hand-built adversarial paths.  The independent
recomputation below does not trust the kernel's own breach counter.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from cppi_lab.config.defaults import SimParams, default_params
from cppi_lab.data.loader import MINUTE_MS, MarketData, MarketWindow
from cppi_lab.engine.kernel_numba import FLOOR_TOLERANCE_USD, SimResult
from tests.conftest import anchored, backtest, ms, random_params, real_window, synthetic_window


def check_floor(result: SimResult, window: MarketWindow, p: SimParams) -> None:
    """Recompute V_t from the raw state columns and test it against F."""
    tel = result.telemetry
    gamma = p.f_taker + p.delta_slip
    equity = tel["cash_usd"] + tel["btc_held"] * tel["close"] * (1.0 - gamma)
    np.testing.assert_allclose(equity, tel["net_equity_usd"], rtol=0, atol=1e-9)
    floor = p.n0 * p.p_liq + p.c0
    gap = result.stats.gap_bar
    upto = tel.shape[0] if gap < 0 else gap
    margin = equity[:upto] - floor
    worst = float(margin.min()) if upto else 0.0
    assert worst >= -FLOOR_TOLERANCE_USD, (
        f"V - F = {worst:.9f} at bar {int(margin.argmin())} (gap bar {gap})"
    )
    assert result.stats.floor_breaches == 0
    assert (tel["btc_held"] >= 0.0).all()
    assert (tel["cash_usd"] >= -1e-9).all()
    if gap >= 0:
        # A gap is exactly a liquidation bar that opened below its trigger.
        assert tel["event_code"][gap] == 7
        assert window.open[gap] < tel["p_floor"][gap]


@settings(max_examples=120, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    market_seed=st.integers(0, 2**31 - 1),
    param_seed=st.integers(0, 2**31 - 1),
    volatility=st.floats(0.0002, 0.006),
    drift=st.floats(-0.0006, 0.0004),
    start_price=st.floats(1_500.0, 300_000.0),
)
def test_floor_holds_on_continuous_markets(
    market_seed: int, param_seed: int, volatility: float, drift: float, start_price: float
) -> None:
    """Bars that open at the previous close can never gap, so the floor must hold on all."""
    rng = np.random.default_rng(param_seed)
    window = synthetic_window(1_500, market_seed, start_price, volatility, drift, gap_free=True)
    p = anchored(random_params(rng), window, rng)
    _, result = backtest(window, p)
    check_floor(result, window, p)
    if window.open[0] >= result.telemetry["p_floor"][0]:
        assert result.stats.gap_bar == -1


REAL_STARTS = [
    ms(2023, 3, 9),
    ms(2024, 3, 13),
    ms(2024, 8, 2),
    ms(2025, 4, 2),
    ms(2025, 10, 8),
    ms(2026, 1, 27),
    ms(2026, 2, 3),
    ms(2026, 4, 10),
    ms(2026, 6, 1),
    ms(2026, 8, 3),
]


@pytest.mark.parametrize("start", REAL_STARTS)
@pytest.mark.parametrize("rho", [0.0, 0.25, 0.6, 1.0])
def test_floor_holds_on_real_windows_across_seeds(
    bundled: MarketData, start: int, rho: float
) -> None:
    base = replace(default_params(), rho=rho, normalize=True)
    window = real_window(bundled, start, 5, base)
    for seed in (1, 42, 1_234_567):
        for anchor in (61_000.0, 64_000.0, 70_000.0, 80_000.0):
            p = replace(base, seed=seed)
            scaled = window.rescaled(anchor / float(window.open[0]))
            _, result = backtest(scaled, p)
            check_floor(result, scaled, p)


@pytest.mark.parametrize("seed", range(30))
def test_floor_holds_for_random_parameters_on_real_data(bundled: MarketData, seed: int) -> None:
    rng = np.random.default_rng(31_000 + seed)
    start = int(rng.integers(bundled.start_ms // MINUTE_MS, bundled.end_ms // MINUTE_MS - 6_000))
    window = bundled.window(start * MINUTE_MS, 3)
    p = anchored(random_params(rng), window, rng)
    _, result = backtest(window, p)
    check_floor(result, window, p)


def flat_window(prices: list[tuple[float, float, float, float]]) -> MarketWindow:
    arr = np.array(prices, dtype=np.float64)
    n = arr.shape[0]
    return MarketWindow(
        ts_ms=np.arange(n, dtype=np.int64) * MINUTE_MS,
        open=np.ascontiguousarray(arr[:, 0]),
        high=np.ascontiguousarray(arr[:, 1]),
        low=np.ascontiguousarray(arr[:, 2]),
        close=np.ascontiguousarray(arr[:, 3]),
        buy_vol=np.full(n, 50.0),
        sell_vol=np.full(n, 50.0),
    )


def test_exit_at_the_trigger_lands_exactly_on_the_floor() -> None:
    p = replace(default_params(), p_ref=61_000.0)  # full allocation, nothing sold beforehand
    trigger = p.p_liq / (1.0 - p.gamma)
    window = flat_window(
        [(62_000, 62_100, 61_900, 62_000), (62_000, 62_050, 60_100, 60_150), (60_150,) * 4]
    )
    _, result = backtest(window, p)
    tel = result.telemetry
    assert tel["event_code"][1] == 7
    assert tel["btc_held"][1] == 0.0
    assert tel["p_floor"][1] == pytest.approx(trigger)
    assert tel["exec_px_usd"][1] == pytest.approx(trigger * (1.0 - p.delta_slip))
    assert tel["cash_usd"][1] == pytest.approx(p.floor_value, abs=1e-9)
    # Realised price after fee and slippage is inside the buffer [P_liq, P_floor].
    realised = tel["cash_usd"][1] / p.n0
    assert p.p_liq - 1e-9 <= realised < trigger
    assert result.stats.gap_bar == -1
    check_floor(result, window, p)


def test_gap_through_the_buffer_is_flagged_and_bounded() -> None:
    p = replace(default_params(), p_ref=61_000.0)
    trigger = p.p_liq / (1.0 - p.gamma)
    window = flat_window(
        [(62_000, 62_100, 61_900, 62_000), (58_000, 58_200, 57_500, 57_900), (57_900,) * 4]
    )
    _, result = backtest(window, p)
    tel = result.telemetry
    assert result.stats.gap_bar == 1
    assert tel["event_code"][1] == 7
    assert tel["exec_px_usd"][1] == pytest.approx(58_000.0 * (1.0 - p.delta_slip))
    shortfall = p.floor_value - tel["cash_usd"][1]
    assert shortfall > 0.0
    assert shortfall == pytest.approx(p.n0 * (trigger - 58_000.0) * (1.0 - p.gamma))
    assert result.stats.floor_breaches == 0  # the conservation count excludes gap bars


def test_starting_below_the_trigger_is_a_gap_at_bar_zero() -> None:
    p = default_params()
    window = flat_window([(59_000, 59_100, 58_900, 59_050), (59_050,) * 4])
    _, result = backtest(window, p)
    assert result.stats.gap_bar == 0
    assert result.telemetry["btc_held"][0] == 0.0


def test_cash_locked_account_never_rebuys() -> None:
    """After an exit exactly at the floor the cushion is zero and stays zero."""
    p = replace(default_params(), p_ref=61_000.0, rho=1.0)
    bars: list[tuple[float, float, float, float]] = [
        (62_000.0, 62_100.0, 61_900.0, 62_000.0),
        (62_000.0, 62_050.0, 60_100.0, 60_150.0),
    ]
    price = 60_150.0
    for _ in range(400):  # a long, clean rally far above the reference
        nxt = price * 1.002
        bars.append((price, nxt * 1.0005, price * 0.9995, nxt))
        price = nxt
    window = flat_window(bars)
    signals, result = backtest(window, p)
    assert signals.s_up[200:].max() > 0.9
    assert result.stats.taker_buys == 0
    assert np.all(result.telemetry["btc_held"][1:] == 0.0)
    check_floor(result, window, p)


def test_maker_buy_followed_by_a_drop_exits_in_the_same_bar() -> None:
    """Step 3b: the low of the fill bar is checked against the post-fill trigger."""
    found = False
    for seed in range(200):
        rng = np.random.default_rng(seed)
        window = synthetic_window(1_200, 500 + seed, 66_000.0, 0.003, 0.0)
        p = anchored(replace(random_params(rng), rho=0.0, dn_min_pct=0.01), window, rng)
        _, result = backtest(window, p)
        check_floor(result, window, p)
        both = (result.flags & 4 != 0) & (result.flags & 64 != 0)
        if both.any():
            bar = int(np.nonzero(both)[0][0])
            assert result.telemetry["event_code"][bar] == 7
            assert result.telemetry["btc_held"][bar] == 0.0
            found = True
            break
    assert found, "no scenario exercised the post-fill trigger re-check"
