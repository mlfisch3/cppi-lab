"""Three-factor maker fill model and its Bernoulli resolution (TSD section 2.5)."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from cppi_lab.config.defaults import default_params
from cppi_lab.data.loader import MINUTE_MS, MarketWindow
from cppi_lab.engine.kernel_numba import fill_probability, run_simulation
from cppi_lab.engine.signals import compute_signals

LAM, BETA, ALPHA = 2.5, 1.8, 1.2


def prob(
    is_sell: bool,
    limit: float,
    qty: float,
    o: float,
    h: float,
    lo: float,
    c: float,
    opp: float = 1_000.0,
    atr: float = 40.0,
    lam: float = LAM,
    beta: float = BETA,
    alpha: float = ALPHA,
) -> float:
    return float(fill_probability(is_sell, limit, qty, o, h, lo, c, opp, atr, lam, beta, alpha))


def test_no_penetration_means_zero_probability() -> None:
    assert prob(True, 100.0, 1.0, 99.0, 100.0, 98.0, 99.5) == 0.0  # high only touches
    assert prob(True, 100.0, 1.0, 99.0, 99.9, 98.0, 99.5) == 0.0
    assert prob(False, 100.0, 1.0, 101.0, 102.0, 100.0, 101.5) == 0.0  # low only touches
    assert prob(False, 100.0, 1.0, 101.0, 102.0, 100.5, 101.5) == 0.0


def test_matches_the_closed_form() -> None:
    # Maker sell at 70,000; bar 69,990 -> 70,060 high, closes 69,970 (adverse 20).
    got = prob(True, 70_000.0, 0.5, 69_990.0, 70_060.0, 69_950.0, 69_970.0, opp=30.0, atr=40.0)
    phi_pen = 1.0 - math.exp(-LAM * 60.0 / 40.0)
    phi_vel = math.exp(-BETA * 20.0 / 40.0)
    phi_size = 1.0 / (1.0 + (0.5 / (0.01 * 30.0)) ** ALPHA)
    assert got == pytest.approx(phi_pen * phi_vel * phi_size, rel=1e-15)
    # Maker buy mirrors it: penetration below the limit, adverse = close above open.
    got = prob(False, 70_000.0, 0.5, 70_010.0, 70_050.0, 69_940.0, 70_030.0, opp=30.0, atr=40.0)
    phi_pen = 1.0 - math.exp(-LAM * 60.0 / 40.0)
    phi_vel = math.exp(-BETA * 20.0 / 40.0)
    assert got == pytest.approx(phi_pen * phi_vel * phi_size, rel=1e-15)


def test_favourable_close_carries_no_velocity_penalty() -> None:
    rising = prob(True, 100.0, 1e-9, 99.0, 101.0, 98.5, 100.8, atr=1.0)
    assert rising == pytest.approx(1.0 - math.exp(-LAM * 1.0), rel=1e-9)


def test_monotonic_in_each_parameter_and_in_size() -> None:
    args = (True, 70_000.0, 0.4, 69_990.0, 70_030.0, 69_950.0, 69_975.0)
    base = prob(*args, opp=60.0)
    assert prob(*args, opp=60.0, lam=5.0) > base  # deeper book -> more fills
    assert prob(*args, opp=60.0, beta=4.0) < base  # harsher velocity penalty
    assert prob(True, 70_000.0, 0.8, *args[3:], opp=60.0) < base  # larger order
    assert prob(*args, opp=600.0) > base  # more counter-party volume
    big = (True, 70_000.0, 5.0, 69_990.0, 70_030.0, 69_950.0, 69_975.0)
    assert prob(*big, opp=60.0, alpha=3.0) < prob(*big, opp=60.0, alpha=1.2)


def test_half_probability_at_the_absorbable_volume() -> None:
    # q equals 1 % of the opposing volume -> phi_size = 1/2 for any alpha.
    for alpha in (0.5, 1.2, 4.0):
        got = prob(True, 100.0, 0.3, 100.0, 5_000.0, 99.0, 100.0, opp=30.0, atr=1.0, alpha=alpha)
        assert got == pytest.approx(0.5, rel=1e-9)


def test_probability_is_bounded_and_zero_volume_is_safe() -> None:
    rng = np.random.default_rng(5)
    for _ in range(500):
        o = float(rng.uniform(50, 150))
        c = float(rng.uniform(50, 150))
        h = max(o, c) + float(rng.uniform(0, 20))
        lo = min(o, c) - float(rng.uniform(0, 20))
        got = prob(
            bool(rng.integers(2)),
            float(rng.uniform(50, 150)),
            float(rng.uniform(0, 50)),
            o,
            h,
            lo,
            c,
            opp=float(rng.choice([0.0, 1.0, 500.0])),
            atr=1.0,
        )
        assert 0.0 <= got <= 1.0
    assert prob(True, 100.0, 1.0, 100.0, 105.0, 99.0, 100.0, opp=0.0) < 1e-9


def one_order_window() -> MarketWindow:
    """Bar 0 forces a maker sell; bar 1 trades through its limit."""
    rows = np.array(
        [
            (66_000.0, 66_050.0, 65_950.0, 66_000.0),
            (66_000.0, 66_080.0, 65_990.0, 66_040.0),
            (66_040.0, 66_060.0, 66_020.0, 66_040.0),
        ]
    )
    n = rows.shape[0]
    return MarketWindow(
        ts_ms=np.arange(n, dtype=np.int64) * MINUTE_MS,
        open=np.ascontiguousarray(rows[:, 0]),
        high=np.ascontiguousarray(rows[:, 1]),
        low=np.ascontiguousarray(rows[:, 2]),
        close=np.ascontiguousarray(rows[:, 3]),
        buy_vol=np.full(n, 400.0),
        sell_vol=np.full(n, 400.0),
    )


def test_uniform_draw_decides_fill_or_miss() -> None:
    window = one_order_window()
    p = replace(default_params(), rho=0.0)  # passive only
    signals = compute_signals(window, p)
    first = run_simulation(window, signals, p, np.array([0.5, 0.0, 0.5]))
    p_fill = float(first.telemetry["p_fill_prob"][1])
    assert 0.0 < p_fill < 1.0
    qty = float(first.telemetry["exec_qty_btc"][1])
    limit = 66_000.0

    at = run_simulation(window, signals, p, np.array([0.5, p_fill, 0.5]))  # U == P_fill fills
    assert at.telemetry["event_code"][1] == 1
    assert at.telemetry["exec_px_usd"][1] == limit
    assert at.telemetry["fee_paid_usd"][1] == pytest.approx(qty * limit * p.f_maker)
    assert at.telemetry["btc_held"][1] == pytest.approx(p.n0 - qty)
    assert at.telemetry["cash_usd"][1] == pytest.approx(qty * limit * (1.0 - p.f_maker))

    above = run_simulation(window, signals, p, np.array([0.5, np.nextafter(p_fill, 1.0), 0.5]))
    assert above.telemetry["event_code"][1] == 5  # MISSED_MAKER_SELL
    assert above.telemetry["exec_px_usd"][1] == limit
    assert above.telemetry["exec_qty_btc"][1] == pytest.approx(qty)
    assert above.telemetry["btc_held"][1] == p.n0
    assert above.telemetry["cash_usd"][1] == 0.0
    assert above.telemetry["fee_paid_usd"][1] == 0.0
    assert above.stats.missed_sells >= 1


def test_untouched_limit_misses_even_with_a_zero_draw() -> None:
    window = one_order_window()
    lowered = replace(
        window,
        high=np.array([66_050.0, 66_000.0, 66_060.0]),
        close=np.array([66_000.0, 65_995.0, 66_040.0]),
    )
    p = replace(default_params(), rho=0.0)
    result = run_simulation(lowered, compute_signals(lowered, p), p, np.zeros(3))
    assert result.telemetry["p_fill_prob"][1] == 0.0
    assert result.telemetry["event_code"][1] == 5
    assert np.isnan(result.telemetry["p_fill_prob"][0])  # no order was resting on bar 0
