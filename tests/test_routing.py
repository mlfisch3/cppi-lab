"""Routing law of TSD section 2.4 at rho = 0, rho = 1 and interior values."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from cppi_lab.config.defaults import SimParams
from cppi_lab.data.loader import MarketData
from cppi_lab.engine.kernel_numba import (
    EVENT_HARD_BUFFER_LIQUIDATION,
    EVENT_MAKER_BUY,
    EVENT_MAKER_SELL,
    EVENT_MISSED_MAKER_BUY,
    EVENT_MISSED_MAKER_SELL,
    EVENT_TAKER_BUY,
    EVENT_TAKER_SELL,
    event_bit,
    execution_urgency,
    route_order_by_rho,
    taker_fraction,
)
from tests.conftest import backtest, ms, real_window

MAKER_BITS = (
    event_bit(EVENT_MAKER_SELL)
    | event_bit(EVENT_MAKER_BUY)
    | event_bit(EVENT_MISSED_MAKER_SELL)
    | event_bit(EVENT_MISSED_MAKER_BUY)
)
TAKER_BITS = event_bit(EVENT_TAKER_SELL) | event_bit(EVENT_TAKER_BUY)
WINDOWS = [ms(2026, 2, 3), ms(2026, 6, 1), ms(2026, 4, 10)]


def test_taker_fraction_boundaries_and_interior() -> None:
    for urgency in (0.0, 0.3, 0.999, 1.0):
        assert taker_fraction(0.0, urgency) == 0.0  # rho = 0: identically zero
        assert taker_fraction(1.0, urgency) == 1.0  # rho = 1: identically one
    assert taker_fraction(0.25, 0.74) == 0.25  # calm: split
    assert taker_fraction(0.25, 0.75) == 1.0  # u >= 1 - rho: escalate
    assert taker_fraction(0.6, 0.39) == 0.6
    assert taker_fraction(0.6, 0.40) == 1.0


def test_urgency_formulas() -> None:
    # Selling: 1 - d (1 - S_down) 0.5^m
    assert execution_urgency(True, 1.0, 0.0, 0.0, 0) == 0.0
    assert execution_urgency(True, 0.8, 0.5, 0.0, 0) == pytest.approx(0.6)
    assert execution_urgency(True, 0.8, 0.5, 0.0, 2) == pytest.approx(0.9)
    assert execution_urgency(True, 0.0, 0.0, 0.0, 0) == 1.0
    # Buying: S_up (1 - 0.5^(m+1))
    assert execution_urgency(False, 0.5, 0.0, 0.8, 0) == pytest.approx(0.4)
    assert execution_urgency(False, 0.5, 0.0, 0.8, 3) == pytest.approx(0.75)


def test_misses_escalate_a_calm_sell_within_two_bars_at_default_rho() -> None:
    thetas = [taker_fraction(0.25, execution_urgency(True, 1.0, 0.0, 0.0, m)) for m in range(4)]
    assert thetas == [0.25, 0.25, 1.0, 1.0]


@pytest.mark.parametrize("rho", [0.05, 0.25, 0.5, 0.9])
def test_split_orders_add_up_exactly(rho: float) -> None:
    dn = 0.123456789
    q_taker, q_maker = route_order_by_rho(True, dn, 1.0, 0.0, 0.0, 0, rho, False)
    assert q_taker == rho * dn
    assert q_taker + q_maker == dn
    # Escalated: nothing left to post.
    q_taker, q_maker = route_order_by_rho(True, dn, 0.0, 0.0, 0.0, 0, rho, False)
    assert (q_taker, q_maker) == (dn, 0.0)


def test_router_extremes_and_maker_dominance() -> None:
    assert route_order_by_rho(True, 2.0, 0.0, 1.0, 0.0, 5, 0.0, False) == (0.0, 2.0)
    assert route_order_by_rho(False, 2.0, 1.0, 0.0, 0.0, 0, 1.0, False) == (2.0, 0.0)
    assert route_order_by_rho(True, 2.0, 1.0, 0.0, 0.0, 0, 0.0, True) == (2.0, 0.0)


@pytest.mark.parametrize("start", WINDOWS)
def test_rho_zero_places_only_maker_orders_above_the_trigger(
    bundled: MarketData, params: SimParams, start: int
) -> None:
    p = replace(params, rho=0.0)
    _, result = backtest(real_window(bundled, start, 8, p), p)
    stats = result.stats
    assert stats.maker_attempts > 50
    assert not (result.flags & TAKER_BITS).any()
    assert stats.taker_sells == stats.taker_buys == 0
    # The only taker executions are emergency exits at the trigger.
    liquidations = result.flags & event_bit(EVENT_HARD_BUFFER_LIQUIDATION) != 0
    tel = result.telemetry
    assert stats.taker_volume_btc == pytest.approx(float(tel["exec_qty_btc"][liquidations].sum()))
    at_trigger = tel["low"][liquidations] <= tel["p_floor"][liquidations]
    assert at_trigger.all()


@pytest.mark.parametrize("start", WINDOWS)
def test_rho_one_places_only_taker_orders(
    bundled: MarketData, params: SimParams, start: int
) -> None:
    p = replace(params, rho=1.0)
    _, result = backtest(real_window(bundled, start, 8, p), p)
    stats = result.stats
    assert stats.taker_fills > 20
    assert not (result.flags & MAKER_BITS).any()
    assert stats.maker_attempts == 0
    assert stats.maker_fees_usd == 0.0
    assert np.isnan(result.telemetry["p_fill_prob"]).all()
    # Deterministic: the seed cannot matter when nothing is passive.
    _, other = backtest(real_window(bundled, start, 8, p), replace(p, seed=7))
    np.testing.assert_array_equal(result.telemetry["btc_held"], other.telemetry["btc_held"])


@pytest.mark.parametrize("rho", [0.25, 0.6])
def test_interior_rho_mixes_both_routes(bundled: MarketData, params: SimParams, rho: float) -> None:
    p = replace(params, rho=rho)
    _, result = backtest(real_window(bundled, ms(2026, 2, 3), 8, p), p)
    stats = result.stats
    assert stats.maker_attempts > 0
    assert stats.taker_fills > 0
    low, high = (
        backtest(real_window(bundled, ms(2026, 2, 3), 8, q), q)[1].stats
        for q in (replace(params, rho=0.0), replace(params, rho=1.0))
    )
    assert low.taker_fills == 0 < stats.taker_fills
    assert high.maker_attempts == 0 < stats.maker_attempts


def test_maker_dominated_fees_route_everything_as_taker(
    bundled: MarketData, params: SimParams
) -> None:
    p = replace(params, rho=0.0, f_maker=0.005, f_taker=0.003, delta_slip=0.001)
    _, result = backtest(real_window(bundled, ms(2026, 2, 3), 6, p), p)
    assert result.stats.taker_fills > 0
    assert result.stats.maker_attempts == 0
