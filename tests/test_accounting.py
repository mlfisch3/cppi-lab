"""Accounting integrity of the telemetry (PRD section 6.2, TSD section 3.2)."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from cppi_lab.config.defaults import SimParams
from cppi_lab.data.loader import MarketData
from cppi_lab.engine.kernel_numba import (
    EVENT_NAMES,
    FLAG_EXECUTED,
    FLAG_LIQUIDATION,
    FLAG_MISSED,
    TELEMETRY_DTYPE,
    TELEMETRY_FIELDS,
    event_bit,
)
from tests.conftest import backtest, ms, real_window

TSD_FIELDS = [
    ("bar_idx", np.int32),
    ("ts_utc_ms", np.int64),
    ("open", np.float64),
    ("high", np.float64),
    ("low", np.float64),
    ("close", np.float64),
    ("buy_vol", np.float64),
    ("sell_vol", np.float64),
    ("s_down", np.float64),
    ("s_up", np.float64),
    ("p_floor", np.float64),
    ("safety_buf_usd", np.float64),
    ("d_norm", np.float64),
    ("n_max", np.float64),
    ("n_ceil", np.float64),
    ("n_rebuy", np.float64),
    ("btc_held", np.float64),
    ("cash_usd", np.float64),
    ("net_equity_usd", np.float64),
    ("cushion_usd", np.float64),
    ("event_code", np.uint8),
    ("exec_qty_btc", np.float64),
    ("exec_px_usd", np.float64),
    ("fee_paid_usd", np.float64),
    ("p_fill_prob", np.float64),
    ("u_rand_draw", np.float64),
]


def test_telemetry_dtype_is_the_tsd_schema() -> None:
    assert np.dtype(TSD_FIELDS) == TELEMETRY_DTYPE
    assert len(TELEMETRY_FIELDS) == 26
    assert EVENT_NAMES[7] == "HARD_BUFFER_LIQUIDATION"
    assert EVENT_NAMES[5:7] == ("MISSED_MAKER_SELL", "MISSED_MAKER_BUY")


@pytest.mark.parametrize("rho", [0.0, 0.25, 1.0])
@pytest.mark.parametrize("start", [ms(2026, 2, 3), ms(2026, 6, 1), ms(2026, 4, 10)])
def test_row_identities(bundled: MarketData, params: SimParams, rho: float, start: int) -> None:
    p = replace(params, rho=rho)
    window = real_window(bundled, start, 6, p)
    _, result = backtest(window, p)
    tel, flags, stats = result.telemetry, result.flags, result.stats
    keep = 1.0 - p.gamma
    floor = p.floor_value

    np.testing.assert_array_equal(tel["bar_idx"], np.arange(window.n))
    np.testing.assert_array_equal(tel["ts_utc_ms"], window.ts_ms)
    for name, source in (
        ("open", window.open),
        ("close", window.close),
        ("buy_vol", window.buy_vol),
        ("sell_vol", window.sell_vol),
    ):
        np.testing.assert_array_equal(tel[name], source)

    np.testing.assert_allclose(
        tel["net_equity_usd"], tel["cash_usd"] + tel["btc_held"] * tel["close"] * keep, atol=1e-9
    )
    np.testing.assert_array_equal(
        tel["cushion_usd"], np.maximum(0.0, tel["net_equity_usd"] - floor)
    )
    np.testing.assert_array_equal(tel["safety_buf_usd"], tel["p_floor"] - p.p_liq)
    baseline = p.p_liq * p.gamma / keep
    assert tel["safety_buf_usd"].min() >= baseline - 1e-9
    assert ((tel["d_norm"] >= 0.0) & (tel["d_norm"] <= 1.0)).all()
    assert (tel["n_rebuy"] <= tel["n_ceil"]).all()
    assert (tel["n_ceil"] <= tel["n_max"] + 1e-12).all()
    assert (tel["n_ceil"] >= 0.0).all()
    assert (tel["btc_held"] >= 0.0).all()
    assert (tel["cash_usd"] >= -1e-9).all()

    # Costs: the per-bar column sums to the kernel's run totals.
    assert tel["fee_paid_usd"].sum() == pytest.approx(stats.total_costs_usd, rel=1e-12)
    assert (tel["fee_paid_usd"] >= 0.0).all()
    assert (tel["fee_paid_usd"][flags & FLAG_EXECUTED == 0] == 0.0).all()

    # Event counts: lossless flags agree with the kernel's counters.
    for code, count in (
        (1, stats.maker_sells),
        (2, stats.taker_sells),
        (3, stats.maker_buys),
        (4, stats.taker_buys),
        (5, stats.missed_sells),
        (6, stats.missed_buys),
    ):
        assert int(np.count_nonzero(flags & event_bit(code))) == count, EVENT_NAMES[code]
    assert int(np.count_nonzero(flags & FLAG_LIQUIDATION)) <= stats.liquidations

    # The primary event of a row is always one of the events that happened.
    primary = tel["event_code"]
    assert ((primary == 0) == (flags == 0)).all()
    happened = primary > 0
    assert ((flags[happened] >> (primary[happened] - 1)) & 1).all()
    # Priority: liquidation over taker fills over maker fills over misses.
    assert (primary[flags & FLAG_LIQUIDATION != 0] == 7).all()
    taker = (flags & (2 | 8) != 0) & (flags & FLAG_LIQUIDATION == 0)
    assert np.isin(primary[taker], (2, 4)).all()

    # Maker attempts carry a probability; bars without one carry NaN.
    attempted = flags & (1 | 4 | FLAG_MISSED) != 0
    assert np.isfinite(tel["p_fill_prob"][attempted]).all()
    assert np.isnan(tel["p_fill_prob"][~attempted]).all()
    filled = flags & (1 | 4) != 0
    assert (tel["u_rand_draw"][filled] <= tel["p_fill_prob"][filled]).all()
    missed = flags & FLAG_MISSED != 0
    fill_or_zero = tel["p_fill_prob"][missed]
    assert ((tel["u_rand_draw"][missed] > fill_or_zero) | (fill_or_zero == 0.0)).all()
    assert np.isnan(tel["exec_px_usd"][primary == 0]).all()
    assert (tel["exec_qty_btc"][primary == 0] == 0.0).all()

    assert stats.final_equity_usd == tel["net_equity_usd"][-1]
    assert stats.min_margin_usd <= (tel["net_equity_usd"] - floor)[0] + 1e-9


def test_holdings_only_move_on_execution_bars(bundled: MarketData, params: SimParams) -> None:
    window = real_window(bundled, ms(2026, 2, 3), 6, params)
    _, result = backtest(window, params)
    held = result.telemetry["btc_held"]
    moved = np.nonzero(np.diff(held) != 0.0)[0] + 1
    assert (result.flags[moved] & FLAG_EXECUTED != 0).all()
    cash = result.telemetry["cash_usd"]
    assert (np.diff(cash)[np.diff(held) == 0.0] == 0.0).all()


def test_taker_sell_is_neutral_for_liquidation_value(
    bundled: MarketData, params: SimParams
) -> None:
    """A taker sale converts coins at exactly the value V already assigned to them."""
    p = replace(params, rho=1.0)
    window = real_window(bundled, ms(2026, 2, 3), 6, p)
    _, result = backtest(window, p)
    tel = result.telemetry
    sells = np.nonzero(tel["event_code"] == 2)[0]
    assert sells.size > 5
    keep = 1.0 - p.gamma
    for bar in sells[sells > 0][:50]:
        before = tel["cash_usd"][bar - 1] + tel["btc_held"][bar - 1] * tel["close"][bar] * keep
        assert tel["net_equity_usd"][bar] == pytest.approx(before, abs=1e-7)
