"""KPI ribbon values (TSD section 6: eight telemetry summary cards)."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from cppi_lab.config.defaults import default_params
from cppi_lab.data.loader import MINUTE_MS, MarketData, MarketWindow
from cppi_lab.engine.controller import SimulationController
from cppi_lab.ui.kpi_ribbon import (
    KPI_SPECS,
    SHORT_CHARS,
    SUBTITLE_CHARS,
    KpiValue,
    compute_kpis,
)
from tests.conftest import ms, random_params


def assert_fits(kpis: dict[str, KpiValue]) -> None:
    """Every sub-line respects the length the two card layouts are sized for."""
    for key, kpi in kpis.items():
        assert len(kpi.subtitle) <= SUBTITLE_CHARS, (key, kpi.subtitle)
        assert 0 < len(kpi.short) <= SHORT_CHARS, (key, kpi.short)


def test_there_are_eight_cards_with_popovers() -> None:
    assert len(KPI_SPECS) == 8
    assert len({spec.key for spec in KPI_SPECS}) == 8
    assert all(spec.popover.startswith("kpi-") for spec in KPI_SPECS)


def test_idle_window_reads_intact(bundled: MarketData) -> None:
    run = SimulationController(bundled).run()
    kpis = compute_kpis(run)
    assert set(kpis) == {spec.key for spec in KPI_SPECS}
    assert kpis["floor"].value == "INTACT"
    assert kpis["floor"].tone == "good"
    assert kpis["floor"].subtitle == "no breach · F $60,000.00"
    assert kpis["floor"].short == "no breach"
    assert kpis["holdings"].subtitle == "max recoverable 1.0000"
    assert kpis["trigger"].subtitle == "+$240.96 over P_liq"
    assert_fits(kpis)
    assert kpis["fills"].value == "0 / 0"
    assert kpis["missed"].value == "0"
    assert kpis["missed"].subtitle == "no maker attempts"
    assert kpis["costs"].value == "$0.00"
    assert kpis["holdings"].value == "1.0000"
    assert kpis["trigger"].value == "$60,240.96"
    last = run.result.telemetry[-1]
    assert kpis["equity"].value == f"${last['net_equity_usd']:,.2f}"
    assert kpis["cushion"].value == f"${last['net_equity_usd'] - 60_000.0:,.2f}"


def test_busy_window_reports_the_kernel_counters(bundled: MarketData) -> None:
    controller = SimulationController(
        bundled, replace(default_params(), start_ms=ms(2026, 2, 3), duration_days=6)
    )
    run = controller.run()
    stats = run.result.stats
    kpis = compute_kpis(run)
    assert kpis["fills"].value == f"{stats.maker_fills} / {stats.taker_fills}"
    assert kpis["missed"].value == str(stats.missed)
    assert kpis["costs"].value == f"${stats.total_costs_usd:,.2f}"
    # The floor held although the emergency exit fired: the card says so, counts the
    # exits and tells whether the strategy has bought back in since.
    assert stats.liquidations == 1
    assert float(run.result.telemetry[-1]["btc_held"]) > 0.0
    assert kpis["floor"].value == "INTACT"
    assert kpis["floor"].tone == "warn"
    assert kpis["floor"].subtitle == "1 liquidation, holds BTC"
    assert kpis["floor"].short == "1 liq. · in BTC"
    assert_fits(kpis)
    ensemble = controller.run_ensemble()
    with_fan = compute_kpis(run, ensemble)
    assert with_fan["floor"].value == "INTACT"
    assert with_fan["floor"].subtitle.startswith("MC 0 breaches · ")
    assert with_fan["floor"].subtitle.endswith(" liq.")
    assert_fits(with_fan)


def test_floor_card_tells_cash_from_btc_after_liquidations(bundled: MarketData) -> None:
    run = SimulationController(
        bundled, replace(default_params(), start_ms=ms(2026, 6, 1), duration_days=30)
    ).run()
    stats = run.result.stats
    assert stats.liquidations >= 5
    assert float(run.result.telemetry[-1]["btc_held"]) == 0.0
    floor = compute_kpis(run)["floor"]
    assert floor.value == "INTACT"
    assert floor.subtitle == f"{stats.liquidations} liquidations, in cash"
    assert floor.short == f"{stats.liquidations} liq. · in cash"


def test_sublines_fit_for_any_account_size(bundled: MarketData) -> None:
    """Sub-lines are sized for the cards, from a thousandth of a coin to a thousand."""
    rng = np.random.default_rng(11)
    starts = (ms(2026, 2, 3), ms(2026, 6, 1), ms(2025, 10, 3), ms(2024, 7, 29))
    for trial in range(24):
        base = replace(
            default_params(),
            start_ms=starts[trial % len(starts)],
            duration_days=int(rng.integers(3, 40)),
            normalize=bool(trial % 2),
        )
        params = random_params(rng, base)
        run = SimulationController(bundled, params).run()
        assert_fits(compute_kpis(run))


def test_gap_is_reported_as_a_breach() -> None:
    n = 6
    prices = np.array([62_000.0, 58_000.0, 57_900.0, 57_900.0, 57_900.0, 57_900.0])
    window = MarketWindow(
        ts_ms=np.arange(n, dtype=np.int64) * MINUTE_MS + ms(2026, 1, 1),
        open=prices.copy(),
        high=prices + 50.0,
        low=prices - 50.0,
        close=prices.copy(),
        buy_vol=np.full(n, 5.0),
        sell_vol=np.full(n, 5.0),
    )
    data = MarketData(
        "gap",
        "test",
        window.ts_ms,
        window.open,
        window.high,
        window.low,
        window.close,
        window.buy_vol,
        window.sell_vol,
    )
    run = SimulationController(
        data, replace(default_params(), start_ms=int(window.ts_ms[0]), p_ref=61_000.0)
    ).run()
    assert run.result.stats.gap_bar == 1
    kpis = compute_kpis(run)
    assert kpis["floor"].value == "GAP BREACH"
    assert kpis["floor"].tone == "bad"
    assert kpis["floor"].subtitle == "gapped: -$2,232.00 vs F"
    assert_fits(kpis)
    assert kpis["cushion"].value == "$0.00"
    assert kpis["holdings"].value == "0.0000"
    assert float(kpis["equity"].value.replace("$", "").replace(",", "")) < 60_000.0
