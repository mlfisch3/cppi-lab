"""KPI ribbon values (TSD section 6: eight telemetry summary cards)."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from cppi_lab.config.defaults import default_params
from cppi_lab.data.loader import MINUTE_MS, MarketData, MarketWindow
from cppi_lab.engine.controller import SimulationController
from cppi_lab.ui.kpi_ribbon import KPI_SPECS, compute_kpis
from tests.conftest import ms


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
    assert kpis["floor"].value == "LIQUIDATED"
    assert kpis["floor"].tone == "warn"
    assert f"×{stats.liquidations}" in kpis["floor"].subtitle
    ensemble = controller.run_ensemble()
    with_fan = compute_kpis(run, ensemble)
    assert "MC 100 seeds: 0 breaches" in with_fan["floor"].subtitle


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
    assert kpis["cushion"].value == "$0.00"
    assert kpis["holdings"].value == "0.0000"
    assert float(kpis["equity"].value.replace("$", "").replace(",", "")) < 60_000.0
