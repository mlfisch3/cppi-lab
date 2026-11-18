"""Ledger widget and chart workspace behaviour (PRD FR-4.2, FR-5.2)."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from PySide6 import QtCore
from pytestqt.qtbot import QtBot

from cppi_lab.config.defaults import default_params
from cppi_lab.data.loader import MarketData
from cppi_lab.engine.controller import RunOutput, SimulationController
from cppi_lab.engine.kernel_numba import FLAG_EXECUTED, FLAG_MISSED, event_bit
from cppi_lab.ui import theme
from cppi_lab.ui.chart_gl import (
    GLLayerItem,
    band_decimate,
    mean_decimate,
    minmax_decimate,
    ribbon_strip,
    segment_quads,
    step_points,
)
from cppi_lab.ui.chart_workspace import PANE_TITLES, ChartWorkspace
from cppi_lab.ui.ledger_window import FILTERS, LedgerWidget
from tests.conftest import ms

pytestmark = pytest.mark.ui


@pytest.fixture(scope="module")
def run(bundled: MarketData) -> RunOutput:
    controller = SimulationController(
        bundled, replace(default_params(), start_ms=ms(2026, 2, 3), duration_days=6)
    )
    return controller.run()


@pytest.fixture
def ledger(qtbot: QtBot, run: RunOutput) -> LedgerWidget:
    widget = LedgerWidget()
    qtbot.addWidget(widget)
    widget.set_result(run.result)
    widget.show()
    return widget


@pytest.fixture
def chart(qtbot: QtBot, run: RunOutput) -> ChartWorkspace:
    widget = ChartWorkspace()
    qtbot.addWidget(widget)
    widget.resize(1400, 900)
    widget.show()
    widget.set_run(run)
    qtbot.wait(30)
    return widget


# -- ledger ---------------------------------------------------------------------------


def test_filter_combo_offers_the_four_prd_filters(ledger: LedgerWidget, run: RunOutput) -> None:
    combo = ledger.filter_combo
    assert [combo.itemText(i) for i in range(combo.count())] == list(FILTERS)
    assert ledger.model.rowCount() == run.result.n_bars
    assert ledger.count_label.text() == f"{run.result.n_bars:,} of {run.result.n_bars:,} timesteps"
    combo.setCurrentText("Missed Maker Orders")
    missed = int(np.count_nonzero(run.result.flags & FLAG_MISSED))
    assert ledger.model.rowCount() == missed
    assert ledger.count_label.text().startswith(f"{missed:,} of")
    combo.setCurrentText("All Timesteps")
    assert ledger.model.rowCount() == run.result.n_bars


def test_header_click_sorts_and_toggles_direction(
    qtbot: QtBot, ledger: LedgerWidget, run: RunOutput
) -> None:
    header = ledger.table.horizontalHeader()
    column = 18  # net_equity_usd
    equity = run.result.telemetry["net_equity_usd"]
    position = QtCore.QPoint(header.sectionViewportPosition(column) + 10, header.height() // 2)
    ledger.table.scrollTo(ledger.model.index(0, column))
    position = QtCore.QPoint(header.sectionViewportPosition(column) + 10, header.height() // 2)
    qtbot.mouseClick(header.viewport(), QtCore.Qt.MouseButton.LeftButton, pos=position)
    first = equity[ledger.model.view_rows]
    assert header.sortIndicatorSection() == column
    qtbot.mouseClick(header.viewport(), QtCore.Qt.MouseButton.LeftButton, pos=position)
    second = equity[ledger.model.view_rows]
    ascending, descending = sorted((first, second), key=lambda a: a[0])
    assert np.all(np.diff(ascending) >= 0)
    assert np.all(np.diff(descending) <= 0)


def test_table_scrolls_both_ways_and_is_read_only(ledger: LedgerWidget) -> None:
    table = ledger.table
    ledger.resize(700, 300)
    assert table.horizontalScrollBar().maximum() > 0  # 26 columns do not fit: scrollable
    assert table.verticalScrollBar().maximum() > 0
    assert table.editTriggers() == table.EditTrigger.NoEditTriggers
    assert table.model().columnCount() == 26


def test_row_selection_emits_the_bar_and_select_bar_is_silent(
    qtbot: QtBot, ledger: LedgerWidget, run: RunOutput
) -> None:
    ledger.set_filter("Executed Trades")
    executed = np.nonzero(run.result.flags & FLAG_EXECUTED)[0]
    with qtbot.waitSignal(ledger.barActivated, timeout=2_000) as activated:
        ledger.table.setCurrentIndex(ledger.model.index(3, 0))
    assert activated.args == [int(executed[3])]
    with qtbot.assertNotEmitted(ledger.barActivated, wait=50):
        ledger.select_bar(int(executed[10]))  # driven by the chart: no echo
    assert ledger.current_bar() == int(executed[10])
    with qtbot.assertNotEmitted(ledger.barActivated, wait=50):
        ledger.select_bar(int(executed[10]) + 1)  # filtered out: nearest shown row
    assert ledger.current_bar() in (int(executed[10]), int(executed[11]))


def test_changing_filter_resyncs_the_chart(qtbot: QtBot, ledger: LedgerWidget) -> None:
    ledger.table.setCurrentIndex(ledger.model.index(5, 0))
    with qtbot.waitSignal(ledger.barActivated, timeout=2_000) as activated:
        ledger.set_filter("Floor Liquidations")
    assert activated.args == [ledger.current_bar()]
    assert ledger.current_bar() != 5


def test_empty_filter_shows_a_hint(qtbot: QtBot, bundled: MarketData) -> None:
    widget = LedgerWidget()
    qtbot.addWidget(widget)
    widget.show()
    widget.set_result(SimulationController(bundled).run().result)  # the idle default window
    widget.set_filter("Executed Trades")
    assert widget.model.rowCount() == 0
    assert widget.empty_label.isVisible()
    assert "Executed Trades" in widget.empty_label.text()
    widget.set_filter("All Timesteps")
    assert not widget.empty_label.isVisible()


@pytest.mark.parametrize("name", ["out.csv", "out.parquet", "no_extension"])
def test_export_from_the_widget(
    qtbot: QtBot, ledger: LedgerWidget, tmp_path: Path, name: str
) -> None:
    ledger.set_filter("Executed Trades")
    with qtbot.waitSignal(ledger.exported, timeout=5_000) as exported:
        rows = ledger.export_to(tmp_path / name)
    written = Path(exported.args[0])
    assert written.exists()
    assert written.suffix in (".csv", ".parquet")
    frame = pl.read_parquet(written) if written.suffix == ".parquet" else pl.read_csv(written)
    assert frame.height == rows == ledger.model.rowCount() == exported.args[1]
    assert set(frame["event_code"].unique().to_list()) <= {1, 2, 3, 4, 7}


# -- chart ------------------------------------------------------------------------------


def test_three_linked_panes_with_headers(chart: ChartWorkspace) -> None:
    assert len(chart.plots) == 3
    assert [h.title.text() for h in chart.headers] == list(PANE_TITLES)
    assert [h.badge.popover_id for h in chart.headers] == [
        "pane-price",
        "pane-holdings",
        "pane-signals",
    ]
    first = chart.plots[0].getViewBox()
    for plot in chart.plots[1:]:
        assert plot.getViewBox().linkedView(0) is first  # one shared, locked x axis
    for header, plot in zip(chart.headers, chart.plots, strict=True):
        assert header.isVisible()
        assert abs(header.y() - plot.sceneBoundingRect().top()) <= 1


def test_zoom_is_synchronised_across_panes(qtbot: QtBot, chart: ChartWorkspace) -> None:
    x0, x1 = chart.visible_range()
    chart.set_visible_range(x0 + 3_600.0, x0 + 9 * 3_600.0)
    qtbot.wait(20)
    for plot in chart.plots:
        (lo, hi), _ = plot.getViewBox().viewRange()
        assert lo == pytest.approx(x0 + 3_600.0, abs=1.0)
        assert hi == pytest.approx(x0 + 9 * 3_600.0, abs=1.0)
    chart.reset_view()
    assert chart.visible_range() == pytest.approx((x0, x1))


def test_layers_carry_the_telemetry(chart: ChartWorkspace, run: RunOutput) -> None:
    tel = run.result.telemetry
    n0 = run.params.n0
    np.testing.assert_array_equal(chart.price._y, tel["close"])
    np.testing.assert_array_equal(chart.price_band._y, tel["low"])
    np.testing.assert_array_equal(chart.price_band._y2, tel["high"])
    np.testing.assert_array_equal(chart.p_floor._y, tel["p_floor"])
    np.testing.assert_array_equal(chart.p_liq._y, np.full(tel.shape[0], run.params.p_liq))
    np.testing.assert_allclose(chart.equity._y, tel["net_equity_usd"] / n0)
    np.testing.assert_array_equal(chart.n_held._y, tel["btc_held"])
    np.testing.assert_array_equal(chart.n_max._y, tel["n_max"])
    np.testing.assert_array_equal(chart.n_ceil._y, tel["n_ceil"])
    np.testing.assert_array_equal(chart.n_rebuy._y, tel["n_rebuy"])
    np.testing.assert_array_equal(chart.deadband._y, tel["n_rebuy"])
    np.testing.assert_array_equal(chart.deadband._y2, tel["n_ceil"])
    np.testing.assert_array_equal(chart.s_down._y, tel["s_down"])
    np.testing.assert_array_equal(chart.s_up._y, tel["s_up"])
    np.testing.assert_array_equal(chart._x, tel["ts_utc_ms"] / 1000.0)


def test_visual_encoding_matches_the_tsd_matrix(chart: ChartWorkspace) -> None:
    def spec(layer: GLLayerItem) -> tuple[str, float, bool]:
        return layer.color.name().upper(), layer.width, layer.dash is not None

    assert spec(chart.price) == ("#F8FAFC", 1.6, False)
    assert spec(chart.p_liq) == ("#DC2626", 1.5, True)
    assert spec(chart.p_floor) == ("#F97316", 2.2, False)
    assert chart.p_floor.kind == "step"
    assert spec(chart.equity) == ("#38BDF8", 2.4, False)
    assert spec(chart.n_ceil) == ("#FB7185", 1.5, True)
    assert spec(chart.n_rebuy) == ("#34D399", 1.5, True)
    assert spec(chart.n_held) == ("#FACC15", 2.5, False)
    assert chart.n_held.kind == "step"
    assert spec(chart.s_down)[:2] == ("#F43F5E", 1.5)
    assert spec(chart.s_up)[:2] == ("#10B981", 1.5)
    assert chart.fill_stems.color.name().upper() == "#A855F7"
    # Eight-digit colours are RRGGBBAA: the band keeps its hue and is translucent.
    assert chart.price_band.color.name().upper() == "#334155"
    assert chart.price_band.color.alpha() == 0x40
    assert chart.mc_fan.color.name().upper() == "#38BDF8"
    assert chart.mc_fan.color.alpha() == 0x25
    assert chart.deadband.color.alpha() == 0x28
    symbols = {code: item.opts["symbol"] for code, item in chart.markers.items()}
    assert symbols == {5: "o", 1: "t", 3: "t1", 4: "d", 2: "d"}
    sizes = {code: item.base_size for code, item in chart.markers.items()}
    assert sizes == {5: 13, 1: 11, 3: 11, 4: 12, 2: 12}
    assert theme.CANVAS == "#0B0F19"
    assert theme.PANEL == "#111827"
    assert theme.GRID == "#1E293B"


def test_markers_are_lossless(chart: ChartWorkspace, run: RunOutput) -> None:
    flags, stats = run.result.flags, run.result.stats
    counts = {code: len(item.getData()[0]) for code, item in chart.markers.items()}
    assert counts[1] == stats.maker_sells
    assert counts[2] == stats.taker_sells
    assert counts[3] == stats.maker_buys
    assert counts[4] == stats.taker_buys
    assert counts[5] == int(np.count_nonzero(flags & FLAG_MISSED))
    assert chart.taker_sell_stems._x.shape[0] == stats.taker_sells
    attempts = int(np.count_nonzero(np.isfinite(run.result.telemetry["p_fill_prob"])))
    assert chart.fill_stems._x.shape[0] == attempts
    assert len(chart.draw_hit.getData()[0]) + len(chart.draw_miss.getData()[0]) == attempts
    assert len(chart.draw_hit.getData()[0]) == stats.maker_fills
    liquidations = int(np.count_nonzero(flags & event_bit(7)))
    assert len(chart.liquidation_lines) == 3  # a full-height flash line in every pane
    for layer in chart.liquidation_lines:
        assert layer._x.shape[0] == liquidations
        assert layer.color.name().upper() == "#EF4444"
    assert len(chart._liquidation_badges) == liquidations
    assert not chart.banner.isVisible()


def test_crosshair_and_readout(chart: ChartWorkspace, run: RunOutput) -> None:
    emitted: list[int] = []
    chart.barHovered.connect(emitted.append)
    chart.set_hover_bar(1_000)
    assert emitted == [1_000]
    assert all(line.isVisible() for line in chart.hover_lines)
    assert all(line.value() == pytest.approx(chart._x[1_000]) for line in chart.hover_lines)
    row = run.result.telemetry[1_000]
    text = chart.headers[0].readout.text()
    assert "2026-02-03 16:40 UTC" in text
    assert f"{row['close']:,.2f}" in text
    assert f"{row['btc_held']:.4f}" in chart.headers[1].readout.text()
    assert f"S_down {row['s_down']:.3f}" in chart.headers[2].readout.text()
    chart.set_hover_bar(-1)
    assert not chart.hover_lines[0].isVisible()
    assert chart.headers[0].readout.text() == ""

    chart.set_selected_bar(2_000)
    assert chart.selected_bar == 2_000
    assert all(line.isVisible() for line in chart.select_lines)
    assert "2026-02-04 09:20" in chart.headers[0].readout.text()
    chart.set_selected_bar(10**9)
    assert chart.selected_bar == -1


def test_bar_lookup_and_centre(chart: ChartWorkspace) -> None:
    x = chart._x
    assert chart.bar_at(x[0] - 500) == 0
    assert chart.bar_at(x[-1] + 500) == len(x) - 1
    assert chart.bar_at(x[100] + 20) == 100
    assert chart.bar_at(x[100] + 40) == 101
    chart.set_visible_range(float(x[0]), float(x[600]))
    chart.centre_on_bar(5_000)
    lo, hi = chart.visible_range()
    assert lo <= x[5_000] <= hi
    assert hi - lo == pytest.approx(x[600] - x[0])


def test_monte_carlo_fan_layer(chart: ChartWorkspace, run: RunOutput, bundled: MarketData) -> None:
    controller = SimulationController(bundled, run.params)
    controller.run()
    ensemble = controller.run_ensemble()
    chart.set_ensemble(ensemble)
    np.testing.assert_allclose(chart.mc_fan._y, ensemble.p05 / run.params.n0)
    np.testing.assert_allclose(chart.mc_fan._y2, ensemble.p95 / run.params.n0)
    chart.set_ensemble(None)
    assert chart.mc_fan._x.shape[0] == 0


def test_keep_view_preserves_zoom_on_parameter_change(
    chart: ChartWorkspace, bundled: MarketData, run: RunOutput
) -> None:
    x = chart._x
    chart.set_visible_range(float(x[500]), float(x[900]))
    other = SimulationController(bundled, replace(run.params, rho=0.9)).run()
    chart.set_run(other, keep_view=True)
    assert chart.visible_range() == pytest.approx((x[500], x[900]))
    chart.set_run(other, keep_view=False)
    assert chart.visible_range() == pytest.approx((x[0], x[-1]))


# -- GPU layer geometry -----------------------------------------------------------------


def test_minmax_decimation_keeps_every_extreme() -> None:
    rng = np.random.default_rng(1)
    x = np.arange(50_000, dtype=np.float64)
    y = np.cumsum(rng.normal(0, 1, x.shape[0]))
    dx, dy = minmax_decimate(x, y, 0, x.shape[0], 500)
    assert dx.shape[0] <= 1_000
    assert np.all(np.diff(dx) > 0)
    assert dy.max() == y.max()
    assert dy.min() == y.min()
    same_x, same_y = minmax_decimate(x, y, 100, 300, 500)  # short range: untouched
    np.testing.assert_array_equal(same_x, x[100:300])
    np.testing.assert_array_equal(same_y, y[100:300])


def test_band_and_mean_decimation() -> None:
    x = np.arange(10_000, dtype=np.float64)
    low, high = np.sin(x / 50.0) - 1.0, np.sin(x / 50.0) + 1.0
    bx, blo, bhi = band_decimate(x, low, high, 0, x.shape[0], 100)
    assert bx.shape[0] == 100
    assert blo.min() == low.min()
    assert bhi.max() == high.max()
    mx, my = mean_decimate(x, np.ones_like(x) * 3.0, 0, x.shape[0], 100)
    assert mx.shape[0] == 100
    np.testing.assert_allclose(my, 3.0)


def test_ribbon_and_segment_tessellation() -> None:
    px = np.array([0.0, 10.0, 10.0])
    py = np.array([0.0, 0.0, 10.0])
    strip = ribbon_strip(px, py, 2.0)
    assert strip.shape == (6, 3)
    np.testing.assert_allclose(strip[0], [0.0, 2.0, 2.0])  # start: offset along the normal
    np.testing.assert_allclose(strip[1], [0.0, -2.0, -2.0])
    np.testing.assert_allclose(strip[2, :2], [8.0, 2.0], atol=1e-6)  # mitred right angle
    np.testing.assert_allclose(strip[3, :2], [12.0, -2.0], atol=1e-6)
    assert ribbon_strip(px[:1], py[:1], 2.0).shape == (0, 3)
    # Repeated points (flat stretches of a step line) are dropped, not turned into spikes.
    doubled = ribbon_strip(np.repeat(px, 2), np.repeat(py, 2), 2.0)
    np.testing.assert_array_equal(doubled, strip)
    quads = segment_quads(np.array([0.0]), np.array([0.0]), np.array([0.0]), np.array([8.0]), 1.0)
    assert quads.shape == (6, 3)
    assert set(np.round(quads[:, 0], 6)) == {-1.0, 1.0}
    sx, sy = step_points(np.array([0.0, 1.0, 2.0]), np.array([5.0, 6.0, 7.0]))
    np.testing.assert_array_equal(sx, [0.0, 1.0, 1.0, 2.0, 2.0])
    np.testing.assert_array_equal(sy, [5.0, 5.0, 6.0, 6.0, 7.0])


def test_layer_bounds_follow_the_visible_range() -> None:
    layer = GLLayerItem("line", "#FFFFFF", 1.0)
    x = np.arange(100, dtype=np.float64)
    y = np.concatenate((np.full(50, 1.0), np.full(50, 9.0)))
    layer.set_polyline(x, y)
    assert layer.dataBounds(0) == (0.0, 99.0)
    assert layer.dataBounds(1) == (1.0, 9.0)
    assert layer.dataBounds(1, orthoRange=(10.0, 20.0)) == (1.0, 1.0)
    assert layer.dataBounds(1, orthoRange=(70.0, 80.0)) == (9.0, 9.0)
    layer.clear()
    assert layer.dataBounds(1) is None
