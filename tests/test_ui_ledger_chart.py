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
from cppi_lab.ui.chart_geometry import (
    build_pyramid,
    decimate_band,
    decimate_extreme,
    decimate_mean,
    range_min_max,
    ribbon_strip,
    segment_quads,
    step_points,
    time_tick_label,
    time_ticks,
    value_tick_label,
    value_ticks,
)
from cppi_lab.ui.chart_gl import Frame, Layer, RibbonOp, SpriteAtlas, SpriteOp
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
    def spec(layer: Layer) -> tuple[str, float, bool]:
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
    symbols = {code: markers.symbol for code, markers in chart.markers.items()}
    assert symbols == {5: "o", 1: "t", 3: "t1", 4: "d", 2: "d"}
    sizes = {code: markers.base_size for code, markers in chart.markers.items()}
    assert sizes == {5: 13, 1: 11, 3: 11, 4: 12, 2: 12}
    assert theme.CANVAS == "#0B0F19"
    assert theme.PANEL == "#111827"
    assert theme.GRID == "#1E293B"


def test_markers_are_lossless(chart: ChartWorkspace, run: RunOutput) -> None:
    flags, stats = run.result.flags, run.result.stats
    counts = {code: markers.x.shape[0] for code, markers in chart.markers.items()}
    assert counts[1] == stats.maker_sells
    assert counts[2] == stats.taker_sells
    assert counts[3] == stats.maker_buys
    assert counts[4] == stats.taker_buys
    assert counts[5] == int(np.count_nonzero(flags & FLAG_MISSED))
    assert chart.taker_sell_stems.size == stats.taker_sells
    attempts = int(np.count_nonzero(np.isfinite(run.result.telemetry["p_fill_prob"])))
    assert chart.fill_stems.size == attempts
    assert chart.draw_hit.x.shape[0] + chart.draw_miss.x.shape[0] == attempts
    assert chart.draw_hit.x.shape[0] == stats.maker_fills
    liquidations = int(np.count_nonzero(flags & event_bit(7)))
    assert liquidations > 0
    assert chart.liquidation_x.shape[0] == liquidations
    assert not chart.banner.isVisible()


def test_the_frame_draws_every_layer_marker_and_flash(chart: ChartWorkspace) -> None:
    scene, _ = chart.frames()
    tags = [op.tag for op in scene.ops]
    for name in (
        "price_band", "floor_zone", "p_liq", "p_floor", "price", "equity_glow", "equity",
        "taker_sell_stems", "deadband", "n_max", "n_rebuy", "n_ceil", "n_held",
        "s_down_fill", "s_up_fill", "s_down", "s_up", "fill_stems",
    ):  # fmt: skip
        assert tags.count(f"layer:{name}") == 1, name
    for name in ("missed", "maker_sell", "maker_buy", "taker_buy", "taker_sell"):
        assert tags.count(f"marker:{name}") == 1, name
    assert "marker:draw_hit" in tags
    assert "marker:draw_miss" in tags
    # Back to front inside pane 1: bands, levels, price, equity, then the markers.
    order = [tags.index(f"layer:{name}") for name in ("price_band", "p_liq", "price", "equity")]
    assert order == sorted(order)
    assert tags.index("layer:equity") < tags.index("marker:taker_sell")
    # A flash line through each of the three panes, clipped to that pane.
    flashes = scene.find("liquidation")
    assert len(flashes) == 3
    assert sorted(op.clip for op in flashes) == [0, 1, 2]
    for op in flashes:
        assert isinstance(op, RibbonOp)
        assert op.count == 6 * chart.liquidation_x.shape[0]
        assert op.width == 1.6
    # Dashed layers reach the shader as dashed, filled bands as fills.
    (p_liq,) = scene.find("layer:p_liq")
    (band,) = scene.find("layer:price_band")
    assert isinstance(p_liq, RibbonOp)
    assert isinstance(band, RibbonOp)
    assert p_liq.dash == (6.0, 4.0)
    assert not p_liq.fill
    assert band.fill
    assert band.strip
    # Grid, axes and tick labels are part of the same frame.
    assert len(scene.find("grid-v")) == 3
    assert scene.find("grid-h")
    assert len(scene.find("axis")) == 4
    labels = [text for op in scene.find("tick-y") for _, _, text, _ in op.texts]  # type: ignore[union-attr]
    assert "60000" in labels
    dates = [text for op in scene.find("tick-x") for _, _, text, _ in op.texts]  # type: ignore[union-attr]
    assert "Feb 5" in dates


def test_every_drawn_vertex_is_finite_and_inside_the_buffers(chart: ChartWorkspace) -> None:
    chart.set_hover_bar(1_000)
    chart.set_selected_bar(2_000)
    for frame in chart.frames():
        assert np.all(np.isfinite(frame.ribbons[: frame.n_ribbons]))
        assert np.all(np.isfinite(frame.sprites[: frame.n_sprites]))
        for op in frame.ops:
            limit = frame.n_ribbons if isinstance(op, RibbonOp) else frame.n_sprites
            assert op.first >= 0
            assert op.first + op.count <= limit
            assert op.count % (1 if isinstance(op, RibbonOp) and op.strip else 3) == 0
            assert -1 <= op.clip < len(frame.clips)


def test_scene_is_rebuilt_only_when_the_view_or_the_data_changes(
    chart: ChartWorkspace, run: RunOutput
) -> None:
    scene, overlay = chart.frames()
    scene_serial, overlay_serial = scene.serial, overlay.serial
    chart.frames()
    assert (scene.serial, overlay.serial) == (scene_serial, overlay_serial)  # nothing changed
    chart.set_hover_bar(1_500)  # the crosshair moves: only the overlay is rebuilt
    chart.frames()
    assert scene.serial == scene_serial
    assert overlay.serial == overlay_serial + 1
    chart.set_hover_bar(-1)
    x0, _ = chart.visible_range()
    chart.set_visible_range(x0 + 3_600.0, x0 + 86_400.0)  # panning rebuilds the scene
    chart.frames()
    assert scene.serial == scene_serial + 1
    chart.set_run(run, keep_view=True)  # so does new data
    chart.frames()
    assert scene.serial == scene_serial + 2


def test_value_axes_follow_the_samples_in_view(chart: ChartWorkspace, run: RunOutput) -> None:
    tel = run.result.telemetry
    x = chart._x
    for lo, hi in ((0, len(x) - 1), (2_000, 2_600), (5_000, 5_090)):
        chart.set_visible_range(float(x[lo]), float(x[hi]))
        pane = chart.pane(0)
        rows = slice(max(0, lo - 1), hi + 2)  # one bar beyond each edge, as drawn
        low = min(tel["low"][rows].min(), run.params.p_liq, tel["p_floor"][rows].min())
        low = min(low, (tel["net_equity_usd"][rows] / run.params.n0).min())
        high = max(tel["high"][rows].max(), (tel["net_equity_usd"][rows] / run.params.n0).max())
        span = high - low
        assert pane.y0 == pytest.approx(low - 0.04 * span)
        assert pane.y1 == pytest.approx(high + 0.04 * span)
        held = chart.pane(1)
        assert held.y0 <= tel["btc_held"][rows].min()
        assert held.y1 >= tel["n_max"][rows].max()
    assert (chart.pane(2).y0, chart.pane(2).y1) == (-0.03, 1.05)  # signals live in [0, 1]


def test_liquidation_badges_cluster_and_never_overlap(qtbot: QtBot, bundled: MarketData) -> None:
    widget = ChartWorkspace()
    qtbot.addWidget(widget)
    widget.resize(900, 560)
    widget.show()
    run = SimulationController(
        bundled, replace(default_params(), start_ms=ms(2026, 6, 1), duration_days=30)
    ).run()
    widget.set_run(run)
    qtbot.wait(30)
    liquidations = int(np.count_nonzero(run.result.flags & event_bit(7)))
    assert liquidations >= 5
    scene, _ = widget.frames()
    badges = widget.badges
    assert 1 <= len(badges) < liquidations  # close flashes share one label
    assert sum(badge.count for badge in badges) == liquidations  # none is lost
    assert any(badge.text.startswith("LIQUIDATION ×") for badge in badges)
    pane = widget.pane(0)
    for badge in badges:
        assert pane.left <= badge.x
        assert badge.x + badge.width <= pane.right + 0.5  # whole inside the pane
        assert badge.text in [
            text
            for op in scene.find("badge-text")
            for _, _, text, _ in op.texts  # type: ignore[union-attr]
        ]
    for i, first in enumerate(badges):
        for second in badges[i + 1 :]:
            apart = first.x + first.width <= second.x or second.x + second.width <= first.x
            stacked = first.y + first.height <= second.y or second.y + second.height <= first.y
            assert apart or stacked
    # The opaque labels are drawn after every flash line, so no line strikes through one.
    tags = [op.tag for op in scene.ops if op.clip == 0]
    assert tags.index("badge") > tags.index("liquidation")
    # Zoomed in on one liquidation the label is the plain one again.
    bar = int(np.nonzero(run.result.flags & event_bit(7))[0][0])
    widget.set_visible_range(float(widget._x[bar]) - 1_800.0, float(widget._x[bar]) + 1_800.0)
    widget.frames()
    assert [badge.text for badge in widget.badges][:1] == ["LIQUIDATION"]


def test_crosshair_and_readout(chart: ChartWorkspace, run: RunOutput) -> None:
    emitted: list[int] = []
    chart.barHovered.connect(emitted.append)
    chart.set_hover_bar(1_000)
    assert emitted == [1_000]
    assert chart.hover_time == pytest.approx(chart._x[1_000])
    _, overlay = chart.frames()
    hover = overlay.find("crosshair-hover")
    assert len(hover) == 3  # one line per pane
    pane = chart.pane(0)
    m11, dx, _, _ = pane.transform
    for op in hover:
        assert isinstance(op, RibbonOp)
        assert op.dash is not None
        rows = overlay.ribbons[op.first : op.first + op.count]
        assert rows[:, 0].mean() == pytest.approx(chart._x[1_000] * m11 + dx, abs=1.0)
    row = run.result.telemetry[1_000]
    text = chart.readouts[0]
    assert "2026-02-03 16:40 UTC" in text
    assert f"{row['close']:,.2f}" in text
    assert f"{row['btc_held']:.4f}" in chart.readouts[1]
    assert f"S_down {row['s_down']:.3f}" in chart.readouts[2]
    drawn = [text for op in overlay.find("readout") for _, _, text, _ in op.texts]  # type: ignore[union-attr]
    assert drawn == chart.readouts
    chart.set_hover_bar(-1)
    assert chart.hover_time is None
    assert chart.readouts == ["", "", ""]
    _, overlay = chart.frames()
    assert not overlay.ops

    chart.set_selected_bar(2_000)
    assert chart.selected_bar == 2_000
    assert chart.selected_time == pytest.approx(chart._x[2_000])
    _, overlay = chart.frames()
    assert len(overlay.find("crosshair-select")) == 3
    assert "2026-02-04 09:20" in chart.readouts[0]
    chart.set_selected_bar(10**9)
    assert chart.selected_bar == -1
    assert chart.selected_time is None


def test_legend_gives_way_to_the_readout_on_a_narrow_pane(
    qtbot: QtBot, chart: ChartWorkspace
) -> None:
    header = chart.headers[0]
    assert header.legend.isVisible()
    chart.resize(1_000, 600)
    qtbot.wait(40)
    assert header.legend.isVisible()  # the legend alone still fits
    chart.set_hover_bar(1_000)
    assert not header.legend.isVisible()  # title, legend and read-out would collide
    chart.set_hover_bar(-1)
    assert header.legend.isVisible()


def test_mouse_moves_the_crosshair_and_a_click_selects_the_bar(
    qtbot: QtBot, chart: ChartWorkspace
) -> None:
    pane = chart.pane(0)
    spot = QtCore.QPoint(int(pane.left + pane.width * 0.5), int(pane.top + pane.height * 0.5))
    viewport = chart.canvas.viewport()
    qtbot.mouseMove(viewport, spot)
    qtbot.wait(20)
    expected = chart.bar_at(pane.x0 + 0.5 * (pane.x1 - pane.x0))
    assert abs(chart.hover_bar - expected) <= 12
    _, overlay = chart.frames()
    assert len(overlay.find("crosshair-level")) == 1  # horizontal line in the price pane
    with qtbot.waitSignal(chart.barSelected, timeout=2_000) as selected:
        qtbot.mouseClick(viewport, QtCore.Qt.MouseButton.LeftButton, pos=spot)
    assert abs(selected.args[0] - expected) <= 12
    assert chart.selected_bar == selected.args[0]


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


# -- level of detail, tessellation and ticks ---------------------------------------------


def test_pyramid_holds_block_extremes_and_sums() -> None:
    rng = np.random.default_rng(7)
    for n in (1, 2, 4, 5, 63, 64, 65, 1_000):
        y = rng.normal(0.0, 1.0, n)
        mins, maxs, sums, offsets = build_pyramid(y)
        assert offsets[0] == 0
        size = n
        for level in range(1, offsets.shape[0]):
            block = 4**level
            size = (size + 3) // 4
            assert offsets[level] - offsets[level - 1] == size
            for b in (0, size // 2, size - 1):
                chunk = y[b * block : (b + 1) * block]
                at = offsets[level - 1] + b
                assert mins[at] == chunk.min()
                assert maxs[at] == chunk.max()
                assert sums[at] == pytest.approx(chunk.sum())
        assert size <= 1


def test_range_min_max_is_exact_for_any_range() -> None:
    rng = np.random.default_rng(3)
    y = np.cumsum(rng.normal(0.0, 1.0, 20_000))
    y[1_234] = np.nan  # ignored, like every non-finite sample
    mins, maxs, _, offsets = build_pyramid(y)
    for _ in range(200):
        lo = int(rng.integers(0, y.shape[0] - 1))
        hi = int(rng.integers(lo + 1, y.shape[0] + 1))
        low, high = range_min_max(y, mins, maxs, offsets, lo, hi)
        if np.all(np.isnan(y[lo:hi])):
            assert np.isnan(low)
            continue
        assert low == np.nanmin(y[lo:hi])
        assert high == np.nanmax(y[lo:hi])
    low, high = range_min_max(y, mins, maxs, offsets, 1_234, 1_235)
    assert np.isnan(low)
    assert np.isnan(high)


@pytest.mark.parametrize("n", [3_000, 50_000, 600_000])
def test_extreme_decimation_keeps_every_extreme_at_every_zoom(n: int) -> None:
    rng = np.random.default_rng(1)
    x = np.arange(n, dtype=np.float64)
    y = np.cumsum(rng.normal(0, 1, n))
    y[n // 3] = y.max() + 50.0  # one-bar spikes must survive any reduction
    y[2 * n // 3] = y.min() - 50.0
    for lo, hi in ((0, n), (n // 10, n - n // 7), (n // 3 - 700, n // 3 + 900)):
        dx, dy = decimate_extreme(x, y, lo, hi, 500)
        assert dx.shape[0] <= 1_000
        assert np.all(np.diff(dx) >= 0)
        # The reduction may look one coarse block beyond each edge, never inside less.
        assert dy.max() >= y[lo:hi].max()
        assert dy.min() <= y[lo:hi].min()
        if (lo, hi) == (0, n):
            assert dy.max() == y.max()
            assert dy.min() == y.min()
    same_x, same_y = decimate_extreme(x, y, 100, 300, 500)  # short range: untouched
    np.testing.assert_array_equal(same_x, x[100:300])
    np.testing.assert_array_equal(same_y, y[100:300])


def test_band_and_mean_decimation() -> None:
    for n in (10_000, 300_000):
        x = np.arange(n, dtype=np.float64)
        low, high = np.sin(x / 50.0) - 1.0, np.sin(x / 50.0) + 1.0
        bx, blo, bhi = decimate_band(x, low, high, 0, n, 100)
        assert bx.shape[0] == 100
        assert blo.min() == low.min()
        assert bhi.max() == high.max()
        mx, my = decimate_mean(x, np.ones_like(x) * 3.0, 0, n, 100)
        assert mx.shape[0] == 100
        np.testing.assert_allclose(my, 3.0)
        # Bucket means of a ramp.  Buckets are aligned to whole pyramid blocks, so
        # each mean sits within a fraction of a bucket of the exact one.
        ramp_x, ramp_y = decimate_mean(x, x, 0, n, 100)
        exact = (np.arange(100) + 0.5) * n / 100 - 0.5
        assert np.abs(ramp_y - exact).max() <= 0.5 * n / 100
        assert np.all(np.diff(ramp_y) > 0)
        assert np.all(np.diff(ramp_x) > 0)
        np.testing.assert_allclose(ramp_x, ramp_y, atol=0.51 * n / 100)  # drawn where it lies


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
    # Points on the straight line between their neighbours add nothing and are dropped too.
    straight = ribbon_strip(
        np.array([0.0, 2.5, 5.0, 10.0, 10.0, 10.0]), np.array([0.0, 0.0, 0.0, 0.0, 4.0, 10.0]), 2.0
    )
    np.testing.assert_allclose(straight, strip, atol=1e-6)
    # A sharp reversal is capped at the mitre limit instead of shooting off to infinity.
    spike = ribbon_strip(np.array([0.0, 10.0, 0.0]), np.array([0.0, 0.0, 0.5]), 2.0)
    assert np.abs(spike[:, :2]).max() <= 10.0 + 2.0 * 2.0 + 1e-6
    quads = segment_quads(np.array([0.0]), np.array([0.0]), np.array([0.0]), np.array([8.0]), 1.0)
    assert quads.shape == (6, 3)
    assert set(np.round(quads[:, 0], 6)) == {-1.0, 1.0}
    sx, sy = step_points(np.array([0.0, 1.0, 2.0]), np.array([5.0, 6.0, 7.0]))
    np.testing.assert_array_equal(sx, [0.0, 1.0, 1.0, 2.0, 2.0])
    np.testing.assert_array_equal(sy, [5.0, 5.0, 6.0, 6.0, 7.0])


def test_layer_bounds_follow_the_visible_range() -> None:
    layer = Layer("line", "#FFFFFF", 1.0)
    x = np.arange(100, dtype=np.float64)
    y = np.concatenate((np.full(50, 1.0), np.full(50, 9.0)))
    layer.set_polyline(x, y)
    assert layer.y_bounds(0.0, 99.0) == (1.0, 9.0)
    assert layer.y_bounds(10.0, 20.0) == (1.0, 1.0)
    assert layer.y_bounds(70.0, 80.0) == (9.0, 9.0)
    band = Layer("band", "#FFFFFF")
    band.set_band(x, y - 2.0, y + 2.0)
    assert band.y_bounds(10.0, 20.0) == (-1.0, 3.0)
    stems = Layer("stems", "#FFFFFF", stem_px=8.0)
    stems.set_stems(x, y)
    assert stems.y_bounds(0.0, 99.0) is None  # decorations do not drive the axis
    layer.clear()
    assert layer.y_bounds(0.0, 99.0) is None
    assert layer.size == 0


def test_layer_tessellation_draws_steps_only_when_bars_are_apart() -> None:
    x = np.arange(4_000, dtype=np.float64)
    y = np.floor(x / 400.0)  # ten flat levels
    layer = Layer("step", "#FFFFFF", 2.0)
    layer.set_polyline(x, y)
    out = np.empty((layer.max_rows(800), 3), dtype=np.float32)
    # 20 bars across 800 pixels: a true staircase, with exactly vertical risers.
    transform = (800.0 / 20.0, -390.0 * 800.0 / 20.0, -10.0, 100.0)
    rows = layer.tessellate(out, 0, transform, 390.0, 410.0, 800)
    xs = np.round(0.5 * (out[0:rows:2, 0] + out[1:rows:2, 0]), 3)
    ys = np.round(0.5 * (out[0:rows:2, 1] + out[1:rows:2, 1]), 3)
    assert rows == 2 * 4  # flat, corner, corner, flat: collinear points are gone
    assert xs[1] == xs[2]  # the riser
    assert ys[0] == ys[1]
    assert ys[2] == ys[3]
    assert ys[1] != ys[2]
    # The whole series across 800 pixels: reduced, never more than two points a pixel.
    wide = layer.tessellate(out, 0, (0.2, 0.0, -10.0, 100.0), 0.0, 3_999.0, 800)
    assert 0 < wide <= 2 * (2 * 800 + 2)


def test_sprite_atlas_lays_out_text_and_markers(qtbot: QtBot) -> None:
    from PySide6 import QtGui

    font = QtGui.QFont()
    font.setPixelSize(10)
    for dpr in (1.0, 2.0):
        atlas = SpriteAtlas(dpr)
        font_id = atlas.add_font(font)
        version = atlas.version
        quads, width = atlas.run(font_id, "68450")
        assert atlas.version > version  # the glyphs were rasterised
        assert quads.shape == (6 * 5, 4)
        assert width == pytest.approx(
            QtGui.QFontMetricsF(atlas._fonts[font_id].device_font).horizontalAdvance("68450") / dpr,
            abs=1.0,
        )
        assert np.all((quads[:, 2:] >= 0.0) & (quads[:, 2:] <= 1.0))  # texture coordinates
        assert np.all(np.diff(quads[0::6, 0]) > 0)  # glyphs advance left to right
        version = atlas.version
        again, _ = atlas.run(font_id, "68450")
        assert again is quads  # cached
        assert atlas.version == version
        assert atlas.text_width(font_id, "") == 0.0
        # The glyphs really are in the image: white ink with some coverage.
        image = atlas.image
        assert image.pixelColor(*map(int, _ink(atlas, quads))).alpha() > 0
        ring = atlas.marker("o", 13.0, theme.MISSED, True)
        solid = atlas.marker("d", 12.0, theme.TAKER_SELL, False)
        assert atlas.marker("o", 13.0, theme.MISSED, True) is ring
        assert ring.width == pytest.approx(15.0, abs=1.0)  # 13 px symbol plus its 2 px rim
        x, y, w, h = solid.source
        assert image.pixelColor(x + w // 2, y + h // 2).name().upper() == theme.TAKER_SELL
        rx, ry, rw, rh = ring.source
        assert image.pixelColor(rx + rw // 2, ry + rh // 2).alpha() == 0  # hollow


def _ink(atlas: SpriteAtlas, quads: np.ndarray) -> tuple[int, int]:
    """Atlas pixel of the most covered texel inside the first glyph of a run."""
    u0, v0 = quads[0, 2] * atlas.SIZE, quads[0, 3] * atlas.SIZE
    u1, v1 = quads[5, 2] * atlas.SIZE, quads[5, 3] * atlas.SIZE
    best, spot = -1, (int(u0), int(v0))
    for py in range(int(v0), int(v1)):
        for px in range(int(u0), int(u1)):
            alpha = atlas.image.pixelColor(px, py).alpha()
            if alpha > best:
                best, spot = alpha, (px, py)
    return spot


def test_frame_merges_text_runs_and_grows_its_buffers(qtbot: QtBot) -> None:
    from PySide6 import QtGui

    font = QtGui.QFont()
    font.setPixelSize(10)
    atlas = SpriteAtlas(1.0)
    font_id = atlas.add_font(font)
    frame = Frame()
    frame.begin(400.0, 300.0, 1.0)
    clip = frame.clip(10.0, 10.0, 380.0, 280.0)
    white = (1.0, 1.0, 1.0, 1.0)
    width = frame.add_text("label", clip, atlas, font_id, "abc", 100.2, 50.6, white, align=1.0)
    frame.add_texts("label", clip, atlas, font_id, [("de", 10.0, 20.0), ("", 0.0, 0.0)], white)
    (op,) = frame.find("label")  # same tag, clip and colour: one draw call
    assert isinstance(op, SpriteOp)
    assert op.count == 6 * 5 == frame.n_sprites
    assert [text for _, _, text, _ in op.texts] == ["abc", "de"]
    pen_x, pen_y, _, _ = op.texts[0]
    assert pen_x == round(100.2 - width)  # right-aligned and snapped to a whole pixel
    assert pen_y == 51.0
    frame.add_rect("box", clip, 5.0, 6.0, 20.0, 10.0, white)
    (box,) = frame.find("box")
    assert isinstance(box, RibbonOp)
    rows = frame.ribbons[box.first : box.first + box.count]
    assert (float(rows[:, 0].min()), float(rows[:, 0].max())) == (5.0, 25.0)
    assert (float(rows[:, 1].min()), float(rows[:, 1].max())) == (6.0, 16.0)
    # Far more segments than the initial capacity: the buffer grows and keeps its content.
    count = frame.ribbons.shape[0]
    xs = np.arange(count, dtype=np.float64)
    frame.add_segments("many", clip, xs, xs * 0.0, xs, xs * 0.0 + 5.0, white)
    assert frame.ribbons.shape[0] >= 6 * count
    np.testing.assert_array_equal(frame.ribbons[box.first : box.first + box.count], rows)
    serial = frame.serial
    frame.begin(400.0, 300.0, 1.0)
    assert frame.serial == serial + 1
    assert not frame.ops
    assert frame.n_ribbons == frame.n_sprites == 0


def test_time_and_value_ticks() -> None:
    day = 86_400.0
    step, ticks = time_ticks(1_770_000_000.0, 1_770_000_000.0 + 30 * day, 1_300.0)
    assert step == 2 * day  # at least 64 px between labels
    assert all(tick % step == 0 for tick in ticks)
    assert time_ticks(5.0, 5.0, 100.0) == (0, [])
    assert time_tick_label(1_770_336_000.0, day) == "Feb 6"
    assert time_tick_label(1_770_336_000.0 + 5_400.0, 1_800.0) == "01:30"
    assert time_tick_label(1_770_336_000.0, 3_600.0) == "Feb 6"  # midnight shows the date
    assert time_tick_label(1_770_336_000.0, 61 * day) == "Feb 2026"

    levels = value_ticks(58_000.0, 92_000.0, 500.0)
    spacings = [spacing for spacing, _, _ in levels]
    assert spacings == [10_000.0, 5_000.0, 1_000.0]
    np.testing.assert_allclose(levels[0][1], [60_000.0, 70_000.0, 80_000.0, 90_000.0])
    np.testing.assert_allclose(levels[1][1], [65_000.0, 75_000.0, 85_000.0])  # majors excluded
    assert levels[0][2] == 1.0  # major grid lines are opaque
    assert levels[0][2] > levels[1][2] > levels[2][2] > 0.0  # finer levels are fainter
    assert value_ticks(1.0, 1.0, 500.0) == []
    assert value_ticks(0.0, 1.0, 0.0) == []
    unit = value_ticks(-0.03, 1.05, 130.0)
    np.testing.assert_allclose(unit[0][1], [0.0, 0.2, 0.4, 0.6, 0.8, 1.0], atol=1e-12)
    assert value_tick_label(90_000.0, 5_000.0) == "90000"
    assert value_tick_label(0.5, 0.5) == "0.5"
    assert value_tick_label(1.0, 0.2) == "1.0"
    assert value_tick_label(0.0, 0.2) == "0"
    assert value_tick_label(1e-17, 0.2) == "0"  # rounding noise at the origin
    assert value_tick_label(0.25, 0.05) == "0.25"
