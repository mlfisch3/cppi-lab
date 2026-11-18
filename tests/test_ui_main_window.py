"""Main-window behaviour: parameter editing, reset, recalculation, docking (FR-2, FR-4, FR-5)."""

from __future__ import annotations

from collections.abc import Iterator
from functools import partial
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from PySide6 import QtCore, QtWidgets
from pytestqt.qtbot import QtBot

from cppi_lab.config.defaults import PARAM_SPECS, SPEC_BY_KEY, default_params
from cppi_lab.data.loader import MarketData
from cppi_lab.ui.chart_workspace import PANE_STRETCH
from cppi_lab.ui.control_sidebar import ParamRow
from cppi_lab.ui.kpi_ribbon import RIBBON_HEIGHT
from cppi_lab.ui.main_window import SCENARIOS, MainWindow
from tests.conftest import ms

pytestmark = pytest.mark.ui


@pytest.fixture
def window(qtbot: QtBot, bundled: MarketData) -> Iterator[MainWindow]:
    win = MainWindow(bundled, use_web=False, async_ensemble=False)
    qtbot.addWidget(win)
    win.resize(1920, 1040)
    win.show()
    qtbot.waitExposed(win)
    yield win
    win.close()


def edit(qtbot: QtBot, window: MainWindow, key: str, shown: float) -> None:
    """Type a value into a parameter's spin box and wait for the recalculation."""
    row = window.sidebar.rows[key]
    with qtbot.waitSignal(window.recalculated, timeout=10_000):
        row.spin.setValue(shown)  # type: ignore[union-attr]


# -- FR-2.1 / FR-2.2 -----------------------------------------------------------------


def test_launches_with_prd_defaults_in_every_control(window: MainWindow) -> None:
    assert window.controller.params == default_params()
    rows = window.sidebar.rows
    assert set(rows) == {s.key for s in PARAM_SPECS} - {"anchor_price"}
    for key, row in rows.items():
        assert row.value == getattr(default_params(), key), key
        assert not row.reset_button.isEnabled(), key
    assert rows["p_liq"].spin.text() == "$60,000.00"  # type: ignore[union-attr]
    assert rows["f_maker"].spin.text() == "0.15 %"  # type: ignore[union-attr]
    assert rows["dn_min_pct"].spin.text() == "3.00 %"  # type: ignore[union-attr]
    assert rows["w_slow_u"].spin.text() == "60 m"  # type: ignore[union-attr]
    assert rows["start_ms"].spin.text() == "2026-01-01 00:00"  # type: ignore[union-attr]
    assert rows["mc_fan"].toggle.isChecked() is False  # type: ignore[union-attr]
    assert window.sidebar.anchor_spin.value() == 68_000.0
    assert not window.sidebar.reset_all_button.isEnabled()


def test_every_numeric_row_has_slider_spin_reset_and_badge(window: MainWindow) -> None:
    for key, row in window.sidebar.rows.items():
        spec = SPEC_BY_KEY[key]
        assert row.badge.popover_id == spec.popover
        assert row.reset_button.toolTip() == "Reset to Default"
        if spec.kind == "bool":
            assert row.toggle is not None
        else:
            assert row.slider is not None, key
            assert row.spin is not None, key
            assert row.slider.maximum() == spec.steps or key in ("start_ms", "p_ref"), key
    assert window.sidebar.reset_all_button.text() == "Reset All Parameters to Defaults"


def test_spin_edit_recalculates_and_moves_the_slider(qtbot: QtBot, window: MainWindow) -> None:
    before = window.run
    edit(qtbot, window, "rho", 0.60)
    assert window.controller.params.rho == pytest.approx(0.60)
    assert window.run is not before
    row = window.sidebar.rows["rho"]
    assert row.slider.value() == 12  # type: ignore[union-attr]  # 0.60 / 0.05
    assert row.reset_button.isEnabled()
    assert row.property("modified") == "true"
    assert window.sidebar.reset_all_button.isEnabled()


def test_slider_edit_recalculates_and_moves_the_spin_box(qtbot: QtBot, window: MainWindow) -> None:
    row = window.sidebar.rows["k_sell"]
    with qtbot.waitSignal(window.recalculated, timeout=10_000):
        row.slider.setValue(10)  # type: ignore[union-attr]  # 0.10 + 10 * 0.05
    assert window.controller.params.k_sell == pytest.approx(0.60)
    assert row.spin.value() == pytest.approx(0.60)  # type: ignore[union-attr]
    percent = window.sidebar.rows["f_taker"]
    with qtbot.waitSignal(window.recalculated, timeout=10_000):
        percent.slider.setValue(45)  # type: ignore[union-attr]
    assert window.controller.params.f_taker == pytest.approx(0.0045)
    assert percent.spin.text() == "0.45 %"  # type: ignore[union-attr]
    integer = window.sidebar.rows["w_brk_u"]
    with qtbot.waitSignal(window.recalculated, timeout=10_000):
        integer.slider.setValue(2)  # type: ignore[union-attr]  # 30 + 2 * 15
    assert window.controller.params.w_brk_u == 60


def test_values_are_clamped_to_the_prd_range(qtbot: QtBot, window: MainWindow) -> None:
    edit(qtbot, window, "lambda_liq", 99.0)
    assert window.controller.params.lambda_liq == 10.0
    assert window.sidebar.rows["lambda_liq"].spin.value() == 10.0  # type: ignore[union-attr]


def test_p_ref_follows_p_liq(qtbot: QtBot, window: MainWindow) -> None:
    edit(qtbot, window, "p_liq", 71_750.0)
    assert window.controller.params.p_ref == 72_750.0
    assert window.sidebar.rows["p_ref"].spin.minimum() == 72_750.0  # type: ignore[union-attr]
    assert window.sidebar.rows["p_ref"].spin.value() == 72_750.0  # type: ignore[union-attr]


def test_reset_one_row(qtbot: QtBot, window: MainWindow) -> None:
    edit(qtbot, window, "rho", 0.9)
    edit(qtbot, window, "beta_vel", 4.0)
    row = window.sidebar.rows["rho"]
    with qtbot.waitSignal(window.recalculated, timeout=10_000):
        qtbot.mouseClick(row.reset_button, QtCore.Qt.MouseButton.LeftButton)
    assert window.controller.params.rho == 0.25
    assert window.controller.params.beta_vel == 4.0
    assert row.spin.value() == 0.25  # type: ignore[union-attr]
    assert not row.reset_button.isEnabled()


def test_reset_all(qtbot: QtBot, window: MainWindow) -> None:
    edit(qtbot, window, "rho", 0.9)
    edit(qtbot, window, "n0", 3.0)
    with qtbot.waitSignal(window.recalculated, timeout=10_000):
        window.sidebar.rows["normalize"].toggle.click()  # type: ignore[union-attr]
    with qtbot.waitSignal(window.recalculated, timeout=10_000):
        qtbot.mouseClick(window.sidebar.reset_all_button, QtCore.Qt.MouseButton.LeftButton)
    assert window.controller.params == default_params()
    assert all(not r.reset_button.isEnabled() for r in window.sidebar.rows.values())
    assert not window.norm_chip.isVisible()


def test_start_time_edit_and_reset(qtbot: QtBot, window: MainWindow) -> None:
    row = window.sidebar.rows["start_ms"]
    moment = QtCore.QDateTime.fromMSecsSinceEpoch(ms(2026, 2, 3, 12, 30), QtCore.QTimeZone.utc())
    with qtbot.waitSignal(window.recalculated, timeout=10_000):
        row.spin.setDateTime(moment)  # type: ignore[union-attr]
    assert window.controller.params.start_ms == ms(2026, 2, 3, 12, 30)
    assert window.run is not None
    assert window.run.window.ts_ms[0] == ms(2026, 2, 3, 12, 30)
    with qtbot.waitSignal(window.recalculated, timeout=10_000):
        row.reset_button.click()
    assert window.controller.params.start_ms == ms(2026, 1, 1)


# -- FR-2.3 ----------------------------------------------------------------------------


def test_recalculation_refreshes_kpis_chart_and_ledger(qtbot: QtBot, window: MainWindow) -> None:
    assert window.kpis.cards["fills"].value.text() == "0 / 0"
    assert window.chart.banner.isVisible()
    with qtbot.waitSignal(window.recalculated, timeout=10_000):
        window.sidebar.rows["normalize"].toggle.click()  # type: ignore[union-attr]
    run = window.run
    assert run is not None
    stats = run.result.stats
    assert stats.maker_attempts > 0
    assert window.kpis.cards["fills"].value.text() == f"{stats.maker_fills} / {stats.taker_fills}"
    assert window.kpis.cards["missed"].value.text() == str(stats.missed)
    assert not window.chart.banner.isVisible()
    assert window.ledger.model.rowCount() == run.result.n_bars
    assert window.norm_chip.isVisible()
    assert window.norm_chip.text().startswith("RESCALED ×0.77")
    assert window.sidebar.anchor_spin.isEnabled()
    assert "43,200 bars" in window.status_chip.text()
    np.testing.assert_array_equal(window.chart.price._y, run.result.telemetry["close"])
    assert run.timings.total_ms < 50.0


def test_burst_of_edits_is_coalesced_into_one_run(qtbot: QtBot, window: MainWindow) -> None:
    generation = window.controller.generation
    with qtbot.waitSignal(window.recalculated, timeout=10_000):
        for key, value in (("rho", 0.5), ("k_sell", 1.5), ("beta_vel", 2.5)):
            window.sidebar.rows[key].spin.setValue(value)  # type: ignore[union-attr]
    qtbot.wait(50)
    assert window.controller.generation == generation + 1
    assert window.controller.params.beta_vel == 2.5


def test_anchor_price_edit(qtbot: QtBot, window: MainWindow) -> None:
    with qtbot.waitSignal(window.recalculated, timeout=10_000):
        window.sidebar.rows["normalize"].toggle.click()  # type: ignore[union-attr]
    with qtbot.waitSignal(window.recalculated, timeout=10_000):
        window.sidebar.anchor_spin.setValue(64_000.0)
    assert window.run is not None
    assert window.run.window.open[0] == pytest.approx(64_000.0)


def test_monte_carlo_fan_toggle(qtbot: QtBot, window: MainWindow) -> None:
    window.apply_scenario(SCENARIOS[5])  # Jan 2026 rescaled: many passive orders
    with qtbot.waitSignal(window.ensembleApplied, timeout=30_000):
        window.sidebar.rows["mc_fan"].toggle.click()  # type: ignore[union-attr]
    assert window.ensemble is not None
    assert window.ensemble.members == 100
    assert window.chart.mc_fan._x.shape[0] == 43_200
    assert window.kpis.cards["floor"].subtitle.text().startswith("MC 0 breaches")
    with qtbot.waitSignal(window.recalculated, timeout=10_000):
        window.sidebar.rows["mc_fan"].toggle.click()  # type: ignore[union-attr]
    assert window.chart.mc_fan._x.shape[0] == 0
    assert "MC" not in window.kpis.cards["floor"].subtitle.text()


def test_monte_carlo_fan_runs_on_a_worker_thread(qtbot: QtBot, bundled: MarketData) -> None:
    win = MainWindow(bundled, use_web=False, async_ensemble=True)
    qtbot.addWidget(win)
    win.show()
    win.apply_scenario(SCENARIOS[5])
    with qtbot.waitSignal(win.ensembleApplied, timeout=60_000):
        win.sidebar.rows["mc_fan"].toggle.click()  # type: ignore[union-attr]
    assert win.ensemble is not None
    assert win.ensemble.floor_breaches == 0
    win.close()


# -- FR-4.1 ----------------------------------------------------------------------------


def test_everything_fits_one_1080p_screen_without_scrolling(
    qtbot: QtBot, window: MainWindow
) -> None:
    qtbot.wait(50)
    sidebar = window.sidebar
    assert not sidebar.scroll_area.verticalScrollBar().isVisible()
    assert sidebar.required_height() <= sidebar.height()
    assert sidebar.body.height() >= sidebar.body.sizeHint().height()
    for row in sidebar.rows.values():
        assert row.isVisible()
        bottom = row.mapTo(window, QtCore.QPoint(0, row.height())).y()
        assert bottom <= window.height()
    assert window.kpis.height() == RIBBON_HEIGHT == 64
    assert all(card.isVisible() for card in window.kpis.cards.values())
    heights = [plot.height() for plot in window.chart.plots]
    assert all(h > 100 for h in heights)
    total = sum(heights)
    for height, stretch in zip(heights, PANE_STRETCH, strict=True):
        assert height / total == pytest.approx(stretch / 100, abs=0.04)
    assert not window.findChildren(QtWidgets.QTabWidget)
    canvas = window.chart.canvas
    assert canvas.horizontalScrollBarPolicy() == QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
    assert canvas.verticalScrollBarPolicy() == QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff


def test_small_screens_scroll_only_the_sidebar(qtbot: QtBot, window: MainWindow) -> None:
    window.resize(1366, 728)
    qtbot.wait(80)
    assert window.sidebar.scroll_area.verticalScrollBar().isVisible()
    assert window.sidebar.reset_all_button.isVisible()
    assert all(plot.height() > 60 for plot in window.chart.plots)
    assert window.kpis.height() == 64


def test_every_explained_item_has_a_question_mark_badge(window: MainWindow) -> None:
    badges = window.findChildren(QtWidgets.QLabel, "helpBadge")
    ids = {badge.popover_id for badge in badges}  # type: ignore[attr-defined]
    assert len(badges) == 28 + 8 + 3  # sidebar rows, KPI cards, pane headers
    assert ids == set(window.registry.popovers)
    assert all(isinstance(b.parent(), (ParamRow, QtWidgets.QWidget)) for b in badges)


# -- datasets and scenarios ----------------------------------------------------------------


def test_scenarios_jump_to_curated_windows(qtbot: QtBot, window: MainWindow) -> None:
    assert window.scenario_combo.count() == len(SCENARIOS) + 1
    with qtbot.waitSignal(window.recalculated, timeout=10_000):
        window.scenario_combo.setCurrentIndex(2)
        window.scenario_combo.activated.emit(2)
    scenario = SCENARIOS[1]
    assert window.controller.params.start_ms == scenario.start_ms
    assert window.controller.params.duration_days == scenario.days
    assert window.run is not None
    assert window.run.result.stats.liquidations >= 1
    assert window.scenario_combo.currentIndex() == 0


def test_empty_state_banner_offers_one_click_remedies(qtbot: QtBot, window: MainWindow) -> None:
    banner = window.chart.banner
    assert banner.isVisible()
    assert "P_ref" in banner.message.text()
    with qtbot.waitSignal(window.recalculated, timeout=10_000):
        banner.normalize_button.click()
    assert window.controller.params.normalize
    assert not banner.isVisible()
    window.reset_all()
    window.recalculate()
    with qtbot.waitSignal(window.recalculated, timeout=10_000):
        window.chart.banner.scenario_button.click()
    assert window.controller.params.start_ms == SCENARIOS[1].start_ms


def test_import_file_becomes_the_active_dataset(
    qtbot: QtBot, window: MainWindow, bundled: MarketData, tmp_path: Path
) -> None:
    lo, hi = bundled.index_at(ms(2026, 2, 4)), bundled.index_at(ms(2026, 2, 7))
    frame = pl.DataFrame(
        {
            "open_time": bundled.ts_ms[lo:hi],
            "open": bundled.open[lo:hi],
            "high": bundled.high[lo:hi],
            "low": bundled.low[lo:hi],
            "close": bundled.close[lo:hi],
            "volume": bundled.buy_vol[lo:hi] + bundled.sell_vol[lo:hi],
            "close_time": bundled.ts_ms[lo:hi] + 59_999,
            "quote_asset_volume": 0.0,
            "number_of_trades": 1,
            "taker_buy_base_asset_volume": bundled.buy_vol[lo:hi],
            "taker_buy_quote_asset_volume": 0.0,
            "ignore": 0,
        }
    )
    path = tmp_path / "BTCUSDT-1m-sample.csv"
    frame.write_csv(path)
    assert window.import_file(path)
    assert window.dataset_combo.count() == 2
    assert window.dataset_combo.currentText() == "BTCUSDT-1m-sample.csv"
    assert window.controller.data.n == hi - lo
    assert window.run is not None
    # The default start (2026-01-01) precedes the file, so it is clamped to its first bar.
    assert window.controller.params.start_ms == ms(2026, 2, 4)
    assert window.run.window.n == hi - lo
    row = window.sidebar.rows["start_ms"]
    assert row.spin.minimumDateTime().toMSecsSinceEpoch() == ms(2026, 2, 4)  # type: ignore[union-attr]
    assert row.spin.maximumDateTime().toMSecsSinceEpoch() == ms(2026, 2, 6, 23, 59)  # type: ignore[union-attr]
    assert not window.scenario_combo.isEnabled()
    np.testing.assert_allclose(window.run.window.sell_vol, bundled.sell_vol[lo:hi], atol=1e-8)
    # Switching back restores the bundled range.
    window.dataset_combo.setCurrentIndex(0)
    window.dataset_combo.activated.emit(0)
    assert window.controller.data is bundled
    assert window.scenario_combo.isEnabled()


def test_bad_import_is_reported_not_raised(
    window: MainWindow, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shown: list[str] = []

    def fake_warning(*args: object) -> int:
        shown.append(str(args[2]))
        return 0

    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", fake_warning)
    bad = tmp_path / "bad.csv"
    bad.write_text("a,b\n1,2\n")
    assert not window.import_file(bad)
    assert shown
    assert "unrecognised file layout" in shown[0]
    assert window.dataset_combo.count() == 1


# -- FR-5.1 docking ------------------------------------------------------------------------


def test_ledger_opens_floating_on_top_and_docks(qtbot: QtBot, window: MainWindow) -> None:
    import PySide6QtAds as ads

    dock = window.ledger_dock
    assert dock.isClosed()
    window.ledger_button.click()
    qtbot.waitUntil(lambda: not dock.isClosed(), timeout=3_000)
    assert dock.isFloating()
    container = dock.floatingDockContainer()
    flags = container.windowFlags()
    for flag in (
        QtCore.Qt.WindowType.Tool,
        QtCore.Qt.WindowType.WindowStaysOnTopHint,
        QtCore.Qt.WindowType.WindowMinimizeButtonHint,
        QtCore.Qt.WindowType.WindowMaximizeButtonHint,
    ):
        assert flags & flag == flag
    assert window.ledger_button.isChecked()
    assert window.ledger.table.horizontalScrollBarPolicy() != (
        QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
    )
    container.resize(700, 300)  # freely resizable
    assert container.width() == 700

    origin = QtCore.QPoint(0, 0)

    def placed(area: object) -> bool:
        panel = window.ledger.mapTo(window, origin)
        bench = window.workbench.mapTo(window, origin)
        if area == ads.DockWidgetArea.LeftDockWidgetArea:
            return panel.x() + window.ledger.width() <= bench.x()
        if area == ads.DockWidgetArea.RightDockWidgetArea:
            return panel.x() >= bench.x() + window.workbench.width()
        return panel.y() >= bench.y() + window.workbench.height()

    for area in (
        ads.DockWidgetArea.LeftDockWidgetArea,
        ads.DockWidgetArea.RightDockWidgetArea,
        ads.DockWidgetArea.BottomDockWidgetArea,
    ):
        window.dock_to(dock, area)
        qtbot.waitUntil(partial(placed, area), timeout=5_000)
        assert not dock.isFloating()
        assert not dock.isClosed()
        assert window.ledger.isVisible()
        assert all(plot.height() > 40 for plot in window.chart.plots)  # panes stay visible

    window.float_dock(dock)
    assert dock.isFloating()
    window.set_dock_visible(dock, False)
    assert dock.isClosed()
    assert not window.ledger_button.isChecked()
    window.set_dock_visible(dock, True)
    assert not dock.isClosed()


def test_docking_menu_lists_every_position(window: MainWindow) -> None:
    labels = [a.text() for a in window.ledger_button.menu().actions() if a.text()]
    assert labels == ["Float (always on top)", "Dock Left", "Dock Right", "Dock Bottom", "Hide"]
    labels = [a.text() for a in window.docs_button.menu().actions() if a.text()]
    assert "Dock Bottom" in labels


# -- FR-5.2 crosshair synchronisation ---------------------------------------------------------


def test_ledger_row_drives_the_chart_and_back(qtbot: QtBot, window: MainWindow) -> None:
    window.apply_scenario(SCENARIOS[1])
    window.set_dock_visible(window.ledger_dock, True)
    ledger, chart = window.ledger, window.chart
    index = ledger.model.index(1_234, 3)
    ledger.table.setCurrentIndex(index)
    assert chart.selected_bar == 1_234
    assert chart.select_lines[0].isVisible()
    assert chart.select_lines[2].value() == pytest.approx(chart._x[1_234])
    assert "2026-01-25 20:34" in chart.headers[0].readout.text()

    chart.barSelected.emit(4_321)
    assert ledger.current_bar() == 4_321

    ledger.set_filter("Floor Liquidations")
    bar = int(np.nonzero(window.run.result.flags & 64)[0][0])  # type: ignore[union-attr]
    ledger.table.setCurrentIndex(ledger.model.index(0, 0))
    assert chart.selected_bar == bar
    x0, x1 = chart.visible_range()
    assert x0 <= chart._x[bar] <= x1
