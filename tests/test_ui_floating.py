"""Floating windows and platform selection (PRD FR-5.1, TSD section 4.2).

The real-mouse checks (title-bar move, edge resize, drag-and-drop docking)
live in ``scripts/drag_dock_tour.py``, which needs an X11 display and
``xdotool``.  The tests here cover the pieces that tour relies on, headless.
"""

from __future__ import annotations

import ctypes
import os
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest
from PySide6 import QtCore, QtGui, QtWidgets
from pytestqt.qtbot import QtBot

from cppi_lab import main as entry
from cppi_lab.data.loader import MarketData
from cppi_lab.ui import x11
from cppi_lab.ui.chart_workspace import ChartWorkspace
from cppi_lab.ui.ledger_window import EXPORT_CAPTION, EXPORT_CAPTION_SHORT, LedgerWidget
from cppi_lab.ui.main_window import MainWindow, floating_window_flags
from cppi_lab.ui.widgets import ResizeGrips, TitleBarPressFix

pytestmark = pytest.mark.ui

Flag = QtCore.Qt.WindowType
LEFT = QtCore.Qt.MouseButton.LeftButton
NO_MODIFIER = QtCore.Qt.KeyboardModifier.NoModifier


# -- choosing the platform ------------------------------------------------------------------


def test_application_attributes_are_set() -> None:
    attribute = QtCore.Qt.ApplicationAttribute
    assert QtCore.QCoreApplication.testAttribute(attribute.AA_ShareOpenGLContexts)
    # Without this the docking system turns sibling widgets into native windows.
    assert QtCore.QCoreApplication.testAttribute(attribute.AA_DontCreateNativeWidgetSiblings)


def _fake_loader(monkeypatch: pytest.MonkeyPatch, loadable: set[str]) -> list[tuple[str, int]]:
    """Replace ``ctypes.CDLL`` by one that only "loads" the given names."""
    attempts: list[tuple[str, int]] = []

    def cdll(name: str, mode: int = 0) -> object:
        attempts.append((name, mode))
        if name not in loadable:
            raise OSError(f"{name}: cannot open shared object file")
        return object()

    monkeypatch.setattr(ctypes, "CDLL", cdll)
    return attempts


def test_xcb_is_preferred_only_when_its_cursor_library_loads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setenv("DISPLAY", ":0")
    monkeypatch.delenv("CPPI_LAB_XCB_CURSOR", raising=False)

    monkeypatch.delenv("QT_QPA_PLATFORM", raising=False)
    _fake_loader(monkeypatch, set())
    entry._prefer_xcb_on_linux()
    assert "QT_QPA_PLATFORM" not in os.environ  # nothing loads: Qt keeps its own choice

    _fake_loader(monkeypatch, {"libxcb-cursor.so.0"})
    entry._prefer_xcb_on_linux()
    assert os.environ["QT_QPA_PLATFORM"] == "xcb"

    monkeypatch.setenv("QT_QPA_PLATFORM", "wayland")  # an explicit choice always wins
    entry._prefer_xcb_on_linux()
    assert os.environ["QT_QPA_PLATFORM"] == "wayland"

    monkeypatch.delenv("QT_QPA_PLATFORM", raising=False)
    monkeypatch.delenv("DISPLAY", raising=False)  # no X server to talk to
    entry._prefer_xcb_on_linux()
    assert "QT_QPA_PLATFORM" not in os.environ

    monkeypatch.setenv("DISPLAY", ":0")
    monkeypatch.setattr(sys, "platform", "darwin")  # only Linux is touched
    entry._prefer_xcb_on_linux()
    assert "QT_QPA_PLATFORM" not in os.environ


def test_a_user_level_cursor_library_is_found_without_ld_library_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("CPPI_LAB_XCB_CURSOR", raising=False)
    home_copy = str(Path(entry.USER_XCB_CURSOR_PATHS[0]).expanduser())
    attempts = _fake_loader(monkeypatch, {home_copy})
    assert entry._load_xcb_cursor()
    assert [name for name, _ in attempts] == ["libxcb-cursor.so.0", home_copy]  # system first
    # Loaded globally, so the dependency of Qt's plugin resolves to it later.
    assert all(mode == ctypes.RTLD_GLOBAL for _, mode in attempts)

    named = str(tmp_path / "libxcb-cursor.so.0")
    monkeypatch.setenv("CPPI_LAB_XCB_CURSOR", named)
    attempts = _fake_loader(monkeypatch, {named})
    assert entry._load_xcb_cursor()
    assert [name for name, _ in attempts] == ["libxcb-cursor.so.0", named]  # before the home copies

    attempts = _fake_loader(monkeypatch, set())
    assert not entry._load_xcb_cursor()
    assert len(attempts) == 2 + len(entry.USER_XCB_CURSOR_PATHS)


# -- what the window manager supports ---------------------------------------------------------


def test_window_manager_query_needs_a_display(monkeypatch: pytest.MonkeyPatch) -> None:
    x11.wm_supported.cache_clear()
    monkeypatch.delenv("DISPLAY", raising=False)
    try:
        assert x11.wm_supported() is None
        assert x11.shows_utility_windows("xcb")  # unknown: assume a conventional one
    finally:
        x11.wm_supported.cache_clear()


@pytest.mark.parametrize(
    ("platform", "supported", "expected"),
    [
        ("xcb", frozenset({x11.UTILITY_WINDOW, "_NET_WM_STATE"}), True),
        ("xcb", frozenset({"_NET_WM_STATE", "_NET_WM_MOVERESIZE"}), False),  # WSLg
        ("xcb", frozenset(), False),  # no EWMH window manager at all
        ("xcb", None, True),
        ("wayland", frozenset(), True),  # only X11 has utility windows to refuse
        ("windows", None, True),
        ("offscreen", frozenset(), True),
    ],
)
def test_tool_windows_are_used_only_where_they_are_shown(
    monkeypatch: pytest.MonkeyPatch,
    platform: str,
    supported: frozenset[str] | None,
    expected: bool,
) -> None:
    monkeypatch.setattr(x11, "wm_supported", lambda: supported)
    assert x11.shows_utility_windows(platform) is expected
    flags = floating_window_flags(platform)
    kind = flags & Flag.WindowType_Mask
    assert kind == (Flag.Tool if expected else Flag.Window)
    for hint in (
        Flag.WindowStaysOnTopHint,
        Flag.WindowMinimizeButtonHint,
        Flag.WindowMaximizeButtonHint,
    ):
        assert flags & hint == hint  # the TSD hints are kept either way


# -- the floating window itself ------------------------------------------------------------


@pytest.fixture
def x11_window(
    qtbot: QtBot, bundled: MarketData, monkeypatch: pytest.MonkeyPatch
) -> Iterator[MainWindow]:
    """A main window built as on X11 under a window manager without utility windows."""
    import PySide6QtAds as ads

    flag = getattr(ads.CDockManager.eConfigFlag, "FloatingContainerForceQWidgetTitleBar", None)
    if flag is None:
        pytest.skip("this docking build has native floating title bars only")
    monkeypatch.setattr(QtGui.QGuiApplication, "platformName", staticmethod(lambda: "xcb"))
    monkeypatch.setattr(x11, "wm_supported", lambda: frozenset({"_NET_WM_STATE"}))
    win = MainWindow(bundled, use_web=False, async_ensemble=False)
    qtbot.addWidget(win)
    win.resize(1600, 900)
    win.show()
    qtbot.waitExposed(win)
    yield win
    win.close()
    ads.CDockManager.setConfigFlag(flag, False)


def test_floating_window_on_x11_has_its_own_title_bar_and_resize_grips(
    qtbot: QtBot, x11_window: MainWindow
) -> None:
    window = x11_window
    window.ledger_button.click()
    dock = window.ledger_dock
    qtbot.waitUntil(lambda: dock.isFloating() and not dock.isClosed(), timeout=3_000)
    container = dock.floatingDockContainer()
    assert not container.hasNativeTitleBar()
    flags = container.windowFlags()
    assert flags & Flag.WindowType_Mask == Flag.Window  # not a utility window here
    assert flags & Flag.WindowStaysOnTopHint == Flag.WindowStaysOnTopHint
    assert flags & Flag.FramelessWindowHint == Flag.FramelessWindowHint
    assert container.titleBarWidget().isVisible()
    assert container.parentWidget() is window.dock_manager  # owned, so it stays above

    (grips,) = [g for g in window._floating_grips if g.parent() is container]
    assert len(grips.grips) == 8
    qtbot.wait(150)  # the window is sized and placed shortly after it floats
    right = grips.grips[1]
    assert right.geometry().right() == container.width() - 1
    assert right.cursor().shape() == QtCore.Qt.CursorShape.SizeHorCursor
    # The grips lie above the content: the corner of the window belongs to a grip.
    corner = QtCore.QPoint(container.width() - 2, container.height() - 2)
    assert container.childAt(corner) is grips.grips[7]
    # Preparing the same window again adds nothing.
    window._prepare_floating_container(container)
    assert len(window._floating_grips) == 1
    container.resize(700, 420)
    qtbot.waitUntil(lambda: right.geometry().right() == container.width() - 1, timeout=1_000)
    assert container.width() == 700  # the grip follows the edge


# -- resize grips ------------------------------------------------------------------------------


def _drag(grip: QtWidgets.QWidget, dx: int, dy: int) -> None:
    """Press a grip, move the pointer by ``(dx, dy)`` on the screen, release."""
    start = QtCore.QPointF(grip.mapToGlobal(grip.rect().center()))
    local = QtCore.QPointF(grip.rect().center())
    kind = QtCore.QEvent.Type

    def send(event_type: QtCore.QEvent.Type, at: QtCore.QPointF, held: bool) -> None:
        buttons = LEFT if held else QtCore.Qt.MouseButton.NoButton
        button = LEFT if event_type != kind.MouseMove else QtCore.Qt.MouseButton.NoButton
        QtCore.QCoreApplication.sendEvent(
            grip, QtGui.QMouseEvent(event_type, local, at, button, buttons, NO_MODIFIER)
        )

    send(kind.MouseButtonPress, start, True)
    for step in (0.5, 1.0):
        send(kind.MouseMove, start + QtCore.QPointF(dx * step, dy * step), True)
    send(kind.MouseButtonRelease, start + QtCore.QPointF(dx, dy), False)


@pytest.fixture
def frameless(qtbot: QtBot) -> QtWidgets.QWidget:
    widget = QtWidgets.QWidget(None, Flag.Window | Flag.FramelessWindowHint)
    qtbot.addWidget(widget)
    widget.setMinimumSize(200, 120)
    widget.setGeometry(300, 200, 500, 320)
    widget.show()
    qtbot.waitExposed(widget)
    return widget


def test_grips_resize_from_every_edge_and_corner(
    qtbot: QtBot, frameless: QtWidgets.QWidget
) -> None:
    grips = ResizeGrips(frameless)
    left, right, top, bottom, top_left, _top_right, _bottom_left, bottom_right = grips.grips
    start = QtCore.QRect(frameless.geometry())

    _drag(right, 60, 25)  # an edge moves along its own axis only
    assert frameless.geometry() == start.adjusted(0, 0, 60, 0)
    _drag(bottom, 40, -30)
    assert frameless.geometry() == start.adjusted(0, 0, 60, -30)
    _drag(left, -50, 10)  # the opposite edge stays where it is
    assert frameless.geometry() == start.adjusted(-50, 0, 60, -30)
    _drag(top, 5, 35)
    assert frameless.geometry() == start.adjusted(-50, 35, 60, -30)
    _drag(bottom_right, -20, 45)
    assert frameless.geometry() == start.adjusted(-50, 35, 40, 15)
    _drag(top_left, 30, -15)
    assert frameless.geometry() == start.adjusted(-20, 20, 40, 15)
    # The grips followed the window.
    assert right.geometry().right() == frameless.width() - 1
    assert bottom.geometry().bottom() == frameless.height() - 1


def test_grips_respect_the_minimum_size(qtbot: QtBot, frameless: QtWidgets.QWidget) -> None:
    grips = ResizeGrips(frameless)
    right_edge = frameless.geometry().right()
    bottom_edge = frameless.geometry().bottom()
    _drag(grips.grips[0], 2_000, 0)  # left edge pushed far past the right one
    assert frameless.width() == 200
    assert frameless.geometry().right() == right_edge
    _drag(grips.grips[2], 0, 2_000)  # top edge pushed far past the bottom one
    assert frameless.height() == 120
    assert frameless.geometry().bottom() == bottom_edge
    _drag(grips.grips[7], -5_000, -5_000)
    assert (frameless.width(), frameless.height()) == (200, 120)


def test_grips_stay_above_content_added_later(qtbot: QtBot, frameless: QtWidgets.QWidget) -> None:
    grips = ResizeGrips(frameless)
    cover = QtWidgets.QLabel("content", frameless)
    cover.setGeometry(frameless.rect())
    cover.show()
    corner = QtCore.QPoint(frameless.width() - 2, frameless.height() - 2)
    qtbot.waitUntil(lambda: frameless.childAt(corner) is grips.grips[7], timeout=1_000)


def test_a_displaced_grip_is_put_back(qtbot: QtBot, frameless: QtWidgets.QWidget) -> None:
    grips = ResizeGrips(frameless)
    right = grips.grips[1]
    home = QtCore.QRect(right.geometry())
    right.setGeometry(3, 3, 5, 5)  # what a stale event from the display server would do
    right.displaced.emit()
    qtbot.waitUntil(lambda: right.geometry() == home, timeout=1_000)


# -- title-bar drag correction ---------------------------------------------------------------


class _Framed(QtWidgets.QWidget):
    """A top-level window that reports a frame of ``above`` pixels over its client area."""

    above = 0

    def frameGeometry(self) -> QtCore.QRect:
        return self.geometry().adjusted(-38, -self.above, 38, 38)


class _Bar(QtWidgets.QWidget):
    def __init__(self, parent: QtWidgets.QWidget) -> None:
        super().__init__(parent)
        self.presses: list[QtCore.QPoint] = []

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        self.presses.append(event.position().toPoint())


def test_title_bar_press_is_lowered_by_the_frame_above_the_window(qtbot: QtBot) -> None:
    host = _Framed()
    qtbot.addWidget(host)
    host.resize(400, 300)
    bar = _Bar(host)
    bar.setGeometry(0, 0, 400, 23)
    bar.installEventFilter(TitleBarPressFix(host))
    host.show()
    qtbot.waitExposed(host)

    qtbot.mousePress(bar, LEFT, pos=QtCore.QPoint(120, 9))
    assert bar.presses == [QtCore.QPoint(120, 9)]  # no frame above: untouched
    host.above = 59
    qtbot.mousePress(bar, LEFT, pos=QtCore.QPoint(120, 9))
    assert bar.presses[-1] == QtCore.QPoint(120, 9 + 59)  # exactly the reported frame
    qtbot.mousePress(bar, QtCore.Qt.MouseButton.RightButton, pos=QtCore.QPoint(50, 5))
    assert bar.presses[-1] == QtCore.QPoint(50, 5)  # only the dragging button
    assert len(bar.presses) == 3  # every press arrived exactly once


# -- the chart after a panel is docked or floated ---------------------------------------------
#
# The end-to-end check - the OpenGL chart is still drawn after a panel is docked
# and floated again - needs a real OpenGL viewport and is part of
# scripts/drag_dock_tour.py.


def test_canvas_rebuilds_its_surface_after_being_reparented(qtbot: QtBot) -> None:
    chart = ChartWorkspace()
    qtbot.addWidget(chart)
    chart.resize(800, 500)
    chart.show()
    canvas = chart.canvas
    qtbot.wait(20)
    assert not canvas._surface_timer.isActive()
    for kind in (
        QtCore.QEvent.Type.WindowAboutToChangeInternal,
        QtCore.QEvent.Type.WindowChangeInternal,
    ):
        canvas._surface_timer.stop()
        QtCore.QCoreApplication.sendEvent(canvas.viewport(), QtCore.QEvent(kind))
        assert canvas._surface_timer.isActive()  # scheduled once the change is complete
    margins = canvas.viewportMargins()
    size = canvas.viewport().size()
    canvas.recreate_surface()  # nothing to rebuild on a raster viewport, and no side effect
    assert canvas.viewportMargins() == margins
    assert canvas.viewport().size() == size
    assert not canvas.accelerated


def test_closing_right_after_floating_or_docking_is_safe(qtbot: QtBot, bundled: MarketData) -> None:
    """Placement is finished from timers; they must tolerate the window being gone."""
    import PySide6QtAds as ads

    window = MainWindow(bundled, use_web=False, async_ensemble=False)
    qtbot.addWidget(window)
    window.resize(1600, 900)
    window.show()
    qtbot.waitExposed(window)
    dock = window.ledger_dock
    window.ledger_button.click()
    window.dock_to(dock, ads.DockWidgetArea.RightDockWidgetArea)
    window.float_dock(dock)
    window.close()  # before the 0, 80 and 120 ms timers have fired
    QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)
    qtbot.wait(200)  # pytest-qt fails the test if a timer callback raises


# -- ledger toolbar in a narrow docked panel ---------------------------------------------------


def test_ledger_toolbar_fits_a_narrow_docked_panel(qtbot: QtBot) -> None:
    ledger = LedgerWidget()
    qtbot.addWidget(ledger)
    ledger.show()
    assert ledger.minimumSizeHint().width() <= 320  # the minimum width of a docked panel
    ledger.resize(980, 420)
    qtbot.wait(10)
    assert ledger.export_button.text() == EXPORT_CAPTION
    ledger.resize(330, 420)
    qtbot.wait(10)
    assert ledger.export_button.text() == EXPORT_CAPTION_SHORT
    assert "CSV" in ledger.export_button.toolTip()
    bar = ledger.export_button.parentWidget()
    assert bar is not None
    assert bar.width() <= 330
    right_end = ledger.export_button.geometry().right()
    assert right_end <= bar.width() - 1  # nothing hangs over the edge of the panel
    for child in (ledger.filter_combo, ledger.count_label, ledger.export_button):
        assert child.geometry().left() >= 0
        assert child.geometry().right() <= right_end
