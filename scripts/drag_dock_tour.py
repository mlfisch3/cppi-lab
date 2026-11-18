"""Exercise the floating windows with a real mouse: move, resize and drag-and-drop docking.

Usage::

    uv run python scripts/drag_dock_tour.py OUT_DIR [--xdotool PATH] [--size 1600x900]
                                                    [--no-web] [--only ledger|docs]

The script builds the actual :class:`MainWindow` on the X11 display and then
drives it from outside with ``xdotool``, which injects genuine pointer events
through the X server (XTEST).  Nothing is called on the docking system
directly: every move, resize and drop below is a mouse press, a series of
mouse motions and a mouse release, exactly as a user's hand produces them.

For the ledger and for the documentation workbench it

1. opens the window floating and checks that it is on screen;
2. moves it by dragging its title bar and checks that it followed the cursor;
3. resizes it by dragging its bottom-right corner and its right edge;
4. drags the title bar onto the left, right and bottom indicator of the
   docking compass, checks before the release that the compass reports that
   drop area, releases, and checks that the panel is docked on that side;
5. floats the panel again before the next drop.

A screenshot is saved in the middle of each docking drag (button still held,
compass and drop preview showing) and after each drop.  The X server of a
rootless session (WSLg) cannot be captured as a whole, so each screenshot is
assembled from the application's own windows, read back from the X server
one by one and placed at their screen positions (see ``x11_capture.py``).

Exit status 0 means every step was verified; any failed check is printed and
makes the status 1.  The X pointer is shared with the real mouse, so keep
your hands off it while the tour runs.  A drag during which a pointer event
arrived from elsewhere (a hand, or the desktop reporting where the real
pointer rests when a window moves under it) is detected and repeated, and the
windows are moved away from that spot.  The script needs the ``xcb`` platform
(see the README), ``xdotool`` and ImageMagick's ``import``.
"""

from __future__ import annotations

import argparse
import math
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

#: Pixels the cursor travels per injected motion event, and the pause after each.
DRAG_STEP_PX = 24
DRAG_STEP_SECONDS = 0.012
#: How often a gesture is attempted when the real mouse moved the pointer during it.
TRIES = 4
#: A pointer event further than this from the injected path came from somewhere else.
POINTER_TOLERANCE_PX = 48


def off_path(point: tuple[int, int], start: tuple[int, int], end: tuple[int, int]) -> float:
    """Distance in pixels from ``point`` to the straight path from ``start`` to ``end``."""
    px, py = point
    ax, ay = start
    bx, by = end
    dx, dy = bx - ax, by - ay
    length = dx * dx + dy * dy
    along = 0.0 if length == 0 else max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / length))
    return math.hypot(px - (ax + along * dx), py - (ay + along * dy))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("out", type=Path)
    parser.add_argument("--xdotool", default=shutil.which("xdotool") or "xdotool")
    parser.add_argument(
        "--capture", default=shutil.which("import") or "import", help="ImageMagick's import"
    )
    parser.add_argument("--size", default="1600x900")
    parser.add_argument("--no-web", action="store_true", help="skip the documentation window")
    parser.add_argument("--only", choices=("ledger", "docs"), default=None)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    from cppi_lab.main import configure_environment, set_application_attributes

    configure_environment()
    import pyqtgraph as pg
    import PySide6QtAds as ads
    from PySide6 import QtCore, QtGui, QtWidgets
    from shiboken6 import isValid

    set_application_attributes()
    from cppi_lab.ui import theme
    from cppi_lab.ui.doc_scheme import register_scheme

    register_scheme()
    pg.setConfigOptions(
        useOpenGL=True,
        enableExperimental=True,
        antialias=True,
        background=theme.CANVAS,
        foreground=theme.TEXT_MUTED,
    )
    app = QtWidgets.QApplication(sys.argv[:1])
    app.setApplicationName("CPPI-Lab")
    app.setApplicationDisplayName("CPPI-Lab")
    theme.apply_theme(app)
    if app.platformName() != "xcb":
        print(f"this tour needs the xcb platform, not {app.platformName()!r}; see the README")
        return 2
    try:
        version = subprocess.run(
            [args.xdotool, "version"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        print(f"xdotool is not usable ({exc}); pass --xdotool PATH")
        return 2
    from x11_capture import X11Capture

    from cppi_lab.data.loader import load_bundled
    from cppi_lab.engine.controller import warmup
    from cppi_lab.ui.main_window import SCENARIOS, MainWindow

    data = load_bundled()
    warmup()
    use_web = not args.no_web
    window = MainWindow(data, use_web=use_web)
    width, _, height = args.size.partition("x")
    window.resize(int(width), int(height))
    window.show()
    capture = X11Capture(app, window, args.xdotool, args.capture)
    problem = capture.usable()
    if problem:
        print(f"cannot take screenshots: {problem}")
        return 2

    failures: list[str] = []

    def pump(seconds: float) -> None:
        end = time.time() + seconds
        while time.time() < end:
            app.processEvents()
            time.sleep(0.01)

    def check(condition: bool, what: str) -> bool:
        print(f"  {'ok  ' if condition else 'FAIL'} {what}", flush=True)
        if not condition:
            failures.append(what)
        return condition

    def xdo(*words: object) -> None:
        """Run one xdotool command line while the Qt event loop keeps turning."""
        process = subprocess.Popen([args.xdotool, *map(str, words)])
        while process.poll() is None:
            app.processEvents()
            time.sleep(0.004)
        pump(0.05)

    position = [0, 0]

    class PointerWatch(QtCore.QObject):
        """Notices when the pointer is not where this script put it.

        The X pointer is shared with the real mouse.  Whenever the desktop
        tells the X server where the real pointer is - a hand moved it, or a
        window appeared or moved underneath it - the server's pointer jumps
        there until the next injected motion, and a docking system that asks
        the server for the pointer position in that moment moves the dragged
        window to the wrong place.  Such a drag says nothing about the
        application, so it is detected here and repeated.
        """

        def __init__(self) -> None:
            super().__init__()
            self.path: tuple[tuple[int, int], tuple[int, int]] = ((0, 0), (0, 0))
            self.foreign: tuple[int, int] | None = None
            self.watching = False
            self._pressed: QtWidgets.QWidget | None = None

        def start(self, pressed: QtWidgets.QWidget | None) -> None:
            """Begin a gesture on ``pressed``, the widget that will receive the drag."""
            self.foreign = None
            self.watching = True
            self._pressed = pressed
            if pressed is not None:
                # While a button is held every mouse event goes to the widget it
                # was pressed on, so that is the only one that has to be watched.
                pressed.installEventFilter(self)

        def stop(self) -> bool:
            """End a gesture; ``True`` if every pointer event came from this script."""
            self.watching = False
            if self._pressed is not None and isValid(self._pressed):
                self._pressed.removeEventFilter(self)  # a drop may have destroyed it
            self._pressed = None
            return self.foreign is None

        def eventFilter(  # noqa: N802 (Qt virtual)
            self, watched: QtCore.QObject, event: QtCore.QEvent
        ) -> bool:
            if (
                self.watching
                and self.foreign is None
                and isinstance(event, QtGui.QMouseEvent)
                and event.spontaneous()
            ):
                # Two things can be off: the position the event carries, and the
                # position the X server reports when asked right now - which is
                # what the docking system asks for while it moves the window.
                for at in (event.globalPosition().toPoint(), QtGui.QCursor.pos()):
                    if off_path((at.x(), at.y()), *self.path) > POINTER_TOLERANCE_PX:
                        self.foreign = (at.x(), at.y())
                        break
            return False

    watch = PointerWatch()

    def move_to(x: int, y: int) -> None:
        """Glide the pointer to ``(x, y)`` in small steps, like a hand does."""
        x0, y0 = position
        watch.path = ((x0, y0), (x, y))
        distance = max(abs(x - x0), abs(y - y0))
        steps = max(1, distance // DRAG_STEP_PX)
        words: list[object] = []
        for i in range(1, steps + 1):
            words += [
                "mousemove",
                round(x0 + (x - x0) * i / steps),
                round(y0 + (y - y0) * i / steps),
                "sleep",
                DRAG_STEP_SECONDS,
            ]
        xdo(*words)
        position[:] = [x, y]

    def press(x: int, y: int) -> None:
        """Start a gesture: jump to ``(x, y)`` and hold the left button down."""
        watch.path = ((x, y), (x, y))
        xdo("mousemove", x, y, "sleep", 0.15)
        position[:] = [x, y]
        # The jump itself may cross anything; from here on the pointer may not stray.
        watch.start(app.widgetAt(QtCore.QPoint(x, y)))
        xdo("mousedown", 1, "sleep", 0.12)

    def release() -> bool:
        """End a gesture; ``True`` if no pointer event came from anywhere else."""
        watch.path = ((position[0], position[1]), (position[0], position[1]))
        xdo("sleep", 0.1, "mouseup", 1)
        pump(0.5)
        clean = watch.stop()
        if not clean:
            print(
                f"  ..   a pointer event from another source arrived at {watch.foreign} "
                "during the drag; repeating",
                flush=True,
            )
            keep_clear_of(watch.foreign)
        return clean

    def keep_clear_of(point: tuple[int, int] | None) -> None:
        """Move the main window so the tour's windows stay away from the real pointer."""
        if point is None:
            return
        area = app.primaryScreen().availableGeometry()
        # The floating windows hang over the main window's edges while they are
        # dragged: this far to the left, to the right and below.
        left, right, below = 420, 790, 770
        top = max(area.top(), min(window.y(), area.bottom() - window.height() - below))

        def clearance(x: int) -> int:
            """How far the point is outside the tour's footprint (negative: inside)."""
            return max(x - left - point[0], point[0] - (x + window.width() + right))

        candidates = [area.left() + left, area.right() - window.width() - right]
        best = max(candidates, key=clearance)
        if clearance(best) > clearance(window.x()):
            window.move(best, top)
            pump(0.8)

    def chart_is_drawn() -> bool:
        """Whether the chart canvas of the main window shows a picture rather than a void."""
        pump(0.3)
        image = capture.read_back(window)
        canvas = window.chart.canvas
        origin = canvas.mapTo(window, QtCore.QPoint(0, 0))
        dark = total = 0
        for y in range(origin.y() + 30, origin.y() + canvas.height() - 30, 16):
            for x in range(origin.x() + 80, origin.x() + canvas.width() - 20, 16):
                colour = image.pixelColor(x, y)
                total += 1
                dark += colour.red() < 3 and colour.green() < 3 and colour.blue() < 3
        return total > 0 and dark / total < 0.05

    def shot(name: str) -> None:
        pump(0.35)
        image = capture.compose()
        image.save(str(args.out / f"{name}.png"))
        print(f"  saved {name}.png ({image.width()}x{image.height()})", flush=True)

    def title_point(container: Any) -> QtCore.QPoint:
        """A point on the floating window's title bar, clear of its buttons."""
        bar = container.titleBarWidget()
        point: QtCore.QPoint
        if bar is not None and bar.isVisible():
            point = bar.mapToGlobal(QtCore.QPoint(max(60, bar.width() // 3), bar.height() // 2))
        else:
            point = container.mapToGlobal(QtCore.QPoint(max(60, container.width() // 3), 10))
        return point

    def indicator(area: Any) -> QtCore.QPoint | None:
        """Centre of the outermost compass indicator for a drop area.

        The compass has one cross for the whole main window (indicators at its
        edges) and one for the dock area under the cursor (indicators around
        its centre); both are top-level overlay windows.  Docking a panel to a
        side of the main window uses the outer one.
        """
        centre = window.mapToGlobal(window.rect().center())
        best: QtCore.QPoint | None = None
        for widget in app.allWidgets():
            if (
                isinstance(widget, QtWidgets.QLabel)
                and widget.isVisible()
                and isinstance(widget.window(), ads.CDockOverlayCross)
                and widget.property("dockWidgetArea") == area.value
            ):
                point = widget.mapToGlobal(widget.rect().center())
                if (
                    best is None
                    or (point - centre).manhattanLength() > (best - centre).manhattanLength()
                ):
                    best = point
        return best

    def docked_on(panel: Any, area: Any) -> tuple[bool, str]:
        """Whether ``panel`` now sits on the given side of the central workbench."""
        origin = QtCore.QPoint(0, 0)
        at = QtCore.QRect(panel.mapTo(window, origin), panel.size())
        bench = QtCore.QRect(window.workbench.mapTo(window, origin), window.workbench.size())
        where = (
            f"panel x {at.left()}..{at.right()} y {at.top()}..{at.bottom()}, workbench "
            f"x {bench.left()}..{bench.right()} y {bench.top()}..{bench.bottom()}"
        )
        if area == ads.DockWidgetArea.LeftDockWidgetArea:
            return at.right() < bench.left(), where
        if area == ads.DockWidgetArea.RightDockWidgetArea:
            return at.left() > bench.right(), where
        return at.top() > bench.bottom(), where

    pump(1.0)
    # Leave room on every side: a window that hangs over the edge of the X screen
    # while it is dragged cannot be read back for the screenshots.  And stay away
    # from where the pointer is now - most likely where the real mouse rests.
    screen_rect = app.primaryScreen().availableGeometry()
    resting = QtGui.QCursor.pos()
    window.move(
        screen_rect.left() + max(0, (screen_rect.width() - window.width()) // 2),
        screen_rect.top() + max(0, (screen_rect.height() - window.height()) // 3),
    )
    pump(0.6)
    keep_clear_of((resting.x(), resting.y()))
    pump(0.6)
    print(f"platform={app.platformName()} {version} window={window.geometry()}", flush=True)
    window.apply_scenario(SCENARIOS[1])
    pump(0.8)
    areas = (
        ("left", ads.DockWidgetArea.LeftDockWidgetArea),
        ("right", ads.DockWidgetArea.RightDockWidgetArea),
        ("bottom", ads.DockWidgetArea.BottomDockWidgetArea),
    )

    def tour(label: str, dock: Any, button: Any, panel: Any) -> None:
        print(f"[{label}]", flush=True)
        settle = 0.6 if label == "ledger" else 1.5
        button.click()
        pump(1.2 if label == "ledger" else 3.0)
        container = dock.floatingDockContainer()
        if not check(dock.isFloating() and container is not None, "opens floating"):
            return
        screen = app.primaryScreen().geometry()
        check(container.isVisible(), "floating window is visible")
        check(
            screen.intersects(container.frameGeometry())
            and window.frameGeometry().intersects(container.frameGeometry()),
            f"it is on screen, over the main window: {container.frameGeometry()}",
        )
        check(
            container.windowFlags() & QtCore.Qt.WindowType.WindowStaysOnTopHint
            == QtCore.Qt.WindowType.WindowStaysOnTopHint,
            "it asks to stay on top",
        )
        shot(f"{label}_1_floating")

        # -- move by the title bar ------------------------------------------------
        delta = QtCore.QPoint(-140, 90)
        for _ in range(TRIES):
            container = dock.floatingDockContainer()
            before = container.pos()
            grip = title_point(container)
            press(grip.x(), grip.y())
            dragging = container.isDraggingActive()
            move_to(grip.x() + delta.x(), grip.y() + delta.y())
            clean = release()
            if clean:
                break
            window.float_dock(dock)  # back over the main window, wherever it is now
            pump(settle)
        check(dragging, "pressing the title bar starts a drag")
        moved = container.pos() - before
        check(
            clean and abs(moved.x() - delta.x()) <= 4 and abs(moved.y() - delta.y()) <= 4,
            f"title-bar drag moved the window by {moved.x()},{moved.y()} "
            f"(cursor travelled {delta.x()},{delta.y()})",
        )
        check(dock.isFloating(), "it is still floating after the move")
        shot(f"{label}_2_moved")

        # -- resize by the corner and by an edge ----------------------------------
        def resize_by(at: QtCore.QPoint, dx: int, dy: int) -> tuple[bool, QtCore.QSize]:
            """Drag from a point inside the window by ``(dx, dy)``; returns the size change."""
            grown = QtCore.QSize()
            clean = False
            for _ in range(TRIES):
                size = container.size()
                start = container.mapToGlobal(at)
                press(start.x(), start.y())
                move_to(start.x() + dx, start.y() + dy)
                clean = release()
                grown = container.size() - size
                if clean:
                    break
            return clean, grown

        clean, grown = resize_by(
            QtCore.QPoint(container.width() - 2, container.height() - 2), -150, -90
        )
        check(
            clean and abs(grown.width() + 150) <= 6 and abs(grown.height() + 90) <= 6,
            f"corner drag resized it by {grown.width()},{grown.height()} (cursor -150,-90)",
        )
        clean, grown = resize_by(
            QtCore.QPoint(container.width() - 2, container.height() // 2), 110, 0
        )
        check(
            clean and abs(grown.width() - 110) <= 6 and abs(grown.height()) <= 2,
            f"right-edge drag resized it by {grown.width()},{grown.height()} (cursor +110,0)",
        )
        shot(f"{label}_3_resized")

        # -- drag onto the compass: left, right, bottom ---------------------------
        for name, area in areas:
            overlay = window.dock_manager.containerOverlay()
            showing = reported = clean = False
            for _ in range(TRIES):
                container = dock.floatingDockContainer()
                if container is None or not dock.isFloating():
                    break
                grip = title_point(container)
                press(grip.x(), grip.y())
                centre = window.workbench.mapToGlobal(window.workbench.rect().center())
                move_to(centre.x(), centre.y())  # the compass appears over the main window
                pump(0.2)
                showing = overlay.isVisible()
                target = indicator(area)
                if target is not None:
                    move_to(target.x(), target.y())
                    pump(0.25)
                    reported = overlay.dropAreaUnderCursor() == area
                if watch.foreign is None and reported:
                    shot(f"{label}_4_{name}_compass")
                clean = release()
                pump(settle)
                if clean:
                    break
                window.float_dock(dock)  # wherever that drop went, start again floating
                pump(2.0 * settle)
            if not check(clean, f"{name}: the drag onto the compass was not interfered with"):
                continue
            check(showing, f"{name}: the docking compass is showing")
            check(reported, f"{name}: the compass reports this drop area under the cursor")
            check(not dock.isFloating() and not dock.isClosed(), f"{name}: the panel is docked")
            placed, where = docked_on(panel, area)
            check(placed, f"{name}: it sits on the {name} side of the workbench ({where})")
            check(not overlay.isVisible(), f"{name}: the compass is gone")
            check(chart_is_drawn(), f"{name}: the chart is drawn beside the docked panel")
            shot(f"{label}_5_{name}_docked")
            window.float_dock(dock)
            pump(1.0 if label == "ledger" else 2.0)
            check(chart_is_drawn(), f"{name}: the chart is drawn after the panel floats again")
        window.set_dock_visible(dock, False)
        pump(0.4)

    if args.only in (None, "ledger"):
        tour("ledger", window.ledger_dock, window.ledger_button, window.ledger)
    if use_web and args.only in (None, "docs"):
        tour("docs", window.docs_dock, window.docs_button, window.docs)

    window.close()
    pump(0.4)
    if use_web:
        from cppi_lab.ui.doc_workbench import release_profile

        QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)
        app.processEvents()
        release_profile()
    print(
        "\nall steps verified"
        if not failures
        else f"\n{len(failures)} step(s) FAILED:\n  " + "\n  ".join(failures)
    )
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
