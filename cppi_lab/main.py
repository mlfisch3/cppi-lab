"""Application entry point: ``uv run cppi-lab`` (TSD section 6).

Start-up order matters and is fixed here:

1. choose the Qt platform plugin and share OpenGL contexts (QtWebEngine and
   the chart canvas live in the same window);
2. register the ``cppidoc://`` scheme - this must precede ``QApplication``;
3. configure pyqtgraph for the OpenGL viewport;
4. create the application, show a splash, load the dataset and compile (or
   load from cache) the Numba kernels;
5. build and show the main window.
"""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import os
import signal
import socket
import sys
from collections.abc import Callable
from pathlib import Path

#: Where a copy of ``libxcb-cursor.so.0`` installed without root is looked for.
USER_XCB_CURSOR_PATHS: tuple[str, ...] = (
    "~/.local/lib/libxcb-cursor.so.0",
    "~/.local/lib/x86_64-linux-gnu/libxcb-cursor.so.0",
    "~/.nix-profile/lib/libxcb-cursor.so.0",
)


def _load_xcb_cursor() -> bool:
    """Load ``libxcb-cursor.so.0``, which Qt's X11 plugin has needed since Qt 6.5.

    The system library is tried first, then the file named by
    ``CPPI_LAB_XCB_CURSOR``, then the per-user locations above.  A library
    loaded here satisfies the plugin's dependency when Qt loads it later, so a
    copy in the home directory works without ``LD_LIBRARY_PATH``.
    """
    candidates = ["libxcb-cursor.so.0"]
    override = os.environ.get("CPPI_LAB_XCB_CURSOR")
    if override:
        candidates.append(override)
    candidates.extend(str(Path(path).expanduser()) for path in USER_XCB_CURSOR_PATHS)
    for candidate in candidates:
        try:
            ctypes.CDLL(candidate, mode=ctypes.RTLD_GLOBAL)
        except OSError:
            continue
        return True
    return False


def _prefer_xcb_on_linux() -> None:
    """Prefer X11 on Linux when it is usable.

    Qt Advanced Docking System needs global cursor positions and window
    placement for drag-and-drop docking of floating windows; a Wayland session
    provides neither.  X11 (including XWayland) does, so it is selected when a
    display is present and Qt's X11 plugin can actually load (it needs
    ``libxcb-cursor0``).  An explicit ``QT_QPA_PLATFORM`` always wins.
    """
    if not sys.platform.startswith("linux") or os.environ.get("QT_QPA_PLATFORM"):
        return
    if not os.environ.get("DISPLAY"):
        return
    if _load_xcb_cursor():
        os.environ["QT_QPA_PLATFORM"] = "xcb"


def _running_under_wsl() -> bool:
    if os.environ.get("WSL_DISTRO_NAME"):
        return True
    try:
        return "microsoft" in Path("/proc/sys/kernel/osrelease").read_text().lower()
    except OSError:
        return False


def _use_software_web_rendering_under_wsl() -> None:
    """Render documentation pages in software under WSL.

    WSLg's D3D12-backed EGL has no ``EGL_EXT_image_dma_buf_import``, which
    QtWebEngine's GPU compositor needs to hand frames to Qt; with it enabled
    every page comes out blank ("Compositor returned null texture").  The
    pages are static text and mathematics, so software rasterisation costs
    nothing visible.  An explicit ``QTWEBENGINE_CHROMIUM_FLAGS`` always wins.
    """
    if not sys.platform.startswith("linux") or "QTWEBENGINE_CHROMIUM_FLAGS" in os.environ:
        return
    if _running_under_wsl():
        os.environ["QTWEBENGINE_CHROMIUM_FLAGS"] = "--disable-gpu"


def configure_environment() -> None:
    """Process-level settings that must be in place before Qt is imported."""
    _prefer_xcb_on_linux()
    _use_software_web_rendering_under_wsl()


def set_application_attributes() -> None:
    """Application attributes that must be set before ``QApplication`` is created.

    * OpenGL contexts are shared: QtWebEngine and the chart canvas live in the
      same window, and a panel keeps its context when it is docked or floated.
    * A widget that needs a native window does not drag its siblings along.
      Without this the docking system turns half of the main window into
      separate X windows the first time a panel floats, and stale geometry
      events from the server can then move child widgets back to where they
      were.
    * Fractional display scale factors are passed through unrounded.
    """
    from PySide6 import QtCore, QtGui

    attribute = QtCore.Qt.ApplicationAttribute
    QtCore.QCoreApplication.setAttribute(attribute.AA_ShareOpenGLContexts)
    QtCore.QCoreApplication.setAttribute(attribute.AA_DontCreateNativeWidgetSiblings)
    QtGui.QGuiApplication.setHighDpiScaleFactorRoundingPolicy(
        QtCore.Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )


#: Exit status after Ctrl+C, as a shell reports a program ended by SIGINT.
INTERRUPTED = 128 + int(signal.SIGINT)


class InterruptWatch:
    """Lets Ctrl+C in the launching terminal reach an idle Qt event loop.

    Python runs a signal handler only when the interpreter next gets control,
    and an idle Qt event loop never gives it any: the interrupt would be
    ignored until the window is touched.  The signal is therefore also written
    to a socket the event loop watches; reading it brings the interpreter
    back, which then runs the handler.  Must be created in the main thread.
    """

    def __init__(self, on_interrupt: Callable[[], None]) -> None:
        from PySide6 import QtCore

        self.requested = False
        self._on_interrupt = on_interrupt
        self._reader, self._writer = socket.socketpair()
        self._reader.setblocking(False)
        self._writer.setblocking(False)
        self._notifier = QtCore.QSocketNotifier(
            self._reader.fileno(), QtCore.QSocketNotifier.Type.Read
        )
        self._notifier.activated.connect(self._drain)
        self._previous_fd = signal.set_wakeup_fd(self._writer.fileno(), warn_on_full_buffer=False)
        self._previous_handler = signal.signal(signal.SIGINT, self._handle)

    def _drain(self) -> None:
        with contextlib.suppress(OSError):  # nothing to read: a spurious wake-up
            self._reader.recv(64)

    def _handle(self, _signum: int, _frame: object) -> None:
        self.requested = True
        self._on_interrupt()

    def close(self) -> None:
        """Put the previous interrupt handling back."""
        # ``None`` stands for a handler that was not installed from Python.
        previous = self._previous_handler
        signal.signal(signal.SIGINT, signal.SIG_DFL if previous is None else previous)
        signal.set_wakeup_fd(self._previous_fd)
        self._notifier.setEnabled(False)
        self._notifier.deleteLater()
        self._reader.close()
        self._writer.close()


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="cppi-lab",
        description="CPPI-Lab: back-testing and educational workbench for a signal-modulated "
        "Bitcoin CPPI strategy with a hysteresis deadband.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="open this CSV/Parquet file instead of the bundled dataset",
    )
    parser.add_argument(
        "--no-opengl",
        action="store_true",
        help="draw the charts with the software rasteriser (slow; for machines without OpenGL)",
    )
    parser.add_argument(
        "--size",
        default="",
        metavar="WxH",
        help="initial window size, e.g. 1920x1080 (default: fit the screen, at most 1920x1080)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    configure_environment()

    from PySide6 import QtCore, QtWidgets

    set_application_attributes()

    from cppi_lab.ui.doc_scheme import register_scheme

    register_scheme()

    import pyqtgraph as pg

    from cppi_lab.ui import theme

    use_opengl = not args.no_opengl and os.environ.get("CPPI_LAB_OPENGL", "1") != "0"
    pg.setConfigOptions(
        useOpenGL=use_opengl,
        enableExperimental=use_opengl,
        antialias=True,
        background=theme.CANVAS,
        foreground=theme.TEXT_MUTED,
    )

    app = QtWidgets.QApplication(sys.argv[:1])
    app.setApplicationName("CPPI-Lab")
    app.setApplicationDisplayName("CPPI-Lab")
    theme.apply_theme(app)
    interrupt = InterruptWatch(app.quit)

    splash = _Splash()
    splash.show()
    splash.say("Loading market data…")

    from cppi_lab.data.loader import DataFormatError, load_bundled, load_file

    try:
        data = load_file(args.dataset) if args.dataset is not None else load_bundled()
    except DataFormatError as exc:
        splash.close()
        QtWidgets.QMessageBox.critical(None, "CPPI-Lab", f"Cannot load the dataset.\n\n{exc}")
        return 2

    splash.say("Preparing simulation kernels (compiled once, then cached)…")
    from cppi_lab.engine.controller import warmup

    warmup()
    if interrupt.requested:  # Ctrl+C while the kernels were compiling
        return INTERRUPTED

    splash.say("Building the workbench…")
    from cppi_lab.ui.main_window import MainWindow

    window = MainWindow(data)
    _place(window, args.size)
    window.show()
    splash.close()
    QtCore.QTimer.singleShot(400, window.popovers.prewarm)
    # ``quit`` does nothing before the event loop runs, so an interrupt that
    # arrived while the window was being built is honoured here.
    code: int = INTERRUPTED if interrupt.requested else app.exec()

    from cppi_lab.ui.doc_workbench import release_profile

    # Web pages must be gone before the profile they belong to.
    QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)
    app.processEvents()
    release_profile()
    return INTERRUPTED if interrupt.requested else code


def _place(window: object, size: str) -> None:
    """Size the window for the screen; maximise it on screens up to 1080p."""
    from PySide6 import QtGui, QtWidgets

    assert isinstance(window, QtWidgets.QMainWindow)
    screen = QtGui.QGuiApplication.primaryScreen()
    available = screen.availableGeometry() if screen is not None else None
    if size:
        width, _, height = size.lower().partition("x")
        window.resize(int(width), int(height))
    elif available is not None and (available.width() <= 1920 or available.height() <= 1080):
        window.setWindowState(window.windowState() | QtGui.Qt.WindowState.WindowMaximized)
        return
    else:
        window.resize(1920, 1040)
    if available is not None:
        frame = window.frameGeometry()
        frame.moveCenter(available.center())
        window.move(frame.topLeft())


class _Splash:
    """A small start-up card; the first launch compiles the kernels for a few seconds."""

    def __init__(self) -> None:
        from PySide6 import QtCore, QtWidgets

        from cppi_lab.ui import theme

        self._app = QtWidgets.QApplication.instance()
        self._card = QtWidgets.QFrame(
            None,
            QtCore.Qt.WindowType.SplashScreen | QtCore.Qt.WindowType.FramelessWindowHint,
        )
        self._card.setStyleSheet(
            f"QFrame {{ background: {theme.PANEL}; border: 1px solid {theme.BORDER}; }}"
            f"QLabel {{ border: none; }}"
        )
        title = QtWidgets.QLabel(
            f'<span style="font-size:22px;font-weight:700;color:{theme.TEXT}">CPPI</span>'
            f'<span style="font-size:22px;font-weight:700;color:{theme.ACCENT}">-Lab</span>'
        )
        subtitle = QtWidgets.QLabel("Signal-modulated Bitcoin CPPI back-testing workbench")
        subtitle.setStyleSheet(f"color: {theme.TEXT_MUTED};")
        self._status = QtWidgets.QLabel("")
        self._status.setStyleSheet(f"color: {theme.TEXT_FAINT}; font-size: 11px;")
        layout = QtWidgets.QVBoxLayout(self._card)
        layout.setContentsMargins(28, 22, 28, 20)
        layout.setSpacing(6)
        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addSpacing(10)
        layout.addWidget(self._status)
        self._card.setFixedSize(440, 136)

    def show(self) -> None:
        from PySide6 import QtGui

        screen = QtGui.QGuiApplication.primaryScreen()
        if screen is not None:
            frame = self._card.frameGeometry()
            frame.moveCenter(screen.availableGeometry().center())
            self._card.move(frame.topLeft())
        self._card.show()
        self._pump()

    def say(self, text: str) -> None:
        self._status.setText(text)
        self._pump()

    def close(self) -> None:
        self._card.close()

    def _pump(self) -> None:
        if self._app is not None:
            for _ in range(3):
                self._app.processEvents()


if __name__ == "__main__":
    sys.exit(main())
