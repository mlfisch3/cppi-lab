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
import ctypes
import os
import sys
from pathlib import Path


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
    try:
        ctypes.CDLL("libxcb-cursor.so.0")
    except OSError:
        return
    os.environ["QT_QPA_PLATFORM"] = "xcb"


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
    _prefer_xcb_on_linux()

    from PySide6 import QtCore, QtGui, QtWidgets

    QtCore.QCoreApplication.setAttribute(QtCore.Qt.ApplicationAttribute.AA_ShareOpenGLContexts)
    QtGui.QGuiApplication.setHighDpiScaleFactorRoundingPolicy(
        QtCore.Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

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

    splash.say("Building the workbench…")
    from cppi_lab.ui.main_window import MainWindow

    window = MainWindow(data)
    _place(window, args.size)
    window.show()
    splash.close()
    QtCore.QTimer.singleShot(400, window.popovers.prewarm)
    code: int = app.exec()

    from cppi_lab.ui.doc_workbench import release_profile

    release_profile()
    return code


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
