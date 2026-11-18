"""Screenshots of the running application, read back from the X server.

Used by ``ui_tour.py`` and ``drag_dock_tour.py``.  ``QWidget.grab()`` cannot
see an OpenGL viewport, and the root window of a rootless X server (WSLg)
cannot be captured as a whole, so a screenshot is assembled from the
application's own top-level windows: each is read back from the X server
with ImageMagick's ``import`` and painted at its screen position, bottom to
top.  Translucent overlays (the docking compass) are rendered by Qt instead,
which keeps their transparency.

Needs the ``xcb`` platform, ``xdotool`` and ImageMagick's ``import``.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import PySide6QtAds as ads
from PySide6 import QtCore, QtGui, QtWidgets


class X11Capture:
    """Reads the application's windows back from the X server and composes them."""

    def __init__(
        self,
        app: QtWidgets.QApplication,
        main_window: QtWidgets.QWidget,
        xdotool: str = "",
        capture: str = "",
    ) -> None:
        self.app = app
        self.main_window = main_window
        self.xdotool = xdotool or shutil.which("xdotool") or "xdotool"
        self.capture = capture or shutil.which("import") or "import"
        self._scratch = Path(tempfile.mkdtemp(prefix="cppi-lab-capture-"))

    def usable(self) -> str:
        """An empty string when screenshots can be taken, otherwise the reason they cannot."""
        if self.app.platformName() != "xcb":
            return f"the Qt platform is {self.app.platformName()!r}, not 'xcb'"
        for tool, probe in ((self.xdotool, "version"), (self.capture, "-version")):
            try:
                subprocess.run([tool, probe], capture_output=True, check=True, timeout=10)
            except (OSError, subprocess.SubprocessError) as exc:
                return f"{tool} is not usable ({exc})"
        return ""

    def window_id(self, widget: Any) -> str:
        """X window id of a top-level widget.

        The docking system re-creates the native window of a floating panel,
        and the id Qt reports for it afterwards is not the one on screen, so a
        floating panel is looked up by its title instead.
        """
        title = widget.windowTitle()
        if widget is not self.main_window and title:
            found = subprocess.run(
                [self.xdotool, "search", "--name", f"^{re.escape(title)}$"],
                capture_output=True,
                text=True,
                check=False,
            ).stdout.split()
            if found:
                return found[-1]
        return str(int(widget.winId()))

    def read_back(self, widget: Any) -> QtGui.QImage:
        """Pixels of one top-level window as the X server holds them."""
        target = self._scratch / "window.png"
        target.unlink(missing_ok=True)
        try:
            subprocess.run(
                [self.capture, "-window", self.window_id(widget), str(target)],
                capture_output=True,
                timeout=10,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            return QtGui.QImage()
        return QtGui.QImage(str(target))

    def top_levels(self) -> list[Any]:
        """The application's visible top-level windows, bottom to top."""
        shown = [w for w in self.app.topLevelWidgets() if w.isVisible() and w.width() > 1]

        def layer(widget: Any) -> int:
            if widget is self.main_window:
                return 0
            if isinstance(widget, ads.CFloatingDockContainer):
                return 1
            return 2  # drop overlays, menus, tool tips

        return sorted(shown, key=layer)

    def compose(self) -> QtGui.QImage:
        """One image of every visible window of the application, placed as on screen."""
        windows = self.top_levels()
        bounds = QtCore.QRect()
        for widget in windows:
            bounds = bounds.united(widget.geometry())
        image = QtGui.QImage(bounds.size(), QtGui.QImage.Format.Format_ARGB32_Premultiplied)
        image.fill(QtGui.QColor("#000000"))
        painter = QtGui.QPainter(image)
        for widget in windows:
            origin = widget.mapToGlobal(QtCore.QPoint(0, 0)) - bounds.topLeft()
            if widget.testAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground):
                painter.setOpacity(widget.windowOpacity())
                painter.drawPixmap(origin, widget.grab())  # keeps the translucency
                painter.setOpacity(1.0)
            else:
                painter.drawImage(origin, self.read_back(widget))
        painter.end()
        return image
