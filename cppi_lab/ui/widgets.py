"""Small shared widgets."""

from __future__ import annotations

from PySide6 import QtCore, QtGui, QtWidgets


class ElidedLabel(QtWidgets.QLabel):
    """A one-line label that shortens its text with an ellipsis instead of clipping it."""

    def minimumSizeHint(self) -> QtCore.QSize:
        return QtCore.QSize(0, super().minimumSizeHint().height())

    def is_elided(self) -> bool:
        """Whether the text is wider than the label and is drawn shortened."""
        return self.fontMetrics().horizontalAdvance(self.text()) > self.contentsRect().width()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        painter.setPen(self.palette().color(self.foregroundRole()))
        rect = self.contentsRect()
        text = self.fontMetrics().elidedText(
            self.text(), QtCore.Qt.TextElideMode.ElideRight, rect.width()
        )
        painter.drawText(rect, int(self.alignment()), text)


class AdaptiveLabel(ElidedLabel):
    """A one-line label that shows the longest of several wordings that fits.

    The wordings go from the most detailed to the tersest; the last one is
    elided if even it is too wide.  The tool tip always has the first.
    """

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__("", parent)
        self._wordings: tuple[str, ...] = ("",)

    def set_wordings(self, *wordings: str) -> None:
        self._wordings = wordings or ("",)
        self.setToolTip(self._wordings[0])
        self._choose()

    def wordings(self) -> tuple[str, ...]:
        return self._wordings

    def _choose(self) -> None:
        metrics = self.fontMetrics()
        room = self.contentsRect().width()
        chosen = next(
            (text for text in self._wordings if metrics.horizontalAdvance(text) <= room),
            self._wordings[-1],
        )
        if chosen != self.text():
            self.setText(chosen)

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        self._choose()


class ResizeWatcher(QtCore.QObject):
    """Emits :attr:`resized` whenever the widget it is given changes size."""

    resized = QtCore.Signal()

    def __init__(self, widget: QtWidgets.QWidget) -> None:
        super().__init__(widget)
        widget.installEventFilter(self)

    def eventFilter(self, watched: QtCore.QObject, event: QtCore.QEvent) -> bool:
        if event.type() == QtCore.QEvent.Type.Resize:
            self.resized.emit()
        return False


class ElasticComboBox(QtWidgets.QComboBox):
    """A combo box that shows its full text when there is room and yields when not."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        # Follow the contents at all times; the default policy measures them once,
        # when the box is first shown, and ignores items added later.
        self.setSizeAdjustPolicy(QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToContents)

    def minimumSizeHint(self) -> QtCore.QSize:
        return QtCore.QSize(120, super().minimumSizeHint().height())

    def sizeHint(self) -> QtCore.QSize:
        hint = super().sizeHint()
        return QtCore.QSize(min(hint.width() + 8, 340), hint.height())

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        # QComboBox cuts a text that does not fit; end it with an ellipsis instead.
        painter = QtWidgets.QStylePainter(self)
        option = QtWidgets.QStyleOptionComboBox()
        self.initStyleOption(option)
        style = QtWidgets.QStyle
        painter.drawComplexControl(style.ComplexControl.CC_ComboBox, option)
        field = self.style().subControlRect(
            style.ComplexControl.CC_ComboBox, option, style.SubControl.SC_ComboBoxEditField, self
        )
        option.currentText = self.fontMetrics().elidedText(
            option.currentText, QtCore.Qt.TextElideMode.ElideRight, field.width()
        )
        painter.drawControl(style.ControlElement.CE_ComboBoxLabel, option)


_EDGE = QtCore.Qt.Edge
_CURSOR = QtCore.Qt.CursorShape
#: Edges each grip moves, in the order the grips are created.
_GRIP_EDGES: tuple[QtCore.Qt.Edge, ...] = (
    _EDGE.LeftEdge,
    _EDGE.RightEdge,
    _EDGE.TopEdge,
    _EDGE.BottomEdge,
    _EDGE.LeftEdge | _EDGE.TopEdge,
    _EDGE.RightEdge | _EDGE.TopEdge,
    _EDGE.LeftEdge | _EDGE.BottomEdge,
    _EDGE.RightEdge | _EDGE.BottomEdge,
)


class _ResizeGrip(QtWidgets.QWidget):
    """An invisible strip along one edge or corner that resizes its window when dragged."""

    def __init__(
        self, target: QtWidgets.QWidget, edges: QtCore.Qt.Edge, outline: QtGui.QColor | None
    ) -> None:
        super().__init__(target)
        self._target = target
        self.edges = edges
        self._outline = outline
        self._pressed_at: QtCore.QPoint | None = None
        self._start = QtCore.QRect()
        horizontal = bool(edges & (_EDGE.LeftEdge | _EDGE.RightEdge))
        vertical = bool(edges & (_EDGE.TopEdge | _EDGE.BottomEdge))
        if horizontal and vertical:
            falling = edges in (_EDGE.LeftEdge | _EDGE.TopEdge, _EDGE.RightEdge | _EDGE.BottomEdge)
            self.setCursor(_CURSOR.SizeFDiagCursor if falling else _CURSOR.SizeBDiagCursor)
        else:
            self.setCursor(_CURSOR.SizeHorCursor if horizontal else _CURSOR.SizeVerCursor)

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self._pressed_at = event.globalPosition().toPoint()
            self._start = self._target.geometry()
            event.accept()

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        if self._pressed_at is None:
            return
        delta = event.globalPosition().toPoint() - self._pressed_at
        rect = QtCore.QRect(self._start)
        hint = self._target.minimumSizeHint()
        min_width = max(self._target.minimumWidth(), hint.width(), 160)
        min_height = max(self._target.minimumHeight(), hint.height(), 100)
        if self.edges & _EDGE.LeftEdge:
            rect.setLeft(min(rect.left() + delta.x(), rect.right() - min_width + 1))
        if self.edges & _EDGE.RightEdge:
            rect.setRight(max(rect.right() + delta.x(), rect.left() + min_width - 1))
        if self.edges & _EDGE.TopEdge:
            rect.setTop(min(rect.top() + delta.y(), rect.bottom() - min_height + 1))
        if self.edges & _EDGE.BottomEdge:
            rect.setBottom(max(rect.bottom() + delta.y(), rect.top() + min_height - 1))
        if rect != self._target.geometry():
            self._target.setGeometry(rect)
        event.accept()

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        self._pressed_at = None
        event.accept()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        # The grips are the frame of a frameless window: draw its outline, one
        # pixel along the outer edge, and leave the rest of the strip see-through.
        if self._outline is None:
            return
        painter = QtGui.QPainter(self)
        painter.setPen(QtGui.QPen(self._outline, 1.0))
        right, bottom = self.width() - 1, self.height() - 1
        if self.edges & _EDGE.LeftEdge:
            painter.drawLine(0, 0, 0, bottom)
        if self.edges & _EDGE.RightEdge:
            painter.drawLine(right, 0, right, bottom)
        if self.edges & _EDGE.TopEdge:
            painter.drawLine(0, 0, right, 0)
        if self.edges & _EDGE.BottomEdge:
            painter.drawLine(0, bottom, right, bottom)

    def moveEvent(self, event: QtGui.QMoveEvent) -> None:
        # If a grip ever is a native window, the display server can report an
        # outdated position for it after the window was resized; put it back.
        super().moveEvent(event)
        if event.spontaneous():
            self.displaced.emit()

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        if event.spontaneous():
            self.displaced.emit()

    #: The display server moved or resized the grip behind the application's back.
    displaced = QtCore.Signal()


class ResizeGrips(QtCore.QObject):
    """Makes a frameless top-level window resizable by its edges and corners.

    A frameless window has no window-manager border to pull.  Eight grips are
    laid along its edges, above its content; dragging one changes the window
    geometry directly, so the behaviour does not depend on the window manager
    supporting an interactive resize for undecorated windows.  The grips are
    see-through except for an optional one-pixel ``outline`` along the outer
    edge, which gives the window the visible frame it otherwise lacks.
    """

    #: Thickness of the edge strips and side length of the corner squares.
    BAND = 5
    CORNER = 14

    def __init__(self, target: QtWidgets.QWidget, outline: QtGui.QColor | None = None) -> None:
        super().__init__(target)
        self._target = target
        self.grips = [_ResizeGrip(target, edges, outline) for edges in _GRIP_EDGES]
        self._rearrange = QtCore.QTimer(self)
        self._rearrange.setSingleShot(True)
        self._rearrange.setInterval(0)
        self._rearrange.timeout.connect(self.arrange)
        for grip in self.grips:
            grip.displaced.connect(self._rearrange.start)
        target.installEventFilter(self)
        self.arrange()

    def arrange(self) -> None:
        """Lay the grips along the current edges and lift them above the content."""
        width, height = self._target.width(), self._target.height()
        band, corner = self.BAND, self.CORNER
        left, right, top, bottom, top_left, top_right, bottom_left, bottom_right = self.grips
        left.setGeometry(0, corner, band, max(0, height - 2 * corner))
        right.setGeometry(width - band, corner, band, max(0, height - 2 * corner))
        top.setGeometry(corner, 0, max(0, width - 2 * corner), band)
        bottom.setGeometry(corner, height - band, max(0, width - 2 * corner), band)
        top_left.setGeometry(0, 0, corner, corner)
        top_right.setGeometry(width - corner, 0, corner, corner)
        bottom_left.setGeometry(0, height - corner, corner, corner)
        bottom_right.setGeometry(width - corner, height - corner, corner, corner)
        for grip in self.grips:
            grip.show()
            grip.raise_()

    def eventFilter(self, watched: QtCore.QObject, event: QtCore.QEvent) -> bool:
        kind = event.type()
        if kind in (QtCore.QEvent.Type.Resize, QtCore.QEvent.Type.Show):
            self.arrange()
        elif kind == QtCore.QEvent.Type.ChildAdded:
            # A widget added later would cover the grips; lift them once it is in place.
            self._rearrange.start()
        return False


class TitleBarPressFix(QtCore.QObject):
    """Keeps a floating window under the cursor while its own title bar is dragged.

    Qt Advanced Docking System moves a floating window with its own title bar
    to ``cursor - press position`` and corrects only for a frame on the sides.
    Some window managers report a frame above an undecorated window as well
    (an invisible shadow margin; WSLg's does), and the window then jumps down
    by that amount the moment the drag starts.  Reporting the press that much
    lower cancels the error exactly, and changes nothing where no frame is
    reported.
    """

    def __init__(self, parent: QtCore.QObject | None = None) -> None:
        super().__init__(parent)
        self._resending = False

    def eventFilter(self, watched: QtCore.QObject, event: QtCore.QEvent) -> bool:
        if (
            self._resending
            or event.type() != QtCore.QEvent.Type.MouseButtonPress
            or not isinstance(watched, QtWidgets.QWidget)
            or not isinstance(event, QtGui.QMouseEvent)
            or event.button() != QtCore.Qt.MouseButton.LeftButton
        ):
            return False
        top_level = watched.window()
        above = top_level.geometry().top() - top_level.frameGeometry().top()
        if above <= 0:
            return False
        shifted = QtGui.QMouseEvent(
            event.type(),
            event.position() + QtCore.QPointF(0.0, above),
            event.globalPosition(),
            event.button(),
            event.buttons(),
            event.modifiers(),
        )
        self._resending = True
        try:
            QtCore.QCoreApplication.sendEvent(watched, shifted)
        finally:
            self._resending = False
        return True
