"""Small shared widgets."""

from __future__ import annotations

from PySide6 import QtCore, QtGui, QtWidgets


class ElidedLabel(QtWidgets.QLabel):
    """A one-line label that shortens its text with an ellipsis instead of clipping it."""

    def minimumSizeHint(self) -> QtCore.QSize:
        return QtCore.QSize(0, super().minimumSizeHint().height())

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        painter.setPen(self.palette().color(self.foregroundRole()))
        rect = self.contentsRect()
        text = self.fontMetrics().elidedText(
            self.text(), QtCore.Qt.TextElideMode.ElideRight, rect.width()
        )
        painter.drawText(rect, int(self.alignment()), text)


class ElasticComboBox(QtWidgets.QComboBox):
    """A combo box that shows its full text when there is room and yields when not."""

    def minimumSizeHint(self) -> QtCore.QSize:
        return QtCore.QSize(120, super().minimumSizeHint().height())

    def sizeHint(self) -> QtCore.QSize:
        hint = super().sizeHint()
        return QtCore.QSize(min(hint.width() + 8, 340), hint.height())
