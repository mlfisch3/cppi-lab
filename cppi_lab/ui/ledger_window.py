"""Tabular ledger: virtualised telemetry table with filters, sort and export (PRD FR-5).

:class:`TelemetryTableModel` is a ``QAbstractTableModel`` laid directly over
the kernel's structured array.  It never copies the telemetry; a view row is
an index into the array, so filtering and sorting only rebuild one ``int64``
index vector and a year of 1-minute bars scrolls instantly.

The ledger is hosted in a Qt Advanced Docking System dock widget by the main
window, which makes it floatable, always-on-top while floating, and dockable
left, right or bottom.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from numpy.typing import NDArray
from PySide6 import QtCore, QtGui, QtWidgets

from cppi_lab.engine.kernel_numba import (
    EVENT_NAMES,
    FLAG_EXECUTED,
    FLAG_LIQUIDATION,
    FLAG_MISSED,
    TELEMETRY_DTYPE,
    TELEMETRY_FIELDS,
    SimResult,
    event_bit,
)
from cppi_lab.ui import theme
from cppi_lab.ui.widgets import ElidedLabel

IntArray = NDArray[np.int64]

FILTER_ALL = "All Timesteps"
FILTER_EXECUTED = "Executed Trades"
FILTER_MISSED = "Missed Maker Orders"
FILTER_LIQUIDATIONS = "Floor Liquidations"
FILTERS: tuple[str, ...] = (FILTER_ALL, FILTER_EXECUTED, FILTER_MISSED, FILTER_LIQUIDATIONS)

_FILTER_MASKS = {
    FILTER_EXECUTED: FLAG_EXECUTED,
    FILTER_MISSED: FLAG_MISSED,
    FILTER_LIQUIDATIONS: FLAG_LIQUIDATION,
}

#: ``(field, header, format kind)`` for every telemetry column, in dtype order.
COLUMNS: tuple[tuple[str, str, str], ...] = (
    ("bar_idx", "t", "int"),
    ("ts_utc_ms", "Time (UTC)", "time"),
    ("open", "Open", "usd"),
    ("high", "High", "usd"),
    ("low", "Low", "usd"),
    ("close", "Close", "usd"),
    ("buy_vol", "Buy Vol", "vol"),
    ("sell_vol", "Sell Vol", "vol"),
    ("s_down", "S_down", "unit"),
    ("s_up", "S_up", "unit"),
    ("p_floor", "P_floor", "usd"),
    ("safety_buf_usd", "Buffer $", "usd"),
    ("d_norm", "d", "unit"),
    ("n_max", "n_max", "btc"),
    ("n_ceil", "n_ceil", "btc"),
    ("n_rebuy", "n_floor", "btc"),
    ("btc_held", "BTC Held", "btc"),
    ("cash_usd", "Cash $", "usd"),
    ("net_equity_usd", "Net Equity $", "usd"),
    ("cushion_usd", "Cushion $", "usd"),
    ("event_code", "Event", "event"),
    ("exec_qty_btc", "Exec Qty", "qty"),
    ("exec_px_usd", "Exec Px", "usd"),
    ("fee_paid_usd", "Fee+Slip $", "fee"),
    ("p_fill_prob", "P_fill", "unit"),
    ("u_rand_draw", "U", "unit"),
)
assert tuple(c[0] for c in COLUMNS) == TELEMETRY_FIELDS

_EVENT_COLOURS = {
    1: theme.MAKER_SELL,
    2: theme.TAKER_SELL,
    3: theme.MAKER_BUY,
    4: theme.TAKER_BUY,
    5: theme.MISSED,
    6: theme.MISSED,
    7: theme.LIQUIDATION,
}
_EVENT_SHORT = {
    1: "MAKER_SELL",
    2: "TAKER_SELL",
    3: "MAKER_BUY",
    4: "TAKER_BUY",
    5: "MISS_SELL",
    6: "MISS_BUY",
    7: "LIQUIDATION",
}


def event_label(code: int, flags: int) -> str:
    """Primary event name plus any further events that shared the bar."""
    if code == 0:
        return ""
    extras = [
        _EVENT_SHORT[other] for other in range(1, 8) if other != code and flags & event_bit(other)
    ]
    name = EVENT_NAMES[code]
    return f"{name} +{' +'.join(extras)}" if extras else name


def format_cell(kind: str, value: Any, flags: int = 0) -> str:
    """Render one telemetry value for display."""
    if kind == "event":
        return event_label(int(value), flags)
    if kind == "int":
        return f"{int(value):,}"
    if kind == "time":
        return datetime.fromtimestamp(int(value) / 1000.0, UTC).strftime("%Y-%m-%d %H:%M")
    number = float(value)
    if number != number:  # NaN marks "not applicable" (no fill price, no maker attempt)
        return ""
    if kind == "usd":
        return f"{number:,.2f}"
    if kind == "vol":
        return f"{number:,.5f}"
    if kind == "btc":
        return f"{number:.6f}"
    if kind == "qty":
        return f"{number:.6f}" if number != 0.0 else ""
    if kind == "fee":
        return f"{number:,.4f}" if number != 0.0 else ""
    return f"{number:.4f}"


class TelemetryTableModel(QtCore.QAbstractTableModel):
    """Zero-copy table model over the telemetry structured array."""

    def __init__(self, parent: QtCore.QObject | None = None) -> None:
        super().__init__(parent)
        self._telemetry = np.zeros(0, dtype=TELEMETRY_DTYPE)
        self._flags = np.zeros(0, dtype=np.uint8)
        self._rows: IntArray = np.zeros(0, dtype=np.int64)
        self._filter = FILTER_ALL
        self._sort_column = -1
        self._sort_order = QtCore.Qt.SortOrder.AscendingOrder
        self._inverse: IntArray | None = None
        self._columns = [self._telemetry[name] for name, _, _ in COLUMNS]

    # -- data ----------------------------------------------------------------------
    def set_result(self, result: SimResult) -> None:
        """Bind a new back-test, keeping the active filter and sort."""
        self.beginResetModel()
        self._telemetry = result.telemetry
        self._flags = result.flags
        self._columns = [self._telemetry[name] for name, _, _ in COLUMNS]
        self._rebuild()
        self.endResetModel()

    def set_filter(self, name: str) -> None:
        """Apply one of :data:`FILTERS`."""
        if name not in FILTERS:
            raise ValueError(f"unknown filter {name!r}")
        self.beginResetModel()
        self._filter = name
        self._rebuild()
        self.endResetModel()

    @property
    def filter_name(self) -> str:
        return self._filter

    def _rebuild(self) -> None:
        mask = _FILTER_MASKS.get(self._filter)
        if mask is None:
            rows = np.arange(self._telemetry.shape[0], dtype=np.int64)
        else:
            rows = np.nonzero(self._flags & mask)[0].astype(np.int64)
        if self._sort_column >= 0 and rows.shape[0] > 1:
            keys = self._columns[self._sort_column][rows]
            order = np.argsort(keys, kind="stable")
            if self._sort_order == QtCore.Qt.SortOrder.DescendingOrder:
                order = order[::-1]
                if keys.dtype.kind == "f":
                    # Keep "not applicable" (NaN) cells at the bottom either way.
                    nan = np.isnan(keys[order])
                    order = np.concatenate((order[~nan], order[nan]))
            rows = rows[order]
        self._rows = rows
        self._inverse = None

    def sort(
        self, column: int, order: QtCore.Qt.SortOrder = QtCore.Qt.SortOrder.AscendingOrder
    ) -> None:
        self.beginResetModel()
        self._sort_column = column
        self._sort_order = order
        self._rebuild()
        self.endResetModel()

    # -- mapping -------------------------------------------------------------------
    def bar_of_row(self, row: int) -> int:
        """Bar index displayed in view row ``row``."""
        return int(self._rows[row])

    def row_of_bar(self, bar: int, nearest: bool = True) -> int:
        """View row showing ``bar``; the nearest shown bar when it is filtered out."""
        if self._rows.shape[0] == 0:
            return -1
        if self._inverse is None:
            inverse = np.full(self._telemetry.shape[0], -1, dtype=np.int64)
            inverse[self._rows] = np.arange(self._rows.shape[0], dtype=np.int64)
            self._inverse = inverse
        if 0 <= bar < self._inverse.shape[0] and self._inverse[bar] >= 0:
            return int(self._inverse[bar])
        if not nearest:
            return -1
        return int(np.argmin(np.abs(self._rows - bar)))

    @property
    def view_rows(self) -> IntArray:
        """Bar index of every view row, in display order."""
        return self._rows

    # -- Qt model API --------------------------------------------------------------
    def rowCount(
        self, parent: QtCore.QModelIndex | QtCore.QPersistentModelIndex = QtCore.QModelIndex()
    ) -> int:
        return 0 if parent.isValid() else int(self._rows.shape[0])

    def columnCount(
        self, parent: QtCore.QModelIndex | QtCore.QPersistentModelIndex = QtCore.QModelIndex()
    ) -> int:
        return 0 if parent.isValid() else len(COLUMNS)

    def data(
        self,
        index: QtCore.QModelIndex | QtCore.QPersistentModelIndex,
        role: int = QtCore.Qt.ItemDataRole.DisplayRole,
    ) -> Any:
        if not index.isValid():
            return None
        bar = int(self._rows[index.row()])
        column = index.column()
        kind = COLUMNS[column][2]
        if role == QtCore.Qt.ItemDataRole.DisplayRole:
            return format_cell(kind, self._columns[column][bar], int(self._flags[bar]))
        if role == QtCore.Qt.ItemDataRole.TextAlignmentRole:
            if kind in ("event", "time"):
                return int(QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter)
            return int(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
        if role == QtCore.Qt.ItemDataRole.ForegroundRole:
            code = int(self._telemetry["event_code"][bar])
            if code and kind in ("event", "qty", "usd", "fee") and column >= 20:
                return QtGui.QColor(_EVENT_COLOURS[code])
            return None
        if role == QtCore.Qt.ItemDataRole.BackgroundRole:
            flags = int(self._flags[bar])
            if flags & FLAG_LIQUIDATION:
                return QtGui.QColor("#3A1418")
            if flags & FLAG_EXECUTED:
                return QtGui.QColor("#101D2E")
            if flags & FLAG_MISSED:
                return QtGui.QColor("#241B0C")
            return None
        if role == QtCore.Qt.ItemDataRole.UserRole:
            return self._columns[column][bar].item()
        return None

    def headerData(
        self,
        section: int,
        orientation: QtCore.Qt.Orientation,
        role: int = QtCore.Qt.ItemDataRole.DisplayRole,
    ) -> Any:
        if orientation != QtCore.Qt.Orientation.Horizontal:
            return None
        if role == QtCore.Qt.ItemDataRole.DisplayRole:
            return COLUMNS[section][1]
        if role == QtCore.Qt.ItemDataRole.ToolTipRole:
            return f"{COLUMNS[section][0]} ({TELEMETRY_DTYPE[section]})"
        return None

    # -- export --------------------------------------------------------------------
    def to_frame(self) -> pl.DataFrame:
        """The rows currently shown, in display order, as a Polars frame."""
        rows = self._rows
        data: dict[str, Any] = {name: self._telemetry[name][rows] for name, _, _ in COLUMNS}
        data["event_flags"] = self._flags[rows]
        return pl.DataFrame(data)

    def export(self, path: str | Path) -> int:
        """Write the shown rows as CSV or Parquet; returns the row count."""
        target = Path(path)
        frame = self.to_frame()
        suffix = target.suffix.lower()
        if suffix == ".parquet":
            frame.write_parquet(target, compression="zstd")
        elif suffix == ".csv":
            frame.write_csv(target)
        else:
            raise ValueError("export path must end in .csv or .parquet")
        return frame.height


EXPORT_CAPTION = "Export CSV / Parquet…"
EXPORT_CAPTION_SHORT = "Export…"
#: Below this widget width the export button shows a short caption.
EXPORT_CAPTION_MIN_WIDTH = 560


class LedgerWidget(QtWidgets.QWidget):
    """Filter bar plus the telemetry table."""

    #: Bar index of the row the user moved to; drives the chart crosshair.
    barActivated = QtCore.Signal(int)
    exported = QtCore.Signal(str, int)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.model = TelemetryTableModel(self)
        self._syncing = False

        bar = QtWidgets.QWidget(self)
        bar.setObjectName("ledgerBar")
        bar.setAttribute(QtCore.Qt.WidgetAttribute.WA_StyledBackground, True)
        self.filter_combo = QtWidgets.QComboBox(bar)
        self.filter_combo.addItems(list(FILTERS))
        self.filter_combo.setAccessibleName("Row filter")
        # The bar must fit a narrow docked panel: the combo may shorten its text,
        # the count elides, and the export button switches to a short caption.
        self.filter_combo.setSizeAdjustPolicy(
            QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self.filter_combo.setMinimumContentsLength(10)
        self.filter_combo.currentTextChanged.connect(self._on_filter)
        self.count_label = ElidedLabel("", bar)
        self.count_label.setProperty("role", "muted")
        self.export_button = QtWidgets.QPushButton(EXPORT_CAPTION, bar)
        self.export_button.setToolTip("Export the rows shown, in this order, as CSV or Parquet")
        short = self.export_button.fontMetrics().horizontalAdvance(EXPORT_CAPTION_SHORT)
        self.export_button.setMinimumWidth(short + 24)
        self.export_button.clicked.connect(self._choose_export)
        bar_layout = QtWidgets.QHBoxLayout(bar)
        bar_layout.setContentsMargins(8, 5, 8, 5)
        bar_layout.setSpacing(8)
        bar_layout.addWidget(QtWidgets.QLabel("Show", bar))
        bar_layout.addWidget(self.filter_combo)
        bar_layout.addWidget(self.count_label, 1)
        bar_layout.addWidget(self.export_button)

        self.table = QtWidgets.QTableView(self)
        self.table.setModel(self.model)
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setWordWrap(False)
        self.table.setHorizontalScrollMode(QtWidgets.QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.table.verticalHeader().setVisible(False)
        self.table.verticalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Fixed)
        self.table.verticalHeader().setDefaultSectionSize(21)
        header = self.table.horizontalHeader()
        header.setSectionsClickable(True)
        header.setSortIndicatorShown(True)
        header.setSortIndicator(-1, QtCore.Qt.SortOrder.AscendingOrder)
        header.setDefaultAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        header.setMinimumSectionSize(48)
        self.table.setSortingEnabled(True)
        header.setSortIndicator(-1, QtCore.Qt.SortOrder.AscendingOrder)
        self._size_columns()
        self.table.selectionModel().currentRowChanged.connect(self._on_current_row)
        self.model.modelReset.connect(self._update_count)
        self.empty_label = QtWidgets.QLabel("", self.table.viewport())
        self.empty_label.setProperty("role", "muted")
        self.empty_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.empty_label.setWordWrap(True)
        self.empty_label.hide()

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(bar)
        layout.addWidget(self.table, 1)
        self.resize(980, 420)

    def _size_columns(self) -> None:
        widths = {
            "int": 64,
            "time": 128,
            "usd": 96,
            "vol": 88,
            "unit": 66,
            "btc": 84,
            "event": 250,
            "qty": 84,
            "fee": 84,
        }
        for column, (_, _, kind) in enumerate(COLUMNS):
            self.table.setColumnWidth(column, widths[kind])

    # -- data ----------------------------------------------------------------------
    def set_result(self, result: SimResult) -> None:
        bar = self.current_bar()
        self.model.set_result(result)
        if bar >= 0:
            self.select_bar(bar)

    def set_filter(self, name: str) -> None:
        self.filter_combo.setCurrentText(name)

    def _on_filter(self, name: str) -> None:
        bar = self.current_bar()
        self.model.set_filter(name)
        if bar >= 0:
            self.select_bar(bar)
            # If the bar is filtered out the selection snaps to the nearest row
            # shown; tell the chart so both keep pointing at the same bar.
            snapped = self.current_bar()
            if snapped >= 0 and snapped != bar:
                self.barActivated.emit(snapped)

    def _update_count(self) -> None:
        shown = self.model.rowCount()
        total = int(self.model._telemetry.shape[0])
        self.count_label.setText(f"{shown:,} of {total:,} timesteps")
        self.count_label.setToolTip(self.count_label.text())
        if shown == 0 and total > 0:
            self.empty_label.setText(
                f"No timestep of this back-test matches \u201c{self.model.filter_name}\u201d."
            )
            self._place_empty_label()
            self.empty_label.show()
        else:
            self.empty_label.hide()

    def _place_empty_label(self) -> None:
        viewport = self.table.viewport()
        self.empty_label.setGeometry(20, 30, max(100, viewport.width() - 40), 40)

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        self._place_empty_label()
        wide = self.width() >= EXPORT_CAPTION_MIN_WIDTH
        caption = EXPORT_CAPTION if wide else EXPORT_CAPTION_SHORT
        if self.export_button.text() != caption:
            self.export_button.setText(caption)

    # -- chart synchronisation -------------------------------------------------------
    def current_bar(self) -> int:
        index = self.table.currentIndex()
        if not index.isValid() or index.row() >= self.model.rowCount():
            return -1
        return self.model.bar_of_row(index.row())

    def select_bar(self, bar: int) -> None:
        """Select and reveal the row for ``bar`` without echoing back to the chart."""
        row = self.model.row_of_bar(bar)
        if row < 0:
            return
        self._syncing = True
        try:
            column = max(0, self.table.currentIndex().column())
            index = self.model.index(row, column)
            self.table.setCurrentIndex(index)
            self.table.scrollTo(index, QtWidgets.QAbstractItemView.ScrollHint.PositionAtCenter)
        finally:
            self._syncing = False

    def _on_current_row(self, current: QtCore.QModelIndex, _previous: QtCore.QModelIndex) -> None:
        if self._syncing or not current.isValid():
            return
        self.barActivated.emit(self.model.bar_of_row(current.row()))

    # -- export --------------------------------------------------------------------
    def _choose_export(self) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export ledger",
            "cppi_lab_ledger.csv",
            "CSV (*.csv);;Parquet (*.parquet)",
        )
        if path:
            self.export_to(path)

    def export_to(self, path: str | Path) -> int:
        """Export the shown rows; the format follows the file extension."""
        target = Path(path)
        if target.suffix.lower() not in (".csv", ".parquet"):
            target = target.with_suffix(".csv")
        count = self.model.export(target)
        self.exported.emit(str(target), count)
        return count
