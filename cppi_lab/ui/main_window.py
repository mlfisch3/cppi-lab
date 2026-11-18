"""Main window: dock orchestration and the recalculation loop (TSD sections 4.1, 4.2).

Layout::

    +--------------------------------------------------------------------+
    | top bar: dataset, import, Coinbase sync, scenario, ledger, docs    |
    +---------------+----------------------------------------------------+
    | parameter     | KPI ribbon (64 px)                                 |
    | sidebar       +----------------------------------------------------+
    |               | three-pane chart canvas (55 : 25 : 20)             |
    +---------------+----------------------------------------------------+

The whole block is the central widget of a Qt Advanced Docking System
manager.  The ledger and the documentation workbench are ``CDockWidget``
instances that open floating (a tool window that stays above the main
window) and can be dragged onto the main window, or sent there from their
toolbar menu, to dock left, right or bottom.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from functools import partial
from pathlib import Path

import PySide6QtAds as ads
from PySide6 import QtCore, QtGui, QtWidgets
from shiboken6 import isValid

from cppi_lab.config.defaults import (
    DEFAULT_START_MS,
    ParamValue,
    SimParams,
    default_params,
)
from cppi_lab.data.loader import (
    DataFormatError,
    FetchError,
    MarketData,
    fetch_coinbase_candles,
    load_file,
)
from cppi_lab.engine.controller import RunOutput, SimulationController
from cppi_lab.engine.monte_carlo import EnsembleResult, run_ensemble
from cppi_lab.ui import theme, x11
from cppi_lab.ui.chart_workspace import ChartWorkspace
from cppi_lab.ui.control_sidebar import ControlSidebar
from cppi_lab.ui.doc_scheme import DocRegistry, load_registry
from cppi_lab.ui.kpi_ribbon import KpiRibbon, compute_kpis
from cppi_lab.ui.ledger_window import LedgerWidget
from cppi_lab.ui.popover_manager import RichPopoverManager
from cppi_lab.ui.widgets import (
    AdaptiveLabel,
    ElasticComboBox,
    ResizeGrips,
    ResizeWatcher,
    TitleBarPressFix,
)

TOP_BAR_HEIGHT = 38
#: Narrowest the two selectors of the top bar are made to keep the status read-out legible.
SCENARIO_MIN_WIDTH = 130
DATASET_MIN_WIDTH = 170


def floating_window_flags(platform_name: str) -> QtCore.Qt.WindowType:
    """Window type and hints of a floating secondary window (TSD section 4.2).

    The TSD asks for ``Tool | WindowStaysOnTopHint | WindowMinimizeButtonHint |
    WindowMaximizeButtonHint``.  ``Tool`` becomes a utility window on X11, and a
    window manager that does not know that type never shows the window at
    all; there an ordinary window is used instead.  It still belongs to the
    main window (it is transient for it), which is what keeps it above.
    """
    flag = QtCore.Qt.WindowType
    kind = flag.Tool if x11.shows_utility_windows(platform_name) else flag.Window
    return (
        kind
        | flag.WindowStaysOnTopHint
        | flag.WindowMinimizeButtonHint
        | flag.WindowMaximizeButtonHint
    )


#: A duration as the status read-out prints it ("12.3").
_TIMING = re.compile(r"\d+\.\d")


def _ms(year: int, month: int, day: int) -> int:
    return int(datetime(year, month, day, tzinfo=UTC).timestamp() * 1000)


@dataclass(frozen=True)
class Scenario:
    """A curated window of the bundled dataset."""

    name: str
    start_ms: int
    days: int
    normalize: bool = False


#: Real periods that exercise the strategy with the PRD default floor and band.
SCENARIOS: tuple[Scenario, ...] = (
    Scenario("PRD default window - Jan 2026 (30 d)", DEFAULT_START_MS, 30),
    Scenario("Feb 2026 crash to $60k (21 d)", _ms(2026, 1, 25), 21),
    Scenario("Apr 2026 chop above the floor (30 d)", _ms(2026, 4, 1), 30),
    Scenario("Jun 2026 breakdown through the floor (30 d)", _ms(2026, 6, 1), 30),
    Scenario("Aug 2026 V-recovery (30 d)", _ms(2026, 8, 1), 30),
    Scenario("Jan 2026 rescaled to the $68k anchor (30 d)", DEFAULT_START_MS, 30, True),
    Scenario("Oct 2025 liquidation cascade, rescaled (14 d)", _ms(2025, 10, 3), 14, True),
    Scenario("Apr 2025 tariff crash, rescaled (21 d)", _ms(2025, 3, 28), 21, True),
    Scenario("Aug 2024 carry-trade crash, rescaled (14 d)", _ms(2024, 7, 29), 14, True),
    Scenario("2023 sideways consolidation, rescaled (60 d)", _ms(2023, 4, 15), 60, True),
)
#: The scenario offered by the empty-state banner.
CRASH_SCENARIO = SCENARIOS[1]


class _EnsembleSignals(QtCore.QObject):
    finished = QtCore.Signal(int, object)


class _EnsembleTask(QtCore.QRunnable):
    """Runs one Monte Carlo ensemble off the GUI thread."""

    def __init__(
        self, generation: int, inputs: tuple[object, ...], signals: _EnsembleSignals
    ) -> None:
        super().__init__()
        self._generation = generation
        self._inputs = inputs
        self._signals = signals

    def run(self) -> None:
        window, signals, params, block = self._inputs
        result = run_ensemble(window, signals, params, u_block=block)  # type: ignore[arg-type]
        self._signals.finished.emit(self._generation, result)


class _FetchWorker(QtCore.QThread):
    """Downloads Coinbase candles without blocking the UI."""

    progressed = QtCore.Signal(int, int)
    succeeded = QtCore.Signal(object)
    failed = QtCore.Signal(str)

    def __init__(
        self, start: datetime, end: datetime, parent: QtCore.QObject | None = None
    ) -> None:
        super().__init__(parent)
        self._start, self._end = start, end
        self._cancel = False

    def cancel(self) -> None:
        self._cancel = True

    def run(self) -> None:
        try:
            data = fetch_coinbase_candles(
                self._start,
                self._end,
                progress=lambda done, total: self.progressed.emit(done, total),
                cancelled=lambda: self._cancel,
            )
        except (FetchError, DataFormatError, ValueError) as exc:
            self.failed.emit(str(exc))
            return
        self.succeeded.emit(data)


class CoinbaseSyncDialog(QtWidgets.QDialog):
    """Pick a UTC range and fetch 1-minute candles from Coinbase (FR-1.2)."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Fetch candles from Coinbase")
        self.setModal(True)
        self.data: MarketData | None = None
        self._worker: _FetchWorker | None = None
        utc = QtCore.QTimeZone.utc()
        now = QtCore.QDateTime.currentDateTimeUtc()
        now = now.addSecs(-now.time().second()).addMSecs(-now.time().msec())
        self.start_edit = QtWidgets.QDateTimeEdit(now.addDays(-3), self)
        self.end_edit = QtWidgets.QDateTimeEdit(now, self)
        for edit in (self.start_edit, self.end_edit):
            edit.setTimeZone(utc)
            edit.setDisplayFormat("yyyy-MM-dd HH:mm")
        note = QtWidgets.QLabel(
            "Downloads BTC-USD 1-minute candles from the public Coinbase Advanced Trade "
            "market endpoint (no account or key).  Coinbase candles carry total volume "
            "only, so the taker buy/sell split is synthesised from the Close Location "
            "Value, as TSD section 2.1 prescribes.",
            self,
        )
        note.setWordWrap(True)
        note.setProperty("role", "muted")
        self.progress = QtWidgets.QProgressBar(self)
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self.status = QtWidgets.QLabel("", self)
        self.status.setWordWrap(True)
        self.fetch_button = QtWidgets.QPushButton("Fetch", self)
        self.fetch_button.setObjectName("primary")
        self.fetch_button.clicked.connect(self._fetch)
        self.cancel_button = QtWidgets.QPushButton("Cancel", self)
        self.cancel_button.clicked.connect(self.reject)
        form = QtWidgets.QFormLayout()
        form.addRow("Start (UTC)", self.start_edit)
        form.addRow("End (UTC)", self.end_edit)
        buttons = QtWidgets.QHBoxLayout()
        buttons.addStretch(1)
        buttons.addWidget(self.cancel_button)
        buttons.addWidget(self.fetch_button)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 12)
        layout.setSpacing(10)
        layout.addWidget(note)
        layout.addLayout(form)
        layout.addWidget(self.progress)
        layout.addWidget(self.status)
        layout.addLayout(buttons)
        self.setFixedWidth(430)

    def _fetch(self) -> None:
        start = self.start_edit.dateTime().toPython()
        end = self.end_edit.dateTime().toPython()
        assert isinstance(start, datetime)
        assert isinstance(end, datetime)
        if end <= start:
            self.status.setText("The end must be after the start.")
            return
        if end - start > timedelta(days=370):
            self.status.setText("Please fetch at most 370 days at a time.")
            return
        self.fetch_button.setEnabled(False)
        self.status.setText("Fetching…")
        self._worker = _FetchWorker(start, end, self)
        self._worker.progressed.connect(self._on_progress)
        self._worker.succeeded.connect(self._on_success)
        self._worker.failed.connect(self._on_failure)
        self._worker.start()

    def _on_progress(self, done: int, total: int) -> None:
        self.progress.setRange(0, total)
        self.progress.setValue(done)

    def _on_success(self, data: MarketData) -> None:
        self.data = data
        self.accept()

    def _on_failure(self, message: str) -> None:
        self.status.setText(f"Fetch failed: {message}")
        self.fetch_button.setEnabled(True)

    def reject(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            self._worker.cancel()
            self._worker.wait(5000)
        super().reject()


class MainWindow(QtWidgets.QMainWindow):
    """The CPPI-Lab application window."""

    #: Emitted after every recalculation with the fresh run.
    recalculated = QtCore.Signal(object)
    #: Emitted when a Monte Carlo ensemble has been applied to the chart.
    ensembleApplied = QtCore.Signal(object)

    def __init__(
        self,
        data: MarketData,
        *,
        use_web: bool = True,
        async_ensemble: bool = True,
        registry: DocRegistry | None = None,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("CPPI-Lab")
        self.resize(1920, 1040)
        self.setMinimumSize(1100, 620)
        self._use_web = use_web
        self._async_ensemble = async_ensemble
        self.registry = registry if registry is not None else load_registry()
        self.controller = SimulationController(data)
        self.datasets: list[MarketData] = [data]
        self.run: RunOutput | None = None
        self.ensemble: EnsembleResult | None = None
        self._ensemble_signals = _EnsembleSignals(self)
        self._ensemble_signals.finished.connect(self._on_ensemble_finished)
        self._ensemble_busy = False
        self._ensemble_stale = False
        self._keep_view = False
        self._last_render_ms = 0.0

        self._build_central()
        self._build_docks()
        self._build_shortcuts()

        self.popovers = RichPopoverManager(self.workbench, self.registry, use_web=use_web)
        for badge in (
            *self.sidebar.badges,
            *self.kpis.badges,
            *(header.badge for header in self.chart.headers),
        ):
            self.popovers.register(badge)
        self.popovers.linkActivated.connect(self.open_documentation)

        self._recalc_timer = QtCore.QTimer(self)
        self._recalc_timer.setSingleShot(True)
        self._recalc_timer.setInterval(0)
        self._recalc_timer.timeout.connect(self.recalculate)

        self._sync_dataset_controls()
        self.recalculate()

    # -- construction --------------------------------------------------------------
    def _build_central(self) -> None:
        self.workbench = QtWidgets.QWidget(self)
        self.workbench.setObjectName("workbench")
        self.workbench.setAttribute(QtCore.Qt.WidgetAttribute.WA_StyledBackground, True)

        bar = QtWidgets.QWidget(self.workbench)
        bar.setObjectName("topBar")
        bar.setAttribute(QtCore.Qt.WidgetAttribute.WA_StyledBackground, True)
        bar.setFixedHeight(TOP_BAR_HEIGHT)
        brand = QtWidgets.QLabel("CPPI", bar)
        brand.setObjectName("brand")
        brand_accent = QtWidgets.QLabel("-Lab", bar)
        brand_accent.setObjectName("brandAccent")
        self.dataset_combo = ElasticComboBox(bar)
        self.dataset_combo.setAccessibleName("Dataset")
        self.dataset_combo.activated.connect(self._on_dataset_chosen)
        self.import_button = QtWidgets.QPushButton("Import CSV / Parquet…", bar)
        self.import_button.clicked.connect(self._choose_import)
        self.sync_button = QtWidgets.QPushButton("Coinbase Sync…", bar)
        self.sync_button.clicked.connect(self._open_sync_dialog)
        self.scenario_combo = ElasticComboBox(bar)
        self.scenario_combo.setAccessibleName("Scenario")
        self.scenario_combo.addItem("Scenario…")
        for scenario in SCENARIOS:
            self.scenario_combo.addItem(scenario.name)
        self.scenario_combo.activated.connect(self._on_scenario_chosen)
        self.norm_chip = QtWidgets.QLabel("", bar)
        self.norm_chip.setObjectName("normChip")
        self.norm_chip.setFixedHeight(20)
        self.norm_chip.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.norm_chip.hide()
        self.status_chip = AdaptiveLabel(bar)
        self.status_chip.setObjectName("statusChip")
        # The read-out takes what the selectors leave; _fit_top_bar decides how much that is.
        self._status_widths: list[int] = []
        self.status_chip.setMinimumWidth(0)
        self.status_chip.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Preferred
        )
        self.status_chip.setAlignment(
            QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter
        )
        self.ledger_button = self._dock_button("Ledger", "Tabular data ledger (Ctrl+L)")
        self.docs_button = self._dock_button("Documentation", "Documentation workbench (F1)")
        row = QtWidgets.QHBoxLayout(bar)
        row.setContentsMargins(12, 0, 10, 0)
        row.setSpacing(8)
        row.addWidget(brand)
        row.addSpacing(-8)
        row.addWidget(brand_accent)
        row.addSpacing(10)
        row.addWidget(self.dataset_combo)
        row.addWidget(self.import_button)
        row.addWidget(self.sync_button)
        row.addSpacing(6)
        row.addWidget(self.scenario_combo)
        row.addWidget(self.norm_chip)
        row.addWidget(self.status_chip, 10)
        row.addSpacing(6)
        row.addWidget(self.ledger_button)
        row.addWidget(self.docs_button)
        self._top_bar = bar
        self._top_row = row
        self._top_bar_watcher = ResizeWatcher(bar)
        self._top_bar_watcher.resized.connect(self._fit_top_bar)

        self.sidebar = ControlSidebar(self.workbench)
        self.sidebar.paramChanged.connect(self._on_param_changed)
        self.sidebar.resetRequested.connect(self._on_reset_one)
        self.sidebar.resetAllRequested.connect(self.reset_all)
        self.kpis = KpiRibbon(self.workbench)
        self.chart = ChartWorkspace(self.workbench)
        self.chart.barSelected.connect(self._on_chart_bar_selected)
        self.chart.normalizeRequested.connect(lambda: self._on_param_changed("normalize", True))
        self.chart.scenarioRequested.connect(lambda: self.apply_scenario(CRASH_SCENARIO))

        right = QtWidgets.QVBoxLayout()
        right.setContentsMargins(0, 0, 0, 0)
        right.setSpacing(0)
        right.addWidget(self.kpis)
        right.addWidget(self.chart, 1)
        body = QtWidgets.QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)
        body.addWidget(self.sidebar)
        body.addLayout(right, 1)
        outer = QtWidgets.QVBoxLayout(self.workbench)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        outer.addWidget(bar)
        outer.addLayout(body, 1)

    def _dock_button(self, text: str, tip: str) -> QtWidgets.QToolButton:
        button = QtWidgets.QToolButton(self)
        button.setText(text)
        button.setToolTip(tip)
        button.setCheckable(True)
        button.setPopupMode(QtWidgets.QToolButton.ToolButtonPopupMode.MenuButtonPopup)
        button.setStyleSheet(
            "QToolButton { padding: 4px 10px; border: 1px solid #243044; border-radius: 4px;"
            " background: #162033; } QToolButton:checked { border-color: #38BDF8; color: #38BDF8; }"
            " QToolButton::menu-button { border: none; width: 14px; }"
        )
        return button

    def _build_docks(self) -> None:
        flag = ads.CDockManager.eConfigFlag
        ads.CDockManager.setConfigFlag(flag.DisableStylesheet, True)
        ads.CDockManager.setConfigFlag(flag.OpaqueSplitterResize, True)
        ads.CDockManager.setConfigFlag(flag.FocusHighlighting, False)
        ads.CDockManager.setConfigFlag(flag.DockAreaHasTabsMenuButton, False)
        ads.CDockManager.setConfigFlag(flag.FloatingContainerHasWidgetTitle, True)
        # On X11 the floating windows carry the docking system's own title bar.
        # Dragging is then handled entirely inside the application, so dropping
        # onto the docking compass does not depend on how a particular window
        # manager reports a move it performs itself.
        own_title_bar = getattr(flag, "FloatingContainerForceQWidgetTitleBar", None)
        if own_title_bar is not None:
            on_x11 = QtGui.QGuiApplication.platformName() == "xcb"
            ads.CDockManager.setConfigFlag(own_title_bar, on_x11)
        # TSD 4.2: a docked panel can also be collapsed into an auto-hide side bar.
        auto_hide = ads.CDockManager.eAutoHideFlag
        ads.CDockManager.setAutoHideConfigFlag(auto_hide.AutoHideFeatureEnabled, True)
        ads.CDockManager.setAutoHideConfigFlag(auto_hide.DockAreaHasAutoHideButton, True)
        ads.CDockManager.setAutoHideConfigFlag(auto_hide.AutoHideCloseButtonCollapsesDock, False)
        self.dock_manager = ads.CDockManager(self)
        self._title_press_fix = TitleBarPressFix(self)
        self._floating_grips: set[ResizeGrips] = set()
        self.dock_manager.floatingWidgetCreated.connect(self._prepare_floating_container)

        feature = ads.CDockWidget.DockWidgetFeature
        self.central_dock = ads.CDockWidget(self.dock_manager, "Workbench")
        self.central_dock.setWidget(self.workbench)
        self.central_dock.setFeature(feature.NoTab, True)
        self.dock_manager.setCentralWidget(self.central_dock)

        self.ledger = LedgerWidget()
        self.ledger.barActivated.connect(self._on_ledger_bar_activated)
        self.ledger.exported.connect(
            lambda path, rows: self.statusBar().showMessage(
                f"Exported {rows:,} rows to {path}", 8000
            )
        )
        self.ledger_dock = self._make_dock(
            "Tabular Data Ledger", self.ledger, self.ledger_button, QtCore.QSize(1060, 470)
        )

        self.docs: QtWidgets.QWidget
        if self._use_web:
            from cppi_lab.ui.doc_workbench import DocWorkbench

            self.docs = DocWorkbench(self.registry)
        else:
            self.docs = QtWidgets.QLabel("Documentation requires QtWebEngine.")
        self.docs_dock = self._make_dock(
            "Documentation Workbench", self.docs, self.docs_button, QtCore.QSize(1180, 720)
        )
        self.statusBar().setSizeGripEnabled(False)
        self.statusBar().setFixedHeight(20)

    def _make_dock(
        self,
        title: str,
        widget: QtWidgets.QWidget,
        button: QtWidgets.QToolButton,
        float_size: QtCore.QSize,
    ) -> ads.CDockWidget:
        dock = ads.CDockWidget(self.dock_manager, title)
        dock.setWidget(widget)
        dock.setMinimumSizeHintMode(
            ads.CDockWidget.eMinimumSizeHintMode.MinimumSizeHintFromDockWidget
        )
        dock.setMinimumSize(320, 180)
        dock._opened_once = False
        dock._float_size = float_size
        # Register with the manager, then keep it closed until first requested.
        self.dock_manager.addDockWidget(ads.DockWidgetArea.BottomDockWidgetArea, dock)
        dock.toggleView(False)
        dock.viewToggled.connect(button.setChecked)
        button.clicked.connect(partial(self._on_dock_button, dock, button))
        menu = QtWidgets.QMenu(button)
        menu.addAction("Float (always on top)", partial(self.float_dock, dock))
        menu.addSeparator()
        for label, area in (
            ("Dock Left", ads.DockWidgetArea.LeftDockWidgetArea),
            ("Dock Right", ads.DockWidgetArea.RightDockWidgetArea),
            ("Dock Bottom", ads.DockWidgetArea.BottomDockWidgetArea),
        ):
            menu.addAction(label, partial(self.dock_to, dock, area))
        menu.addSeparator()
        menu.addAction("Hide", partial(self.set_dock_visible, dock, False))
        button.setMenu(menu)
        return dock

    def _build_shortcuts(self) -> None:
        def bind(sequence: str, slot: object) -> None:
            action = QtGui.QAction(self)
            action.setShortcut(QtGui.QKeySequence(sequence))
            action.triggered.connect(slot)
            self.addAction(action)

        bind("Ctrl+L", lambda: self.set_dock_visible(self.ledger_dock, self.ledger_dock.isClosed()))
        bind("F1", lambda: self.set_dock_visible(self.docs_dock, self.docs_dock.isClosed()))
        bind("Ctrl+0", self.chart.reset_view)
        bind("Ctrl+Shift+R", self.reset_all)
        bind("Escape", lambda: self.popovers.close_all())

    # -- docking -------------------------------------------------------------------
    def _on_dock_button(
        self, dock: ads.CDockWidget, button: QtWidgets.QToolButton, *_: object
    ) -> None:
        # Qt may or may not pass the ``checked`` flag to a partial; read the button.
        self.set_dock_visible(dock, button.isChecked())

    def set_dock_visible(self, dock: ads.CDockWidget, visible: bool) -> None:
        """Show or hide a secondary window; the first show floats it (TSD 4.2)."""
        if visible and not dock._opened_once:
            self.float_dock(dock)
            return
        dock.toggleView(visible)
        if visible and dock.isFloating():
            self._apply_floating_flags(dock)

    def float_dock(self, dock: ads.CDockWidget) -> None:
        """Tear a dock widget off into an always-on-top floating tool window."""
        dock._opened_once = True
        if dock.isClosed():
            dock.toggleView(True)
        if not dock.isFloating():
            dock.setFloating()
        self._finish_floating(dock)
        # The docking system and the window manager both finish their part of the
        # float asynchronously; settle flags, visibility and placement after them.
        QtCore.QTimer.singleShot(0, partial(self._finish_floating, dock))
        QtCore.QTimer.singleShot(80, partial(self._finish_floating, dock))

    def _finish_floating(self, dock: ads.CDockWidget) -> None:
        # Called again from timers; the window may have been closed in between.
        if not isValid(dock) or not dock.isFloating() or dock.isClosed():
            return
        self._apply_floating_flags(dock)
        container = dock.floatingDockContainer()
        if container is None:
            return
        if not container.isVisible():
            container.show()
        self._place_floating(dock)

    def _place_floating(self, dock: ads.CDockWidget) -> None:
        """Size a floating window and put it over the right part of the main window."""
        container = dock.floatingDockContainer()
        if container is None or not dock.isFloating():
            return
        size: QtCore.QSize = dock._float_size
        container.resize(size.width(), size.height() + 30)
        origin = self.mapToGlobal(QtCore.QPoint(0, 0))
        offset = 70 if dock is self.ledger_dock else 130
        container.move(
            origin.x() + max(20, self.width() - size.width() - offset), origin.y() + offset
        )
        container.raise_()

    def dock_to(self, dock: ads.CDockWidget, area: ads.DockWidgetArea) -> None:
        """Dock a secondary window into a split panel of the main window."""
        dock._opened_once = True
        self.dock_manager.addDockWidget(area, dock)
        dock.toggleView(True)
        # Size the panel once the new splitter has been laid out.
        QtCore.QTimer.singleShot(0, partial(self._size_docked_panel, dock, area))
        QtCore.QTimer.singleShot(120, partial(self._size_docked_panel, dock, area))

    def _size_docked_panel(self, dock: ads.CDockWidget, area: ads.DockWidgetArea) -> None:
        """Give a freshly docked panel a useful share; the chart reflows into the rest."""
        if not isValid(dock):  # called from a timer: the window may be gone by now
            return
        target = dock.dockAreaWidget()
        if target is None or dock.isFloating() or dock.isClosed():
            return
        if area == ads.DockWidgetArea.BottomDockWidgetArea:
            total = self.dock_manager.height()
            panel = min(max(220, total * 30 // 100), 380)
            sizes = [total - panel, panel]
        else:
            total = self.dock_manager.width()
            panel = min(max(420, total * 34 // 100), 700)
            left = area == ads.DockWidgetArea.LeftDockWidgetArea
            sizes = [panel, total - panel] if left else [total - panel, panel]
        self.dock_manager.setSplitterSizes(target, sizes)

    def _prepare_floating_container(self, container: ads.CFloatingDockContainer) -> None:
        """Give a floating window with the docking system's own title bar its handles.

        Such a window is frameless: it gets edge and corner grips for resizing
        and the title-bar correction described in :class:`TitleBarPressFix`.
        A window with a native title bar is left to the window manager.
        """
        native = getattr(container, "hasNativeTitleBar", None)
        if native is None or native() or container.property("cppiHandles"):
            return
        container.setProperty("cppiHandles", True)
        # The grips are Python objects owned by a window the docking system
        # creates and destroys; without a reference of our own their Python
        # side (and with it their behaviour) can be collected while the window
        # is still there.
        grips = ResizeGrips(container, QtGui.QColor(theme.FLOAT_BORDER))
        self._floating_grips.add(grips)
        container.destroyed.connect(partial(self._floating_grips.discard, grips))
        bar = container.titleBarWidget()
        if bar is not None:
            bar.installEventFilter(self._title_press_fix)

    def _apply_floating_flags(self, dock: ads.CDockWidget) -> None:
        """Window flags of TSD section 4.2 for a floating container."""
        container = dock.floatingDockContainer()
        if container is None:
            return
        self._prepare_floating_container(container)
        wanted = floating_window_flags(QtGui.QGuiApplication.platformName())
        current = container.windowFlags()
        # Keep the docking system's own hints (frameless, close button, ...) and
        # replace only the window type.
        target = (current & ~QtCore.Qt.WindowType.WindowType_Mask) | wanted
        if current == target:
            return
        geometry = container.geometry()
        container.setWindowFlags(target)
        container.setWindowTitle(f"{dock.windowTitle()} - CPPI-Lab")
        container.setGeometry(geometry)
        container.show()

    def open_documentation(self, volume: str, anchor: str) -> None:
        """Open or focus the documentation workbench at an exact section."""
        self.set_dock_visible(self.docs_dock, True)
        if hasattr(self.docs, "open_at"):
            self.docs.open_at(volume, anchor)
        container = self.docs_dock.floatingDockContainer()
        if container is not None:
            container.raise_()
            container.activateWindow()

    # -- datasets ------------------------------------------------------------------
    def _sync_dataset_controls(self) -> None:
        self.dataset_combo.blockSignals(True)
        self.dataset_combo.clear()
        for data in self.datasets:
            self.dataset_combo.addItem(data.name)
        self.dataset_combo.setCurrentIndex(self.datasets.index(self.controller.data))
        self.dataset_combo.blockSignals(False)
        data = self.controller.data
        lo, hi = self.controller.start_bounds_ms
        default_start = min(max(DEFAULT_START_MS, lo), hi)
        self.sidebar.set_start_bounds(lo, hi, default_start)
        self.scenario_combo.setEnabled(data is self.datasets[0])
        first = datetime.fromtimestamp(lo / 1000, UTC).strftime("%Y-%m-%d")
        last = datetime.fromtimestamp(hi / 1000, UTC).strftime("%Y-%m-%d")
        flow = "taker split synthesised from CLV" if data.synthesized_flow else "real taker split"
        self.dataset_combo.setToolTip(
            f"{data.source}\n{data.n:,} bars, {first} to {last} UTC, "
            f"{data.missing_minutes:,} missing minutes, {flow}"
        )

    def add_dataset(self, data: MarketData) -> None:
        """Register a dataset and make it the active one."""
        self.datasets.append(data)
        self.controller.set_data(data)
        self._sync_dataset_controls()
        self._keep_view = False
        self.recalculate()
        self.statusBar().showMessage(
            f"Loaded {data.name}: {data.n:,} bars"
            + (" (CLV-synthesised flow)" if data.synthesized_flow else ""),
            8000,
        )

    def import_file(self, path: str | Path) -> bool:
        """Import a CSV/Parquet file (FR-1.2); returns ``False`` on a format error."""
        try:
            data = load_file(path)
        except DataFormatError as exc:
            QtWidgets.QMessageBox.warning(self, "Import failed", str(exc))
            return False
        self.add_dataset(data)
        return True

    def _choose_import(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Import historical data",
            "",
            "Market data (*.csv *.tsv *.txt *.parquet *.pq);;All files (*)",
        )
        if path:
            self.import_file(path)

    def _open_sync_dialog(self) -> None:
        dialog = CoinbaseSyncDialog(self)
        if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted and dialog.data is not None:
            self.add_dataset(dialog.data)

    def _on_dataset_chosen(self, index: int) -> None:
        if 0 <= index < len(self.datasets) and self.datasets[index] is not self.controller.data:
            self.controller.set_data(self.datasets[index])
            self._sync_dataset_controls()
            self._keep_view = False
            self.recalculate()

    def _on_scenario_chosen(self, index: int) -> None:
        if index >= 1:
            self.apply_scenario(SCENARIOS[index - 1])
        self.scenario_combo.setCurrentIndex(0)

    def apply_scenario(self, scenario: Scenario) -> None:
        """Jump to a curated window; strategy parameters are left untouched."""
        if self.controller.data is not self.datasets[0]:
            self.controller.set_data(self.datasets[0])
            self._sync_dataset_controls()
        params = replace(
            self.controller.params,
            start_ms=scenario.start_ms,
            duration_days=scenario.days,
            normalize=scenario.normalize,
        )
        self.controller.set_params(params)
        self._keep_view = False
        self.recalculate()

    # -- parameters ----------------------------------------------------------------
    def _on_param_changed(self, key: str, value: ParamValue) -> None:
        before = self.controller.params
        after = self.controller.update(key, value)
        self._keep_view = self._same_window(before, after)
        self._recalc_timer.start()

    def _on_reset_one(self, key: str) -> None:
        before = self.controller.params
        if key == "start_ms":
            lo, hi = self.controller.start_bounds_ms
            after = self.controller.update(key, min(max(DEFAULT_START_MS, lo), hi))
        elif key == "normalize":
            defaults = default_params()
            after = self.controller.set_params(
                replace(before, normalize=defaults.normalize, anchor_price=defaults.anchor_price)
            )
        else:
            after = self.controller.reset(key)
        self._keep_view = self._same_window(before, after)
        self._recalc_timer.start()

    def reset_all(self) -> None:
        """Reset every parameter to its PRD default (FR-2.2)."""
        self.controller.reset()
        self._keep_view = False
        self._recalc_timer.start()

    @staticmethod
    def _same_window(a: SimParams, b: SimParams) -> bool:
        return (a.start_ms, a.duration_days) == (b.start_ms, b.duration_days)

    # -- recalculation -------------------------------------------------------------
    def recalculate(self) -> RunOutput:
        """Re-run the back-test and refresh every view (FR-2.3)."""
        self._recalc_timer.stop()
        run = self.controller.run()
        started = time.perf_counter()
        self.run = run
        params = run.params
        self.sidebar.set_params(params)
        if not params.mc_fan or not self._keep_view:
            self.ensemble = None
        self.chart.set_run(run, keep_view=self._keep_view)
        if not params.mc_fan:
            self.chart.set_ensemble(None)
        self.kpis.set_kpis(compute_kpis(run, self.ensemble if params.mc_fan else None))
        self.ledger.set_result(run.result)
        self._last_render_ms = (time.perf_counter() - started) * 1e3
        self._update_status(run)
        self._keep_view = True
        if params.mc_fan:
            self._request_ensemble()
        self.recalculated.emit(run)
        return run

    def _fit_top_bar(self) -> None:
        """Share the top bar between the two selectors and the status read-out.

        The read-out has four wordings, from every timing down to the bar
        count alone.  It gets the longest one for which there is room once
        the scenario selector has given up what it can spare, so it is never
        left with a fragment of a word; the dataset selector keeps its full
        name unless the window is too narrow even for the bar count.
        """
        dataset, scenario = self.dataset_combo, self.scenario_combo
        # What the three share: the bar less everything else in it.  The layout's
        # minimum is exactly that "everything else" plus the minimum widths of
        # the two selectors (the read-out's own minimum is zero).
        self._top_row.invalidate()
        others = (
            self._top_row.minimumSize().width()
            - dataset.minimumSizeHint().width()
            - scenario.minimumSizeHint().width()
        )
        available = self._top_bar.width() - others
        natural_dataset = dataset.sizeHint().width()
        natural_scenario = scenario.sizeHint().width()
        least_dataset = min(natural_dataset, DATASET_MIN_WIDTH)
        least_scenario = min(natural_scenario, SCENARIO_MIN_WIDTH)
        # The dataset name is worth more than a longer read-out: the read-out is
        # shortened first, and the dataset selector gives way only when even the
        # bar count alone would not fit beside it.
        beside_full_name = available - natural_dataset - least_scenario
        most = available - least_dataset - least_scenario
        target = next(
            (width for width in self._status_widths if width <= beside_full_name),
            min(self._status_widths[-1], max(0, most)) if self._status_widths else 0,
        )
        for_scenario = min(
            natural_scenario, max(least_scenario, available - natural_dataset - target)
        )
        for_dataset = min(natural_dataset, max(least_dataset, available - for_scenario - target))
        if dataset.maximumWidth() != for_dataset:
            dataset.setMaximumWidth(for_dataset)
        if scenario.maximumWidth() != for_scenario:
            scenario.setMaximumWidth(for_scenario)

    def _update_status(self, run: RunOutput) -> None:
        params = run.params
        timings = run.timings
        bars = f"{run.window.n:,} bars"
        recalc = f"recalc {timings.total_ms:.1f} ms"
        refresh = f"refresh {self._last_render_ms:.1f} ms"
        self.status_chip.set_wordings(
            f"{bars} · {recalc} (signals {timings.signals_ms:.1f} · "
            f"kernel {timings.kernel_ms:.1f}) · {refresh}",
            f"{bars} · {recalc} · {refresh}",
            f"{bars} · {recalc}",
            bars,
        )
        # Room each wording needs.  The timings are measured as three-digit values
        # so the bar does not re-flow whenever one of them gains or loses a digit.
        metrics = self.status_chip.fontMetrics()
        self._status_widths = [
            metrics.horizontalAdvance(_TIMING.sub("888.8", text)) + 8
            for text in self.status_chip.wordings()
        ]
        if params.normalize:
            self.norm_chip.setText(f"RESCALED ×{run.window.price_scale:.4f}")
            self.norm_chip.setToolTip(
                "FR-1.3: every price in this window is multiplied by this factor so the first "
                f"open equals the {params.anchor_price:,.0f} USD anchor.  Returns, volatility "
                "and volume imbalance are unchanged."
            )
            self.norm_chip.show()
        else:
            self.norm_chip.hide()
        self._fit_top_bar()

    def _request_ensemble(self) -> None:
        if not self._async_ensemble:
            self._apply_ensemble(self.controller.generation, self.controller.run_ensemble())
            return
        if self._ensemble_busy:
            self._ensemble_stale = True
            return
        self._ensemble_busy = True
        self._ensemble_stale = False
        task = _EnsembleTask(
            self.controller.generation, self.controller.ensemble_inputs(), self._ensemble_signals
        )
        QtCore.QThreadPool.globalInstance().start(task)

    def _on_ensemble_finished(self, generation: int, result: EnsembleResult) -> None:
        self._ensemble_busy = False
        if self._ensemble_stale and self.controller.params.mc_fan:
            self._request_ensemble()
            return
        self._apply_ensemble(generation, result)

    def _apply_ensemble(self, generation: int, result: EnsembleResult) -> None:
        if generation != self.controller.generation or self.run is None:
            return
        if not self.controller.params.mc_fan:
            return
        self.ensemble = result
        self.chart.set_ensemble(result)
        self.kpis.set_kpis(compute_kpis(self.run, result))
        self.ensembleApplied.emit(result)

    # -- crosshair synchronisation -----------------------------------------------------
    def _on_chart_bar_selected(self, bar: int) -> None:
        self.ledger.select_bar(bar)

    def _on_ledger_bar_activated(self, bar: int) -> None:
        self.chart.set_selected_bar(bar)
        self.chart.centre_on_bar(bar)

    # -- shutdown ------------------------------------------------------------------
    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self.popovers.close_all()
        QtCore.QThreadPool.globalInstance().waitForDone(5000)
        if hasattr(self.docs, "shutdown"):
            self.docs.shutdown()
        self.dock_manager.deleteLater()
        super().closeEvent(event)
