"""Dual documentation viewer with live counterpart preview (PRD FR-6, TSD section 5.2).

A ``QSplitter`` hosts two ``QWebEngineView`` instances:

* ``primary_view`` (65 %) shows the volume the user is reading;
* ``preview_view`` (35 %) always shows the *other* volume.

The page script in the primary view reports, over a ``QWebChannel`` bridge,
which ``<section data-concept-id>`` is at the top of the reading viewport and
which ``cppidoc://`` link the mouse is over.  :class:`DocWorkbench` answers by
scrolling the preview to the matching section and highlighting it, so the
counterpart of whatever is being read is always in view.
"""

from __future__ import annotations

import json

from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtWebChannel import QWebChannel
from PySide6.QtWebEngineCore import QWebEnginePage, QWebEngineProfile
from PySide6.QtWebEngineWidgets import QWebEngineView

from cppi_lab.ui import theme
from cppi_lab.ui.doc_scheme import (
    SCHEME,
    VOLUMES,
    DocRegistry,
    counterpart,
    doc_url,
    install_handler,
    parse_doc_url,
)
from cppi_lab.ui.widgets import ElidedLabel

PRIMARY_SHARE = 65
PREVIEW_SHARE = 35
#: Below this width the two viewers are stacked instead of placed side by side.
STACK_BELOW_PX = 820

_profile: QWebEngineProfile | None = None


def shared_profile(registry: DocRegistry) -> QWebEngineProfile:
    """The off-the-record profile every documentation and popover page uses."""
    global _profile
    if _profile is None:
        app = QtCore.QCoreApplication.instance()
        _profile = QWebEngineProfile(app)
        install_handler(_profile, registry)
    return _profile


def release_profile() -> None:
    """Drop the shared profile (call after every page that used it is gone)."""
    global _profile
    if _profile is not None:
        _profile.deleteLater()
        _profile = None


class DocPage(QWebEnginePage):
    """A page that only ever navigates inside ``cppidoc://``."""

    def acceptNavigationRequest(
        self, url: QtCore.QUrl | str, kind: QWebEnginePage.NavigationType, is_main_frame: bool
    ) -> bool:
        target = QtCore.QUrl(url)
        accepted = target.scheme() == SCHEME and parse_doc_url(target) is not None
        if accepted and kind != QWebEnginePage.NavigationType.NavigationTypeTyped:
            self.userNavigation.emit()  # a link, Back or Forward - not a setUrl() call
        return accepted

    def javaScriptConsoleMessage(
        self,
        level: QWebEnginePage.JavaScriptConsoleMessageLevel,
        message: str,
        line: int,
        source: str,
    ) -> None:
        if level == QWebEnginePage.JavaScriptConsoleMessageLevel.ErrorMessageLevel:
            self.consoleError.emit(f"{source}:{line}: {message}")

    consoleError = QtCore.Signal(str)
    userNavigation = QtCore.Signal()


class DocSyncBridge(QtCore.QObject):
    """Python end of the ``QWebChannel`` used by the primary view's script."""

    sectionFocused = QtCore.Signal(str)
    linkHovered = QtCore.Signal(str, str)
    linkLeft = QtCore.Signal()
    ready = QtCore.Signal(str)
    previewCollapsedChanged = QtCore.Signal(bool)

    def __init__(self, registry: DocRegistry, parent: QtCore.QObject | None = None) -> None:
        super().__init__(parent)
        self._registry = registry
        self._collapsed = False

    @QtCore.Slot(str)
    def notifySectionFocused(self, concept_id: str) -> None:
        self.sectionFocused.emit(concept_id)

    @QtCore.Slot(str, str)
    def notifyLinkHovered(self, volume: str, anchor: str) -> None:
        self.linkHovered.emit(volume, anchor)

    @QtCore.Slot()
    def notifyLinkLeft(self) -> None:
        self.linkLeft.emit()

    @QtCore.Slot(str)
    def notifyReady(self, volume: str) -> None:
        self.ready.emit(volume)

    @QtCore.Slot(str, str, result=str)
    def sectionSnippet(self, volume: str, anchor: str) -> str:
        """Title and summary of a section, for the floating snippet card."""
        return json.dumps(self._registry.snippet(volume, anchor))

    def _get_collapsed(self) -> bool:
        return self._collapsed

    def set_collapsed(self, collapsed: bool) -> None:
        if collapsed != self._collapsed:
            self._collapsed = collapsed
            self.previewCollapsedChanged.emit(collapsed)

    previewCollapsed = QtCore.Property(bool, _get_collapsed, notify=previewCollapsedChanged)


class DocWorkbench(QtWidgets.QWidget):
    """Toolbar, primary viewer and counterpart preview."""

    #: Emitted when the primary view reports a new section at the reading line.
    sectionFocused = QtCore.Signal(str)
    #: Emitted once the preview has been scrolled to a concept: ``(volume, concept)``.
    previewSynced = QtCore.Signal(str, str)
    #: Emitted when the primary view finished loading a volume.
    primaryLoaded = QtCore.Signal(str)
    volumeChanged = QtCore.Signal(str)

    def __init__(self, registry: DocRegistry, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._registry = registry
        self._volume = "theory"
        self._focused = registry.concept_ids()[0]
        self._peeking = False
        self._preview_loaded_volume = ""
        self._preview_ready = False
        self._pending_preview: str | None = None
        #: Target of the last ``open_at`` that has not been confirmed by the view yet.
        self._wanted: tuple[str, str] | None = None
        profile = shared_profile(registry)

        self.primary_view = QWebEngineView(self)
        self.primary_page = DocPage(profile, self.primary_view)
        self.primary_view.setPage(self.primary_page)
        self.preview_view = QWebEngineView(self)
        self.preview_page = DocPage(profile, self.preview_view)
        self.preview_view.setPage(self.preview_page)
        for page in (self.primary_page, self.preview_page):
            page.setBackgroundColor(QtGui.QColor(theme.CANVAS))
        for view in (self.primary_view, self.preview_view):
            view.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.NoContextMenu)

        self.bridge = DocSyncBridge(registry, self)
        self.channel = QWebChannel(self)
        self.channel.registerObject("bridge", self.bridge)
        self.primary_page.setWebChannel(self.channel)
        self.bridge.sectionFocused.connect(self._on_section_focused)
        self.bridge.linkHovered.connect(self._on_link_hovered)
        self.bridge.linkLeft.connect(self._on_link_left)
        self.primary_page.userNavigation.connect(self._forget_wanted)

        self.splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal, self)
        self.splitter.addWidget(self.primary_view)
        self.splitter.addWidget(self.preview_view)
        self.splitter.setStretchFactor(0, PRIMARY_SHARE)
        self.splitter.setStretchFactor(1, PREVIEW_SHARE)
        self.splitter.setChildrenCollapsible(True)
        self.splitter.setSizes([PRIMARY_SHARE * 10, PREVIEW_SHARE * 10])

        self._build_toolbar()
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.splitter, 1)
        self.resize(1040, 620)

        self.primary_view.urlChanged.connect(self._on_primary_url)
        self.primary_view.loadFinished.connect(self._on_primary_loaded)
        self.preview_view.loadFinished.connect(self._on_preview_loaded)
        self.primary_page.findTextFinished.connect(self._on_find_finished)
        self.open_at("theory", self._focused)

    # -- toolbar -------------------------------------------------------------------
    def _build_toolbar(self) -> None:
        self.toolbar = QtWidgets.QWidget(self)
        self.toolbar.setObjectName("docBar")
        self.toolbar.setAttribute(QtCore.Qt.WidgetAttribute.WA_StyledBackground, True)
        self.back_button = QtWidgets.QPushButton("←", self.toolbar)
        self.back_button.setToolTip("Back")
        self.back_button.setAccessibleName("Back")
        self.back_button.setFixedWidth(34)
        self.back_button.clicked.connect(self.primary_view.back)
        self.forward_button = QtWidgets.QPushButton("→", self.toolbar)
        self.forward_button.setToolTip("Forward")
        self.forward_button.setAccessibleName("Forward")
        self.forward_button.setFixedWidth(34)
        self.forward_button.clicked.connect(self.primary_view.forward)
        self.switch_button = QtWidgets.QPushButton("", self.toolbar)
        self.switch_button.setObjectName("primary")
        self.switch_button.setToolTip("Switch Primary Volume (Theory ⇄ Technical)")
        self.switch_button.setAccessibleName("Switch Primary Volume")
        self.switch_button.clicked.connect(self.switch_volume)
        self.volume_label = ElidedLabel("", self.toolbar)
        self.volume_label.setObjectName("paneTitle")
        self.search_edit = QtWidgets.QLineEdit(self.toolbar)
        self.search_edit.setPlaceholderText("Search this volume…")
        self.search_edit.setClearButtonEnabled(True)
        self.search_edit.setMinimumWidth(96)
        self.search_edit.setMaximumWidth(230)
        self.search_edit.setAccessibleName("Search")
        self.search_edit.textChanged.connect(lambda text: self.find_text(text))
        self.search_edit.returnPressed.connect(lambda: self.find_text(self.search_edit.text()))
        self.search_status = QtWidgets.QLabel("", self.toolbar)
        self.search_status.setProperty("role", "muted")
        self.search_status.setMinimumWidth(52)
        self.preview_button = QtWidgets.QPushButton("Counterpart", self.toolbar)
        self.preview_button.setCheckable(True)
        self.preview_button.setChecked(True)
        self.preview_button.setToolTip("Toggle Counterpart Preview Split")
        self.preview_button.setAccessibleName("Toggle Counterpart Preview Split")
        self.preview_button.toggled.connect(self.set_preview_visible)
        row = QtWidgets.QHBoxLayout(self.toolbar)
        row.setContentsMargins(8, 5, 8, 5)
        row.setSpacing(6)
        row.addWidget(self.back_button)
        row.addWidget(self.forward_button)
        row.addSpacing(4)
        row.addWidget(self.volume_label, 2)
        row.addWidget(self.switch_button)
        row.addWidget(self.search_edit, 3)
        row.addWidget(self.search_status)
        row.addWidget(self.preview_button)

    def _refresh_toolbar(self) -> None:
        history = self.primary_view.history()
        self.back_button.setEnabled(history.canGoBack())
        self.forward_button.setEnabled(history.canGoForward())
        other = counterpart(self._volume)
        self.volume_label.setText(self._registry.volume_title(self._volume))
        self.volume_label.setToolTip(self._registry.volume_title(self._volume))
        short = "Technical" if other == "tech" else "Theory"
        self.switch_button.setText(f"⇄ {short}")

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        self._fit_splitter()

    def _fit_splitter(self) -> None:
        """Side by side when wide; stacked when docked into a narrow side panel."""
        wanted = (
            QtCore.Qt.Orientation.Horizontal
            if self.width() >= STACK_BELOW_PX
            else QtCore.Qt.Orientation.Vertical
        )
        if self.splitter.orientation() != wanted:
            self.splitter.setOrientation(wanted)
            self._apply_shares()

    def _apply_shares(self) -> None:
        horizontal = self.splitter.orientation() == QtCore.Qt.Orientation.Horizontal
        total = max(1, self.splitter.width() if horizontal else self.splitter.height())
        self.splitter.setSizes([total * PRIMARY_SHARE // 100, total * PREVIEW_SHARE // 100])

    # -- navigation ----------------------------------------------------------------
    @property
    def volume(self) -> str:
        """The volume shown in the primary view."""
        return self._volume

    @property
    def focused_concept(self) -> str:
        return self._focused

    def open_at(self, volume: str, anchor: str = "") -> None:
        """Show ``volume`` in the primary view, scrolled to ``anchor``."""
        if volume not in VOLUMES:
            raise ValueError(f"unknown volume {volume!r}")
        if anchor:
            self._focused = anchor
        self._wanted = (volume, anchor)
        self.primary_view.setUrl(doc_url(volume, anchor))

    def _forget_wanted(self) -> None:
        self._wanted = None

    def _reached(self, target: tuple[str, str]) -> bool:
        current = parse_doc_url(self.primary_view.url())
        return (
            current is not None
            and current[0] == target[0]
            and (not target[1] or current[1] == target[1])
        )

    def switch_volume(self) -> None:
        """Swap primary and counterpart, staying on the same concept."""
        self.open_at(counterpart(self._volume), self._focused)

    def _on_primary_url(self, url: QtCore.QUrl) -> None:
        target = parse_doc_url(url)
        if target is None:
            return
        volume, anchor = target
        if anchor and self._registry.concept(anchor) is not None:
            self._focused = anchor
        if volume != self._volume or not self._preview_loaded_volume:
            self._volume = volume
            self.volumeChanged.emit(volume)
            self._load_preview(self._focused)
        elif anchor:
            self._sync_preview(self._focused)
        self._refresh_toolbar()

    def _on_primary_loaded(self, ok: bool) -> None:
        self._refresh_toolbar()
        wanted = self._wanted
        if wanted is not None:
            if self._reached(wanted):
                self._wanted = None
            else:
                # A load that was already in flight (typically the very first
                # page) finished after the request and replaced it: ask again.
                self.primary_view.setUrl(doc_url(*wanted))
                return
        if ok:
            self.primaryLoaded.emit(self._volume)

    # -- counterpart preview ---------------------------------------------------------
    def _load_preview(self, concept_id: str) -> None:
        volume = counterpart(self._volume)
        self._preview_ready = False
        self._pending_preview = concept_id
        self._preview_loaded_volume = volume
        self.preview_view.setUrl(doc_url(volume, concept_id))

    def _on_preview_loaded(self, ok: bool) -> None:
        self._preview_ready = ok
        if ok and self._pending_preview is not None:
            pending, self._pending_preview = self._pending_preview, None
            self._sync_preview(pending)

    def _sync_preview(self, concept_id: str, volume: str | None = None) -> None:
        """Scroll the preview to ``concept_id`` and highlight it."""
        target_volume = volume or counterpart(self._volume)
        if target_volume != self._preview_loaded_volume:
            # A hover peek at the primary's own volume: nothing to show there.
            return
        if not self._preview_ready:
            self._pending_preview = concept_id
            return
        script = f"window.cppiDoc ? window.cppiDoc.focusSection({json.dumps(concept_id)}) : false"
        self.preview_page.runJavaScript(
            script, 0, lambda ok: self.previewSynced.emit(target_volume, concept_id) if ok else None
        )

    def _on_section_focused(self, concept_id: str) -> None:
        if self._registry.concept(concept_id) is None:
            return
        self._focused = concept_id
        if self._wanted == (self._volume, concept_id):
            self._wanted = None  # the page confirms it arrived (same-document jump)
        self.sectionFocused.emit(concept_id)
        if not self._peeking:
            self._sync_preview(concept_id)

    def _on_link_hovered(self, volume: str, anchor: str) -> None:
        if not anchor:
            return
        self._peeking = True
        self._sync_preview(anchor, volume)

    def _on_link_left(self) -> None:
        if self._peeking:
            self._peeking = False
            self._sync_preview(self._focused)

    def set_preview_visible(self, visible: bool) -> None:
        """Show or collapse the counterpart preview pane."""
        self.preview_view.setVisible(visible)
        self.bridge.set_collapsed(not visible)
        if self.preview_button.isChecked() != visible:
            self.preview_button.setChecked(visible)
        if visible:
            self._apply_shares()
            self._sync_preview(self._focused)

    # -- search --------------------------------------------------------------------
    def find_text(self, text: str, backward: bool = False) -> None:
        """Highlight ``text`` in the primary volume and jump to the next match."""
        flags = QWebEnginePage.FindFlag(0)
        if backward:
            flags |= QWebEnginePage.FindFlag.FindBackward
        self.primary_page.findText(text, flags)
        if not text:
            self.search_status.setText("")

    def _on_find_finished(self, result: object) -> None:
        total = result.numberOfMatches()  # type: ignore[attr-defined]
        active = result.activeMatch()  # type: ignore[attr-defined]
        if not self.search_edit.text():
            self.search_status.setText("")
        elif total == 0:
            self.search_status.setText("no match")
        else:
            self.search_status.setText(f"{active} / {total}")

    def shutdown(self) -> None:
        """Release the web pages before the shared profile goes away."""
        for view, page in (
            (self.primary_view, self.primary_page),
            (self.preview_view, self.preview_page),
        ):
            view.setPage(None)  # type: ignore[arg-type]
            page.deleteLater()
