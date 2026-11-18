"""Hover-intent ``?`` popovers with rendered math and deep links (TSD section 4.3).

State machine, exactly as specified:

* hovering a ``?`` badge starts a 150 ms debounce, after which the popover
  opens 8 px to the right of the badge;
* leaving the badge starts a 300 ms grace timer; entering the popover within
  that time cancels it, so the cursor can travel into the popover to click a
  link or the pin button;
* when the grace timer expires with the cursor outside both the badge and
  the popover, the popover fades out and closes - unless it was pinned, in
  which case it stays as a movable floating card.

The popover body is a ``QWebEngineView`` showing a KaTeX page served over
``cppidoc://``.  When QtWebEngine is switched off (tests, restricted
environments) a ``QTextBrowser`` with the same text and links is used.
"""

from __future__ import annotations

from typing import Any

from PySide6 import QtCore, QtGui, QtWidgets

from cppi_lab.ui import theme
from cppi_lab.ui.doc_scheme import SCHEME, DocRegistry, parse_doc_url, popover_text

SHOW_DELAY_MS = 150
GRACE_MS = 300
FADE_MS = 120
ANCHOR_GAP_PX = 8
POPOVER_WIDTH = 392
POPOVER_MIN_BODY = 90
POPOVER_MAX_BODY = 620
_TITLE_BAR_HEIGHT = 24


class HelpBadge(QtWidgets.QLabel):
    """The small circular ``?`` icon placed next to every explained item."""

    hoverEntered = QtCore.Signal(object)
    hoverLeft = QtCore.Signal(object)
    clicked = QtCore.Signal(object)

    def __init__(self, popover_id: str, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__("?", parent)
        self.popover_id = popover_id
        self.setObjectName("helpBadge")
        self.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.setFixedSize(14, 14)
        self.setCursor(QtCore.Qt.CursorShape.WhatsThisCursor)
        self.setAccessibleName(f"Help: {popover_id}")

    def set_active(self, active: bool) -> None:
        self.setProperty("active", "true" if active else "false")
        self.style().unpolish(self)
        self.style().polish(self)

    def enterEvent(self, event: QtGui.QEnterEvent) -> None:
        self.hoverEntered.emit(self)
        super().enterEvent(event)

    def leaveEvent(self, event: QtCore.QEvent) -> None:
        self.hoverLeft.emit(self)
        super().leaveEvent(event)

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self.clicked.emit(self)
        super().mousePressEvent(event)


class _TitleBar(QtWidgets.QWidget):
    """Drag handle of a popover; dragging moves the card inside its host."""

    def __init__(self, popover: RichPopoverWidget) -> None:
        super().__init__(popover)
        self._popover = popover
        self._grab: QtCore.QPoint | None = None
        self.setObjectName("popoverTitleBar")
        self.setFixedHeight(_TITLE_BAR_HEIGHT)

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == QtCore.Qt.MouseButton.LeftButton and self._popover.pinned:
            self._grab = event.globalPosition().toPoint() - self._popover.pos()
            self.setCursor(QtCore.Qt.CursorShape.ClosedHandCursor)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        if self._grab is not None:
            self._popover.move_within_host(event.globalPosition().toPoint() - self._grab)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        self._grab = None
        self.setCursor(
            QtCore.Qt.CursorShape.OpenHandCursor
            if self._popover.pinned
            else QtCore.Qt.CursorShape.ArrowCursor
        )


class RichPopoverWidget(QtWidgets.QFrame):
    """One popover card: title bar with pin/close, and a rich body."""

    hoverEntered = QtCore.Signal(object)
    hoverLeft = QtCore.Signal(object)
    linkActivated = QtCore.Signal(str, str)
    pinToggled = QtCore.Signal(object, bool)
    closeRequested = QtCore.Signal(object)

    def __init__(self, host: QtWidgets.QWidget, registry: DocRegistry, use_web: bool) -> None:
        super().__init__(host)
        self._registry = registry
        self._use_web = use_web
        self.popover_id = ""
        self.pinned = False
        self.setObjectName("popover")
        self.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setFixedWidth(POPOVER_WIDTH)
        self.hide()

        self._bar = _TitleBar(self)
        self._caption = QtWidgets.QLabel("", self._bar)
        self._caption.setObjectName("popoverTitle")
        self._caption.setProperty("role", "muted")
        self.pin_button = QtWidgets.QToolButton(self._bar)
        self.pin_button.setObjectName("popoverButton")
        self.pin_button.setText("Pin")
        self.pin_button.setCheckable(True)
        self.pin_button.setToolTip("Keep this card open as a movable floating card")
        self.pin_button.toggled.connect(self._on_pin)
        self.close_button = QtWidgets.QToolButton(self._bar)
        self.close_button.setObjectName("popoverButton")
        self.close_button.setText("✕")
        self.close_button.setToolTip("Close")
        self.close_button.clicked.connect(lambda: self.closeRequested.emit(self))
        bar_layout = QtWidgets.QHBoxLayout(self._bar)
        bar_layout.setContentsMargins(10, 3, 5, 0)
        bar_layout.setSpacing(2)
        bar_layout.addWidget(self._caption, 1)
        bar_layout.addWidget(self.pin_button)
        bar_layout.addWidget(self.close_button)

        self._web: Any = None
        self._text: QtWidgets.QTextBrowser | None = None
        self.body: QtWidgets.QWidget
        if use_web:
            self.body = self._make_web_body()
        else:
            self.body = self._make_text_body()

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(1, 1, 1, 1)
        layout.setSpacing(0)
        layout.addWidget(self._bar)
        layout.addWidget(self.body, 1)
        self._set_body_height(POPOVER_MIN_BODY + 60)

    # -- body ---------------------------------------------------------------------
    def _make_web_body(self) -> QtWidgets.QWidget:
        from PySide6.QtWebEngineCore import QWebEnginePage
        from PySide6.QtWebEngineWidgets import QWebEngineView

        outer = self

        class _Page(QWebEnginePage):
            def acceptNavigationRequest(
                self,
                url: QtCore.QUrl | str,
                kind: QWebEnginePage.NavigationType,
                is_main: bool,
            ) -> bool:
                destination = QtCore.QUrl(url)
                target = parse_doc_url(destination)
                clicked = kind == QWebEnginePage.NavigationType.NavigationTypeLinkClicked
                if target is not None and clicked:
                    outer.linkActivated.emit(target[0], target[1])
                    return False
                return destination.scheme() == SCHEME and destination.host() == "popover"

        from cppi_lab.ui.doc_workbench import shared_profile

        view = QWebEngineView(self)
        page = _Page(shared_profile(self._registry), view)
        page.setBackgroundColor(QtGui.QColor(theme.PANEL_RAISED))
        page.loadFinished.connect(self._measure_soon)
        view.setPage(page)
        view.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.NoContextMenu)
        self._web = view
        return view

    def _make_text_body(self) -> QtWidgets.QWidget:
        browser = QtWidgets.QTextBrowser(self)
        browser.setOpenLinks(False)
        browser.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        browser.setStyleSheet(
            f"QTextBrowser {{ background: {theme.PANEL_RAISED}; padding: 4px 10px; }}"
        )
        browser.anchorClicked.connect(self._on_anchor)
        self._text = browser
        return browser

    def _on_anchor(self, url: QtCore.QUrl) -> None:
        target = parse_doc_url(url)
        if target is not None:
            self.linkActivated.emit(target[0], target[1])

    def _measure_soon(self, ok: bool = True) -> None:
        """Fit the card to its page; repeat once the web fonts have settled."""
        if not ok or self._web is None:
            return
        self._measure()
        for delay in (120, 400):
            QtCore.QTimer.singleShot(delay, self._measure)

    def _measure(self) -> None:
        if self._web is None or self._web.page() is None:
            return
        # The body's own box, not the document height, which never goes below
        # the viewport and would stop a card from shrinking for shorter content.
        self._web.page().runJavaScript(
            "Math.ceil(document.body.getBoundingClientRect().height)", 0, self._apply_measured
        )

    def _apply_measured(self, height: object) -> None:
        if isinstance(height, (int, float)) and height > 0:
            self._set_body_height(int(height) + 2)

    def _set_body_height(self, height: int) -> None:
        bounded = max(POPOVER_MIN_BODY, min(POPOVER_MAX_BODY, height))
        self.body.setFixedHeight(bounded)
        self.setFixedHeight(bounded + _TITLE_BAR_HEIGHT + 2)
        self.move_within_host(self.pos())

    # -- content -------------------------------------------------------------------
    def show_entry(self, popover_id: str) -> None:
        """Load one registry entry into the card."""
        self.popover_id = popover_id
        self._caption.setText("")
        if self._web is not None:
            self._web.setUrl(QtCore.QUrl(f"{SCHEME}://popover/{popover_id}"))
        elif self._text is not None:
            self._text.setHtml(popover_text(self._registry, popover_id))
            height = int(self._text.document().size().height()) + 16
            self._set_body_height(height)

    def fade_out(self) -> None:
        """Dim the body just before the card is hidden."""
        if self._web is not None:
            self._web.page().runJavaScript(
                "document.body.style.transition='opacity 110ms';document.body.style.opacity='0';"
            )

    # -- geometry ------------------------------------------------------------------
    def move_within_host(self, pos: QtCore.QPoint) -> None:
        """Move the card, clamped so it stays fully inside the host widget."""
        host = self.parentWidget()
        if host is None:
            self.move(pos)
            return
        x = max(4, min(pos.x(), host.width() - self.width() - 4))
        y = max(4, min(pos.y(), host.height() - self.height() - 4))
        self.move(x, y)

    def place_next_to(self, badge: QtWidgets.QWidget) -> None:
        """Anchor the card 8 px to the right of a badge (left if it would not fit)."""
        host = self.parentWidget()
        if host is None:
            return
        top_left = badge.mapTo(host, QtCore.QPoint(0, 0))
        x = top_left.x() + badge.width() + ANCHOR_GAP_PX
        if x + self.width() > host.width() - 4:
            x = top_left.x() - ANCHOR_GAP_PX - self.width()
        y = top_left.y() - 10
        self.move_within_host(QtCore.QPoint(x, y))

    # -- events --------------------------------------------------------------------
    def _on_pin(self, checked: bool) -> None:
        self.pinned = checked
        self._caption.setText("Pinned - drag to move" if checked else "")
        self._bar.setCursor(
            QtCore.Qt.CursorShape.OpenHandCursor if checked else QtCore.Qt.CursorShape.ArrowCursor
        )
        self.pinToggled.emit(self, checked)

    def enterEvent(self, event: QtGui.QEnterEvent) -> None:
        self.hoverEntered.emit(self)
        super().enterEvent(event)

    def leaveEvent(self, event: QtCore.QEvent) -> None:
        self.hoverLeft.emit(self)
        super().leaveEvent(event)

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        if event.key() == QtCore.Qt.Key.Key_Escape:
            self.closeRequested.emit(self)
            return
        super().keyPressEvent(event)


class RichPopoverManager(QtCore.QObject):
    """Owns the hover-intent timers and the open popover cards."""

    #: ``(volume, anchor)`` of a documentation link clicked inside a popover.
    linkActivated = QtCore.Signal(str, str)
    popoverShown = QtCore.Signal(str)
    popoverClosed = QtCore.Signal(str)

    def __init__(
        self, host: QtWidgets.QWidget, registry: DocRegistry, use_web: bool = True
    ) -> None:
        super().__init__(host)
        self._host = host
        self._registry = registry
        self._use_web = use_web
        self._current: RichPopoverWidget | None = None
        self._current_badge: HelpBadge | None = None
        self._pending_badge: HelpBadge | None = None
        self._pinned: list[RichPopoverWidget] = []
        self._over_badge = False
        self._over_popover = False

        self.show_timer = self._timer(SHOW_DELAY_MS, self._open_pending)
        self.grace_timer = self._timer(GRACE_MS, self._grace_expired)
        self.fade_timer = self._timer(FADE_MS, self._finish_close)

    def _timer(self, interval: int, slot: Any) -> QtCore.QTimer:
        timer = QtCore.QTimer(self)
        timer.setSingleShot(True)
        timer.setTimerType(QtCore.Qt.TimerType.PreciseTimer)
        timer.setInterval(interval)
        timer.timeout.connect(slot)
        return timer

    # -- public state ----------------------------------------------------------------
    @property
    def current(self) -> RichPopoverWidget | None:
        """The transient (unpinned) popover, whether or not it is visible."""
        return self._current

    @property
    def pinned(self) -> list[RichPopoverWidget]:
        return list(self._pinned)

    def is_open(self) -> bool:
        return self._current is not None and self._current.isVisible()

    def prewarm(self) -> None:
        """Create the transient card ahead of time so the first hover is instant."""
        self._ensure_card()

    def register(self, badge: HelpBadge) -> None:
        """Start managing a ``?`` badge."""
        badge.hoverEntered.connect(self.badge_entered)
        badge.hoverLeft.connect(self.badge_left)
        badge.clicked.connect(self.badge_clicked)

    # -- badge events ---------------------------------------------------------------
    def badge_entered(self, badge: HelpBadge) -> None:
        self._over_badge = True
        self.grace_timer.stop()
        if self.is_open() and self._current_badge is badge and not self.fade_timer.isActive():
            return
        self._pending_badge = badge
        self.show_timer.start()

    def badge_left(self, badge: HelpBadge) -> None:
        self._over_badge = False
        if self._pending_badge is badge:
            self.show_timer.stop()
            self._pending_badge = None
        if self.is_open():
            self.grace_timer.start()

    def badge_clicked(self, badge: HelpBadge) -> None:
        self._pending_badge = badge
        self.show_timer.stop()
        self._open_pending()

    # -- popover events -------------------------------------------------------------
    def popover_entered(self, popover: RichPopoverWidget) -> None:
        if popover is self._current:
            self._over_popover = True
            self.grace_timer.stop()

    def popover_left(self, popover: RichPopoverWidget) -> None:
        if popover is self._current:
            self._over_popover = False
            if not popover.pinned:
                self.grace_timer.start()

    # -- transitions ------------------------------------------------------------------
    def _ensure_card(self) -> RichPopoverWidget:
        if self._current is None:
            card = RichPopoverWidget(self._host, self._registry, self._use_web)
            card.hoverEntered.connect(self.popover_entered)
            card.hoverLeft.connect(self.popover_left)
            card.linkActivated.connect(self.linkActivated)
            card.pinToggled.connect(self._pin_toggled)
            card.closeRequested.connect(self._close_card)
            self._current = card
        return self._current

    def _open_pending(self) -> None:
        badge = self._pending_badge
        self._pending_badge = None
        if badge is None:
            return
        self.fade_timer.stop()
        self.grace_timer.stop()
        card = self._ensure_card()
        if self._current_badge is not None and self._current_badge is not badge:
            self._current_badge.set_active(False)
        self._current_badge = badge
        badge.set_active(True)
        card.show_entry(badge.popover_id)
        card.place_next_to(badge)
        card.show()
        card.raise_()
        self.popoverShown.emit(badge.popover_id)

    def _grace_expired(self) -> None:
        card = self._current
        if card is None or not card.isVisible() or card.pinned:
            return
        if self._over_badge or self._over_popover:
            return
        card.fade_out()
        self.fade_timer.start()

    def _finish_close(self) -> None:
        card = self._current
        if card is None or card.pinned:
            return
        if self._over_badge or self._over_popover:
            card.show_entry(card.popover_id)
            return
        self._hide_current()

    def _hide_current(self) -> None:
        card = self._current
        if card is None:
            return
        card.hide()
        if self._current_badge is not None:
            self._current_badge.set_active(False)
            self._current_badge = None
        self._over_popover = False
        self.popoverClosed.emit(card.popover_id)

    def _pin_toggled(self, card: RichPopoverWidget, pinned: bool) -> None:
        if pinned and card is self._current:
            # The card becomes an independent floating card; a fresh transient
            # one is created on the next hover.
            self.grace_timer.stop()
            self.fade_timer.stop()
            self._pinned.append(card)
            self._current = None
            if self._current_badge is not None:
                self._current_badge.set_active(False)
                self._current_badge = None
            self._over_popover = False
        elif not pinned and card in self._pinned:
            self._close_card(card)

    def _close_card(self, card: RichPopoverWidget) -> None:
        if card in self._pinned:
            self._pinned.remove(card)
            popover_id = card.popover_id
            card.hide()
            card.deleteLater()
            self.popoverClosed.emit(popover_id)
        elif card is self._current:
            self.grace_timer.stop()
            self.fade_timer.stop()
            self._hide_current()

    def close_all(self) -> None:
        """Close the transient popover and every pinned card."""
        self.show_timer.stop()
        self.grace_timer.stop()
        self.fade_timer.stop()
        for card in list(self._pinned):
            self._close_card(card)
        self._hide_current()
