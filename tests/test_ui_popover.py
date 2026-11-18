"""Hover-intent ``?`` popovers (PRD FR-6.1, TSD section 4.3).

Timing is asserted against the timers themselves and against lower bounds on
elapsed time.  Nothing waits a fixed interval and then expects a state, so the
tests cannot go flaky on a loaded machine.
"""

from __future__ import annotations

import time
from collections.abc import Iterator

import pytest
from PySide6 import QtCore, QtGui, QtWidgets
from pytestqt.qtbot import QtBot

from cppi_lab.ui.doc_scheme import load_registry
from cppi_lab.ui.popover_manager import (
    ANCHOR_GAP_PX,
    FADE_MS,
    GRACE_MS,
    SHOW_DELAY_MS,
    HelpBadge,
    RichPopoverManager,
)

pytestmark = pytest.mark.ui
# Qt's precise timers may fire a millisecond early; allow for that and no more.
SLACK_S = 0.003


class Rig:
    def __init__(self, qtbot: QtBot) -> None:
        self.host = QtWidgets.QWidget()
        self.host.resize(1200, 800)
        qtbot.addWidget(self.host)
        self.badge = HelpBadge("param-rho", self.host)
        self.badge.move(200, 300)
        self.other = HelpBadge("param-seed", self.host)
        self.other.move(200, 500)
        self.manager = RichPopoverManager(self.host, load_registry(), use_web=False)
        self.manager.register(self.badge)
        self.manager.register(self.other)
        self.host.show()
        qtbot.waitExposed(self.host)

    def open(self, qtbot: QtBot, badge: HelpBadge | None = None) -> None:
        self.manager.badge_entered(badge or self.badge)
        qtbot.waitUntil(self.manager.is_open, timeout=5_000)


@pytest.fixture
def rig(qtbot: QtBot) -> Iterator[Rig]:
    made = Rig(qtbot)
    yield made
    made.manager.close_all()


def test_timer_constants_match_the_specification(rig: Rig) -> None:
    assert (SHOW_DELAY_MS, GRACE_MS, ANCHOR_GAP_PX) == (150, 300, 8)
    manager = rig.manager
    assert manager.show_timer.interval() == 150
    assert manager.grace_timer.interval() == 300
    for timer in (manager.show_timer, manager.grace_timer, manager.fade_timer):
        assert timer.isSingleShot()
        assert timer.timerType() == QtCore.Qt.TimerType.PreciseTimer


def test_popover_opens_after_the_150_ms_debounce(qtbot: QtBot, rig: Rig) -> None:
    started = time.perf_counter()
    rig.manager.badge_entered(rig.badge)
    assert not rig.manager.is_open()  # nothing appears on the hover itself
    assert rig.manager.show_timer.isActive()
    with qtbot.waitSignal(rig.manager.popoverShown, timeout=5_000) as shown:
        pass
    elapsed = time.perf_counter() - started
    assert shown.args == ["param-rho"]
    assert elapsed >= SHOW_DELAY_MS / 1000 - SLACK_S
    assert rig.manager.is_open()
    assert rig.badge.property("active") == "true"


def test_popover_is_anchored_eight_pixels_right_of_the_badge(qtbot: QtBot, rig: Rig) -> None:
    rig.open(qtbot)
    card = rig.manager.current
    assert card is not None
    assert card.x() == rig.badge.x() + rig.badge.width() + ANCHOR_GAP_PX
    # Near the right edge it flips to the left instead of leaving the window.
    rig.manager.close_all()
    rig.badge.move(rig.host.width() - 40, 300)
    rig.open(qtbot)
    card = rig.manager.current
    assert card is not None
    assert card.x() + card.width() == rig.badge.x() - ANCHOR_GAP_PX
    assert card.geometry().right() <= rig.host.width()
    assert card.geometry().bottom() <= rig.host.height()


def test_brushing_past_a_badge_shows_nothing(qtbot: QtBot, rig: Rig) -> None:
    rig.manager.badge_entered(rig.badge)
    rig.manager.badge_left(rig.badge)
    assert not rig.manager.show_timer.isActive()
    qtbot.wait(SHOW_DELAY_MS + 120)
    assert not rig.manager.is_open()


def test_leaving_the_badge_starts_the_300_ms_grace_period(qtbot: QtBot, rig: Rig) -> None:
    rig.open(qtbot)
    started = time.perf_counter()
    rig.manager.badge_left(rig.badge)
    assert rig.manager.grace_timer.isActive()
    assert rig.manager.is_open()  # still open: the grace period has only begun
    with qtbot.waitSignal(rig.manager.popoverClosed, timeout=5_000) as closed:
        pass
    elapsed = time.perf_counter() - started
    assert closed.args == ["param-rho"]
    assert elapsed >= (GRACE_MS + FADE_MS) / 1000 - 2 * SLACK_S
    assert not rig.manager.is_open()
    assert rig.badge.property("active") == "false"


def test_entering_the_popover_within_the_grace_period_keeps_it_open(qtbot: QtBot, rig: Rig) -> None:
    rig.open(qtbot)
    card = rig.manager.current
    assert card is not None
    rig.manager.badge_left(rig.badge)
    rig.manager.popover_entered(card)  # the cursor arrived inside the 300 ms
    assert not rig.manager.grace_timer.isActive()
    qtbot.wait(GRACE_MS + FADE_MS + 150)  # no timer is running, so this cannot race
    assert rig.manager.is_open()
    # Leaving the popover starts the grace period again, and then it closes.
    with qtbot.waitSignal(rig.manager.popoverClosed, timeout=5_000):
        rig.manager.popover_left(card)
    assert not rig.manager.is_open()


def test_returning_to_the_badge_cancels_the_grace_period(qtbot: QtBot, rig: Rig) -> None:
    rig.open(qtbot)
    rig.manager.badge_left(rig.badge)
    rig.manager.badge_entered(rig.badge)
    assert not rig.manager.grace_timer.isActive()
    qtbot.wait(GRACE_MS + FADE_MS + 150)
    assert rig.manager.is_open()


def test_real_enter_and_leave_events_drive_the_manager(qtbot: QtBot, rig: Rig) -> None:
    origin = QtCore.QPointF(3, 3)
    enter = QtGui.QEnterEvent(origin, origin, origin)
    QtWidgets.QApplication.sendEvent(rig.badge, enter)
    assert rig.manager.show_timer.isActive()
    qtbot.waitUntil(rig.manager.is_open, timeout=5_000)
    card = rig.manager.current
    assert card is not None
    QtWidgets.QApplication.sendEvent(rig.badge, QtCore.QEvent(QtCore.QEvent.Type.Leave))
    assert rig.manager.grace_timer.isActive()
    QtWidgets.QApplication.sendEvent(card, QtGui.QEnterEvent(origin, origin, origin))
    assert not rig.manager.grace_timer.isActive()
    QtWidgets.QApplication.sendEvent(card, QtCore.QEvent(QtCore.QEvent.Type.Leave))
    assert rig.manager.grace_timer.isActive()


def test_click_opens_immediately(rig: Rig) -> None:
    QtWidgets.QApplication.sendEvent(
        rig.badge,
        QtGui.QMouseEvent(
            QtCore.QEvent.Type.MouseButtonPress,
            QtCore.QPointF(3, 3),
            QtCore.QPointF(3, 3),
            QtCore.Qt.MouseButton.LeftButton,
            QtCore.Qt.MouseButton.LeftButton,
            QtCore.Qt.KeyboardModifier.NoModifier,
        ),
    )
    assert rig.manager.is_open()
    assert not rig.manager.show_timer.isActive()


def test_moving_to_another_badge_swaps_the_content(qtbot: QtBot, rig: Rig) -> None:
    rig.open(qtbot)
    rig.manager.badge_left(rig.badge)
    with qtbot.waitSignal(rig.manager.popoverShown, timeout=5_000) as shown:
        rig.manager.badge_entered(rig.other)
    assert shown.args == ["param-seed"]
    card = rig.manager.current
    assert card is not None
    assert card.popover_id == "param-seed"
    assert rig.badge.property("active") == "false"
    assert rig.other.property("active") == "true"


def test_pinned_popover_survives_and_can_be_moved(qtbot: QtBot, rig: Rig) -> None:
    rig.open(qtbot)
    card = rig.manager.current
    assert card is not None
    card.pin_button.click()
    assert card.pinned
    assert rig.manager.pinned == [card]
    assert rig.manager.current is None  # a fresh transient card serves the next hover
    rig.manager.badge_left(rig.badge)
    assert not rig.manager.grace_timer.isActive()
    qtbot.wait(GRACE_MS + FADE_MS + 150)
    assert card.isVisible()

    card.move_within_host(QtCore.QPoint(600, 250))  # a floating card: movable ...
    assert card.pos() == QtCore.QPoint(600, 250)
    card.move_within_host(QtCore.QPoint(5_000, -50))  # ... but kept inside the window
    assert card.geometry().right() <= rig.host.width()
    assert card.y() >= 0

    rig.open(qtbot, rig.other)  # a second popover opens alongside the pinned one
    assert rig.manager.current is not card
    assert card.isVisible()
    assert rig.manager.is_open()

    card.close_button.click()
    assert rig.manager.pinned == []


def test_links_are_forwarded_with_volume_and_anchor(qtbot: QtBot, rig: Rig) -> None:
    rig.open(qtbot)
    card = rig.manager.current
    assert card is not None
    body = card.body
    assert isinstance(body, QtWidgets.QTextBrowser)
    html = body.toHtml()
    assert "Open in Technical Documentation" in html
    assert "Open in Strategy Theory Documentation" in html
    assert "Risk-Aversion Parameter" in body.toPlainText()
    with qtbot.waitSignal(rig.manager.linkActivated, timeout=2_000) as activated:
        body.anchorClicked.emit(QtCore.QUrl("cppidoc://tech/#sec-2-4-rho"))
    assert activated.args == ["tech", "sec-2-4-rho"]
    with qtbot.waitSignal(rig.manager.linkActivated, timeout=2_000) as activated:
        body.anchorClicked.emit(QtCore.QUrl("cppidoc://theory/#sec-2-4-rho"))
    assert activated.args == ["theory", "sec-2-4-rho"]


def test_escape_and_close_button_dismiss(qtbot: QtBot, rig: Rig) -> None:
    rig.open(qtbot)
    card = rig.manager.current
    assert card is not None
    qtbot.keyClick(card, QtCore.Qt.Key.Key_Escape)
    assert not rig.manager.is_open()
    rig.open(qtbot)
    card.close_button.click()
    assert not rig.manager.is_open()
