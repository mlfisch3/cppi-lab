"""The entry point: what the launching terminal can do to the running application."""

from __future__ import annotations

import os
import signal
import threading
import time

import pytest
from PySide6 import QtCore
from pytestqt.qtbot import QtBot

from cppi_lab import main as entry

pytestmark = pytest.mark.ui


def test_ctrl_c_reaches_an_idle_event_loop(qtbot: QtBot) -> None:
    """An interrupt that arrives while no Python code runs still ends the loop at once.

    The signal is sent from another thread while the main thread sits in a Qt
    event loop that has nothing to do.  Without the watch the handler would not
    run before the safety timer brings the interpreter back, seconds later.
    """
    loop = QtCore.QEventLoop()
    watch = entry.InterruptWatch(loop.quit)
    safety = QtCore.QTimer()
    safety.setSingleShot(True)
    safety.timeout.connect(loop.quit)
    safety.start(5000)
    sender = threading.Timer(0.2, os.kill, (os.getpid(), signal.SIGINT))
    try:
        started = time.monotonic()
        sender.start()
        loop.exec()
        waited = time.monotonic() - started
    finally:
        sender.join()
        safety.stop()
        watch.close()
    assert watch.requested
    assert waited < 2.0, f"the interrupt was acted on only after {waited:.1f} s"


def test_interrupt_watch_puts_the_previous_handling_back(qtbot: QtBot) -> None:
    handler_before = signal.getsignal(signal.SIGINT)
    watch = entry.InterruptWatch(lambda: None)
    assert signal.getsignal(signal.SIGINT) is not handler_before
    watch.close()
    assert signal.getsignal(signal.SIGINT) is handler_before
    assert signal.set_wakeup_fd(-1) == -1  # nothing of the watch is left installed
    assert not watch.requested


def test_exit_status_after_an_interrupt_is_the_shell_convention() -> None:
    assert entry.INTERRUPTED == 130
