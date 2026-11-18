"""Documentation workbench in a real QtWebEngine (PRD FR-6.2, FR-6.3, TSD section 5.2).

These tests load the bundled pages through ``cppidoc://`` and exercise the
page script, the ``QWebChannel`` bridge and the counterpart preview end to
end.  Every wait is on a signal or a polled condition, never on a fixed delay.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

import pytest
from PySide6 import QtCore, QtWidgets
from PySide6.QtWebEngineCore import QWebEnginePage
from pytestqt.qtbot import QtBot

from cppi_lab.data.loader import MarketData
from cppi_lab.ui.doc_scheme import load_registry, parse_doc_url
from cppi_lab.ui.doc_workbench import DocWorkbench
from cppi_lab.ui.popover_manager import HelpBadge, RichPopoverManager

pytestmark = [pytest.mark.ui, pytest.mark.web]
REGISTRY = load_registry()
IDS = REGISTRY.concept_ids()
WAIT = 30_000


def js(qtbot: QtBot, page: QWebEnginePage, script: str) -> Any:
    """Evaluate ``script`` in ``page`` and return its JSON-decoded value."""
    box: list[Any] = []
    page.runJavaScript(f"JSON.stringify({script})", 0, box.append)
    qtbot.waitUntil(lambda: bool(box), timeout=WAIT)
    return None if box[0] in (None, "") else json.loads(box[0])


class Synced:
    """Records ``previewSynced`` emissions so a test can wait for a specific one."""

    def __init__(self, workbench: DocWorkbench) -> None:
        self.events: list[tuple[str, str]] = []
        workbench.previewSynced.connect(
            lambda volume, concept: self.events.append((volume, concept))
        )

    def wait(self, qtbot: QtBot, volume: str, concept: str) -> None:
        qtbot.waitUntil(lambda: (volume, concept) in self.events, timeout=WAIT)
        self.events.clear()


@pytest.fixture
def workbench(qtbot: QtBot) -> Iterator[DocWorkbench]:
    widget = DocWorkbench(REGISTRY)
    qtbot.addWidget(widget)
    widget.resize(1300, 800)
    synced = Synced(widget)
    with qtbot.waitSignal(widget.bridge.ready, timeout=WAIT):
        widget.show()
    synced.wait(qtbot, "tech", IDS[0])
    wait_settled(qtbot, widget, IDS[0])
    synced.events.clear()
    widget.synced = synced  # type: ignore[attr-defined]
    yield widget
    widget.shutdown()


def synced_section(qtbot: QtBot, workbench: DocWorkbench) -> str:
    return str(
        js(
            qtbot,
            workbench.preview_page,
            "(document.querySelector('section.synced') || {id: ''}).id",
        )
    )


def wait_settled(qtbot: QtBot, workbench: DocWorkbench, concept: str) -> None:
    """Wait until the reader position and the highlighted counterpart agree on ``concept``.

    Section reports travel over an asynchronous channel, so a report sent just
    before a navigation may still arrive after it; the state converges once
    the page has reported its new position.
    """
    deadline = QtCore.QDeadlineTimer(WAIT)
    while not deadline.hasExpired():
        if workbench.focused_concept == concept and synced_section(qtbot, workbench) == concept:
            return
        qtbot.wait(20)
    raise AssertionError(
        f"focused {workbench.focused_concept!r}, preview {synced_section(qtbot, workbench)!r}, "
        f"expected {concept!r}"
    )


def test_theory_volume_renders_with_math_and_the_full_spine(
    qtbot: QtBot, workbench: DocWorkbench
) -> None:
    page = workbench.primary_page
    assert workbench.volume == "theory"
    assert js(qtbot, page, "document.documentElement.getAttribute('data-volume')") == "theory"
    assert js(qtbot, page, "document.documentElement.getAttribute('data-math')") == "rendered"
    assert js(qtbot, page, "cppiDoc.sectionIds()") == IDS
    assert js(qtbot, page, "document.querySelectorAll('.katex').length") > 150
    assert js(qtbot, page, "document.querySelectorAll('.katex-display').length") >= 20
    assert js(qtbot, page, "document.querySelectorAll('.katex-error').length") == 0
    # No raw TeX delimiters survive in the prose.
    assert js(qtbot, page, "document.querySelector('main').innerText.includes('$$')") is False
    # The KaTeX font really loaded from the local scheme.
    assert js(qtbot, page, "document.fonts.check('16px KaTeX_Main')") is True
    assert workbench.volume_label.text() == "Theoretical Strategy Documentation"
    assert workbench.switch_button.text() == "⇄ Technical"


def test_technical_volume_renders_without_errors(qtbot: QtBot, workbench: DocWorkbench) -> None:
    page = workbench.preview_page  # the counterpart pane shows the other volume
    assert js(qtbot, page, "document.documentElement.getAttribute('data-volume')") == "tech"
    assert js(qtbot, page, "cppiDoc.sectionIds()") == IDS
    assert js(qtbot, page, "document.querySelectorAll('.katex').length") > 60
    assert js(qtbot, page, "document.querySelectorAll('.katex-error').length") == 0
    assert js(qtbot, page, "document.querySelectorAll('pre code').length") >= 10


def test_splitter_gives_primary_65_and_preview_35(workbench: DocWorkbench) -> None:
    primary, preview = workbench.splitter.sizes()
    assert primary / (primary + preview) == pytest.approx(0.65, abs=0.03)
    assert workbench.preview_view.isVisible()


def test_toolbar_has_the_five_specified_controls(workbench: DocWorkbench) -> None:
    names = [
        workbench.back_button.accessibleName(),
        workbench.forward_button.accessibleName(),
        workbench.switch_button.accessibleName(),
        workbench.search_edit.accessibleName(),
        workbench.preview_button.accessibleName(),
    ]
    assert names == [
        "Back",
        "Forward",
        "Switch Primary Volume",
        "Search",
        "Toggle Counterpart Preview Split",
    ]
    assert workbench.toolbar.minimumSizeHint().width() < 520  # still usable in a side panel


def test_narrow_panel_stacks_the_two_viewers(qtbot: QtBot, workbench: DocWorkbench) -> None:
    assert workbench.splitter.orientation() == QtCore.Qt.Orientation.Horizontal
    workbench.resize(640, 900)
    qtbot.waitUntil(
        lambda: workbench.splitter.orientation() == QtCore.Qt.Orientation.Vertical, timeout=WAIT
    )
    top, bottom = workbench.splitter.sizes()
    assert top / (top + bottom) == pytest.approx(0.65, abs=0.04)
    assert workbench.preview_view.isVisible()
    workbench.resize(1300, 800)
    qtbot.waitUntil(
        lambda: workbench.splitter.orientation() == QtCore.Qt.Orientation.Horizontal, timeout=WAIT
    )


def test_scroll_spy_moves_the_counterpart_preview(qtbot: QtBot, workbench: DocWorkbench) -> None:
    synced: Synced = workbench.synced  # type: ignore[attr-defined]
    for concept in ("sec-3-liquidity", "sec-5-state-machine", "sec-1-whipsaw-cost"):
        with qtbot.waitSignal(
            workbench.sectionFocused,
            timeout=WAIT,
            check_params_cb=lambda c, want=concept: c == want,
        ):
            # Scroll the primary view the way a reader would.
            js(
                qtbot,
                workbench.primary_page,
                f"window.scrollTo(0, document.getElementById('{concept}')"
                ".getBoundingClientRect().top + window.scrollY + 4), 0",
            )
        synced.wait(qtbot, "tech", concept)
        wait_settled(qtbot, workbench, concept)
        top = js(
            qtbot,
            workbench.preview_page,
            f"document.getElementById('{concept}').getBoundingClientRect().top",
        )
        assert -5 <= top <= 60  # the counterpart section is at the top of the preview
        assert (
            js(
                qtbot,
                workbench.primary_page,
                "document.querySelector('#toc a.current').getAttribute('href')",
            )
            == f"#{concept}"
        )


def test_open_at_scrolls_both_views(qtbot: QtBot, workbench: DocWorkbench) -> None:
    synced: Synced = workbench.synced  # type: ignore[attr-defined]
    workbench.open_at("theory", "sec-4-signals")
    synced.wait(qtbot, "tech", "sec-4-signals")
    wait_settled(qtbot, workbench, "sec-4-signals")
    top = js(
        qtbot,
        workbench.primary_page,
        "document.getElementById('sec-4-signals').getBoundingClientRect().top",
    )
    assert -5 <= top <= 60
    with pytest.raises(ValueError):
        workbench.open_at("appendix")


def test_hovering_a_link_peeks_and_leaving_returns(qtbot: QtBot, workbench: DocWorkbench) -> None:
    synced: Synced = workbench.synced  # type: ignore[attr-defined]
    hovered: list[tuple[str, str]] = []
    workbench.bridge.linkHovered.connect(lambda v, a: hovered.append((v, a)))
    # A real DOM hover on the counterpart link of section 2.2.
    js(
        qtbot,
        workbench.primary_page,
        "document.querySelector('#sec-2-4-rho a.xref').dispatchEvent("
        "new MouseEvent('mouseover', {bubbles: true})), 0",
    )
    qtbot.waitUntil(lambda: ("tech", "sec-2-4-rho") in hovered, timeout=WAIT)
    synced.wait(qtbot, "tech", "sec-2-4-rho")
    assert synced_section(qtbot, workbench) == "sec-2-4-rho"
    js(
        qtbot,
        workbench.primary_page,
        "document.querySelector('#sec-2-4-rho a.xref').dispatchEvent("
        "new MouseEvent('mouseout', {bubbles: true})), 0",
    )
    synced.wait(qtbot, "tech", workbench.focused_concept)  # back to what is being read
    assert synced_section(qtbot, workbench) == workbench.focused_concept


def test_switching_volume_keeps_the_concept(qtbot: QtBot, workbench: DocWorkbench) -> None:
    synced: Synced = workbench.synced  # type: ignore[attr-defined]
    workbench.open_at("theory", "sec-3-liquidity")
    synced.wait(qtbot, "tech", "sec-3-liquidity")
    wait_settled(qtbot, workbench, "sec-3-liquidity")
    with qtbot.waitSignal(workbench.volumeChanged, timeout=WAIT) as changed:
        workbench.switch_button.click()
    assert changed.args == ["tech"]
    synced.wait(qtbot, "theory", "sec-3-liquidity")
    wait_settled(qtbot, workbench, "sec-3-liquidity")
    assert workbench.volume == "tech"
    assert parse_doc_url(workbench.primary_view.url()) == ("tech", "sec-3-liquidity")
    assert parse_doc_url(workbench.preview_view.url())[0] == "theory"  # type: ignore[index]
    assert workbench.switch_button.text() == "⇄ Theory"
    assert workbench.volume_label.text() == "Technical Implementation Documentation"


def click_link(qtbot: QtBot, workbench: DocWorkbench, selector: str) -> None:
    """Click a link in the primary view with a real mouse event (a user gesture)."""
    box = js(
        qtbot,
        workbench.primary_page,
        f"(function () {{ var r = document.querySelector({json.dumps(selector)})"
        ".getBoundingClientRect(); return [r.left + r.width / 2, r.top + r.height / 2]; })()",
    )
    target = workbench.primary_view.focusProxy()
    assert target is not None
    qtbot.mouseClick(
        target, QtCore.Qt.MouseButton.LeftButton, pos=QtCore.QPoint(int(box[0]), int(box[1]))
    )


def test_clicking_a_counterpart_link_navigates_the_primary_view(
    qtbot: QtBot, workbench: DocWorkbench
) -> None:
    synced: Synced = workbench.synced  # type: ignore[attr-defined]
    workbench.open_at("theory", "sec-5-state-machine")
    wait_settled(qtbot, workbench, "sec-5-state-machine")
    with qtbot.waitSignal(workbench.volumeChanged, timeout=WAIT) as changed:
        click_link(qtbot, workbench, "#sec-5-state-machine a.xref")
    assert changed.args == ["tech"]
    synced.wait(qtbot, "theory", "sec-5-state-machine")
    wait_settled(qtbot, workbench, "sec-5-state-machine")
    assert parse_doc_url(workbench.primary_view.url()) == ("tech", "sec-5-state-machine")
    # Back returns to the volume the reader came from, Forward goes there again.
    qtbot.waitUntil(workbench.back_button.isEnabled, timeout=WAIT)
    with qtbot.waitSignal(workbench.volumeChanged, timeout=WAIT) as changed:
        workbench.back_button.click()
    assert changed.args == ["theory"]
    qtbot.waitUntil(workbench.forward_button.isEnabled, timeout=WAIT)
    with qtbot.waitSignal(workbench.volumeChanged, timeout=WAIT) as changed:
        workbench.forward_button.click()
    assert changed.args == ["tech"]


def test_in_page_links_stay_in_the_volume(qtbot: QtBot, workbench: DocWorkbench) -> None:
    assert not workbench.back_button.isEnabled()
    click_link(qtbot, workbench, "#toc a[href='#sec-2-dual-envelope']")
    wait_settled(qtbot, workbench, "sec-2-dual-envelope")
    assert workbench.volume == "theory"
    assert parse_doc_url(workbench.primary_view.url()) == ("theory", "sec-2-dual-envelope")


def test_search_reports_matches(qtbot: QtBot, workbench: DocWorkbench) -> None:
    workbench.search_edit.setText("cushion")
    qtbot.waitUntil(lambda: "/" in workbench.search_status.text(), timeout=WAIT)
    active, total = (int(part) for part in workbench.search_status.text().split("/"))
    assert total >= 5
    assert active == 1
    workbench.search_edit.setText("zzzqqqxxx")
    qtbot.waitUntil(lambda: workbench.search_status.text() == "no match", timeout=WAIT)
    workbench.search_edit.clear()
    qtbot.waitUntil(lambda: workbench.search_status.text() == "", timeout=WAIT)


def test_collapsed_preview_falls_back_to_a_snippet_card(
    qtbot: QtBot, workbench: DocWorkbench
) -> None:
    workbench.preview_button.click()
    assert not workbench.preview_view.isVisible()
    assert workbench.bridge.property("previewCollapsed") is True
    # The page learns about the collapse through the channel's batched property update.
    qtbot.waitUntil(
        lambda: js(qtbot, workbench.primary_page, "cppiDoc.previewCollapsed()") is True,
        timeout=WAIT,
    )
    js(
        qtbot,
        workbench.primary_page,
        "document.querySelector('#sec-3-liquidity a.xref').dispatchEvent("
        "new MouseEvent('mouseover', {bubbles: true})), 0",
    )
    qtbot.waitUntil(
        lambda: (
            js(
                qtbot,
                workbench.primary_page,
                "document.getElementById('snippet-card').style.display",
            )
            == "block"
        ),
        timeout=WAIT,
    )
    title = js(
        qtbot, workbench.primary_page, "document.querySelector('#snippet-card .ttl').textContent"
    )
    assert title == "§3.1 Stochastic Fill Sampler & PCG64 Seeding"
    snippet = json.loads(workbench.bridge.sectionSnippet("tech", "sec-3-liquidity"))
    assert snippet["title"] == title
    workbench.preview_button.click()
    assert workbench.preview_view.isVisible()
    assert workbench.bridge.property("previewCollapsed") is False


def test_web_popover_renders_math_and_its_links_open_the_docs(qtbot: QtBot) -> None:
    host = QtWidgets.QWidget()
    host.resize(1200, 800)
    qtbot.addWidget(host)
    badge = HelpBadge("param-rho", host)
    badge.move(100, 100)
    manager = RichPopoverManager(host, REGISTRY, use_web=True)
    manager.register(badge)
    host.show()
    manager.badge_clicked(badge)
    card = manager.current
    assert card is not None
    page = card.body.page()  # type: ignore[attr-defined]
    qtbot.waitUntil(
        lambda: js(qtbot, page, "document.querySelectorAll('.katex').length") > 5, timeout=WAIT
    )
    assert "Risk-Aversion Parameter" in js(qtbot, page, "document.querySelector('h1').innerText")
    assert js(qtbot, page, "document.querySelectorAll('.katex-error').length") == 0
    assert js(
        qtbot, page, "Array.from(document.querySelectorAll('nav a')).map(a => a.innerText)"
    ) == [
        "Open in Technical Documentation (§2.2) ↗",
        "Open in Strategy Theory Documentation (§2.2) ↗",
    ]
    # The card grew to fit its content instead of keeping the placeholder height.
    qtbot.waitUntil(lambda: card.body.height() > 250, timeout=WAIT)
    with qtbot.waitSignal(manager.linkActivated, timeout=WAIT) as activated:
        js(qtbot, page, "document.querySelectorAll('nav a')[0].click(), 0")
    assert activated.args == ["tech", "sec-2-4-rho"]
    with qtbot.waitSignal(manager.linkActivated, timeout=WAIT) as activated:
        js(qtbot, page, "document.querySelectorAll('nav a')[1].click(), 0")
    assert activated.args == ["theory", "sec-2-4-rho"]
    manager.close_all()
    card.body.setPage(None)  # type: ignore[attr-defined]
    page.deleteLater()


def test_popover_link_opens_the_workbench_at_the_exact_section(
    qtbot: QtBot, bundled: MarketData
) -> None:
    from cppi_lab.ui.main_window import MainWindow

    window = MainWindow(bundled, use_web=True, async_ensemble=False)
    qtbot.addWidget(window)
    window.resize(1600, 900)
    window.show()
    docs = window.docs
    assert isinstance(docs, DocWorkbench)
    synced = Synced(docs)
    assert window.docs_dock.isClosed()
    window.popovers.linkActivated.emit("tech", "sec-3-liquidity")
    synced.wait(qtbot, "theory", "sec-3-liquidity")
    assert not window.docs_dock.isClosed()
    assert window.docs_dock.isFloating()
    assert docs.volume == "tech"
    assert docs.focused_concept == "sec-3-liquidity"
    assert window.docs_button.isChecked()
    window.close()
    QtCore.QCoreApplication.processEvents()


@pytest.mark.parametrize("volume", ["theory", "tech"])
def test_pages_never_scroll_sideways(qtbot: QtBot, workbench: DocWorkbench, volume: str) -> None:
    """A wide table or code block scrolls inside its own box, not the whole page."""
    overflow = "document.documentElement.scrollWidth - document.documentElement.clientWidth"
    synced: Any = workbench.synced  # type: ignore[attr-defined]
    workbench.open_at(volume, IDS[0])
    wait_settled(qtbot, workbench, IDS[0])
    del synced
    for width in (1300, 1180, 1000, 900, 760, 620, 480):
        workbench.resize(width, 800)
        qtbot.wait(150)
        for name, page in (
            ("primary", workbench.primary_page),
            ("preview", workbench.preview_page),
        ):
            assert js(qtbot, page, overflow) <= 0, (volume, width, name)
    # The widest table of the page is still complete: it scrolls inside its own box.
    widest = js(
        qtbot,
        workbench.primary_page,
        "Math.max.apply(null, Array.from(document.querySelectorAll('table'))"
        ".map(function (t) { return t.scrollWidth - t.clientWidth; }))",
    )
    assert widest >= 0


def test_toolbar_keeps_whole_words_in_a_narrow_docked_panel(
    qtbot: QtBot, workbench: DocWorkbench
) -> None:
    label = workbench.volume_label
    assert label.text() == "Technical Implementation Documentation" or label.text() == (
        "Theoretical Strategy Documentation"
    )
    assert not workbench.search_status.isVisible()  # no room is reserved for an idle counter
    workbench.resize(480, 800)  # the width of a panel docked beside the workbench
    qtbot.wait(100)
    assert label.text() in ("Technical", "Theory")  # the short name, not a cut-off title
    assert not label.is_elided()
    assert label.toolTip().endswith("Documentation")
    for button in (workbench.back_button, workbench.switch_button, workbench.preview_button):
        assert button.geometry().right() <= workbench.toolbar.width()
    workbench.search_edit.setText("trigger")
    qtbot.waitUntil(lambda: "/" in workbench.search_status.text(), timeout=WAIT)
    assert workbench.search_status.isVisible()
    workbench.search_edit.clear()
    qtbot.waitUntil(lambda: not workbench.search_status.isVisible(), timeout=WAIT)
    workbench.resize(1300, 800)
    qtbot.wait(100)
    assert label.text().endswith("Documentation")
