"""Static consistency of the documentation system (PRD FR-6, TSD section 5.1)."""

from __future__ import annotations

import re
from html.parser import HTMLParser

import pytest
from PySide6 import QtCore

from cppi_lab.config.defaults import PARAM_SPECS
from cppi_lab.ui.chart_workspace import PANE_POPOVERS
from cppi_lab.ui.doc_scheme import (
    ASSETS_DIR,
    DOCS_DIR,
    VOLUMES,
    DocSchemeHandler,
    counterpart,
    doc_url,
    load_registry,
    parse_doc_url,
    popover_html,
    popover_text,
)
from cppi_lab.ui.kpi_ribbon import KPI_SPECS

REGISTRY = load_registry()
TSD_CONCEPT_IDS = [
    "sec-1-floor-buffer",
    "sec-1-whipsaw-cost",
    "sec-2-dual-envelope",
    "sec-2-4-rho",
    "sec-3-liquidity",
    "sec-4-signals",
]


class Page(HTMLParser):
    """Collects sections, anchors, links and visible text of a documentation page."""

    def __init__(self) -> None:
        super().__init__()
        self.volume = ""
        self.sections: list[str] = []
        self.ids: set[str] = set()
        self.links: list[str] = []
        self.xrefs: dict[str, str] = {}
        self.text: dict[str, list[str]] = {}
        self._section = ""
        self._skip = 0
        self.assets: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        a = dict(attrs)
        if tag == "html":
            self.volume = a.get("data-volume") or ""
        if a.get("id"):
            self.ids.add(a["id"] or "")
        if tag == "section" and a.get("data-concept-id"):
            self._section = a["data-concept-id"] or ""
            assert a.get("id") == self._section, "section id must equal its data-concept-id"
            self.sections.append(self._section)
            self.text[self._section] = []
        if tag == "a" and a.get("href"):
            href = a["href"] or ""
            self.links.append(href)
            if a.get("class") == "xref":
                self.xrefs[self._section] = href
        if tag in ("script", "link"):
            ref = a.get("src") or a.get("href")
            if ref:
                self.assets.append(ref)
        if tag in ("script", "style", "pre", "code"):
            self._skip += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in ("script", "style", "pre", "code"):
            self._skip -= 1
        if tag == "section":
            self._section = ""

    def handle_data(self, data: str) -> None:
        if self._section and not self._skip:
            self.text[self._section].append(data)


def load_page(volume: str) -> Page:
    page = Page()
    page.feed((DOCS_DIR / f"{volume}.html").read_text(encoding="utf-8"))
    return page


PAGES = {volume: load_page(volume) for volume in VOLUMES}


def test_registry_spine_has_eight_sections_including_the_six_tsd_concepts() -> None:
    ids = REGISTRY.concept_ids()
    assert len(ids) == len(set(ids)) == 8
    assert ids[:6] == TSD_CONCEPT_IDS
    for concept in REGISTRY.concepts:
        for side in (concept.theory, concept.tech):
            assert side.number
            assert side.title
            assert len(side.summary) > 40
    assert REGISTRY.concept("sec-2-4-rho").theory.number == "2.2"  # type: ignore[union-attr]
    assert "route_order_by_rho" in REGISTRY.concept("sec-2-4-rho").tech.title  # type: ignore[union-attr]


@pytest.mark.parametrize("volume", VOLUMES)
def test_both_volumes_share_the_spine_in_order(volume: str) -> None:
    page = PAGES[volume]
    assert page.volume == volume
    assert page.sections == REGISTRY.concept_ids()


@pytest.mark.parametrize("volume", VOLUMES)
def test_every_section_links_to_its_exact_counterpart(volume: str) -> None:
    page = PAGES[volume]
    other = counterpart(volume)
    for concept_id in REGISTRY.concept_ids():
        target = parse_doc_url(QtCore.QUrl(page.xrefs[concept_id]))
        assert target == (other, concept_id)
        assert concept_id in PAGES[other].ids


@pytest.mark.parametrize("volume", VOLUMES)
def test_every_link_resolves(volume: str) -> None:
    page = PAGES[volume]
    for href in page.links:
        if href.startswith("#"):
            assert href[1:] in page.ids, href
            continue
        target = parse_doc_url(QtCore.QUrl(href))
        assert target is not None, f"unexpected link {href}"
        assert target[1] in PAGES[target[0]].ids, href


@pytest.mark.parametrize("volume", VOLUMES)
def test_sections_are_substantial_prose_with_math(volume: str) -> None:
    page = PAGES[volume]
    raw = (DOCS_DIR / f"{volume}.html").read_text(encoding="utf-8")
    assert "stub" not in raw.lower()
    assert "lorem" not in raw.lower()
    assert "TODO" not in raw
    total = 0
    for concept_id in REGISTRY.concept_ids():
        words = len(" ".join(page.text[concept_id]).split())
        assert words >= 170, f"{volume}#{concept_id} has only {words} words"
        total += words
    assert total >= 2_400
    # Rendered mathematics: display equations in the theory, inline maths in both.
    assert raw.count("$$") % 2 == 0
    assert raw.count("$") % 2 == 0
    if volume == "theory":
        assert raw.count("$$") >= 40  # at least twenty display equations
    assert raw.count("$") - 2 * raw.count("$$") >= 80


def test_volumes_cover_the_topics_required_by_the_prd() -> None:
    theory = (DOCS_DIR / "theory.html").read_text(encoding="utf-8").lower()
    for topic in (
        "wealth floor",
        "safety buffer",
        "hysteresis",
        "order flow",
        "momentum",
        "risk aversion",
        "queue",
        "velocity",
        "whipsaw",
        "gap",
    ):
        assert topic.replace("order flow", "flow imbalance") in theory, topic
    tech = (DOCS_DIR / "tech.html").read_text(encoding="utf-8")
    for topic in (
        "@njit",
        "TELEMETRY_DTYPE",
        "float64",
        "PCG64",
        "fastmath",
        "QTY_EPS",
        "paramChanged",
        "route_order_by_rho",
        "prange",
        "IntersectionObserver",
        "QWebChannel",
        "Parquet",
    ):
        assert topic in tech, topic


@pytest.mark.parametrize("volume", VOLUMES)
def test_page_assets_exist(volume: str) -> None:
    for ref in PAGES[volume].assets:
        assert ref.startswith("assets/")
        name = ref[len("assets/") :]
        assert name == "qwebchannel.js" or (ASSETS_DIR / name).is_file(), ref


def test_katex_is_bundled_locally_with_its_fonts() -> None:
    css = (ASSETS_DIR / "katex" / "katex.min.css").read_text(encoding="utf-8")
    fonts = set(re.findall(r"url\(fonts/([^)]+\.woff2)\)", css))
    assert len(fonts) >= 20
    for font in fonts:
        assert (ASSETS_DIR / "katex" / "fonts" / font).is_file(), font
    for name in ("katex.min.js", "auto-render.min.js", "LICENSE", "VERSION"):
        assert (ASSETS_DIR / "katex" / name).is_file()
    for page in ("theory.html", "tech.html"):
        text = (DOCS_DIR / page).read_text()
        assert "http://" not in text  # nothing is fetched remotely
        assert "https://" not in text


def test_every_badge_has_a_popover_entry() -> None:
    used = {spec.popover for spec in PARAM_SPECS}
    used |= {spec.popover for spec in KPI_SPECS}
    used |= set(PANE_POPOVERS)
    assert used == set(REGISTRY.popovers)
    assert len(used) == 37


@pytest.mark.parametrize("popover_id", sorted(REGISTRY.popovers))
def test_popover_entry_is_complete(popover_id: str) -> None:
    entry = REGISTRY.popover(popover_id)
    assert entry["title"]
    assert len(entry["summary"]) > 40
    assert REGISTRY.concept(entry["concept"]) is not None
    page = popover_html(REGISTRY, popover_id)
    assert f"cppidoc://tech/#{entry['concept']}" in page
    assert f"cppidoc://theory/#{entry['concept']}" in page
    assert "Open in Technical Documentation" in page
    assert "Open in Strategy Theory Documentation" in page
    assert page.count("$$") % 2 == 0
    for text in (entry["title"], entry["summary"], *entry["details"]):
        assert text.count("$") % 2 == 0, text
    assert entry["concept"] in popover_text(REGISTRY, popover_id)


def test_rho_popover_matches_the_tsd_example() -> None:
    entry = REGISTRY.popover("param-rho")
    assert entry["default"] == "0.25"
    assert entry["concept"] == "sec-2-4-rho"
    joined = " ".join(entry["details"])
    assert "only places a taker sell order when the liquidation level + safety buffer" in joined
    assert "$\\rho = 1$" in joined
    assert "$0 < \\rho < 1$" in joined


def test_popover_defaults_agree_with_the_parameter_table() -> None:
    assert REGISTRY.popover("param-p-liq")["default"] == "$60,000.00"
    assert REGISTRY.popover("param-seed")["default"] == "42"
    assert REGISTRY.popover("param-ema-u")["default"] == "15 m / 60 m"
    assert REGISTRY.popover("param-start")["default"] == "2026-01-01 00:00"


def test_url_helpers() -> None:
    url = doc_url("tech", "sec-3-liquidity")
    assert url.toString() == "cppidoc://tech/#sec-3-liquidity"
    assert parse_doc_url(url) == ("tech", "sec-3-liquidity")
    assert parse_doc_url(QtCore.QUrl("cppidoc://tech#sec-2-4-rho")) == ("tech", "sec-2-4-rho")
    assert parse_doc_url(QtCore.QUrl("cppidoc://theory")) == ("theory", "")
    assert parse_doc_url(QtCore.QUrl("https://example.com/#x")) is None
    assert parse_doc_url(QtCore.QUrl("cppidoc://popover/param-rho")) is None
    assert counterpart("theory") == "tech"
    assert counterpart("tech") == "theory"


def test_scheme_handler_serves_pages_assets_and_popovers() -> None:
    handler = DocSchemeHandler(REGISTRY)
    mime, body = handler.resolve(QtCore.QUrl("cppidoc://theory/"))  # type: ignore[misc]
    assert mime == b"text/html"
    assert b'data-volume="theory"' in body
    mime, body = handler.resolve(QtCore.QUrl("cppidoc://tech"))  # type: ignore[misc]
    assert b'data-volume="tech"' in body
    mime, body = handler.resolve(QtCore.QUrl("cppidoc://tech/assets/katex/katex.min.css"))  # type: ignore[misc]
    assert mime == b"text/css"
    mime, body = handler.resolve(  # type: ignore[misc]
        QtCore.QUrl("cppidoc://popover/assets/katex/fonts/KaTeX_Main-Regular.woff2")
    )
    assert mime == b"font/woff2"
    assert body[:4] == b"wOF2"
    mime, body = handler.resolve(QtCore.QUrl("cppidoc://theory/assets/qwebchannel.js"))  # type: ignore[misc]
    assert b"QWebChannel" in body
    mime, body = handler.resolve(QtCore.QUrl("cppidoc://popover/param-rho"))  # type: ignore[misc]
    assert b"Risk-Aversion Parameter" in body
    assert handler.resolve(QtCore.QUrl("cppidoc://popover/nope")) is None
    assert handler.resolve(QtCore.QUrl("cppidoc://theory/assets/../../config/defaults.py")) is None
    assert handler.resolve(QtCore.QUrl("cppidoc://theory/assets/missing.css")) is None
    assert handler.resolve(QtCore.QUrl("cppidoc://elsewhere/")) is None


def test_snippets_come_from_the_registry() -> None:
    snippet = REGISTRY.snippet("tech", "sec-3-liquidity")
    assert snippet["title"].startswith("§3.1 Stochastic Fill Sampler")
    assert snippet["volume"] == "Technical Implementation Documentation"
    assert "PCG64" in snippet["summary"]
    assert REGISTRY.snippet("theory", "missing")["title"] == ""
