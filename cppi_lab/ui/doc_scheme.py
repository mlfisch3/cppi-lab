"""The ``cppidoc://`` URL scheme and the documentation registry (TSD sections 1.1, 5.1).

Both documentation volumes, the KaTeX assets and the ``?`` popover pages are
served to ``QWebEngineView`` instances from the installed package through a
custom scheme, so nothing is fetched from the network and no temporary files
are written.

URL layout::

    cppidoc://theory#<concept-id>      Theoretical Strategy Documentation
    cppidoc://tech#<concept-id>        Technical Implementation Documentation
    cppidoc://popover/<popover-id>     one generated "?" popover page
    cppidoc://<any-host>/assets/...    static assets (KaTeX, CSS, JS)
"""

from __future__ import annotations

import html
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from PySide6 import QtCore
from PySide6.QtWebEngineCore import (
    QWebEngineProfile,
    QWebEngineUrlRequestJob,
    QWebEngineUrlScheme,
    QWebEngineUrlSchemeHandler,
)

SCHEME = "cppidoc"
PACKAGE_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = PACKAGE_ROOT / "docs"
ASSETS_DIR = DOCS_DIR / "assets"
REGISTRY_PATH = PACKAGE_ROOT / "config" / "doc_registry.json"

VOLUMES: tuple[str, ...] = ("theory", "tech")

_MIME = {
    ".html": b"text/html",
    ".css": b"text/css",
    ".js": b"text/javascript",
    ".json": b"application/json",
    ".woff2": b"font/woff2",
    ".svg": b"image/svg+xml",
    ".png": b"image/png",
}


def counterpart(volume: str) -> str:
    """The other documentation volume."""
    return "tech" if volume == "theory" else "theory"


def doc_url(volume: str, anchor: str = "") -> QtCore.QUrl:
    """``cppidoc://<volume>#<anchor>`` as a ``QUrl``."""
    url = QtCore.QUrl(f"{SCHEME}://{volume}/")
    if anchor:
        url.setFragment(anchor)
    return url


def parse_doc_url(url: QtCore.QUrl) -> tuple[str, str] | None:
    """Split a documentation URL into ``(volume, anchor)``; ``None`` if it is not one."""
    if url.scheme() != SCHEME or url.host() not in VOLUMES:
        return None
    if url.path() not in ("", "/"):
        return None
    return url.host(), url.fragment()


def register_scheme() -> None:
    """Register ``cppidoc://``.  Must run before the ``QApplication`` exists."""
    if QWebEngineUrlScheme.schemeByName(SCHEME.encode()).name():
        return
    scheme = QWebEngineUrlScheme(SCHEME.encode())
    scheme.setSyntax(QWebEngineUrlScheme.Syntax.Host)
    flag = QWebEngineUrlScheme.Flag
    scheme.setFlags(flag.SecureScheme | flag.CorsEnabled | flag.LocalAccessAllowed)
    QWebEngineUrlScheme.registerScheme(scheme)


@dataclass(frozen=True)
class ConceptSide:
    """One volume's view of a concept: section number, title and summary."""

    number: str
    title: str
    summary: str


@dataclass(frozen=True)
class Concept:
    """A section of the shared documentation spine (TSD section 5.1)."""

    id: str
    theory: ConceptSide
    tech: ConceptSide

    def side(self, volume: str) -> ConceptSide:
        return self.theory if volume == "theory" else self.tech


class DocRegistry:
    """Typed access to ``doc_registry.json``."""

    def __init__(self, path: Path = REGISTRY_PATH) -> None:
        raw: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        self.volumes: dict[str, dict[str, str]] = raw["volumes"]
        self.concepts: list[Concept] = [
            Concept(
                id=entry["id"],
                theory=ConceptSide(**entry["theory"]),
                tech=ConceptSide(**entry["tech"]),
            )
            for entry in raw["concepts"]
        ]
        self.popovers: dict[str, dict[str, Any]] = raw["popovers"]
        self._by_id = {concept.id: concept for concept in self.concepts}

    def concept(self, concept_id: str) -> Concept | None:
        return self._by_id.get(concept_id)

    def concept_ids(self) -> list[str]:
        return [concept.id for concept in self.concepts]

    def volume_title(self, volume: str) -> str:
        return self.volumes[volume]["title"]

    def popover(self, popover_id: str) -> dict[str, Any]:
        return self.popovers[popover_id]

    def snippet(self, volume: str, concept_id: str) -> dict[str, str]:
        """Title and summary shown in the hover snippet card (TSD section 5.2)."""
        concept = self.concept(concept_id)
        if concept is None:
            return {"title": "", "summary": "", "volume": self.volume_title(volume)}
        side = concept.side(volume)
        return {
            "title": f"§{side.number} {side.title}",
            "summary": side.summary,
            "volume": self.volume_title(volume),
        }


@lru_cache(maxsize=1)
def load_registry() -> DocRegistry:
    """The shared, parsed documentation registry."""
    return DocRegistry()


_POPOVER_PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<link rel="stylesheet" href="assets/katex/katex.min.css">
<link rel="stylesheet" href="assets/popover.css">
<script src="assets/katex/katex.min.js"></script>
<script src="assets/katex/auto-render.min.js"></script>
</head><body>
<h1>{title}</h1>
{default}
<p class="summary">{summary}</p>
{math}
{details}
{implementation}
<nav>
<a href="cppidoc://tech/#{concept}">Open in Technical Documentation (§{tech_number}) ↗</a>
<a href="cppidoc://theory/#{concept}">Open in Strategy Theory Documentation (§{theory_number}) ↗</a>
</nav>
<script>
renderMathInElement(document.body, {{
  delimiters: [
    {{left: "$$", right: "$$", display: true}},
    {{left: "$", right: "$", display: false}}
  ],
  throwOnError: false
}});
</script>
</body></html>
"""


def popover_html(registry: DocRegistry, popover_id: str) -> str:
    """Render one popover entry as a self-contained KaTeX page."""
    entry = registry.popover(popover_id)
    concept = registry.concept(entry["concept"])
    default = entry.get("default", "")
    details = entry.get("details", [])
    math = entry.get("math", "")
    implementation = entry.get("implementation", "")
    return _POPOVER_PAGE.format(
        title=entry["title"],
        default=f'<p class="default">Default: <code>{html.escape(default)}</code></p>'
        if default
        else "",
        summary=entry["summary"],
        math=f'<div class="math">$${math}$$</div>' if math else "",
        details="<ul>" + "".join(f"<li>{item}</li>" for item in details) + "</ul>"
        if details
        else "",
        implementation=f'<p class="impl"><b>Implementation.</b> {implementation}</p>'
        if implementation
        else "",
        concept=entry["concept"],
        tech_number=concept.tech.number if concept else "",
        theory_number=concept.theory.number if concept else "",
    )


def popover_text(registry: DocRegistry, popover_id: str) -> str:
    """Plain rich-text rendering of a popover, used when QtWebEngine is off."""
    entry = registry.popover(popover_id)
    parts = [f"<h3>{entry['title']}</h3>"]
    if entry.get("default"):
        parts.append(f"<p>Default: <code>{html.escape(entry['default'])}</code></p>")
    parts.append(f"<p>{entry['summary']}</p>")
    if entry.get("math"):
        parts.append(f"<p><code>{html.escape(entry['math'])}</code></p>")
    for item in entry.get("details", []):
        parts.append(f"<p>• {item}</p>")
    if entry.get("implementation"):
        parts.append(f"<p><b>Implementation.</b> {entry['implementation']}</p>")
    concept = entry["concept"]
    parts.append(
        f'<p><a href="cppidoc://tech/#{concept}">Open in Technical Documentation ↗</a><br>'
        f'<a href="cppidoc://theory/#{concept}">Open in Strategy Theory Documentation ↗</a></p>'
    )
    return "".join(parts)


def _qwebchannel_js() -> bytes:
    handle = QtCore.QFile(":/qtwebchannel/qwebchannel.js")
    if not handle.open(QtCore.QIODevice.OpenModeFlag.ReadOnly):
        return b""
    try:
        return bytes(handle.readAll().data())
    finally:
        handle.close()


class DocSchemeHandler(QWebEngineUrlSchemeHandler):
    """Serves documentation, assets and popover pages for ``cppidoc://``."""

    def __init__(self, registry: DocRegistry, parent: QtCore.QObject | None = None) -> None:
        super().__init__(parent)
        self._registry = registry

    def resolve(self, url: QtCore.QUrl) -> tuple[bytes, bytes] | None:
        """Return ``(mime, body)`` for a URL, or ``None`` when it does not exist."""
        host, path = url.host(), url.path()
        if path.startswith("/assets/"):
            relative = path[len("/assets/") :]
            if relative == "qwebchannel.js":
                return b"text/javascript", _qwebchannel_js()
            target = (ASSETS_DIR / relative).resolve()
            if ASSETS_DIR not in target.parents or not target.is_file():
                return None
            mime = _MIME.get(target.suffix.lower())
            return (mime, target.read_bytes()) if mime else None
        if host in VOLUMES and path in ("", "/"):
            page = DOCS_DIR / self._registry.volumes[host]["file"]
            return b"text/html", page.read_bytes()
        if host == "popover":
            popover_id = path.strip("/")
            if popover_id in self._registry.popovers:
                return b"text/html", popover_html(self._registry, popover_id).encode("utf-8")
        return None

    def requestStarted(self, job: QWebEngineUrlRequestJob) -> None:
        resolved = self.resolve(job.requestUrl())
        if resolved is None:
            job.fail(QWebEngineUrlRequestJob.Error.UrlNotFound)
            return
        mime, body = resolved
        buffer = QtCore.QBuffer(job)
        buffer.setData(body)
        buffer.open(QtCore.QIODevice.OpenModeFlag.ReadOnly)
        job.reply(mime, buffer)


def install_handler(profile: QWebEngineProfile, registry: DocRegistry) -> DocSchemeHandler:
    """Attach the scheme handler to a profile (once per profile)."""
    existing = profile.urlSchemeHandler(SCHEME.encode())
    if isinstance(existing, DocSchemeHandler):
        return existing
    handler = DocSchemeHandler(registry, profile)
    profile.installUrlSchemeHandler(SCHEME.encode(), handler)
    return handler
