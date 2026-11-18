"""GPU rendering of the chart canvas: layers, the sprite atlas, frames and back ends.

pyqtgraph's built-in OpenGL curve path draws ``GL_LINE_STRIP`` primitives and
therefore depends on the driver's ``glLineWidth`` range.  Many drivers (every
core profile, and the D3D12 translation layer used by WSLg) cap that range at
exactly one pixel and have no notion of dashes, so the line weights and dash
styles of the TSD visual-encoding matrix cannot be honoured that way.  Its
axis, text and scatter items go through ``QPainter``, which on an OpenGL
viewport costs several milliseconds per frame in state changes alone.

The canvas is therefore drawn in one pass:

* a :class:`Frame` is a display list - ribbon triangles in one vertex array,
  textured quads (glyphs and markers) in another, and the ordered draw
  operations that refer to them;
* :class:`GLBackend` uploads a frame once and replays it with two small
  shaders inside a single native-painting block;
* :class:`RasterBackend` draws the same frame with ``QPainter`` when the
  viewport is not an OpenGL widget (``--no-opengl``, offscreen tests).

The workspace keeps two frames: the *scene* (grid, layers, markers, axes),
rebuilt when the view or the data changes, and the *overlay* (crosshair and
read-outs), rebuilt when the cursor moves.  A crosshair step therefore
uploads a few hundred bytes and replays buffers that are already on the GPU.

Line weight, dashing and anti-aliasing come from the ribbon shader and are
identical on every driver.  Text and markers are rasterised by Qt into a
:class:`SpriteAtlas` once and drawn as textured quads.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pyqtgraph as pg
from numpy.typing import NDArray
from pyqtgraph.graphicsItems.ScatterPlotItem import renderSymbol
from pyqtgraph.Qt import OpenGLHelpers
from PySide6 import QtCore, QtGui, QtOpenGL, QtWidgets

from cppi_lab.ui import theme
from cppi_lab.ui.chart_geometry import (
    AA_MARGIN,
    Vertices,
    build_pyramid,
    range_min_max,
    segments_into,
    sprites_into,
    tessellate_band,
    tessellate_polyline,
)

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
Rgba = tuple[float, float, float, float]
#: ``(m11, dx, m22, dy)``: pixel = data * m + d on each axis.
Transform = tuple[float, float, float, float]

LayerKind = Literal["line", "step", "band", "segments", "stems"]
ReduceMode = Literal["extreme", "mean"]

GL_TRIANGLES = 0x0004
GL_TRIANGLE_STRIP = 0x0005
GL_ONE = 1
GL_ONE_MINUS_SRC_ALPHA = 0x0303
GL_SCISSOR_TEST = 0x0C11
GL_BLEND = 0x0BE2
GL_DEPTH_TEST = 0x0B71
GL_CULL_FACE = 0x0B44
GL_FLOAT = 0x1406
GL_TEXTURE0 = 0x84C0

_RIBBON_STRIDE = 12
_SPRITE_STRIDE = 16

_RIBBON_VERT_LEGACY = """
attribute vec2 a_position;
attribute float a_across;
uniform vec2 u_viewport;
varying float v_across;
void main() {
    v_across = a_across;
    gl_Position = vec4(a_position.x * u_viewport.x - 1.0,
                       1.0 - a_position.y * u_viewport.y, 0.0, 1.0);
}
"""
_RIBBON_FRAG_LEGACY = """
#ifdef GL_ES
precision mediump float;
#endif
uniform vec4 u_color;
uniform float u_half;
uniform vec2 u_dash;
varying float v_across;
void main() {
    float coverage = clamp(u_half + 0.5 - abs(v_across), 0.0, 1.0);
    if (u_dash.y > 0.0) {
        float phase = mod(gl_FragCoord.x + gl_FragCoord.y, u_dash.y);
        if (phase > u_dash.x) discard;
    }
    gl_FragColor = u_color * coverage;
}
"""
_RIBBON_VERT_140 = """#version 140
in vec2 a_position;
in float a_across;
uniform vec2 u_viewport;
out float v_across;
void main() {
    v_across = a_across;
    gl_Position = vec4(a_position.x * u_viewport.x - 1.0,
                       1.0 - a_position.y * u_viewport.y, 0.0, 1.0);
}
"""
_RIBBON_FRAG_140 = """#version 140
uniform vec4 u_color;
uniform float u_half;
uniform vec2 u_dash;
in float v_across;
out vec4 fragColor;
void main() {
    float coverage = clamp(u_half + 0.5 - abs(v_across), 0.0, 1.0);
    if (u_dash.y > 0.0) {
        float phase = mod(gl_FragCoord.x + gl_FragCoord.y, u_dash.y);
        if (phase > u_dash.x) discard;
    }
    fragColor = u_color * coverage;
}
"""
_SPRITE_VERT_LEGACY = """
attribute vec2 a_position;
attribute vec2 a_uv;
uniform vec2 u_viewport;
varying vec2 v_uv;
void main() {
    v_uv = a_uv;
    gl_Position = vec4(a_position.x * u_viewport.x - 1.0,
                       1.0 - a_position.y * u_viewport.y, 0.0, 1.0);
}
"""
_SPRITE_FRAG_LEGACY = """
#ifdef GL_ES
precision mediump float;
#endif
uniform sampler2D u_atlas;
uniform vec4 u_color;
varying vec2 v_uv;
void main() {
    gl_FragColor = texture2D(u_atlas, v_uv) * u_color;
}
"""
_SPRITE_VERT_140 = """#version 140
in vec2 a_position;
in vec2 a_uv;
uniform vec2 u_viewport;
out vec2 v_uv;
void main() {
    v_uv = a_uv;
    gl_Position = vec4(a_position.x * u_viewport.x - 1.0,
                       1.0 - a_position.y * u_viewport.y, 0.0, 1.0);
}
"""
_SPRITE_FRAG_140 = """#version 140
uniform sampler2D u_atlas;
uniform vec4 u_color;
in vec2 v_uv;
out vec4 fragColor;
void main() {
    fragColor = texture(u_atlas, v_uv) * u_color;
}
"""


def premultiplied(color: QtGui.QColor, opacity: float = 1.0) -> Rgba:
    """Premultiplied float RGBA of a colour; both shaders blend premultiplied."""
    alpha = color.alphaF() * opacity
    return (color.redF() * alpha, color.greenF() * alpha, color.blueF() * alpha, alpha)


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------


class Series:
    """A full-resolution series together with its level-of-detail pyramid."""

    __slots__ = ("maxs", "mins", "offsets", "sums", "y")

    def __init__(self, y: FloatArray) -> None:
        self.y: FloatArray = np.ascontiguousarray(y, dtype=np.float64)
        self.mins, self.maxs, self.sums, self.offsets = build_pyramid(self.y)

    def min_max(self, lo: int, hi: int) -> tuple[float, float]:
        """Exact finite extremes of ``y[lo:hi]`` (``nan, nan`` when there are none)."""
        low, high = range_min_max(self.y, self.mins, self.maxs, self.offsets, lo, hi)
        return float(low), float(high)


_EMPTY = np.empty(0, dtype=np.float64)


class Layer:
    """One styled series of a pane: its data, its style and its tessellation."""

    def __init__(
        self,
        kind: LayerKind,
        color: str | QtGui.QColor,
        width: float = 1.0,
        dash: tuple[float, float] | None = None,
        stem_px: float = 0.0,
        reduce: ReduceMode = "extreme",
    ) -> None:
        self.kind: LayerKind = kind
        # ``pg.mkColor`` reads eight-digit hex as RRGGBBAA (the TSD notation);
        # ``QColor`` would read it as AARRGGBB.
        self.color: QtGui.QColor = pg.mkColor(color)
        #: The colour as the shaders want it; kept in step by :meth:`set_alpha`.
        self.rgba: Rgba = premultiplied(self.color)
        #: How samples are combined when several bars share a pixel.
        self.reduce: ReduceMode = reduce
        self.width = float(width)
        #: ``(on, off)`` dash lengths in pixels, or ``None`` for a solid line.
        self.dash = dash
        #: Length in pixels of the downward stem drawn by the ``stems`` kind.
        self.stem_px = float(stem_px)
        self._x: FloatArray = _EMPTY
        self._y: FloatArray = _EMPTY
        self._y2: FloatArray = _EMPTY
        self._x2: FloatArray = _EMPTY
        self._series: Series | None = None
        self._series2: Series | None = None

    # -- data ------------------------------------------------------------------
    def set_polyline(self, x: FloatArray, y: FloatArray | Series) -> None:
        """Data of a ``line`` or ``step`` layer (``x`` ascending)."""
        series = y if isinstance(y, Series) else Series(y)
        self._x = np.ascontiguousarray(x, dtype=np.float64)
        self._y, self._y2, self._x2 = series.y, _EMPTY, _EMPTY
        self._series, self._series2 = series, None

    def set_band(
        self, x: FloatArray, y_low: FloatArray | Series, y_high: FloatArray | Series
    ) -> None:
        """Data of a ``band`` layer: the area between two curves."""
        low = y_low if isinstance(y_low, Series) else Series(y_low)
        high = y_high if isinstance(y_high, Series) else Series(y_high)
        self._x = np.ascontiguousarray(x, dtype=np.float64)
        self._y, self._y2, self._x2 = low.y, high.y, _EMPTY
        self._series, self._series2 = low, high

    def set_segments(self, x0: FloatArray, y0: FloatArray, x1: FloatArray, y1: FloatArray) -> None:
        """Data of a ``segments`` layer: independent segments, ``x0`` ascending."""
        self._x = np.ascontiguousarray(x0, dtype=np.float64)
        self._y = np.ascontiguousarray(y0, dtype=np.float64)
        self._x2 = np.ascontiguousarray(x1, dtype=np.float64)
        self._y2 = np.ascontiguousarray(y1, dtype=np.float64)
        self._series = self._series2 = None

    def set_stems(self, x: FloatArray, y: FloatArray) -> None:
        """Data of a ``stems`` layer: a fixed-pixel stem below each point, ``x`` ascending."""
        self._x = np.ascontiguousarray(x, dtype=np.float64)
        self._y = np.ascontiguousarray(y, dtype=np.float64)
        self._x2 = self._y2 = _EMPTY
        self._series = self._series2 = None

    def clear(self) -> None:
        self._x = self._y = self._y2 = self._x2 = _EMPTY
        self._series = self._series2 = None

    @property
    def size(self) -> int:
        return int(self._x.shape[0])

    def set_alpha(self, alpha: float) -> None:
        """Change the opacity of the layer's colour."""
        self.color.setAlphaF(alpha)
        self.rgba = premultiplied(self.color)

    def ranged(self) -> tuple[tuple[FloatArray, Series], ...]:
        """The (x, series) pairs that take part in auto-ranging the value axis."""
        if self._series is None or self._x.shape[0] == 0:
            return ()
        if self._series2 is None:
            return ((self._x, self._series),)
        return ((self._x, self._series), (self._x, self._series2))

    # -- queries -----------------------------------------------------------------
    def _visible(self, x_min: float, x_max: float) -> tuple[int, int]:
        """Index range in view, one sample beyond each edge so lines reach the border."""
        lo = max(0, int(np.searchsorted(self._x, x_min, side="left")) - 1)
        hi = min(self._x.shape[0], int(np.searchsorted(self._x, x_max, side="right")) + 1)
        return lo, hi

    def y_bounds(self, x_min: float, x_max: float) -> tuple[float, float] | None:
        """Extremes of the samples inside an x-range, for auto-ranging the value axis.

        Stems and free segments are decorations and do not take part.
        """
        series = self._series
        if series is None or self._x.shape[0] == 0:
            return None
        lo, hi = self._visible(x_min, x_max)
        if hi <= lo:
            return None
        low, high = series.min_max(lo, hi)
        if self._series2 is not None:
            low2, high2 = self._series2.min_max(lo, hi)
            low, high = np.nanmin([low, low2]), np.nanmax([high, high2])
        if not (math.isfinite(low) and math.isfinite(high)):
            return None
        return float(low), float(high)

    def max_rows(self, pixels: int) -> int:
        """Upper bound of the vertex rows :meth:`tessellate` can write."""
        if self.kind in ("line", "step"):
            return 8 * pixels + 8
        if self.kind == "band":
            return 4 * pixels + 8
        return 6 * int(self._x.shape[0])

    def visible(self, x_min: float, x_max: float) -> tuple[int, int]:
        """Index range of the samples a view of ``[x_min, x_max]`` has to draw."""
        return self._visible(x_min, x_max)

    def tessellate(
        self,
        out: Vertices,
        offset: int,
        transform: Transform,
        x_min: float,
        x_max: float,
        pixels: int,
        span: tuple[int, int] | None = None,
    ) -> int:
        """Write the triangles of the part in view into ``out``; returns the row count.

        ``span`` is the result of :meth:`visible` when the caller already has it
        (layers of one pane usually share their x array).
        """
        if self._x.shape[0] == 0:
            return 0
        m11, dx, m22, dy = transform
        half = 0.5 * self.width + AA_MARGIN
        kind = self.kind
        if kind == "segments" or kind == "stems":
            lo = int(np.searchsorted(self._x, x_min, side="left"))
            hi = int(np.searchsorted(self._x, x_max, side="right"))
            if hi <= lo:
                return 0
            px = self._x[lo:hi] * m11 + dx
            py = self._y[lo:hi] * m22 + dy
            if kind == "stems":
                return int(segments_into(out, offset, px, py, px, py + self.stem_px, half))
            px2 = self._x2[lo:hi] * m11 + dx
            py2 = self._y2[lo:hi] * m22 + dy
            return int(segments_into(out, offset, px, py, px2, py2, half))
        series = self._series
        assert series is not None
        lo, hi = span if span is not None else self._visible(x_min, x_max)
        mean = self.reduce == "mean"
        if kind == "band":
            high = self._series2
            assert high is not None
            return int(
                tessellate_band(
                    out,
                    offset,
                    self._x,
                    series.y,
                    high.y,
                    series.mins,
                    series.sums,
                    high.maxs,
                    high.sums,
                    series.offsets,
                    lo,
                    hi,
                    pixels,
                    mean,
                    m11,
                    dx,
                    m22,
                    dy,
                )
            )
        return int(
            tessellate_polyline(
                out,
                offset,
                self._x,
                series.y,
                series.mins,
                series.maxs,
                series.sums,
                series.offsets,
                lo,
                hi,
                pixels,
                mean,
                kind == "step",
                m11,
                dx,
                m22,
                dy,
                half,
            )
        )


@dataclass
class MarkerSet:
    """One kind of execution marker: a symbol drawn at each ``(x, y)``, ``x`` ascending."""

    symbol: str
    base_size: float
    colour: str
    hollow: bool = False
    x: FloatArray = field(default_factory=lambda: _EMPTY)
    y: FloatArray = field(default_factory=lambda: _EMPTY)

    def set_data(self, x: FloatArray, y: FloatArray) -> None:
        self.x = np.ascontiguousarray(x, dtype=np.float64)
        self.y = np.ascontiguousarray(y, dtype=np.float64)


# --------------------------------------------------------------------------
# Sprite atlas
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Sprite:
    """A rectangle of the atlas: texture coordinates and size in logical pixels."""

    u0: float
    v0: float
    u1: float
    v1: float
    width: float
    height: float
    #: Position of the rectangle inside the atlas image, in device pixels.
    source: tuple[int, int, int, int]


class _AtlasFont:
    """Glyph table of one font inside the atlas."""

    def __init__(self, font: QtGui.QFont, dpr: float) -> None:
        self.font = QtGui.QFont(font)
        device = QtGui.QFont(font)
        device.setPixelSize(max(1, round(font.pixelSize() * dpr)))
        self.device_font = device
        self.metrics = QtGui.QFontMetricsF(device)
        self.ascent = math.ceil(self.metrics.ascent())
        self.descent = math.ceil(self.metrics.descent())
        #: Code point to glyph index; ``-1`` while a glyph has not been rasterised.
        self.lookup = np.full(0x10000, -1, dtype=np.int32)
        self.advance: list[float] = []
        self.rects: list[tuple[int, int, int, int]] = []
        self.advance_array = np.empty(0, dtype=np.float64)
        self.rect_array = np.empty((0, 4), dtype=np.float64)
        self.runs: dict[str, tuple[NDArray[np.float32], float]] = {}


class SpriteAtlas:
    """One texture holding every glyph and marker the canvas draws.

    Qt rasterises each glyph and each marker symbol once, at the device pixel
    ratio, into a premultiplied RGBA image.  Glyphs are stored white so one
    uniform colour tints a whole run of text.  :attr:`version` changes
    whenever the image does, which tells the OpenGL back end to upload it
    again; :attr:`generation` changes when the atlas had to start over, which
    invalidates every texture coordinate handed out before.
    """

    SIZE = 1024
    PAD = 2

    def __init__(self, dpr: float = 1.0) -> None:
        self.dpr = float(dpr)
        self.image = QtGui.QImage(
            self.SIZE, self.SIZE, QtGui.QImage.Format.Format_RGBA8888_Premultiplied
        )
        self.image.fill(QtCore.Qt.GlobalColor.transparent)
        self.version = 0
        self.generation = 0
        self._cursor_x = 0
        self._cursor_y = 0
        self._row_height = 0
        self._fonts: list[_AtlasFont] = []
        self._markers: dict[tuple[Any, ...], Sprite] = {}

    # -- packing -----------------------------------------------------------------
    def _allocate(self, width: int, height: int) -> tuple[int, int]:
        """A free cell on the current shelf, starting a new shelf when it is full."""
        if self._cursor_x + width > self.SIZE:
            self._cursor_x = 0
            self._cursor_y += self._row_height
            self._row_height = 0
        if self._cursor_y + height > self.SIZE:
            self._reset()
        x, y = self._cursor_x, self._cursor_y
        self._cursor_x += width
        self._row_height = max(self._row_height, height)
        return x, y

    def _reset(self) -> None:
        """Start over when the image is full; every earlier coordinate becomes invalid."""
        self.image.fill(QtCore.Qt.GlobalColor.transparent)
        self._cursor_x = self._cursor_y = self._row_height = 0
        self._markers.clear()
        self._fonts = [_AtlasFont(entry.font, self.dpr) for entry in self._fonts]
        self.generation += 1
        self.version += 1

    def _sprite(self, x: int, y: int, width: int, height: int) -> Sprite:
        size = float(self.SIZE)
        return Sprite(
            x / size,
            y / size,
            (x + width) / size,
            (y + height) / size,
            width / self.dpr,
            height / self.dpr,
            (x, y, width, height),
        )

    # -- text --------------------------------------------------------------------
    def add_font(self, font: QtGui.QFont) -> int:
        """Register a font (with a pixel size); returns its id."""
        self._fonts.append(_AtlasFont(font, self.dpr))
        return len(self._fonts) - 1

    def font(self, font_id: int) -> QtGui.QFont:
        return self._fonts[font_id].font

    def ascent(self, font_id: int) -> float:
        """Ascent in logical pixels."""
        return self._fonts[font_id].ascent / self.dpr

    def descent(self, font_id: int) -> float:
        return self._fonts[font_id].descent / self.dpr

    def _rasterise(self, entry: _AtlasFont, codes: list[int]) -> None:
        painter = QtGui.QPainter(self.image)
        painter.setFont(entry.device_font)
        painter.setPen(QtGui.QColor("#FFFFFF"))
        painter.setRenderHint(QtGui.QPainter.RenderHint.TextAntialiasing, True)
        height = entry.ascent + entry.descent + 2 * self.PAD
        generation = self.generation
        try:
            for code in codes:
                glyph = chr(code)
                advance = entry.metrics.horizontalAdvance(glyph)
                width = math.ceil(advance) + 2 * self.PAD
                x, y = self._allocate(width, height)
                if self.generation != generation:
                    break  # the atlas started over; the caller retries
                painter.drawText(QtCore.QPointF(x + self.PAD, y + self.PAD + entry.ascent), glyph)
                entry.lookup[code] = len(entry.advance)
                entry.advance.append(advance)
                entry.rects.append((x, y, width, height))
        finally:
            painter.end()
        entry.advance_array = np.array(entry.advance, dtype=np.float64)
        entry.rect_array = np.array(entry.rects, dtype=np.float64).reshape(-1, 4)
        self.version += 1

    def run(self, font_id: int, text: str) -> tuple[NDArray[np.float32], float]:
        """Glyph quads of ``text`` and its advance width.

        The quads are ``(6 n, 4)`` rows ``(x, y, u, v)`` in logical pixels
        relative to the pen position on the baseline.
        """
        entry = self._fonts[font_id]
        cached = entry.runs.get(text)
        if cached is not None:
            return cached
        codes = [min(ord(ch), 0xFFFF) for ch in text]
        index = entry.lookup[codes]
        while np.any(index < 0):
            missing = sorted({code for code, slot in zip(codes, index, strict=True) if slot < 0})
            self._rasterise(self._fonts[font_id], missing)
            entry = self._fonts[font_id]  # replaced if the atlas started over
            index = entry.lookup[codes]
        advance = entry.advance_array[index]
        pen = np.concatenate(([0.0], np.cumsum(advance)[:-1]))
        rects = entry.rect_array[index]
        left = np.round(pen) - self.PAD
        top = float(-entry.ascent - self.PAD)
        right = left + rects[:, 2]
        bottom = top + rects[:, 3]
        size = float(self.SIZE)
        u0, v0 = rects[:, 0] / size, rects[:, 1] / size
        u1, v1 = (rects[:, 0] + rects[:, 2]) / size, (rects[:, 1] + rects[:, 3]) / size
        quads = np.empty((len(codes), 6, 4), dtype=np.float32)
        tops = np.full(len(codes), top)
        for corner, (cx, cy, cu, cv) in enumerate(
            (
                (left, tops, u0, v0),
                (right, tops, u1, v0),
                (left, bottom, u0, v1),
                (left, bottom, u0, v1),
                (right, tops, u1, v0),
                (right, bottom, u1, v1),
            )
        ):
            quads[:, corner, 0] = cx / self.dpr
            quads[:, corner, 1] = cy / self.dpr
            quads[:, corner, 2] = cu
            quads[:, corner, 3] = cv
        result = (quads.reshape(-1, 4), float(advance.sum()) / self.dpr)
        if len(entry.runs) > 8192:
            entry.runs.clear()
        entry.runs[text] = result
        return result

    def text_width(self, font_id: int, text: str) -> float:
        """Advance width of ``text`` in logical pixels."""
        return self.run(font_id, text)[1]

    # -- markers -----------------------------------------------------------------
    def marker(self, symbol: str, size: float, colour: str, hollow: bool) -> Sprite:
        """A pyqtgraph scatter symbol; filled with a canvas-coloured rim, or a ring."""
        key = (symbol, round(size, 3), colour, hollow)
        sprite = self._markers.get(key)
        if sprite is not None:
            return sprite
        if hollow:
            pen = pg.mkPen(colour, width=2.0)
            brush = QtGui.QBrush(QtCore.Qt.BrushStyle.NoBrush)
        else:
            pen = pg.mkPen(theme.CANVAS, width=1.0)
            brush = pg.mkBrush(colour)
        image = renderSymbol(symbol, size, pen, brush, dpr=self.dpr)
        image.setDevicePixelRatio(1.0)
        x, y = self._allocate(image.width() + 2, image.height() + 2)
        painter = QtGui.QPainter(self.image)
        painter.drawImage(QtCore.QPoint(x + 1, y + 1), image)
        painter.end()
        sprite = self._sprite(x + 1, y + 1, image.width(), image.height())
        self._markers[key] = sprite
        self.version += 1
        return sprite


# --------------------------------------------------------------------------
# Frames
# --------------------------------------------------------------------------


@dataclass(slots=True)
class RibbonOp:
    """Draw a range of the ribbon vertex array with one style."""

    tag: str
    clip: int
    first: int
    count: int
    strip: bool
    fill: bool
    rgba: Rgba
    width: float
    dash: tuple[float, float] | None


@dataclass(slots=True)
class SpriteOp:
    """Draw a range of the sprite vertex array with one tint.

    ``texts`` and ``markers`` describe the same content for the raster back end.
    """

    tag: str
    clip: int
    first: int
    count: int
    rgba: Rgba
    texts: list[tuple[float, float, str, int]] = field(default_factory=list)
    markers: list[tuple[Sprite, FloatArray, FloatArray]] = field(default_factory=list)


Op = RibbonOp | SpriteOp
_WHITE: Rgba = (1.0, 1.0, 1.0, 1.0)


class Frame:
    """A display list: vertex arrays plus the ordered operations that draw them."""

    def __init__(self) -> None:
        self.ribbons: Vertices = np.empty((1 << 16, 3), dtype=np.float32)
        self.sprites: Vertices = np.empty((1 << 13, 4), dtype=np.float32)
        self.n_ribbons = 0
        self.n_sprites = 0
        self.ops: list[Op] = []
        #: Clip rectangles ``(x, y, width, height)`` in logical pixels.
        self.clips: list[tuple[float, float, float, float]] = []
        self.width = 0.0
        self.height = 0.0
        self.dpr = 1.0
        #: Incremented by every :meth:`begin`; the back ends upload when it changes.
        self.serial = 0

    def begin(self, width: float, height: float, dpr: float) -> None:
        self.width, self.height, self.dpr = float(width), float(height), float(dpr)
        self.n_ribbons = self.n_sprites = 0
        self.ops = []
        self.clips = []
        self.serial += 1

    def clip(self, x: float, y: float, width: float, height: float) -> int:
        """Register a clip rectangle; returns the id operations refer to."""
        self.clips.append((x, y, width, height))
        return len(self.clips) - 1

    def find(self, tag: str) -> list[Op]:
        """Operations carrying ``tag`` (introspection for tests and diagnostics)."""
        return [op for op in self.ops if op.tag == tag]

    def _reserve_ribbons(self, rows: int) -> None:
        needed = self.n_ribbons + rows
        if needed > self.ribbons.shape[0]:
            grown = np.empty((max(needed, 2 * self.ribbons.shape[0]), 3), dtype=np.float32)
            grown[: self.n_ribbons] = self.ribbons[: self.n_ribbons]
            self.ribbons = grown

    def _reserve_sprites(self, rows: int) -> None:
        needed = self.n_sprites + rows
        if needed > self.sprites.shape[0]:
            grown = np.empty((max(needed, 2 * self.sprites.shape[0]), 4), dtype=np.float32)
            grown[: self.n_sprites] = self.sprites[: self.n_sprites]
            self.sprites = grown

    # -- ribbons -----------------------------------------------------------------
    def add_layer(
        self,
        tag: str,
        clip: int,
        layer: Layer,
        transform: Transform,
        x_min: float,
        x_max: float,
        pixels: int,
        span: tuple[int, int] | None = None,
    ) -> None:
        """Tessellate the part of ``layer`` in view and queue its draw."""
        if layer.size == 0:
            return
        self._reserve_ribbons(layer.max_rows(pixels))
        rows = layer.tessellate(self.ribbons, self.n_ribbons, transform, x_min, x_max, pixels, span)
        if rows < 3:
            return
        fill = layer.kind == "band"
        self.ops.append(
            RibbonOp(
                tag,
                clip,
                self.n_ribbons,
                rows,
                layer.kind in ("line", "step", "band"),
                fill,
                layer.rgba,
                layer.width,
                None if fill else layer.dash,
            )
        )
        self.n_ribbons += rows

    def add_segments(
        self,
        tag: str,
        clip: int,
        x0: FloatArray,
        y0: FloatArray,
        x1: FloatArray,
        y1: FloatArray,
        rgba: Rgba,
        width: float = 1.0,
        dash: tuple[float, float] | None = None,
    ) -> None:
        """Queue independent straight segments given in pixels."""
        count = int(x0.shape[0])
        if count == 0:
            return
        self._reserve_ribbons(6 * count)
        rows = int(
            segments_into(self.ribbons, self.n_ribbons, x0, y0, x1, y1, 0.5 * width + AA_MARGIN)
        )
        self.ops.append(RibbonOp(tag, clip, self.n_ribbons, rows, False, False, rgba, width, dash))
        self.n_ribbons += rows

    def add_rect(
        self, tag: str, clip: int, x: float, y: float, width: float, height: float, rgba: Rgba
    ) -> None:
        """Queue a filled rectangle given in pixels."""
        self._reserve_ribbons(6)
        base = self.n_ribbons
        corners = ((x, y), (x + width, y), (x, y + height), (x + width, y + height))
        for row, corner in enumerate((0, 1, 2, 2, 1, 3)):
            self.ribbons[base + row, 0] = corners[corner][0]
            self.ribbons[base + row, 1] = corners[corner][1]
            self.ribbons[base + row, 2] = 0.0
        self.ops.append(RibbonOp(tag, clip, base, 6, False, True, rgba, 0.0, None))
        self.n_ribbons += 6

    # -- sprites -----------------------------------------------------------------
    def _sprite_op(self, tag: str, clip: int, rgba: Rgba) -> SpriteOp:
        last = self.ops[-1] if self.ops else None
        if (
            isinstance(last, SpriteOp)
            and last.tag == tag
            and last.clip == clip
            and last.rgba == rgba
            and last.first + last.count == self.n_sprites
        ):
            return last
        op = SpriteOp(tag, clip, self.n_sprites, 0, rgba)
        self.ops.append(op)
        return op

    def add_text(
        self,
        tag: str,
        clip: int,
        atlas: SpriteAtlas,
        font: int,
        text: str,
        x: float,
        baseline: float,
        rgba: Rgba,
        align: float = 0.0,
    ) -> float:
        """Queue one line of text; returns its width.

        ``align`` is the fraction of the width that lies left of ``x``: 0 starts
        the text there, 0.5 centres it, 1 ends it there.  The pen position is
        snapped to device pixels so glyphs stay crisp.
        """
        if not text:
            return 0.0
        quads, width = atlas.run(font, text)
        dpr = self.dpr
        pen_x = round((x - align * width) * dpr) / dpr
        pen_y = round(baseline * dpr) / dpr
        rows = quads.shape[0]
        self._reserve_sprites(rows)
        target = self.sprites[self.n_sprites : self.n_sprites + rows]
        target[:] = quads
        target[:, 0] += pen_x
        target[:, 1] += pen_y
        op = self._sprite_op(tag, clip, rgba)
        op.count += rows
        op.texts.append((pen_x, pen_y, text, font))
        self.n_sprites += rows
        return width

    def add_texts(
        self,
        tag: str,
        clip: int,
        atlas: SpriteAtlas,
        font: int,
        items: Sequence[tuple[str, float, float]],
        rgba: Rgba,
        align: float = 0.0,
    ) -> None:
        """Queue many ``(text, x, baseline)`` lines of one font and colour at once.

        Equivalent to calling :meth:`add_text` for each, without the per-call cost;
        this is how the tick labels of a frame are added.
        """
        if not items:
            return
        dpr = self.dpr
        runs = []
        pens_x = []
        pens_y = []
        counts = []
        texts = []
        for text, x, baseline in items:
            if not text:
                continue
            quads, width = atlas.run(font, text)
            pen_x = round((x - align * width) * dpr) / dpr
            pen_y = round(baseline * dpr) / dpr
            runs.append(quads)
            pens_x.append(pen_x)
            pens_y.append(pen_y)
            counts.append(quads.shape[0])
            texts.append((pen_x, pen_y, text, font))
        if not runs:
            return
        rows = int(sum(counts))
        self._reserve_sprites(rows)
        target = self.sprites[self.n_sprites : self.n_sprites + rows]
        np.concatenate(runs, out=target)
        target[:, 0] += np.repeat(np.array(pens_x, dtype=np.float32), counts)
        target[:, 1] += np.repeat(np.array(pens_y, dtype=np.float32), counts)
        op = self._sprite_op(tag, clip, rgba)
        op.count += rows
        op.texts.extend(texts)
        self.n_sprites += rows

    def add_markers(
        self, tag: str, clip: int, sprite: Sprite, px: FloatArray, py: FloatArray
    ) -> None:
        """Queue one sprite centred on each ``(px, py)``."""
        count = int(px.shape[0])
        if count == 0:
            return
        self._reserve_sprites(6 * count)
        rows = int(
            sprites_into(
                self.sprites,
                self.n_sprites,
                px,
                py,
                sprite.width,
                sprite.height,
                sprite.u0,
                sprite.v0,
                sprite.u1,
                sprite.v1,
            )
        )
        op = self._sprite_op(tag, clip, _WHITE)
        op.count += rows
        op.markers.append((sprite, px, py))
        self.n_sprites += rows


# --------------------------------------------------------------------------
# Back ends
# --------------------------------------------------------------------------


class _FrameBuffers:
    """GPU copies of one frame's two vertex arrays."""

    def __init__(self) -> None:
        kind = QtOpenGL.QOpenGLBuffer.Type.VertexBuffer
        self.ribbons = QtOpenGL.QOpenGLBuffer(kind)
        self.sprites = QtOpenGL.QOpenGLBuffer(kind)
        self.ribbon_capacity = 0
        self.sprite_capacity = 0
        self.serial = -1
        self.frame: Frame | None = None

    def create(self) -> None:
        pattern = QtOpenGL.QOpenGLBuffer.UsagePattern.StreamDraw
        for buffer in (self.ribbons, self.sprites):
            buffer.create()
            buffer.setUsagePattern(pattern)

    def destroy(self) -> None:
        self.ribbons.destroy()
        self.sprites.destroy()

    def upload(self, frame: Frame) -> None:
        """Replace both buffers; the old storage is orphaned so the driver never waits."""
        nbytes = frame.n_ribbons * _RIBBON_STRIDE
        self.ribbons.bind()
        self.ribbon_capacity = max(self.ribbon_capacity, nbytes, 1 << 16)
        self.ribbons.allocate(self.ribbon_capacity)
        if nbytes:
            self.ribbons.write(0, frame.ribbons, nbytes)  # type: ignore[arg-type]
        self.ribbons.release()
        nbytes = frame.n_sprites * _SPRITE_STRIDE
        self.sprites.bind()
        self.sprite_capacity = max(self.sprite_capacity, nbytes, 1 << 14)
        self.sprites.allocate(self.sprite_capacity)
        if nbytes:
            self.sprites.write(0, frame.sprites, nbytes)  # type: ignore[arg-type]
        self.sprites.release()
        self.serial = frame.serial
        self.frame = frame


class GLBackend(QtCore.QObject):
    """Replays frames on one OpenGL viewport widget."""

    def __init__(self, widget: QtWidgets.QWidget) -> None:
        super().__init__(widget)
        self._context: QtGui.QOpenGLContext | None = None
        self._functions: Any = None
        self._vao = QtOpenGL.QOpenGLVertexArrayObject(self)
        self._ribbon: Any = None
        self._sprite: Any = None
        self._uniforms: dict[str, int] = {}
        self._buffers: list[_FrameBuffers] = []
        self._texture: Any = None
        self._atlas_key: tuple[int, int] | None = None

    # -- resources ---------------------------------------------------------------
    @staticmethod
    def _program(vertex: str, fragment: str, attributes: Sequence[str]) -> Any:
        program = QtOpenGL.QOpenGLShaderProgram()
        stage = QtOpenGL.QOpenGLShader.ShaderTypeBit
        if not program.addShaderFromSourceCode(stage.Vertex, vertex):
            raise RuntimeError(program.log())
        if not program.addShaderFromSourceCode(stage.Fragment, fragment):
            raise RuntimeError(program.log())
        for location, name in enumerate(attributes):
            program.bindAttributeLocation(name, location)
        if not program.link():
            raise RuntimeError(program.log())
        return program

    def _setup(self, context: QtGui.QOpenGLContext) -> None:
        """(Re)create every GPU object when the widget got a new context."""
        if self._context is context:
            return
        if self._context is not None:
            self._context.aboutToBeDestroyed.disconnect(self._release)
            self._release()
        self._context = context
        context.aboutToBeDestroyed.connect(self._release)
        self._functions = context.extraFunctions()
        modern = (not context.isOpenGLES()) and context.format().version() >= (3, 1)
        if modern:
            ribbon = (_RIBBON_VERT_140, _RIBBON_FRAG_140)
            sprite = (_SPRITE_VERT_140, _SPRITE_FRAG_140)
        else:
            ribbon = (_RIBBON_VERT_LEGACY, _RIBBON_FRAG_LEGACY)
            sprite = (_SPRITE_VERT_LEGACY, _SPRITE_FRAG_LEGACY)
        self._ribbon = self._program(*ribbon, ("a_position", "a_across"))
        self._sprite = self._program(*sprite, ("a_position", "a_uv"))
        self._uniforms = {
            "ribbon.viewport": self._ribbon.uniformLocation("u_viewport"),
            "ribbon.color": self._ribbon.uniformLocation("u_color"),
            "ribbon.half": self._ribbon.uniformLocation("u_half"),
            "ribbon.dash": self._ribbon.uniformLocation("u_dash"),
            "sprite.viewport": self._sprite.uniformLocation("u_viewport"),
            "sprite.color": self._sprite.uniformLocation("u_color"),
            "sprite.atlas": self._sprite.uniformLocation("u_atlas"),
        }
        self._vao.create()
        self._buffers = []
        self._texture = None
        self._atlas_key = None

    def _release(self) -> None:
        widget: Any = self.parent()
        widget.makeCurrent()
        for buffers in self._buffers:
            buffers.destroy()
        self._buffers = []
        if self._texture is not None:
            self._texture.destroy()
            self._texture = None
        self._atlas_key = None
        self._vao.destroy()
        self._ribbon = None
        self._sprite = None
        self._context = None
        self._functions = None
        widget.doneCurrent()

    def _upload_atlas(self, atlas: SpriteAtlas) -> None:
        key = (id(atlas), atlas.version)
        if key == self._atlas_key and self._texture is not None:
            return
        if self._texture is not None:
            self._texture.destroy()
        texture_type = QtOpenGL.QOpenGLTexture
        texture = texture_type(texture_type.Target.Target2D)
        texture.setFormat(texture_type.TextureFormat.RGBA8_UNorm)
        texture.setSize(atlas.SIZE, atlas.SIZE)
        texture.setMipLevels(1)
        texture.setMinMagFilters(texture_type.Filter.Linear, texture_type.Filter.Linear)
        texture.setWrapMode(texture_type.WrapMode.ClampToEdge)
        texture.allocateStorage(texture_type.PixelFormat.RGBA, texture_type.PixelType.UInt8)
        # The raw bytes are uploaded as they are: QOpenGLTexture.setData(QImage)
        # would convert to straight alpha, and filtering straight alpha leaves
        # dark fringes around every glyph and marker.
        pixels = bytes(atlas.image.constBits())
        texture.setData(
            texture_type.PixelFormat.RGBA,
            texture_type.PixelType.UInt8,
            pixels,  # type: ignore[call-overload]
        )
        self._texture = texture
        self._atlas_key = key

    # -- drawing -----------------------------------------------------------------
    def draw(self, frames: Sequence[Frame], atlas: SpriteAtlas) -> None:
        """Replay ``frames`` in order.  Call between begin/endNativePainting."""
        widget: Any = self.parent()
        self._setup(widget.context())
        glf = self._functions
        while len(self._buffers) < len(frames):
            buffers = _FrameBuffers()
            buffers.create()
            self._buffers.append(buffers)
        self._upload_atlas(atlas)
        if self._vao.isCreated():
            self._vao.bind()
        glf.glDisable(GL_DEPTH_TEST)
        glf.glDisable(GL_CULL_FACE)
        glf.glEnable(GL_BLEND)
        glf.glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA)
        glf.glActiveTexture(GL_TEXTURE0)
        self._texture.bind()
        uniforms = self._uniforms
        height = widget.height()
        for buffers, frame in zip(self._buffers, frames, strict=False):
            if buffers.serial != frame.serial or buffers.frame is not frame:
                buffers.upload(frame)
            if not frame.ops or frame.width <= 0 or frame.height <= 0:
                continue
            dpr = frame.dpr
            scale_x, scale_y = 2.0 / frame.width, 2.0 / frame.height
            scissors = [
                (
                    round(x * dpr),
                    round((height - y - h) * dpr),
                    max(0, round(w * dpr)),
                    max(0, round(h * dpr)),
                )
                for x, y, w, h in frame.clips
            ]
            bound: Any = None
            clip = -2
            style: tuple[Any, ...] | None = None
            for op in frame.ops:
                if op.clip != clip:
                    clip = op.clip
                    if clip < 0:
                        glf.glDisable(GL_SCISSOR_TEST)
                    else:
                        glf.glEnable(GL_SCISSOR_TEST)
                        glf.glScissor(*scissors[clip])
                if isinstance(op, RibbonOp):
                    program = self._ribbon
                    if bound is not program:
                        bound = program
                        style = None
                        program.bind()
                        buffers.ribbons.bind()
                        program.enableAttributeArray(0)
                        program.setAttributeBuffer(0, GL_FLOAT, 0, 2, _RIBBON_STRIDE)
                        program.enableAttributeArray(1)
                        program.setAttributeBuffer(1, GL_FLOAT, 8, 1, _RIBBON_STRIDE)
                        program.setUniformValue(uniforms["ribbon.viewport"], scale_x, scale_y)
                    wanted = (op.rgba, op.fill, op.width, op.dash)
                    if wanted != style:
                        style = wanted
                        program.setUniformValue(uniforms["ribbon.color"], *op.rgba)
                        program.setUniformValue1f(
                            uniforms["ribbon.half"], 1.0 if op.fill else 0.5 * op.width
                        )
                        if op.dash is None:
                            program.setUniformValue(uniforms["ribbon.dash"], 0.0, 0.0)
                        else:
                            on, off = op.dash
                            program.setUniformValue(
                                uniforms["ribbon.dash"], on * dpr, (on + off) * dpr
                            )
                    glf.glDrawArrays(
                        GL_TRIANGLE_STRIP if op.strip else GL_TRIANGLES, op.first, op.count
                    )
                else:
                    program = self._sprite
                    if bound is not program:
                        bound = program
                        style = None
                        program.bind()
                        buffers.sprites.bind()
                        program.enableAttributeArray(0)
                        program.setAttributeBuffer(0, GL_FLOAT, 0, 2, _SPRITE_STRIDE)
                        program.enableAttributeArray(1)
                        program.setAttributeBuffer(1, GL_FLOAT, 8, 2, _SPRITE_STRIDE)
                        program.setUniformValue(uniforms["sprite.viewport"], scale_x, scale_y)
                        program.setUniformValue1i(uniforms["sprite.atlas"], 0)
                    if op.rgba != style:
                        style = op.rgba
                        program.setUniformValue(uniforms["sprite.color"], *op.rgba)
                    glf.glDrawArrays(GL_TRIANGLES, op.first, op.count)
            if bound is not None:
                bound.release()
        glf.glDisable(GL_SCISSOR_TEST)
        glf.glDisable(GL_BLEND)
        self._texture.release()
        if self._vao.isCreated():
            self._vao.release()


class RasterBackend:
    """Draws frames with ``QPainter``; the fallback when there is no OpenGL viewport."""

    @staticmethod
    def _colour(rgba: Rgba) -> QtGui.QColor:
        r, g, b, a = rgba
        if a <= 0.0:
            return QtGui.QColor(0, 0, 0, 0)
        return QtGui.QColor.fromRgbF(min(1.0, r / a), min(1.0, g / a), min(1.0, b / a), a)

    def draw(self, painter: QtGui.QPainter, frames: Sequence[Frame], atlas: SpriteAtlas) -> None:
        painter.save()
        painter.resetTransform()
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QtGui.QPainter.RenderHint.SmoothPixmapTransform, True)
        for frame in frames:
            clip = -2
            for op in frame.ops:
                if op.clip != clip:
                    clip = op.clip
                    if clip < 0:
                        painter.setClipping(False)
                    else:
                        painter.setClipRect(QtCore.QRectF(*frame.clips[clip]))
                if isinstance(op, RibbonOp):
                    self._ribbon(painter, frame, op)
                else:
                    self._sprites(painter, op, atlas)
        painter.restore()

    def _ribbon(self, painter: QtGui.QPainter, frame: Frame, op: RibbonOp) -> None:
        rows = frame.ribbons[op.first : op.first + op.count].astype(np.float64)
        colour = self._colour(op.rgba)
        if op.fill:
            painter.setPen(QtCore.Qt.PenStyle.NoPen)
            painter.setBrush(colour)
            if op.strip:
                outline_x = np.concatenate((rows[0::2, 0], rows[1::2, 0][::-1]))
                outline_y = np.concatenate((rows[0::2, 1], rows[1::2, 1][::-1]))
                painter.drawPath(pg.arrayToQPath(outline_x, outline_y, finiteCheck=False))
            else:
                for quad in rows.reshape(-1, 6, 3):
                    low, high = quad[:, :2].min(axis=0), quad[:, :2].max(axis=0)
                    painter.drawRect(
                        QtCore.QRectF(low[0], low[1], high[0] - low[0], high[1] - low[1])
                    )
            return
        pen = QtGui.QPen(colour, op.width)
        pen.setJoinStyle(QtCore.Qt.PenJoinStyle.MiterJoin)
        pen.setCapStyle(QtCore.Qt.PenCapStyle.FlatCap)
        if op.dash is not None:
            pen.setDashPattern([op.dash[0] / op.width, op.dash[1] / op.width])
        painter.setPen(pen)
        painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
        if op.strip:
            centre = 0.5 * (rows[0::2, :2] + rows[1::2, :2])
            painter.drawPath(pg.arrayToQPath(centre[:, 0], centre[:, 1], finiteCheck=False))
        else:
            quads = rows.reshape(-1, 6, 3)
            start = 0.5 * (quads[:, 0, :2] + quads[:, 1, :2])
            end = 0.5 * (quads[:, 2, :2] + quads[:, 5, :2])
            points = np.empty((2 * start.shape[0], 2))
            points[0::2] = start
            points[1::2] = end
            painter.drawPath(
                pg.arrayToQPath(points[:, 0], points[:, 1], connect="pairs", finiteCheck=False)
            )

    def _sprites(self, painter: QtGui.QPainter, op: SpriteOp, atlas: SpriteAtlas) -> None:
        if op.texts:
            painter.setPen(self._colour(op.rgba))
            for x, baseline, text, font in op.texts:
                painter.setFont(atlas.font(font))
                painter.drawText(QtCore.QPointF(x, baseline), text)
        for sprite, px, py in op.markers:
            source = QtCore.QRectF(*sprite.source)
            for x, y in zip(px.tolist(), py.tolist(), strict=True):
                target = QtCore.QRectF(
                    x - 0.5 * sprite.width, y - 0.5 * sprite.height, sprite.width, sprite.height
                )
                painter.drawImage(target, atlas.image, source)


class ChartCanvas(pg.GraphicsLayoutWidget):
    """A pyqtgraph layout whose whole content is drawn by one frame replay.

    The plots inside it only provide geometry, ranges and mouse interaction;
    ``frames`` returns the display lists to draw on top of the empty scene.
    """

    def __init__(
        self,
        parent: QtWidgets.QWidget | None,
        frames: Callable[[], tuple[Sequence[Frame], SpriteAtlas]],
    ) -> None:
        super().__init__(parent)
        self._frames = frames
        self._gl: GLBackend | None = None
        self._raster = RasterBackend()

    @property
    def accelerated(self) -> bool:
        """Whether the viewport is an OpenGL widget."""
        return isinstance(self.viewport(), OpenGLHelpers.GraphicsViewGLWidget)

    def drawForeground(self, painter: QtGui.QPainter, rect: QtCore.QRectF) -> None:
        frames, atlas = self._frames()
        viewport = self.viewport()
        if isinstance(viewport, OpenGLHelpers.GraphicsViewGLWidget):
            if self._gl is None or self._gl.parent() is not viewport:
                self._gl = GLBackend(viewport)
            painter.beginNativePainting()
            try:
                self._gl.draw(frames, atlas)
            finally:
                painter.endNativePainting()
        else:
            self._raster.draw(painter, frames, atlas)
