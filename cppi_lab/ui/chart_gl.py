"""GPU layer primitives for the chart canvas.

pyqtgraph's built-in OpenGL curve path draws ``GL_LINE_STRIP`` primitives and
therefore depends on the driver's ``glLineWidth`` range.  Many drivers (every
core profile, and the D3D12 translation layer used by WSLg) cap that range at
exactly one pixel and have no notion of dashes, so the line weights and dash
styles of the TSD visual-encoding matrix cannot be honoured that way.

:class:`GLLayerItem` instead tessellates each layer into triangles in screen
space - a mitred ribbon for a polyline, a strip for a filled band, quads for
independent segments - and draws them with a small shader that anti-aliases
the ribbon edges analytically and cuts dashes in window space.  Line weight,
dashing and smoothing are then identical on every driver.

Each layer keeps its full-resolution arrays and reduces them to at most two
points per horizontal pixel (a min/max decimation that preserves every
extreme) for the range currently in view, so the per-frame cost is bounded by
the canvas width rather than by the back-test length.

When the viewport is not an OpenGL widget (``useOpenGL=False``, offscreen
tests, image export) the same geometry is drawn with ``QPainter``.
"""

from __future__ import annotations

import math
from typing import Any, Literal

import numpy as np
import pyqtgraph as pg
from numba import njit
from numpy.typing import NDArray
from pyqtgraph.Qt import OpenGLConstants as GLC
from pyqtgraph.Qt import OpenGLHelpers
from PySide6 import QtCore, QtGui, QtOpenGL, QtWidgets

FloatArray = NDArray[np.float64]

LayerKind = Literal["line", "step", "band", "segments", "stems", "vlines"]
ReduceMode = Literal["extreme", "mean"]

_PROGRAM_KEY = "CppiLabLayer"
#: Extra half-pixel of ribbon on each side that carries the anti-aliasing ramp.
_AA_MARGIN = 0.75
#: Width in pixels of one averaging bucket of a ``reduce="mean"`` layer.
MEAN_PIXELS = 3
#: Mitre length is capped at this multiple of the half width at sharp turns.
MITRE_LIMIT = 2.0
#: Block size of the coarse level used when very many bars share one pixel.
COARSE_BLOCK = 64
#: Layers shorter than this never need the coarse level.
COARSE_MIN_BARS = 131_072
#: Every vertical grid line is drawn across the whole pane.
_EVERYWHERE = (-1e13, -1e13, 2e13, 2e13)

_VERT_LEGACY = """
attribute vec2 a_position;
attribute float a_across;
uniform mat4 u_mvp;
varying float v_across;
void main() {
    v_across = a_across;
    gl_Position = u_mvp * vec4(a_position, 0.0, 1.0);
}
"""
_FRAG_LEGACY = """
#ifdef GL_ES
precision mediump float;
#endif
uniform vec4 u_color;
uniform float u_half;
uniform vec2 u_dash;
varying float v_across;
void main() {
    float alpha = clamp(u_half + 0.5 - abs(v_across), 0.0, 1.0);
    if (u_dash.y > 0.0) {
        float phase = mod(gl_FragCoord.x + gl_FragCoord.y, u_dash.y);
        if (phase > u_dash.x) discard;
    }
    gl_FragColor = vec4(u_color.rgb, u_color.a * alpha);
}
"""
_VERT_140 = """#version 140
in vec2 a_position;
in float a_across;
uniform mat4 u_mvp;
out float v_across;
void main() {
    v_across = a_across;
    gl_Position = u_mvp * vec4(a_position, 0.0, 1.0);
}
"""
_FRAG_140 = """#version 140
uniform vec4 u_color;
uniform float u_half;
uniform vec2 u_dash;
in float v_across;
out vec4 fragColor;
void main() {
    float alpha = clamp(u_half + 0.5 - abs(v_across), 0.0, 1.0);
    if (u_dash.y > 0.0) {
        float phase = mod(gl_FragCoord.x + gl_FragCoord.y, u_dash.y);
        if (phase > u_dash.x) discard;
    }
    fragColor = vec4(u_color.rgb, u_color.a * alpha);
}
"""


# --------------------------------------------------------------------------
# Decimation and bounds (Numba)
# --------------------------------------------------------------------------


@njit(cache=True, nogil=True)
def minmax_decimate(
    x: FloatArray, y: FloatArray, lo: int, hi: int, buckets: int
) -> tuple[FloatArray, FloatArray]:
    """Reduce ``y[lo:hi]`` to at most ``2 * buckets`` points keeping every extreme.

    Each bucket contributes its minimum and maximum in time order, so spikes
    survive any zoom level.  Short ranges are returned untouched.
    """
    count = hi - lo
    if count <= 2 * buckets or buckets < 1:
        return x[lo:hi].copy(), y[lo:hi].copy()
    out_x = np.empty(2 * buckets, dtype=np.float64)
    out_y = np.empty(2 * buckets, dtype=np.float64)
    filled = 0
    for b in range(buckets):
        start = lo + (b * count) // buckets
        stop = lo + ((b + 1) * count) // buckets
        if stop <= start:
            continue
        i_min = start
        i_max = start
        for i in range(start + 1, stop):
            if y[i] < y[i_min]:
                i_min = i
            if y[i] > y[i_max]:
                i_max = i
        first = i_min if i_min < i_max else i_max
        second = i_max if i_min < i_max else i_min
        out_x[filled] = x[first]
        out_y[filled] = y[first]
        filled += 1
        if second != first:
            out_x[filled] = x[second]
            out_y[filled] = y[second]
            filled += 1
    return out_x[:filled], out_y[:filled]


@njit(cache=True, nogil=True)
def band_decimate(
    x: FloatArray, y_low: FloatArray, y_high: FloatArray, lo: int, hi: int, buckets: int
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Reduce a band to one ``(x, min(low), max(high))`` triple per bucket."""
    count = hi - lo
    if count <= buckets or buckets < 1:
        return x[lo:hi].copy(), y_low[lo:hi].copy(), y_high[lo:hi].copy()
    out_x = np.empty(buckets, dtype=np.float64)
    out_lo = np.empty(buckets, dtype=np.float64)
    out_hi = np.empty(buckets, dtype=np.float64)
    filled = 0
    for b in range(buckets):
        start = lo + (b * count) // buckets
        stop = lo + ((b + 1) * count) // buckets
        if stop <= start:
            continue
        low = y_low[start]
        high = y_high[start]
        for i in range(start + 1, stop):
            if y_low[i] < low:
                low = y_low[i]
            if y_high[i] > high:
                high = y_high[i]
        out_x[filled] = x[start]
        out_lo[filled] = low
        out_hi[filled] = high
        filled += 1
    return out_x[:filled], out_lo[:filled], out_hi[:filled]


@njit(cache=True, nogil=True)
def mean_decimate(
    x: FloatArray, y: FloatArray, lo: int, hi: int, buckets: int
) -> tuple[FloatArray, FloatArray]:
    """Reduce ``y[lo:hi]`` to one bucket mean per pixel.

    Used for fast-oscillating series (signals, envelope bounds) whose min/max
    envelope would fill the whole pane and hide everything behind it.
    """
    count = hi - lo
    if count <= 2 * buckets or buckets < 1:
        return x[lo:hi].copy(), y[lo:hi].copy()
    out_x = np.empty(buckets, dtype=np.float64)
    out_y = np.empty(buckets, dtype=np.float64)
    filled = 0
    for b in range(buckets):
        start = lo + (b * count) // buckets
        stop = lo + ((b + 1) * count) // buckets
        if stop <= start:
            continue
        acc = 0.0
        for i in range(start, stop):
            acc += y[i]
        out_x[filled] = x[start + (stop - start) // 2]
        out_y[filled] = acc / (stop - start)
        filled += 1
    return out_x[:filled], out_y[:filled]


@njit(cache=True, nogil=True)
def band_mean_decimate(
    x: FloatArray, y_low: FloatArray, y_high: FloatArray, lo: int, hi: int, buckets: int
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Reduce a band to bucket means of both edges (see :func:`mean_decimate`)."""
    count = hi - lo
    if count <= 2 * buckets or buckets < 1:
        return x[lo:hi].copy(), y_low[lo:hi].copy(), y_high[lo:hi].copy()
    out_x = np.empty(buckets, dtype=np.float64)
    out_lo = np.empty(buckets, dtype=np.float64)
    out_hi = np.empty(buckets, dtype=np.float64)
    filled = 0
    for b in range(buckets):
        start = lo + (b * count) // buckets
        stop = lo + ((b + 1) * count) // buckets
        if stop <= start:
            continue
        acc_lo = 0.0
        acc_hi = 0.0
        for i in range(start, stop):
            acc_lo += y_low[i]
            acc_hi += y_high[i]
        out_x[filled] = x[start + (stop - start) // 2]
        out_lo[filled] = acc_lo / (stop - start)
        out_hi[filled] = acc_hi / (stop - start)
        filled += 1
    return out_x[:filled], out_lo[:filled], out_hi[:filled]


@njit(cache=True, nogil=True)
def coarse_level(
    x: FloatArray, y: FloatArray, block: int
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """Per-block ``(x_start, min, max, mean)`` of ``y``; the last block may be short."""
    n = x.shape[0]
    count = (n + block - 1) // block
    bx = np.empty(count, dtype=np.float64)
    lo = np.empty(count, dtype=np.float64)
    hi = np.empty(count, dtype=np.float64)
    mean = np.empty(count, dtype=np.float64)
    for b in range(count):
        start = b * block
        stop = min(n, start + block)
        low = y[start]
        high = y[start]
        acc = 0.0
        for i in range(start, stop):
            v = y[i]
            if v < low:
                low = v
            if v > high:
                high = v
            acc += v
        bx[b] = x[start]
        lo[b] = low
        hi[b] = high
        mean[b] = acc / (stop - start)
    return bx, lo, hi, mean


@njit(cache=True, nogil=True)
def envelope_decimate(
    x: FloatArray, y_min: FloatArray, y_max: FloatArray, lo: int, hi: int, buckets: int
) -> tuple[FloatArray, FloatArray]:
    """Min/max polyline from pre-reduced ``(min, max)`` blocks (see :func:`coarse_level`)."""
    count = hi - lo
    out_x = np.empty(2 * buckets, dtype=np.float64)
    out_y = np.empty(2 * buckets, dtype=np.float64)
    filled = 0
    for b in range(buckets):
        start = lo + (b * count) // buckets
        stop = lo + ((b + 1) * count) // buckets
        if stop <= start:
            continue
        low = y_min[start]
        high = y_max[start]
        for i in range(start + 1, stop):
            if y_min[i] < low:
                low = y_min[i]
            if y_max[i] > high:
                high = y_max[i]
        out_x[filled] = x[start]
        out_y[filled] = low
        filled += 1
        if high != low:
            out_x[filled] = x[start + (stop - start) // 2]
            out_y[filled] = high
            filled += 1
    return out_x[:filled], out_y[:filled]


@njit(cache=True, nogil=True)
def finite_min_max(y: FloatArray, lo: int, hi: int) -> tuple[float, float]:
    """Finite minimum and maximum of ``y[lo:hi]`` (``nan, nan`` if none)."""
    low = np.inf
    high = -np.inf
    for i in range(lo, hi):
        v = y[i]
        if v == v:
            if v < low:
                low = v
            if v > high:
                high = v
    if low > high:
        return np.nan, np.nan
    return low, high


# --------------------------------------------------------------------------
# Tessellation (NumPy, screen space)
# --------------------------------------------------------------------------


@njit(cache=True, nogil=True)
def ribbon_strip(px: FloatArray, py: FloatArray, half: float) -> NDArray[np.float32]:
    """Triangle strip of a mitred ribbon of half width ``half`` around a polyline.

    Returns ``(2 * n, 3)`` float32 rows ``(x, y, across)`` where ``across`` is
    the signed pixel distance from the centre line.  Repeated points (a flat
    stretch of a step line produces one per bar) are dropped first: a
    zero-length segment has no direction and would leave a spike.
    """
    total = px.shape[0]
    xs = np.empty(total, dtype=np.float64)
    ys = np.empty(total, dtype=np.float64)
    n = 0
    for i in range(total):
        if n == 0 or abs(px[i] - xs[n - 1]) + abs(py[i] - ys[n - 1]) > 1e-6:
            xs[n] = px[i]
            ys[n] = py[i]
            n += 1
    out = np.empty((2 * n if n >= 2 else 0, 3), dtype=np.float32)
    if n < 2:
        return out
    prev_nx = 0.0
    prev_ny = 0.0
    for i in range(n):
        if i < n - 1:
            dx = xs[i + 1] - xs[i]
            dy = ys[i + 1] - ys[i]
            length = math.sqrt(dx * dx + dy * dy)
            if length < 1e-9:
                length = 1e-9
            next_nx = -dy / length
            next_ny = dx / length
        else:
            next_nx = prev_nx
            next_ny = prev_ny
        if i == 0:
            prev_nx = next_nx
            prev_ny = next_ny
        # Vertex normal: normalised sum of the adjacent segment normals.
        vx = prev_nx + next_nx
        vy = prev_ny + next_ny
        norm = math.sqrt(vx * vx + vy * vy)
        if norm < 1e-6:
            vx = prev_nx
            vy = prev_ny
        else:
            vx /= norm
            vy /= norm
        # Mitre scale 1 / cos(theta / 2), capped at sharp reversals.
        cos_half = abs(vx * prev_nx + vy * prev_ny)
        if cos_half < 1.0 / MITRE_LIMIT:
            cos_half = 1.0 / MITRE_LIMIT
        scale = half / cos_half
        out[2 * i, 0] = xs[i] + vx * scale
        out[2 * i, 1] = ys[i] + vy * scale
        out[2 * i, 2] = half
        out[2 * i + 1, 0] = xs[i] - vx * scale
        out[2 * i + 1, 1] = ys[i] - vy * scale
        out[2 * i + 1, 2] = -half
        prev_nx = next_nx
        prev_ny = next_ny
    return out


def band_strip(px: FloatArray, py_low: FloatArray, py_high: FloatArray) -> NDArray[np.float32]:
    """Triangle strip filling the area between two curves sharing ``px``."""
    n = px.shape[0]
    if n < 2:
        return np.empty((0, 3), dtype=np.float32)
    out = np.zeros((2 * n, 3), dtype=np.float32)
    out[0::2, 0] = px
    out[0::2, 1] = py_low
    out[1::2, 0] = px
    out[1::2, 1] = py_high
    return out


def segment_quads(
    x0: FloatArray, y0: FloatArray, x1: FloatArray, y1: FloatArray, half: float
) -> NDArray[np.float32]:
    """Two triangles per independent segment, ``(6 * n, 3)`` rows."""
    n = x0.shape[0]
    if n == 0:
        return np.empty((0, 3), dtype=np.float32)
    dx = x1 - x0
    dy = y1 - y0
    length = np.hypot(dx, dy)
    length[length < 1e-9] = 1e-9
    nx = -dy / length * half
    ny = dx / length * half
    corners = np.empty((n, 4, 3), dtype=np.float32)
    corners[:, 0, 0], corners[:, 0, 1], corners[:, 0, 2] = x0 + nx, y0 + ny, half
    corners[:, 1, 0], corners[:, 1, 1], corners[:, 1, 2] = x0 - nx, y0 - ny, -half
    corners[:, 2, 0], corners[:, 2, 1], corners[:, 2, 2] = x1 + nx, y1 + ny, half
    corners[:, 3, 0], corners[:, 3, 1], corners[:, 3, 2] = x1 - nx, y1 - ny, -half
    return np.ascontiguousarray(corners[:, (0, 1, 2, 2, 1, 3), :].reshape(n * 6, 3))


def step_points(x: FloatArray, y: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Expand ``(x, y)`` into a hold-then-jump ("post") step polyline."""
    n = x.shape[0]
    if n < 2:
        return x, y
    sx = np.empty(2 * n - 1)
    sy = np.empty(2 * n - 1)
    sx[0::2] = x
    sy[0::2] = y
    sx[1::2] = x[1:]
    sy[1::2] = y[:-1]
    return sx, sy


# --------------------------------------------------------------------------
# The graphics item
# --------------------------------------------------------------------------


class _GLState(QtCore.QObject):
    """Per-item vertex array and buffer bound to one OpenGL context."""

    def __init__(self, widget: QtWidgets.QWidget) -> None:
        super().__init__(widget)
        self.context: QtGui.QOpenGLContext | None = None
        self.vao = QtOpenGL.QOpenGLVertexArrayObject(self)
        self.vbo = QtOpenGL.QOpenGLBuffer(QtOpenGL.QOpenGLBuffer.Type.VertexBuffer)
        self.capacity = 0

    def setup(self, context: QtGui.QOpenGLContext) -> Any:
        widget: Any = self.parent()
        program: Any = widget.retrieveProgram(_PROGRAM_KEY)
        if program is None:
            program = QtOpenGL.QOpenGLShaderProgram()
            modern = (not context.isOpenGLES()) and context.format().version() >= (3, 1)
            vert, frag = (_VERT_140, _FRAG_140) if modern else (_VERT_LEGACY, _FRAG_LEGACY)
            stage = QtOpenGL.QOpenGLShader.ShaderTypeBit
            if not program.addShaderFromSourceCode(stage.Vertex, vert):
                raise RuntimeError(program.log())
            if not program.addShaderFromSourceCode(stage.Fragment, frag):
                raise RuntimeError(program.log())
            program.bindAttributeLocation("a_position", 0)
            program.bindAttributeLocation("a_across", 1)
            if not program.link():
                raise RuntimeError(program.log())
            widget.storeProgram(_PROGRAM_KEY, program)
        if self.context is context:
            return program
        if self.context is not None:
            self.context.aboutToBeDestroyed.disconnect(self.cleanup)
            self.cleanup()
        self.context = context
        context.aboutToBeDestroyed.connect(self.cleanup)
        self.vao.create()
        self.vbo.create()
        self.capacity = 0
        self.vao.bind()
        self.vbo.bind()
        program.enableAttributeArray(0)
        program.setAttributeBuffer(0, GLC.GL_FLOAT, 0, 2, 12)
        program.enableAttributeArray(1)
        program.setAttributeBuffer(1, GLC.GL_FLOAT, 8, 1, 12)
        self.vbo.release()
        self.vao.release()
        return program

    def cleanup(self) -> None:
        widget: Any = self.parent()
        widget.makeCurrent()
        self.vbo.destroy()
        self.vao.destroy()
        self.context = None
        self.capacity = 0
        widget.doneCurrent()

    def upload(self, vertices: NDArray[np.float32]) -> None:
        self.vbo.bind()
        if vertices.nbytes > self.capacity:
            self.capacity = max(vertices.nbytes, 2 * self.capacity)
            self.vbo.allocate(self.capacity)
        self.vbo.write(0, vertices, vertices.nbytes)  # type: ignore[arg-type]
        self.vbo.release()


class GLLayerItem(pg.GraphicsObject):
    """One styled chart layer drawn as screen-space triangles."""

    def __init__(
        self,
        kind: LayerKind,
        color: str | QtGui.QColor,
        width: float = 1.0,
        dash: tuple[float, float] | None = None,
        stem_px: float = 0.0,
        reduce: ReduceMode = "extreme",
    ) -> None:
        super().__init__()
        self.kind: LayerKind = kind
        # ``pg.mkColor`` reads eight-digit hex as RRGGBBAA (the TSD notation);
        # ``QColor`` would read it as AARRGGBB.
        self.color: QtGui.QColor = pg.mkColor(color)
        #: How samples are combined when several bars share a pixel.
        self.reduce: ReduceMode = reduce
        self.width = float(width)
        #: ``(on, off)`` dash lengths in pixels, or ``None`` for a solid line.
        self.dash = dash
        #: Length in pixels of the downward stem drawn by the ``stems`` kind.
        self.stem_px = float(stem_px)
        self._x: FloatArray = np.empty(0)
        self._y: FloatArray = np.empty(0)
        self._y2: FloatArray = np.empty(0)
        self._x2: FloatArray = np.empty(0)
        self._bounds = QtCore.QRectF()
        self._state: _GLState | None = None
        self._cache_key: tuple[Any, ...] | None = None
        self._vertices: NDArray[np.float32] = np.empty((0, 3), dtype=np.float32)
        self._version = 0
        self._coarse: tuple[FloatArray, ...] | None = None
        self._range_key: tuple[int, int, int] | None = None
        self._range_value: tuple[float, float] | None = None
        self._uploaded: tuple[Any, ...] | None = None

    # -- data ------------------------------------------------------------------
    def set_polyline(self, x: FloatArray, y: FloatArray) -> None:
        """Data of a ``line`` or ``step`` layer (``x`` ascending)."""
        self._set(x, y, np.empty(0), np.empty(0))

    def set_band(self, x: FloatArray, y_low: FloatArray, y_high: FloatArray) -> None:
        """Data of a ``band`` layer: the area between two curves."""
        self._set(x, y_low, y_high, np.empty(0))

    def set_segments(self, x0: FloatArray, y0: FloatArray, x1: FloatArray, y1: FloatArray) -> None:
        """Data of a ``segments`` layer: independent line segments."""
        self._set(x0, y0, y1, x1)

    def set_stems(self, x: FloatArray, y: FloatArray) -> None:
        """Data of a ``stems`` layer: a fixed-pixel stem below each point."""
        self._set(x, y, np.empty(0), np.empty(0))

    def set_vlines(self, x: FloatArray) -> None:
        """Data of a ``vlines`` layer: full-height vertical lines at ``x``."""
        self._set(x, np.zeros(len(x)), np.empty(0), np.empty(0))

    def clear(self) -> None:
        self._set(np.empty(0), np.empty(0), np.empty(0), np.empty(0))

    def invalidate(self) -> None:
        """Rebuild the geometry on the next paint (after a style change)."""
        self._cache_key = None
        self.update()

    def _set(self, x: FloatArray, y: FloatArray, y2: FloatArray, x2: FloatArray) -> None:
        self._x = np.ascontiguousarray(x, dtype=np.float64)
        self._y = np.ascontiguousarray(y, dtype=np.float64)
        self._y2 = np.ascontiguousarray(y2, dtype=np.float64)
        self._x2 = np.ascontiguousarray(x2, dtype=np.float64)
        # A second, 64-times coarser level keeps the per-frame cost of a
        # year-long window independent of its half a million bars.
        self._coarse = None
        if self.kind in ("line", "step", "band") and self._x.shape[0] >= COARSE_MIN_BARS:
            bx, y_min, y_max, y_mean = coarse_level(self._x, self._y, COARSE_BLOCK)
            if self.kind == "band":
                _, _y2_min, y2_max, y2_mean = coarse_level(self._x, self._y2, COARSE_BLOCK)
            else:
                y2_max = y2_mean = y_mean
            self._coarse = (bx, y_min, y_max, y_mean, y2_max, y2_mean)
        self._version += 1
        self._cache_key = None
        self.prepareGeometryChange()
        self._bounds = self._compute_bounds()
        self.informViewBoundsChanged()
        self.update()

    def _compute_bounds(self) -> QtCore.QRectF:
        if self._x.shape[0] == 0:
            return QtCore.QRectF()
        if self.kind == "vlines":
            return QtCore.QRectF(*_EVERYWHERE)
        x_lo, x_hi = float(self._x[0]), float(self._x[-1])
        if self.kind == "segments" and self._x2.shape[0]:
            x_lo = float(min(self._x.min(), self._x2.min()))
            x_hi = float(max(self._x.max(), self._x2.max()))
        elif self.kind in ("segments", "stems"):
            x_lo, x_hi = float(self._x.min()), float(self._x.max())
        y_lo, y_hi = finite_min_max(self._y, 0, self._y.shape[0])
        if self._y2.shape[0]:
            lo2, hi2 = finite_min_max(self._y2, 0, self._y2.shape[0])
            y_lo, y_hi = np.nanmin([y_lo, lo2]), np.nanmax([y_hi, hi2])
        if not (np.isfinite(y_lo) and np.isfinite(y_hi)):
            return QtCore.QRectF()
        return QtCore.QRectF(x_lo, y_lo, max(x_hi - x_lo, 1e-9), max(y_hi - y_lo, 1e-9))

    # -- pyqtgraph hooks ---------------------------------------------------------
    def boundingRect(self) -> QtCore.QRectF:
        return self._bounds

    def dataBounds(
        self, ax: int, frac: float = 1.0, orthoRange: tuple[float, float] | None = None
    ) -> tuple[float, float] | None:
        """Data extent along one axis, optionally limited to an x-range."""
        if self._x.shape[0] == 0 or self.kind == "vlines":
            return None
        if ax == 0:
            return self._bounds.left(), self._bounds.right()
        if self.kind in ("stems", "segments"):
            return self._bounds.top(), self._bounds.bottom()
        lo, hi = 0, self._x.shape[0]
        if orthoRange is not None:
            lo = max(0, int(np.searchsorted(self._x, orthoRange[0], side="left")) - 1)
            hi = min(hi, int(np.searchsorted(self._x, orthoRange[1], side="right")) + 1)
            if hi <= lo:
                return None
        # Auto-ranging asks for this on every frame, including frames in which
        # only the crosshair moved; remember the answer for the visible range.
        key = (self._version, lo, hi)
        if key == self._range_key:
            return self._range_value
        coarse = self._coarse
        if coarse is not None and hi - lo >= 8 * COARSE_BLOCK:
            _, y_min, y_max, _, y2_max, _ = coarse
            lo_b = lo // COARSE_BLOCK
            hi_b = min(y_min.shape[0], (hi + COARSE_BLOCK - 1) // COARSE_BLOCK)
            y_lo, _ = finite_min_max(y_min, lo_b, hi_b)
            _, y_hi = finite_min_max(y2_max if self.kind == "band" else y_max, lo_b, hi_b)
        else:
            y_lo, y_hi = finite_min_max(self._y, lo, hi)
            if self._y2.shape[0]:
                lo2, hi2 = finite_min_max(self._y2, lo, hi)
                y_lo, y_hi = np.nanmin([y_lo, lo2]), np.nanmax([y_hi, hi2])
        value = (float(y_lo), float(y_hi)) if np.isfinite(y_lo) and np.isfinite(y_hi) else None
        self._range_key = key
        self._range_value = value
        return value

    def pixelPadding(self) -> float:
        return self.width

    # -- geometry ----------------------------------------------------------------
    def _visible(self, view: pg.ViewBox) -> tuple[int, int, int]:
        """Visible index range and the horizontal pixel budget."""
        (x_min, x_max), _ = view.viewRange()
        lo = max(0, int(np.searchsorted(self._x, x_min, side="left")) - 1)
        hi = min(self._x.shape[0], int(np.searchsorted(self._x, x_max, side="right")) + 1)
        return lo, hi, max(16, int(view.width()))

    def _build(self, view: pg.ViewBox, tr: QtGui.QTransform) -> NDArray[np.float32]:
        """Tessellate the visible part of the layer in scene (pixel) space."""
        m11, m22, dx, dy = tr.m11(), tr.m22(), tr.dx(), tr.dy()
        half = 0.5 * self.width + _AA_MARGIN
        kind = self.kind
        if kind == "segments":
            return segment_quads(
                self._x * m11 + dx,
                self._y * m22 + dy,
                self._x2 * m11 + dx,
                self._y2 * m22 + dy,
                half,
            )
        if kind == "stems":
            px = self._x * m11 + dx
            py = self._y * m22 + dy
            return segment_quads(px, py, px, py + self.stem_px, half)
        if kind == "vlines":
            px = np.round(self._x * m11 + dx) + 0.5  # centred on a pixel: a crisp 1 px line
            area = view.sceneBoundingRect()
            top = np.full(px.shape[0], area.top())
            return segment_quads(px, top, px, np.full(px.shape[0], area.bottom()), half)
        lo, hi, pixels = self._visible(view)
        if hi - lo < 2:
            return np.empty((0, 3), dtype=np.float32)
        coarse = self._coarse
        if coarse is not None and hi - lo >= 4 * COARSE_BLOCK * pixels:
            bx, y_min, y_max, y_mean, y2_max, y2_mean = coarse
            lo_b, hi_b = (
                lo // COARSE_BLOCK,
                min(bx.shape[0], (hi + COARSE_BLOCK - 1) // COARSE_BLOCK),
            )
            mean_buckets = max(48, pixels // MEAN_PIXELS)
            if kind == "band":
                if self.reduce == "mean":
                    x, y_low, y_high = band_mean_decimate(
                        bx, y_mean, y2_mean, lo_b, hi_b, mean_buckets
                    )
                else:
                    x, y_low, y_high = band_decimate(bx, y_min, y2_max, lo_b, hi_b, pixels)
                return band_strip(x * m11 + dx, y_low * m22 + dy, y_high * m22 + dy)
            if self.reduce == "mean":
                x, y = mean_decimate(bx, y_mean, lo_b, hi_b, mean_buckets)
            else:
                x, y = envelope_decimate(bx, y_min, y_max, lo_b, hi_b, pixels)
            coarse_strip: NDArray[np.float32] = ribbon_strip(x * m11 + dx, y * m22 + dy, half)
            return coarse_strip
        if kind == "band":
            if self.reduce == "mean":
                x, y_low, y_high = band_mean_decimate(
                    self._x, self._y, self._y2, lo, hi, max(48, pixels // MEAN_PIXELS)
                )
            else:
                x, y_low, y_high = band_decimate(self._x, self._y, self._y2, lo, hi, pixels)
            return band_strip(x * m11 + dx, y_low * m22 + dy, y_high * m22 + dy)
        if self.reduce == "mean":
            x, y = mean_decimate(self._x, self._y, lo, hi, max(48, pixels // MEAN_PIXELS))
        else:
            x, y = minmax_decimate(self._x, self._y, lo, hi, pixels)
        if kind == "step" and hi - lo <= 2 * pixels:
            x, y = step_points(x, y)
        strip: NDArray[np.float32] = ribbon_strip(x * m11 + dx, y * m22 + dy, half)
        return strip

    def _geometry(self, view: pg.ViewBox) -> NDArray[np.float32]:
        tr = self.sceneTransform()
        key = (
            self._version,
            tr.m11(),
            tr.m22(),
            tr.dx(),
            tr.dy(),
            int(view.width()),
            int(view.height()),
        )
        if key != self._cache_key:
            self._vertices = self._build(view, tr)
            self._cache_key = key
        return self._vertices

    # -- painting ----------------------------------------------------------------
    def paint(
        self,
        painter: QtGui.QPainter,
        option: QtWidgets.QStyleOptionGraphicsItem,
        widget: QtWidgets.QWidget | None = None,
    ) -> None:
        view = self.getViewBox()
        if view is None or self._x.shape[0] == 0:
            return
        vertices = self._geometry(view)
        if vertices.shape[0] < 3:
            return
        if isinstance(widget, OpenGLHelpers.GraphicsViewGLWidget):
            painter.beginNativePainting()
            try:
                self._paint_gl(widget, view, vertices)
            finally:
                painter.endNativePainting()
        else:
            self._paint_raster(painter, vertices)

    def _paint_gl(
        self,
        widget: OpenGLHelpers.GraphicsViewGLWidget,
        view: pg.ViewBox,
        vertices: NDArray[np.float32],
    ) -> None:
        if self._state is None or self._state.parent() is not widget:
            self._state = _GLState(widget)
            self._uploaded = None
        state = self._state
        program = state.setup(widget.context())
        if self._uploaded != self._cache_key or state.capacity == 0:
            state.upload(vertices)
            self._uploaded = self._cache_key

        glf = widget.getFunctions()
        widget.setViewboxClip(view)
        projection = QtGui.QMatrix4x4()
        projection.ortho(widget.rect())

        state.vao.bind()
        program.bind()
        OpenGLHelpers.setUniformValue(program, "u_mvp", projection)
        OpenGLHelpers.setUniformValue(program, "u_color", self.color)
        is_fill = self.kind == "band"
        program.setUniformValue1f(
            program.uniformLocation("u_half"), 1.0 if is_fill else 0.5 * self.width
        )
        dash_location = program.uniformLocation("u_dash")
        if self.dash is None or is_fill:
            program.setUniformValue(dash_location, 0.0, 0.0)
        else:
            dpr = widget.devicePixelRatioF()
            on, off = self.dash
            program.setUniformValue(dash_location, on * dpr, (on + off) * dpr)
        glf.glEnable(GLC.GL_BLEND)
        glf.glBlendFuncSeparate(
            GLC.GL_SRC_ALPHA, GLC.GL_ONE_MINUS_SRC_ALPHA, 1, GLC.GL_ONE_MINUS_SRC_ALPHA
        )
        separate = self.kind in ("segments", "stems", "vlines")
        mode = GLC.GL_TRIANGLES if separate else GLC.GL_TRIANGLE_STRIP
        glf.glDrawArrays(mode, 0, vertices.shape[0])
        glf.glDisable(GLC.GL_BLEND)
        glf.glDisable(GLC.GL_SCISSOR_TEST)
        program.release()
        state.vao.release()

    def _paint_raster(self, painter: QtGui.QPainter, vertices: NDArray[np.float32]) -> None:
        """QPainter fallback; vertices are in scene coordinates."""
        painter.save()
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
        # ``vertices`` are scene coordinates; the painter arrives in item
        # coordinates, so undo the item-to-scene part of its transform.
        to_device = self.sceneTransform().inverted()[0] * painter.worldTransform()
        painter.setWorldTransform(to_device, False)
        if self.kind == "band":
            polygon = QtGui.QPolygonF()
            low = vertices[0::2]
            high = vertices[1::2][::-1]
            for row in low:
                polygon.append(QtCore.QPointF(float(row[0]), float(row[1])))
            for row in high:
                polygon.append(QtCore.QPointF(float(row[0]), float(row[1])))
            painter.setPen(QtCore.Qt.PenStyle.NoPen)
            painter.setBrush(self.color)
            painter.drawPolygon(polygon)
        else:
            pen = QtGui.QPen(self.color, self.width)
            pen.setCosmetic(True)
            pen.setJoinStyle(QtCore.Qt.PenJoinStyle.MiterJoin)
            if self.dash is not None:
                pen.setDashPattern([self.dash[0] / self.width, self.dash[1] / self.width])
            painter.setPen(pen)
            if self.kind in ("segments", "stems", "vlines"):
                quads = vertices.reshape(-1, 6, 3)
                start = 0.5 * (quads[:, 0, :2] + quads[:, 1, :2])
                end = 0.5 * (quads[:, 2, :2] + quads[:, 5, :2])
                for a, b in zip(start, end, strict=True):
                    painter.drawLine(
                        QtCore.QPointF(float(a[0]), float(a[1])),
                        QtCore.QPointF(float(b[0]), float(b[1])),
                    )
            else:
                centre = 0.5 * (vertices[0::2, :2] + vertices[1::2, :2])
                polygon = QtGui.QPolygonF()
                for row in centre:
                    polygon.append(QtCore.QPointF(float(row[0]), float(row[1])))
                painter.drawPolyline(polygon)
        painter.restore()
