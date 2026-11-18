"""Level of detail, tessellation and tick arithmetic for the chart canvas.

Nothing in this module knows about Qt or OpenGL.  It turns full-resolution
series into the triangles of one frame, in screen space, fast enough to be
done again on every pan or zoom step:

* **Pyramids.**  Every series carries block minima, maxima and sums at block
  sizes 4, 16, 64, ...  A frame never scans more than about eight samples per
  horizontal pixel, whatever the zoom level and however long the back-test,
  and the extremes of the data survive at every level.
* **Decimation.**  A layer is reduced to at most two points per pixel for the
  range in view: a min/max envelope that keeps every spike, or bucket means
  for fast-oscillating series whose envelope would paint the pane solid.
* **Tessellation.**  A polyline becomes a mitred ribbon, a band a strip, and
  independent segments become quads, written straight into a shared vertex
  buffer of ``(x, y, across)`` rows; ``across`` is the signed pixel distance
  from the centre line, which the fragment shader turns into an anti-aliased
  edge.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime
from functools import lru_cache

import numpy as np
from numba import njit
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
Vertices = NDArray[np.float32]

#: Extra ribbon width on each side that carries the anti-aliasing ramp.
AA_MARGIN = 0.75
#: Width in pixels of one averaging bucket of a mean-reduced layer.
MEAN_PIXELS = 3
#: A mean-reduced layer never uses fewer buckets than this.
MEAN_MIN_BUCKETS = 48
#: Mitre length is capped at this multiple of the half width at sharp turns.
MITRE_LIMIT = 2.0
#: Each pyramid level is ``2 ** LEVEL_SHIFT`` times coarser than the one below.
LEVEL_SHIFT = 2


# --------------------------------------------------------------------------
# Pyramids
# --------------------------------------------------------------------------


@njit(cache=True, nogil=True)
def build_pyramid(y: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray, IntArray]:
    """Block minima, maxima and sums of ``y`` at block sizes 4, 16, 64, ...

    Returns ``(mins, maxs, sums, offsets)``.  Level ``k`` (``k >= 1``, block
    size ``4 ** k``) occupies ``[offsets[k - 1], offsets[k])`` of the three
    arrays.  NaN samples are ignored by the minima and maxima.
    """
    n = y.shape[0]
    levels = 0
    total = 0
    size = n
    while size > 1:
        size = (size + 3) >> LEVEL_SHIFT
        total += size
        levels += 1
    mins = np.empty(total, dtype=np.float64)
    maxs = np.empty(total, dtype=np.float64)
    sums = np.empty(total, dtype=np.float64)
    offsets = np.zeros(levels + 1, dtype=np.int64)
    pos = 0
    below = n
    below_off = 0
    for k in range(levels):
        size = (below + 3) >> LEVEL_SHIFT
        offsets[k] = pos
        for b in range(size):
            start = b << LEVEL_SHIFT
            stop = min(below, start + 4)
            low = np.inf
            high = -np.inf
            acc = 0.0
            if k == 0:
                for i in range(start, stop):
                    v = y[i]
                    if v < low:
                        low = v
                    if v > high:
                        high = v
                    acc += v
            else:
                for i in range(below_off + start, below_off + stop):
                    if mins[i] < low:
                        low = mins[i]
                    if maxs[i] > high:
                        high = maxs[i]
                    acc += sums[i]
            mins[pos + b] = low
            maxs[pos + b] = high
            sums[pos + b] = acc
        below_off = pos
        pos += size
        below = size
    offsets[levels] = pos
    return mins, maxs, sums, offsets


@njit(cache=True, nogil=True)
def range_min_max(
    y: FloatArray, mins: FloatArray, maxs: FloatArray, offsets: IntArray, lo: int, hi: int
) -> tuple[float, float]:
    """Exact finite minimum and maximum of ``y[lo:hi]`` (``nan, nan`` if none)."""
    low = np.inf
    high = -np.inf
    count = hi - lo
    levels = offsets.shape[0] - 1
    level = 0
    while level < levels and (count >> (LEVEL_SHIFT * (level + 1))) >= 32:
        level += 1
    shift = LEVEL_SHIFT * level
    first = (lo + (1 << shift) - 1) >> shift  # first block lying wholly inside
    last = hi >> shift  # one past the last such block
    if level == 0 or last <= first:
        for i in range(lo, hi):
            v = y[i]
            if v < low:
                low = v
            if v > high:
                high = v
    else:
        for i in range(lo, first << shift):
            v = y[i]
            if v < low:
                low = v
            if v > high:
                high = v
        base = offsets[level - 1]
        for b in range(base + first, base + last):
            if mins[b] < low:
                low = mins[b]
            if maxs[b] > high:
                high = maxs[b]
        for i in range(last << shift, hi):
            v = y[i]
            if v < low:
                low = v
            if v > high:
                high = v
    if low > high:
        return np.nan, np.nan
    return low, high


@njit(cache=True, nogil=True)
def _level_for(count: int, buckets: int, levels: int) -> int:
    """Coarsest level that still leaves at least two blocks in every bucket."""
    level = 0
    while level < levels and (count >> (LEVEL_SHIFT * (level + 1))) >= 2 * buckets:
        level += 1
    return level


# --------------------------------------------------------------------------
# Decimation
# --------------------------------------------------------------------------


@njit(cache=True, nogil=True)
def _extreme_into(
    x: FloatArray,
    y: FloatArray,
    mins: FloatArray,
    maxs: FloatArray,
    offsets: IntArray,
    lo: int,
    hi: int,
    buckets: int,
    tx: FloatArray,
    ty: FloatArray,
) -> int:
    """Min/max polyline of ``y[lo:hi]`` with at most ``2 * buckets`` points."""
    count = hi - lo
    if count <= 2 * buckets or buckets < 1:
        for i in range(count):
            tx[i] = x[lo + i]
            ty[i] = y[lo + i]
        return count
    n = x.shape[0]
    level = _level_for(count, buckets, offsets.shape[0] - 1)
    filled = 0
    if level == 0:
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
            first = min(i_min, i_max)
            second = max(i_min, i_max)
            tx[filled] = x[first]
            ty[filled] = y[first]
            filled += 1
            if second != first:
                tx[filled] = x[second]
                ty[filled] = y[second]
                filled += 1
        return filled
    shift = LEVEL_SHIFT * level
    base = offsets[level - 1]
    b_lo = lo >> shift
    b_hi = min(offsets[level] - base, (hi + (1 << shift) - 1) >> shift)
    blocks = b_hi - b_lo
    for b in range(buckets):
        start = b_lo + (b * blocks) // buckets
        stop = b_lo + ((b + 1) * blocks) // buckets
        if stop <= start:
            continue
        low = np.inf
        high = -np.inf
        at_low = start
        at_high = start
        for i in range(start, stop):
            if mins[base + i] < low:
                low = mins[base + i]
                at_low = i
            if maxs[base + i] > high:
                high = maxs[base + i]
                at_high = i
        if low > high:
            continue
        x_first = x[min(n - 1, start << shift)]
        x_second = x[min(n - 1, ((start + stop) >> 1) << shift)]
        tx[filled] = x_first
        ty[filled] = low if at_low <= at_high else high
        filled += 1
        if high != low:
            tx[filled] = x_second
            ty[filled] = high if at_low <= at_high else low
            filled += 1
    return filled


@njit(cache=True, nogil=True)
def _mean_into(
    x: FloatArray,
    y: FloatArray,
    sums: FloatArray,
    offsets: IntArray,
    lo: int,
    hi: int,
    buckets: int,
    tx: FloatArray,
    ty: FloatArray,
) -> int:
    """Bucket means of ``y[lo:hi]``: at most ``buckets`` points unless the range is short."""
    count = hi - lo
    if count <= 2 * buckets or buckets < 1:
        for i in range(count):
            tx[i] = x[lo + i]
            ty[i] = y[lo + i]
        return count
    n = x.shape[0]
    level = _level_for(count, buckets, offsets.shape[0] - 1)
    filled = 0
    if level == 0:
        for b in range(buckets):
            start = lo + (b * count) // buckets
            stop = lo + ((b + 1) * count) // buckets
            if stop <= start:
                continue
            acc = 0.0
            for i in range(start, stop):
                acc += y[i]
            tx[filled] = x[start + (stop - start) // 2]
            ty[filled] = acc / (stop - start)
            filled += 1
        return filled
    shift = LEVEL_SHIFT * level
    base = offsets[level - 1]
    b_lo = lo >> shift
    b_hi = min(offsets[level] - base, (hi + (1 << shift) - 1) >> shift)
    blocks = b_hi - b_lo
    for b in range(buckets):
        start = b_lo + (b * blocks) // buckets
        stop = b_lo + ((b + 1) * blocks) // buckets
        if stop <= start:
            continue
        acc = 0.0
        for i in range(start, stop):
            acc += sums[base + i]
        samples = min(n, stop << shift) - (start << shift)
        tx[filled] = x[min(n - 1, ((start + stop) >> 1) << shift)]
        ty[filled] = acc / samples
        filled += 1
    return filled


@njit(cache=True, nogil=True)
def _band_extreme_into(
    x: FloatArray,
    y_low: FloatArray,
    y_high: FloatArray,
    low_mins: FloatArray,
    high_maxs: FloatArray,
    offsets: IntArray,
    lo: int,
    hi: int,
    buckets: int,
    tx: FloatArray,
    t_low: FloatArray,
    t_high: FloatArray,
) -> int:
    """One ``(x, min(low), max(high))`` triple per bucket."""
    count = hi - lo
    if count <= buckets or buckets < 1:
        for i in range(count):
            tx[i] = x[lo + i]
            t_low[i] = y_low[lo + i]
            t_high[i] = y_high[lo + i]
        return count
    n = x.shape[0]
    level = _level_for(count, buckets, offsets.shape[0] - 1)
    filled = 0
    shift = LEVEL_SHIFT * level
    if level == 0:
        base = 0
        b_lo = lo
        blocks = count
    else:
        base = offsets[level - 1]
        b_lo = lo >> shift
        blocks = min(offsets[level] - base, (hi + (1 << shift) - 1) >> shift) - b_lo
    for b in range(buckets):
        start = b_lo + (b * blocks) // buckets
        stop = b_lo + ((b + 1) * blocks) // buckets
        if stop <= start:
            continue
        low = np.inf
        high = -np.inf
        if level == 0:
            for i in range(start, stop):
                if y_low[i] < low:
                    low = y_low[i]
                if y_high[i] > high:
                    high = y_high[i]
        else:
            for i in range(base + start, base + stop):
                if low_mins[i] < low:
                    low = low_mins[i]
                if high_maxs[i] > high:
                    high = high_maxs[i]
        if low > high:
            continue
        tx[filled] = x[min(n - 1, start << shift)]
        t_low[filled] = low
        t_high[filled] = high
        filled += 1
    return filled


@njit(cache=True, nogil=True)
def _band_mean_into(
    x: FloatArray,
    y_low: FloatArray,
    y_high: FloatArray,
    low_sums: FloatArray,
    high_sums: FloatArray,
    offsets: IntArray,
    lo: int,
    hi: int,
    buckets: int,
    tx: FloatArray,
    t_low: FloatArray,
    t_high: FloatArray,
) -> int:
    """Bucket means of both edges of a band."""
    count = hi - lo
    if count <= 2 * buckets or buckets < 1:
        for i in range(count):
            tx[i] = x[lo + i]
            t_low[i] = y_low[lo + i]
            t_high[i] = y_high[lo + i]
        return count
    n = x.shape[0]
    level = _level_for(count, buckets, offsets.shape[0] - 1)
    filled = 0
    shift = LEVEL_SHIFT * level
    if level == 0:
        base = 0
        b_lo = lo
        blocks = count
    else:
        base = offsets[level - 1]
        b_lo = lo >> shift
        blocks = min(offsets[level] - base, (hi + (1 << shift) - 1) >> shift) - b_lo
    for b in range(buckets):
        start = b_lo + (b * blocks) // buckets
        stop = b_lo + ((b + 1) * blocks) // buckets
        if stop <= start:
            continue
        acc_low = 0.0
        acc_high = 0.0
        if level == 0:
            for i in range(start, stop):
                acc_low += y_low[i]
                acc_high += y_high[i]
            samples = stop - start
        else:
            for i in range(base + start, base + stop):
                acc_low += low_sums[i]
                acc_high += high_sums[i]
            samples = min(n, stop << shift) - (start << shift)
        tx[filled] = x[min(n - 1, ((start + stop) >> 1) << shift)]
        t_low[filled] = acc_low / samples
        t_high[filled] = acc_high / samples
        filled += 1
    return filled


# --------------------------------------------------------------------------
# Tessellation
# --------------------------------------------------------------------------


@njit(cache=True, nogil=True)
def ribbon_into(
    out: Vertices, offset: int, px: FloatArray, py: FloatArray, count: int, half: float
) -> int:
    """Write the triangle strip of a mitred ribbon; returns the number of rows.

    ``px`` and ``py`` are pixel coordinates and are compacted in place.
    Repeated points (a flat stretch of a step line produces one per bar) are
    dropped, because a zero-length segment has no direction and would leave a
    spike; so are points that lie on the straight line between their
    neighbours, which shortens flat levels and long steps to their corners
    without changing a single pixel.
    """
    n = 0
    for i in range(count):
        x = px[i]
        y = py[i]
        if n > 0 and abs(x - px[n - 1]) + abs(y - py[n - 1]) <= 1e-6:
            continue
        if n >= 2:
            ax = px[n - 1] - px[n - 2]
            ay = py[n - 1] - py[n - 2]
            bx = x - px[n - 1]
            by = y - py[n - 1]
            # Same direction (not a reversal) and no measurable turn: extend the segment.
            if ax * bx + ay * by > 0.0 and abs(ax * by - ay * bx) <= 1e-9 * (abs(ax) + abs(ay)) * (
                abs(bx) + abs(by)
            ):
                px[n - 1] = x
                py[n - 1] = y
                continue
        px[n] = x
        py[n] = y
        n += 1
    if n < 2:
        return 0
    # The mitre offset at a vertex between unit normals p and q is
    # half * (p + q) / (1 + p.q): the bisector scaled by 1 / cos(theta / 2).
    # At sharp reversals 1 + p.q approaches zero, so the offset is capped at
    # MITRE_LIMIT half widths instead.
    sharp = 2.0 / (MITRE_LIMIT * MITRE_LIMIT)
    prev_nx = 0.0
    prev_ny = 0.0
    for i in range(n):
        if i < n - 1:
            dx = px[i + 1] - px[i]
            dy = py[i + 1] - py[i]
            inverse = 1.0 / max(math.sqrt(dx * dx + dy * dy), 1e-9)
            next_nx = -dy * inverse
            next_ny = dx * inverse
        else:
            next_nx = prev_nx
            next_ny = prev_ny
        if i == 0:
            prev_nx = next_nx
            prev_ny = next_ny
        vx = prev_nx + next_nx
        vy = prev_ny + next_ny
        spread = 1.0 + prev_nx * next_nx + prev_ny * next_ny
        if spread >= sharp:
            scale = half / spread
        else:
            norm = math.sqrt(vx * vx + vy * vy)
            if norm < 1e-6:  # an exact reversal has no bisector: square the end off
                vx = prev_nx
                vy = prev_ny
                scale = half
            else:
                scale = half * MITRE_LIMIT / norm
        row = offset + 2 * i
        out[row, 0] = px[i] + vx * scale
        out[row, 1] = py[i] + vy * scale
        out[row, 2] = half
        out[row + 1, 0] = px[i] - vx * scale
        out[row + 1, 1] = py[i] - vy * scale
        out[row + 1, 2] = -half
        prev_nx = next_nx
        prev_ny = next_ny
    return 2 * n


@njit(cache=True, nogil=True)
def segments_into(
    out: Vertices,
    offset: int,
    x0: FloatArray,
    y0: FloatArray,
    x1: FloatArray,
    y1: FloatArray,
    half: float,
) -> int:
    """Two triangles per independent segment (pixel coordinates); returns the rows."""
    n = x0.shape[0]
    for i in range(n):
        dx = x1[i] - x0[i]
        dy = y1[i] - y0[i]
        length = math.sqrt(dx * dx + dy * dy)
        if length < 1e-9:
            length = 1e-9
        nx = -dy / length * half
        ny = dx / length * half
        row = offset + 6 * i
        out[row, 0] = x0[i] + nx
        out[row, 1] = y0[i] + ny
        out[row, 2] = half
        out[row + 1, 0] = x0[i] - nx
        out[row + 1, 1] = y0[i] - ny
        out[row + 1, 2] = -half
        out[row + 2, 0] = x1[i] + nx
        out[row + 2, 1] = y1[i] + ny
        out[row + 2, 2] = half
        out[row + 3, 0] = x1[i] + nx
        out[row + 3, 1] = y1[i] + ny
        out[row + 3, 2] = half
        out[row + 4, 0] = x0[i] - nx
        out[row + 4, 1] = y0[i] - ny
        out[row + 4, 2] = -half
        out[row + 5, 0] = x1[i] - nx
        out[row + 5, 1] = y1[i] - ny
        out[row + 5, 2] = -half
    return int(6 * n)


@njit(cache=True, nogil=True)
def sprites_into(
    out: Vertices,
    offset: int,
    px: FloatArray,
    py: FloatArray,
    width: float,
    height: float,
    u0: float,
    v0: float,
    u1: float,
    v1: float,
) -> int:
    """Two textured triangles centred on each point; rows are ``(x, y, u, v)``."""
    n = px.shape[0]
    half_w = 0.5 * width
    half_h = 0.5 * height
    for i in range(n):
        left = px[i] - half_w
        right = left + width
        top = py[i] - half_h
        bottom = top + height
        row = offset + 6 * i
        out[row, 0] = left
        out[row, 1] = top
        out[row, 2] = u0
        out[row, 3] = v0
        out[row + 1, 0] = right
        out[row + 1, 1] = top
        out[row + 1, 2] = u1
        out[row + 1, 3] = v0
        out[row + 2, 0] = left
        out[row + 2, 1] = bottom
        out[row + 2, 2] = u0
        out[row + 2, 3] = v1
        out[row + 3, 0] = left
        out[row + 3, 1] = bottom
        out[row + 3, 2] = u0
        out[row + 3, 3] = v1
        out[row + 4, 0] = right
        out[row + 4, 1] = top
        out[row + 4, 2] = u1
        out[row + 4, 3] = v0
        out[row + 5, 0] = right
        out[row + 5, 1] = bottom
        out[row + 5, 2] = u1
        out[row + 5, 3] = v1
    return int(6 * n)


@njit(cache=True, nogil=True)
def tessellate_polyline(
    out: Vertices,
    offset: int,
    x: FloatArray,
    y: FloatArray,
    mins: FloatArray,
    maxs: FloatArray,
    sums: FloatArray,
    offsets: IntArray,
    lo: int,
    hi: int,
    pixels: int,
    mean: bool,
    step: bool,
    m11: float,
    dx: float,
    m22: float,
    dy: float,
    half: float,
) -> int:
    """Decimate ``y[lo:hi]`` for ``pixels`` columns and write its ribbon into ``out``.

    ``mean`` selects bucket means instead of the min/max envelope.  ``step``
    draws a hold-then-jump line while the bars are at least half a pixel
    apart; below that a step and a straight line are indistinguishable.
    """
    count = hi - lo
    if count < 2:
        return 0
    tx = np.empty(4 * pixels + 4, dtype=np.float64)
    ty = np.empty(4 * pixels + 4, dtype=np.float64)
    if mean:
        buckets = max(MEAN_MIN_BUCKETS, pixels // MEAN_PIXELS)
        n = _mean_into(x, y, sums, offsets, lo, hi, buckets, tx, ty)
    else:
        n = _extreme_into(x, y, mins, maxs, offsets, lo, hi, pixels, tx, ty)
    if step and count <= 2 * pixels:
        # In place, back to front: (x[i], y[i-1]) then (x[i], y[i]).
        for i in range(n - 1, 0, -1):
            tx[2 * i] = tx[i]
            ty[2 * i] = ty[i]
            tx[2 * i - 1] = tx[i]
            ty[2 * i - 1] = ty[i - 1]
        n = 2 * n - 1
    for i in range(n):
        tx[i] = tx[i] * m11 + dx
        ty[i] = ty[i] * m22 + dy
    return int(ribbon_into(out, offset, tx, ty, n, half))


@njit(cache=True, nogil=True)
def tessellate_band(
    out: Vertices,
    offset: int,
    x: FloatArray,
    y_low: FloatArray,
    y_high: FloatArray,
    low_mins: FloatArray,
    low_sums: FloatArray,
    high_maxs: FloatArray,
    high_sums: FloatArray,
    offsets: IntArray,
    lo: int,
    hi: int,
    pixels: int,
    mean: bool,
    m11: float,
    dx: float,
    m22: float,
    dy: float,
) -> int:
    """Decimate a band for ``pixels`` columns and write its filled strip into ``out``."""
    count = hi - lo
    if count < 2:
        return 0
    size = 2 * pixels + 4
    tx = np.empty(size, dtype=np.float64)
    t_low = np.empty(size, dtype=np.float64)
    t_high = np.empty(size, dtype=np.float64)
    if mean:
        buckets = max(MEAN_MIN_BUCKETS, pixels // MEAN_PIXELS)
        n = _band_mean_into(
            x, y_low, y_high, low_sums, high_sums, offsets, lo, hi, buckets, tx, t_low, t_high
        )
    else:
        n = _band_extreme_into(
            x, y_low, y_high, low_mins, high_maxs, offsets, lo, hi, pixels, tx, t_low, t_high
        )
    if n < 2:
        return 0
    for i in range(n):
        px = tx[i] * m11 + dx
        row = offset + 2 * i
        out[row, 0] = px
        out[row, 1] = t_low[i] * m22 + dy
        out[row, 2] = 0.0
        out[row + 1, 0] = px
        out[row + 1, 1] = t_high[i] * m22 + dy
        out[row + 1, 2] = 0.0
    return int(2 * n)


# --------------------------------------------------------------------------
# Array-returning wrappers (tests, one-off geometry)
# --------------------------------------------------------------------------


def decimate_extreme(
    x: FloatArray, y: FloatArray, lo: int, hi: int, buckets: int
) -> tuple[FloatArray, FloatArray]:
    """Min/max reduction of ``y[lo:hi]`` to at most ``2 * buckets`` points."""
    x = np.ascontiguousarray(x, dtype=np.float64)
    y = np.ascontiguousarray(y, dtype=np.float64)
    mins, maxs, _, offsets = build_pyramid(y)
    size = max(2 * buckets, hi - lo) + 4
    tx, ty = np.empty(size), np.empty(size)
    n = _extreme_into(x, y, mins, maxs, offsets, lo, hi, buckets, tx, ty)
    return tx[:n], ty[:n]


def decimate_mean(
    x: FloatArray, y: FloatArray, lo: int, hi: int, buckets: int
) -> tuple[FloatArray, FloatArray]:
    """Bucket means of ``y[lo:hi]`` (the samples themselves when the range is short)."""
    x = np.ascontiguousarray(x, dtype=np.float64)
    y = np.ascontiguousarray(y, dtype=np.float64)
    _, _, sums, offsets = build_pyramid(y)
    size = max(2 * buckets, hi - lo) + 4
    tx, ty = np.empty(size), np.empty(size)
    n = _mean_into(x, y, sums, offsets, lo, hi, buckets, tx, ty)
    return tx[:n], ty[:n]


def decimate_band(
    x: FloatArray, y_low: FloatArray, y_high: FloatArray, lo: int, hi: int, buckets: int
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """One ``(x, min(low), max(high))`` triple per bucket of a band."""
    x = np.ascontiguousarray(x, dtype=np.float64)
    y_low = np.ascontiguousarray(y_low, dtype=np.float64)
    y_high = np.ascontiguousarray(y_high, dtype=np.float64)
    low_mins, _, _, offsets = build_pyramid(y_low)
    _, high_maxs, _, _ = build_pyramid(y_high)
    size = max(buckets, hi - lo) + 4
    tx, t_low, t_high = np.empty(size), np.empty(size), np.empty(size)
    n = _band_extreme_into(
        x, y_low, y_high, low_mins, high_maxs, offsets, lo, hi, buckets, tx, t_low, t_high
    )
    return tx[:n], t_low[:n], t_high[:n]


def ribbon_strip(px: FloatArray, py: FloatArray, half: float) -> Vertices:
    """Triangle strip ``(2 n, 3)`` of a mitred ribbon around a pixel-space polyline."""
    out = np.empty((2 * px.shape[0], 3), dtype=np.float32)
    rows = ribbon_into(
        out,
        0,
        np.array(px, dtype=np.float64),
        np.array(py, dtype=np.float64),
        px.shape[0],
        float(half),
    )
    return out[:rows]


def segment_quads(
    x0: FloatArray, y0: FloatArray, x1: FloatArray, y1: FloatArray, half: float
) -> Vertices:
    """Two triangles per independent segment, ``(6 n, 3)`` rows."""
    out = np.empty((6 * x0.shape[0], 3), dtype=np.float32)
    rows = segments_into(
        out,
        0,
        np.ascontiguousarray(x0, dtype=np.float64),
        np.ascontiguousarray(y0, dtype=np.float64),
        np.ascontiguousarray(x1, dtype=np.float64),
        np.ascontiguousarray(y1, dtype=np.float64),
        float(half),
    )
    return out[:rows]


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
# Ticks
# --------------------------------------------------------------------------

#: Time-axis spacings, in seconds, tried from the finest to the coarsest.
_TIME_STEPS: tuple[int, ...] = (
    60, 120, 300, 600, 900, 1_800, 3_600, 7_200, 10_800, 21_600, 43_200,
    86_400, 2 * 86_400, 5 * 86_400, 7 * 86_400, 14 * 86_400, 30 * 86_400,
    61 * 86_400, 91 * 86_400, 182 * 86_400, 365 * 86_400,
)  # fmt: skip
#: Minimum distance between two time labels.
TIME_TICK_MIN_PX = 64.0


def time_ticks(x_min: float, x_max: float, width_px: float) -> tuple[int, list[float]]:
    """Round UTC tick positions for a visible range: ``(step_seconds, values)``."""
    span = x_max - x_min
    if span <= 0.0 or width_px <= 0.0:
        return 0, []
    target = span * TIME_TICK_MIN_PX / width_px
    step = next((s for s in _TIME_STEPS if s >= target), _TIME_STEPS[-1])
    first = math.ceil(x_min / step) * step
    count = int((x_max - first) // step) + 1
    return step, [float(first + i * step) for i in range(max(0, count))]


@lru_cache(maxsize=4096)
def time_tick_label(value: float, spacing: float) -> str:
    """UTC label of one time tick: month and year, day, or hour and minute."""
    try:
        moment = datetime.fromtimestamp(value, UTC)
    except (OverflowError, OSError, ValueError):
        return ""
    if spacing >= 28 * 86_400:
        return moment.strftime("%b %Y")
    if spacing >= 86_400 or (moment.hour == 0 and moment.minute == 0):
        return moment.strftime("%b %d").replace(" 0", " ")
    return moment.strftime("%H:%M")


def value_ticks(
    v_min: float, v_max: float, length_px: float
) -> list[tuple[float, FloatArray, float]]:
    """Major, minor and sub-minor ticks of a value axis.

    Returns ``(spacing, values, opacity)`` per level, coarsest first; a value
    appears only in the coarsest level it belongs to.  The spacings follow the
    same 1-2-5 ladder and density rule as pyqtgraph's ``AxisItem`` so the grid
    reads the way it always did: major lines opaque, finer ones fainter and
    fading out as they crowd together.
    """
    span = abs(v_max - v_min)
    if not math.isfinite(span) or span <= 0.0 or length_px <= 0.0:
        return []
    low, high = min(v_min, v_max), max(v_min, v_max)
    intervals = max(2.25, 2.25 * math.sqrt(length_px / 300.0))
    major_max = span / intervals
    _, exponent = math.frexp(major_max)
    unit = 10.0 ** (math.floor((exponent - 1) / 3.32192809488736) - 1)
    if 100.0 * unit <= major_max:
        major_factor = 10
        unit *= 10.0
    else:
        major_factor = next((f for f in (50, 20, 10) if f * unit <= major_max), 10)
    spacings = [major_factor * unit]
    minor_min = 2.0 * span / length_px  # at most one minor tick per two pixels
    minor_trials = (5, 10) if major_factor == 10 else (10, 20, 50)
    minor = next((f * unit for f in minor_trials if f * unit >= minor_min), minor_trials[-1] * unit)
    spacings.append(minor)
    extra_trials = {10: (1, 2, 5, 10), 20: (2, 5, 10, 20), 50: (5, 10, 50)}[major_factor]
    extra = next(
        (f * unit for f in extra_trials if f * unit >= minor_min or f * unit == minor),
        extra_trials[-1] * unit,
    )
    if extra < minor:
        spacings.append(extra)

    levels: list[tuple[float, FloatArray, float]] = []
    for index, spacing in enumerate(spacings):
        start = math.ceil(low / spacing) * spacing
        count = max(0, int((high - start) / spacing) + 1)
        values = start + spacing * np.arange(count, dtype=np.float64)
        # Drop the values a coarser level already owns.
        for coarser in spacings[:index]:
            ratio = values / coarser
            values = values[np.abs(ratio - np.round(ratio)) * coarser > spacing * 0.01]
        crowding = min(1.0, max(0.0, 0.05 * length_px / (values.shape[0] + 1)))
        levels.append((spacing, values, crowding / (index + 1)))
    return levels


def value_tick_label(value: float, spacing: float) -> str:
    """Label of one value tick with just enough decimals for its spacing."""
    if abs(value) < 1e-9 * max(1.0, spacing):
        return "0"
    places = max(0, math.ceil(-math.log10(spacing))) if spacing > 0 else 0
    return f"{value:.{places}f}"
