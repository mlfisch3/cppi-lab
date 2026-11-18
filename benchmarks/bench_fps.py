"""Measure chart pan, zoom and crosshair frame rates on a real display.

Usage::

    uv run python benchmarks/bench_fps.py [--out DIR] [--frames N] [--no-opengl]

PRD section 6.1 asks that "OpenGL chart pan/zoom and crosshair inspection
must maintain 60 FPS", i.e. a frame must cost no more than 16.7 ms.

The script shows the real three-pane canvas at the size it has in a
1920x1080 window, loads a busy 30-day back-test (and a 365-day one), and then
replays three interactions frame by frame:

* **pan**   - the visible range slides across the back-test;
* **zoom**  - the visible range shrinks and grows around a fixed point;
* **crosshair** - the hover line sweeps across the bars.

For every frame it records

* the *paint time*: the duration of the canvas paint event, which covers the
  decimation and tessellation of every layer and the submission of all
  OpenGL commands, and
* the *frame time*: wall-clock from requesting the change to the end of the
  event-loop pass that painted it.

Vertical sync is switched off for the measurement so a fast frame is not
rounded up to the display's refresh interval; the frame-time figure is
therefore the rate the application could sustain, and the requirement is met
when its 95th percentile is at or below 16.7 ms.  The measurement needs a
display and an OpenGL context.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from collections.abc import Callable
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

FRAME_BUDGET_MS = 1000.0 / 60.0


def summarise(samples: list[float]) -> dict[str, float]:
    return {
        "median_ms": statistics.median(samples),
        "mean_ms": statistics.fmean(samples),
        "p95_ms": float(np.percentile(samples, 95)),
        "max_ms": max(samples),
    }


def pan_step(chart: Any, x0: float, span: float, fraction: float, n: int) -> Callable[[int], None]:
    """Slide a window of ``fraction`` of the back-test from one end to the other and back."""
    width = span * fraction

    def step(i: int) -> None:
        left = x0 + (span - width) * (0.5 - 0.5 * np.cos(2 * np.pi * i / n))
        chart.set_visible_range(left, left + width)

    return step


def zoom_step(chart: Any, x0: float, span: float, n: int) -> Callable[[int], None]:
    """Zoom from the whole back-test down to 2 % of it and out again, about a fixed point."""

    def step(i: int) -> None:
        factor = 0.02 + 0.98 * (0.5 + 0.5 * np.cos(2 * np.pi * i / n))
        centre = x0 + span * 0.62
        chart.set_visible_range(centre - span * factor * 0.62, centre + span * factor * 0.38)

    return step


def crosshair_step(chart: Any, n: int) -> Callable[[int], None]:
    """Sweep the hover crosshair across the visible bars."""

    def step(i: int) -> None:
        lo, hi = chart.visible_range()
        chart.set_hover_bar(chart.bar_at(lo + (hi - lo) * ((i * 7) % n) / n))

    return step


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--frames", type=int, default=240)
    parser.add_argument("--no-opengl", action="store_true")
    args = parser.parse_args(argv)

    from cppi_lab.main import configure_environment

    configure_environment()
    import pyqtgraph as pg
    from PySide6 import QtCore, QtGui, QtWidgets

    QtCore.QCoreApplication.setAttribute(QtCore.Qt.ApplicationAttribute.AA_ShareOpenGLContexts)
    surface = QtGui.QSurfaceFormat.defaultFormat()
    surface.setSwapInterval(0)
    QtGui.QSurfaceFormat.setDefaultFormat(surface)

    from cppi_lab.ui import theme

    use_gl = not args.no_opengl
    pg.setConfigOptions(
        useOpenGL=use_gl,
        enableExperimental=use_gl,
        antialias=True,
        background=theme.CANVAS,
        foreground=theme.TEXT_MUTED,
    )
    app = QtWidgets.QApplication(sys.argv[:1])
    theme.apply_theme(app)

    from pyqtgraph.widgets.GraphicsView import GraphicsView

    from cppi_lab.config.defaults import default_params
    from cppi_lab.data.loader import load_bundled
    from cppi_lab.engine.controller import SimulationController, warmup
    from cppi_lab.ui.chart_workspace import ChartWorkspace

    paint_ms: list[float] = []
    original_paint = GraphicsView.paintEvent

    def timed_paint(self: Any, event: Any) -> None:
        started = time.perf_counter()
        original_paint(self, event)
        paint_ms.append((time.perf_counter() - started) * 1e3)

    GraphicsView.paintEvent = timed_paint

    data = load_bundled()
    warmup()
    chart = ChartWorkspace()
    chart.setWindowTitle("CPPI-Lab frame-rate benchmark")
    chart.resize(1450, 940)  # the canvas of a 1920x1080 main window
    chart.show()

    def pump(seconds: float) -> None:
        end = time.time() + seconds
        while time.time() < end:
            app.processEvents()
            time.sleep(0.005)

    def measure(step: Callable[[int], None], frames: int) -> dict[str, Any]:
        pump(0.3)
        paint_ms.clear()
        frame_ms = []
        for i in range(frames):
            started = time.perf_counter()
            step(i)
            app.processEvents()
            frame_ms.append((time.perf_counter() - started) * 1e3)
        painted = list(paint_ms)
        return {
            "frames": frames,
            "paints": len(painted),
            "frame": summarise(frame_ms),
            "paint": summarise(painted) if painted else {},
            "fps_at_p95": 1000.0 / float(np.percentile(frame_ms, 95)),
            "met": float(np.percentile(frame_ms, 95)) <= FRAME_BUDGET_MS,
        }

    report: dict[str, Any] = {
        "environment": {
            "date_utc": datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
            "platform": platform.platform(),
            "qt_platform": app.platformName(),
            "opengl": use_gl,
            "canvas": f"{chart.width()}x{chart.height()}",
            "viewport": type(chart.canvas.viewport()).__name__,
        },
        "budget_ms": FRAME_BUDGET_MS,
        "cases": {},
    }
    if use_gl:
        from OpenGL import GL

        viewport = chart.canvas.viewport()
        viewport.makeCurrent()
        report["environment"]["gl_renderer"] = GL.glGetString(GL.GL_RENDERER).decode()
        report["environment"]["gl_version"] = GL.glGetString(GL.GL_VERSION).decode()
        report["environment"]["gl_line_width_range"] = [
            float(v) for v in GL.glGetFloatv(GL.GL_ALIASED_LINE_WIDTH_RANGE)
        ]
        viewport.doneCurrent()

    windows = {
        "30 days, busy (Jan 2026 rescaled)": replace(default_params(), normalize=True),
        "365 days (from 2025-10-01, rescaled)": replace(
            default_params(),
            normalize=True,
            duration_days=365,
            start_ms=int(datetime(2025, 10, 1, tzinfo=UTC).timestamp() * 1000),
        ),
    }
    for label, params in windows.items():
        run = SimulationController(data, params).run()
        chart.set_run(run)
        pump(0.8)
        x = chart._x
        x0, x1 = float(x[0]), float(x[-1])
        span = x1 - x0
        n = args.frames
        cases: dict[str, Callable[[int], None]] = {
            "pan, whole window in view": pan_step(chart, x0, span, 0.999, n),
            "pan, 20 % of the window in view": pan_step(chart, x0, span, 0.20, n),
            "pan, 6 hours in view": pan_step(chart, x0, span, 6 * 3600 / span, n),
            "zoom in and out": zoom_step(chart, x0, span, n),
        }
        crosshair = crosshair_step(chart, n)
        results = {}
        for name, step in cases.items():
            results[name] = measure(step, n)
        chart.reset_view()
        results["crosshair, whole window in view"] = measure(crosshair, n)
        chart.set_visible_range(x0 + span * 0.4, x0 + span * 0.4 + min(span, 2 * 86400))
        results["crosshair, 2 days in view"] = measure(crosshair, n)
        chart.set_hover_bar(-1)
        report["cases"][label] = {
            "bars": run.result.n_bars,
            "orders": run.result.stats.maker_attempts + run.result.stats.taker_fills,
            "results": results,
        }

    env = report["environment"]
    lines = [
        "CPPI-Lab chart frame-rate benchmark",
        "===================================",
        f"date      : {env['date_utc']}",
        f"platform  : {env['platform']} (Qt platform {env['qt_platform']})",
        f"canvas    : {env['canvas']} px, viewport {env['viewport']}, OpenGL {env['opengl']}",
    ]
    if use_gl:
        lines.append(f"renderer  : {env['gl_renderer']} - {env['gl_version']}")
        lines.append(f"driver line-width range: {env['gl_line_width_range']}")
    lines.append(
        f"budget    : {FRAME_BUDGET_MS:.1f} ms per frame (60 FPS), judged at the 95th percentile"
    )
    worst = 0.0
    for label, block in report["cases"].items():
        lines += ["", f"[{label}]  {block['bars']:,} bars, {block['orders']:,} orders"]
        for name, row in block["results"].items():
            frame, paint = row["frame"], row["paint"]
            worst = max(worst, frame["p95_ms"])
            lines.append(
                f"  {name:<34} frame median {frame['median_ms']:6.2f}  p95 {frame['p95_ms']:6.2f}  "
                f"max {frame['max_ms']:6.2f} ms | paint median {paint.get('median_ms', 0):6.2f}  "
                f"p95 {paint.get('p95_ms', 0):6.2f} ms | {row['fps_at_p95']:5.0f} FPS at p95  "
                f"{'MET' if row['met'] else 'MISSED'}"
            )
    report["worst_p95_frame_ms"] = worst
    report["met"] = worst <= FRAME_BUDGET_MS
    lines += ["", f"worst p95 frame time {worst:.2f} ms -> {'MET' if report['met'] else 'MISSED'}"]
    text = "\n".join(lines) + "\n"
    print(text)
    if args.out is not None:
        args.out.mkdir(parents=True, exist_ok=True)
        name = "bench_fps" if use_gl else "bench_fps_software"
        (args.out / f"{name}.txt").write_text(text)
        (args.out / f"{name}.json").write_text(json.dumps(report, indent=2) + "\n")
    chart.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
