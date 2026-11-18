"""Measure chart pan, zoom and crosshair frame rates on a real display.

Usage::

    uv run python benchmarks/bench_fps.py [--out DIR] [--frames N] [--no-opengl]
                                          [--skip-window]

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
  event-loop pass that painted and presented it.

Vertical sync is switched off for the measurement so a fast frame is not
rounded up to the display's refresh interval; the frame-time figure is
therefore the rate the application could sustain, and the requirement is met
when its 95th percentile is at or below 16.7 ms.

Two further measurements put those figures in context and are reported
alongside them; neither changes the verdict on the canvas cases:

* the **presentation floor**: an empty OpenGL widget of the same size that
  only clears itself.  Whatever the chart draws, a frame cannot be faster
  than this on the machine and display server in use;
* the **full window**: the same interactions inside the complete 1920x1040
  application window, where the display server has more pixels to present.

The measurement needs a display and an OpenGL context.
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
CANVAS_SIZE = (1450, 940)  # the canvas of a 1920x1080 main window
WINDOW_SIZE = (1920, 1040)


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
    parser.add_argument(
        "--skip-window", action="store_true", help="do not measure inside the full main window"
    )
    args = parser.parse_args(argv)

    from cppi_lab.main import configure_environment

    configure_environment()
    import pyqtgraph as pg
    from PySide6 import QtCore, QtGui, QtOpenGLWidgets, QtWidgets

    QtCore.QCoreApplication.setAttribute(QtCore.Qt.ApplicationAttribute.AA_ShareOpenGLContexts)
    surface = QtGui.QSurfaceFormat.defaultFormat()
    surface.setSwapInterval(0)
    QtGui.QSurfaceFormat.setDefaultFormat(surface)

    from cppi_lab.ui import theme
    from cppi_lab.ui.doc_scheme import register_scheme

    register_scheme()
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

    class EmptyScene(QtOpenGLWidgets.QOpenGLWidget):
        """An OpenGL widget that only clears itself: the cheapest frame there is."""

        def paintGL(self) -> None:  # noqa: N802 (Qt virtual)
            functions = self.context().functions()
            functions.glClearColor(0.04, 0.06, 0.10, 1.0)
            functions.glClear(0x4000)

    def floor(size: tuple[int, int], frames: int) -> dict[str, float]:
        """Frame time of presenting an empty OpenGL widget of ``size``."""
        widget: Any = EmptyScene() if use_gl else QtWidgets.QWidget()
        widget.setWindowTitle("CPPI-Lab frame-rate benchmark (empty scene)")
        widget.resize(*size)
        widget.show()
        pump(0.6)
        samples = []
        for _ in range(frames):
            started = time.perf_counter()
            widget.update()
            app.processEvents()
            samples.append((time.perf_counter() - started) * 1e3)
        widget.close()
        pump(0.2)
        return summarise(samples)

    data = load_bundled()
    warmup()

    report: dict[str, Any] = {
        "environment": {
            "date_utc": datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
            "platform": platform.platform(),
            "qt_platform": app.platformName(),
            "opengl": use_gl,
            "canvas": f"{CANVAS_SIZE[0]}x{CANVAS_SIZE[1]}",
        },
        "budget_ms": FRAME_BUDGET_MS,
        "cases": {},
    }
    report["floor"] = {"canvas": floor(CANVAS_SIZE, args.frames)}

    chart = ChartWorkspace()
    chart.setWindowTitle("CPPI-Lab frame-rate benchmark")
    chart.resize(*CANVAS_SIZE)
    chart.show()
    pump(0.5)
    env = report["environment"]
    env["canvas"] = f"{chart.width()}x{chart.height()}"
    env["viewport"] = type(chart.canvas.viewport()).__name__
    if use_gl:
        from OpenGL import GL

        viewport = chart.canvas.viewport()
        viewport.makeCurrent()
        env["gl_renderer"] = GL.glGetString(GL.GL_RENDERER).decode()
        env["gl_version"] = GL.glGetString(GL.GL_VERSION).decode()
        env["gl_line_width_range"] = [
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

    def run_cases(target: Any, n: int) -> dict[str, Any]:
        """The six interactions on ``target`` (a chart showing a back-test)."""
        x = target._x
        x0, x1 = float(x[0]), float(x[-1])
        span = x1 - x0
        cases: dict[str, Callable[[int], None]] = {
            "pan, whole window in view": pan_step(target, x0, span, 0.999, n),
            "pan, 20 % of the window in view": pan_step(target, x0, span, 0.20, n),
            "pan, 6 hours in view": pan_step(target, x0, span, 6 * 3600 / span, n),
            "zoom in and out": zoom_step(target, x0, span, n),
        }
        crosshair = crosshair_step(target, n)
        results = {name: measure(step, n) for name, step in cases.items()}
        target.reset_view()
        results["crosshair, whole window in view"] = measure(crosshair, n)
        target.set_visible_range(x0 + span * 0.4, x0 + span * 0.4 + min(span, 2 * 86400))
        results["crosshair, 2 days in view"] = measure(crosshair, n)
        target.set_hover_bar(-1)
        target.reset_view()
        return results

    for label, params in windows.items():
        run = SimulationController(data, params).run()
        chart.set_run(run)
        pump(0.8)
        report["cases"][label] = {
            "bars": run.result.n_bars,
            "orders": run.result.stats.maker_attempts + run.result.stats.taker_fills,
            "results": run_cases(chart, args.frames),
        }
    chart.close()
    pump(0.2)

    if not args.skip_window:
        from cppi_lab.ui.main_window import MainWindow

        report["floor"]["window"] = floor(WINDOW_SIZE, args.frames)
        window = MainWindow(data, use_web=False)
        window.setWindowTitle("CPPI-Lab frame-rate benchmark (full window)")
        window.resize(*WINDOW_SIZE)
        window.show()
        pump(1.0)
        report["window"] = {
            "size": f"{window.width()}x{window.height()}",
            "canvas": f"{window.chart.canvas.width()}x{window.chart.canvas.height()}",
            "cases": {},
        }
        for label, params in windows.items():
            window.controller.set_params(params)
            run = window.recalculate()
            pump(0.8)
            report["window"]["cases"][label] = {
                "bars": run.result.n_bars,
                "results": run_cases(window.chart, args.frames),
            }
        window.close()
        pump(0.3)

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

    def row(name: str, result: dict[str, Any]) -> str:
        frame, paint = result["frame"], result["paint"]
        return (
            f"  {name:<34} frame median {frame['median_ms']:6.2f}  p95 {frame['p95_ms']:6.2f}  "
            f"max {frame['max_ms']:6.2f} ms | paint median {paint.get('median_ms', 0):6.2f}  "
            f"p95 {paint.get('p95_ms', 0):6.2f} ms | {result['fps_at_p95']:5.0f} FPS at p95  "
            f"{'MET' if result['met'] else 'MISSED'}"
        )

    def floor_row(name: str, stats: dict[str, float]) -> str:
        return (
            f"  {name:<34} frame median {stats['median_ms']:6.2f}  p95 {stats['p95_ms']:6.2f}  "
            f"max {stats['max_ms']:6.2f} ms"
        )

    worst = 0.0
    for label, block in report["cases"].items():
        lines += ["", f"[{label}]  {block['bars']:,} bars, {block['orders']:,} orders"]
        for name, result in block["results"].items():
            worst = max(worst, result["frame"]["p95_ms"])
            lines.append(row(name, result))
    report["worst_p95_frame_ms"] = worst
    report["met"] = worst <= FRAME_BUDGET_MS
    lines += ["", f"worst p95 frame time {worst:.2f} ms -> {'MET' if report['met'] else 'MISSED'}"]

    lines += [
        "",
        "Context (not part of the verdict above)",
        "---------------------------------------",
        "[presentation floor: an empty OpenGL widget that only clears itself]",
        floor_row(f"{CANVAS_SIZE[0]}x{CANVAS_SIZE[1]} (the canvas)", report["floor"]["canvas"]),
    ]
    if "window" in report:
        lines.append(
            floor_row(f"{WINDOW_SIZE[0]}x{WINDOW_SIZE[1]} (the window)", report["floor"]["window"])
        )
        worst_window = 0.0
        for label, block in report["window"]["cases"].items():
            lines += [
                "",
                f"[full {report['window']['size']} main window, canvas "
                f"{report['window']['canvas']}: {label}]",
            ]
            for name, result in block["results"].items():
                worst_window = max(worst_window, result["frame"]["p95_ms"])
                lines.append(row(name, result))
        report["window"]["worst_p95_frame_ms"] = worst_window
        report["window"]["met"] = worst_window <= FRAME_BUDGET_MS
        lines += [
            "",
            f"worst p95 frame time in the full window {worst_window:.2f} ms -> "
            f"{'MET' if report['window']['met'] else 'MISSED'}",
        ]
    text = "\n".join(lines) + "\n"
    print(text)
    if args.out is not None:
        args.out.mkdir(parents=True, exist_ok=True)
        name = "bench_fps" if use_gl else "bench_fps_software"
        if use_gl:
            name += f"_{env['qt_platform']}"
        (args.out / f"{name}.txt").write_text(text)
        (args.out / f"{name}.json").write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
