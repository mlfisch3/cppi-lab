"""Drive the real application through its main states and save a screenshot of each.

Usage::

    uv run python scripts/ui_tour.py OUT_DIR [--size 1920x1040] [--only NAME ...]
                                             [--shot-cmd "CMD {title} {out}"]

The script builds the actual :class:`MainWindow` on the current display, with
OpenGL and QtWebEngine enabled exactly as ``cppi-lab`` does, and performs the
same actions a user would: editing parameters, hovering ``?`` badges, opening,
floating and docking the ledger and the documentation, following links.

Screenshots are taken with ``QWidget.grab()`` by default.  That cannot see an
OpenGL viewport on every platform (it returns black under WSLg), so
``--shot-cmd`` may name an external command that captures the real window
from the compositor; ``{title}`` is replaced by the window-title substring
and ``{out}`` by the PNG path.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("out", type=Path)
    parser.add_argument("--size", default="1920x1040")
    parser.add_argument("--only", nargs="*", default=None, help="run only these steps")
    parser.add_argument("--shot-cmd", default="", help="external capture command")
    parser.add_argument("--no-opengl", action="store_true")
    parser.add_argument("--settle", type=float, default=0.9, help="seconds to wait per step")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    from cppi_lab.main import configure_environment

    configure_environment()
    import pyqtgraph as pg
    import PySide6QtAds as ads
    from PySide6 import QtCore, QtWidgets

    QtCore.QCoreApplication.setAttribute(QtCore.Qt.ApplicationAttribute.AA_ShareOpenGLContexts)
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
    app.setApplicationName("CPPI-Lab")
    app.setApplicationDisplayName("CPPI-Lab")
    theme.apply_theme(app)

    from cppi_lab.data.loader import load_bundled
    from cppi_lab.engine.controller import warmup
    from cppi_lab.ui.main_window import SCENARIOS, MainWindow
    from cppi_lab.ui.popover_manager import HelpBadge

    data = load_bundled()
    warmup()
    window = MainWindow(data)
    width, _, height = args.size.partition("x")
    window.resize(int(width), int(height))
    window.show()

    def pump(seconds: float | None = None) -> None:
        end = time.time() + (args.settle if seconds is None else seconds)
        while time.time() < end:
            app.processEvents()
            time.sleep(0.01)

    pump(1.5)
    print(f"platform={app.platformName()} opengl={use_gl} size={window.width()}x{window.height()}")

    def shot(name: str) -> None:
        pump()
        target = args.out / f"{name}.png"
        if args.shot_cmd:
            command = args.shot_cmd.format(title="CPPI-Lab", out=str(target))
            subprocess.run(shlex.split(command), check=False, capture_output=True)
        else:
            window.grab().save(str(target))
            for index, container in enumerate(window.dock_manager.floatingWidgets()):
                if container.isVisible():
                    container.grab().save(str(args.out / f"{name}__floating{index}.png"))
        print(f"saved {target.name}")

    def badge(popover_id: str) -> HelpBadge:
        for candidate in window.findChildren(HelpBadge):
            if candidate.popover_id == popover_id:
                return candidate
        raise KeyError(popover_id)

    def toggle(key: str) -> QtWidgets.QAbstractButton:
        switch = window.sidebar.rows[key].toggle
        assert switch is not None
        return switch

    docs: Any = window.docs

    def hover(popover_id: str) -> None:
        target = badge(popover_id)
        window.popovers.badge_entered(target)
        pump(0.6)

    steps: dict[str, Callable[[], None]] = {}

    def step(name: str) -> Callable[[Callable[[], None]], Callable[[], None]]:
        def register(fn: Callable[[], None]) -> Callable[[], None]:
            steps[name] = fn
            return fn

        return register

    @step("01_default_window")
    def _default() -> None:
        window.reset_all()
        window.recalculate()
        shot("01_default_window")

    @step("02_feb_2026_crash")
    def _crash() -> None:
        window.apply_scenario(SCENARIOS[1])
        shot("02_feb_2026_crash")

    @step("03_rescaled_with_mc_fan")
    def _fan() -> None:
        window.apply_scenario(SCENARIOS[5])
        toggle("mc_fan").click()
        pump(1.5)
        shot("03_rescaled_with_mc_fan")

    @step("04_zoom_crosshair")
    def _zoom() -> None:
        chart = window.chart
        run = window.run
        assert run is not None
        bars = (run.result.flags & 64).nonzero()[0]
        centre = int(bars[0]) if len(bars) else run.result.n_bars // 2
        x = float(chart._x[centre])
        chart.set_visible_range(x - 5 * 3600, x + 3 * 3600)
        pump(0.3)
        chart.set_hover_bar(max(0, centre - 90))
        chart.set_selected_bar(centre)
        shot("04_zoom_crosshair")
        chart.set_hover_bar(-1)
        chart.reset_view()

    @step("05_popover_rho")
    def _popover() -> None:
        hover("param-rho")
        shot("05_popover_rho")

    @step("06_popover_pinned_and_second")
    def _pinned() -> None:
        hover("param-rho")
        card = window.popovers.current
        assert card is not None
        card.pin_button.click()
        card.move_within_host(QtCore.QPoint(520, 520))
        hover("kpi-floor")
        shot("06_popover_pinned_and_second")
        hover("pane-holdings")
        shot("07_popover_pane_header")
        window.popovers.close_all()

    @step("08_ledger_floating")
    def _ledger() -> None:
        toggle("mc_fan").setChecked(False)
        window.ledger_button.click()
        pump(0.6)
        container = window.ledger_dock.floatingDockContainer()
        if container is not None:
            print(
                f"ledger floating geometry={container.geometry()} visible={container.isVisible()}"
            )
        window.ledger.set_filter("Executed Trades")
        window.ledger.table.setCurrentIndex(window.ledger.model.index(12, 0))
        shot("08_ledger_floating")
        window.ledger.set_filter("Missed Maker Orders")
        shot("09_ledger_missed_filter")

    @step("10_ledger_docked")
    def _docked() -> None:
        window.ledger.set_filter("Floor Liquidations")
        window.dock_to(window.ledger_dock, ads.DockWidgetArea.BottomDockWidgetArea)
        window.ledger.table.setCurrentIndex(window.ledger.model.index(0, 0))
        shot("10_ledger_docked_bottom")
        window.ledger.set_filter("All Timesteps")
        window.dock_to(window.ledger_dock, ads.DockWidgetArea.RightDockWidgetArea)
        shot("11_ledger_docked_right")
        window.dock_to(window.ledger_dock, ads.DockWidgetArea.LeftDockWidgetArea)
        shot("12_ledger_docked_left")
        window.set_dock_visible(window.ledger_dock, False)

    @step("13_docs_floating")
    def _docs() -> None:
        window.open_documentation("theory", "sec-2-4-rho")
        pump(2.5)
        shot("13_docs_theory_rho")
        docs.open_at("tech", "sec-5-state-machine")
        pump(2.0)
        shot("14_docs_tech_state_machine")
        docs.bridge.notifyLinkHovered("theory", "sec-3-liquidity")
        pump(0.8)
        shot("15_docs_hover_peek")
        docs.bridge.notifyLinkLeft()
        docs.search_edit.setText("trigger")
        pump(0.8)
        shot("16_docs_search")
        docs.search_edit.clear()

    @step("17_docs_docked")
    def _docs_docked() -> None:
        window.open_documentation("theory", "sec-1-floor-buffer")
        pump(1.0)
        window.dock_to(window.docs_dock, ads.DockWidgetArea.RightDockWidgetArea)
        pump(1.5)
        shot("17_docs_docked_right")
        docs.set_preview_visible(False)
        pump(0.6)
        shot("18_docs_preview_collapsed")
        docs.set_preview_visible(True)
        window.set_dock_visible(window.docs_dock, False)

    @step("19_popover_link_opens_docs")
    def _link() -> None:
        hover("param-lambda-liq")
        shot("19_popover_lambda")
        window.popovers.linkActivated.emit("tech", "sec-3-liquidity")
        pump(2.5)
        shot("20_docs_from_popover_link")
        window.popovers.close_all()
        window.set_dock_visible(window.docs_dock, False)

    @step("21_small_sizes")
    def _small() -> None:
        window.apply_scenario(SCENARIOS[3])
        for label, (w, h) in (("1536x864", (1536, 824)), ("1366x768", (1366, 728))):
            window.resize(w, h)
            pump(0.8)
            shot(f"21_window_{label}")
        window.resize(int(width), int(height))

    for name, action in steps.items():
        if args.only and not any(token in name for token in args.only):
            continue
        action()

    window.close()
    pump(0.3)
    return 0


if __name__ == "__main__":
    sys.exit(main())
