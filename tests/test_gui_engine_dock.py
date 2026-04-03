"""GUI tests for EngineDock error reporting and progress indicators."""

from __future__ import annotations

import os

import pytest

pyside6 = pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication, QMainWindow, QStatusBar

from prometheus_equilibrium.gui.engine import EngineDock


@pytest.fixture(scope="module")
def qapp():
    """Provide a headless-capable QApplication for widget tests."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


class _DummyResultsText:
    def __init__(self):
        self._text = ""

    def setText(self, text: str) -> None:
        self._text = text

    def toPlainText(self) -> str:
        return self._text


class _DummyAnalysisPage:
    def __init__(self):
        self.results_text = _DummyResultsText()

    def update_convergence_plots(self, *_args, **_kwargs):
        pass

    def update_expansion_plots(self, *_args, **_kwargs):
        pass

    def update_performance_plots(self, *_args, **_kwargs):
        pass


class _DummyLibraryPage:
    def refresh_species_list(self):
        pass


class _DummyMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.current_units = "SI"
        self.page_analysis = _DummyAnalysisPage()
        self.page_library = _DummyLibraryPage()
        self.setStatusBar(QStatusBar(self))


@pytest.fixture
def dock(qapp):
    """Create a fresh EngineDock instance for each test."""
    _ = qapp
    main = _DummyMainWindow()
    return EngineDock(main)


def test_render_error_report_mirrors_message_and_traceback(dock):
    dock._render_error_report(
        "Performance Solver Error",
        "Exit solve did not converge",
        "Traceback line 1\nTraceback line 2",
    )
    text = dock.main_window.page_analysis.results_text.toPlainText()
    assert "=== Performance Solver Error ===" in text
    assert "Exit solve did not converge" in text
    assert "Traceback line 1" in text


def test_progress_single_point_uses_indeterminate_spinner_mode(dock):
    dock._start_run_progress(1)
    assert not dock.run_progress_label.isHidden()
    assert not dock.run_progress_bar.isHidden()
    assert dock.run_progress_bar.minimum() == 0
    assert dock.run_progress_bar.maximum() == 0
    assert "single point" in dock.run_progress_label.text().lower()

    dock._finish_run_progress()
    assert dock.run_progress_label.isHidden()
    assert dock.run_progress_bar.isHidden()


def test_progress_sweep_uses_determinate_bar_and_updates(dock):
    dock._start_run_progress(5)
    assert dock.run_progress_bar.minimum() == 0
    assert dock.run_progress_bar.maximum() == 5
    assert dock.run_progress_bar.value() == 0

    dock._update_run_progress(3, 5, "Case 3/5")
    assert dock.run_progress_bar.value() == 3
    assert "3/5" in dock.run_progress_label.text()
    assert dock.run_progress_bar.format() == "Case 3/5"


def test_on_perf_finished_error_payload_updates_gui(dock):
    payload = {
        "ok": False,
        "message": "Exit solve did not converge (6).",
        "traceback": "Traceback details",
    }
    dock.on_perf_finished(payload)

    assert dock.res_tc.text() == "ERROR"
    assert "Performance Error" in dock.main_window.statusBar().currentMessage()
    text = dock.main_window.page_analysis.results_text.toPlainText()
    assert "Exit solve did not converge" in text
    assert "Traceback details" in text
