"""GUI tests for formulation-first optimizer page behaviors."""

from __future__ import annotations

import os

import pytest

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication, QComboBox, QScrollArea

from prometheus_equilibrium.gui.pages.optimizer import OptimizerPage


@pytest.fixture(scope="module")
def qapp():
    """Provide a headless-capable QApplication for widget tests."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


class _DummySimulatorPage:
    def __init__(self):
        self._payload = {
            "schema_version": 1,
            "propellant_type": "solid",
            "components": [
                {"name": "AP", "mass_fraction": 0.70},
                {"name": "AL", "mass_fraction": 0.16},
                {"name": "HTPB", "mass_fraction": 0.14},
            ],
        }

    def composition_snapshot(self):
        return self._payload


class _DummyMainWindow:
    def __init__(self):
        self.page_simulator = _DummySimulatorPage()


@pytest.fixture
def page(qapp):
    """Create an OptimizerPage with minimal dependencies."""
    _ = qapp
    return OptimizerPage(_DummyMainWindow(), prop_db=None)


def test_load_from_simulator_adds_group_label_column_entries(page):
    page.load_from_simulator()

    assert page.table_vars.columnCount() == 4
    assert page.table_vars.rowCount() == 3
    assert page.table_vars.item(0, 3).text() == ""
    assert page.combo_closure.currentData() is not None


def test_optimizer_page_uses_scroll_area(page):
    assert len(page.findChildren(QScrollArea)) >= 1
    assert page.canvas_best.minimumHeight() >= 260


def test_collect_problem_compiles_label_rules(page):
    page.table_vars.setRowCount(0)
    page._append_variable_row("HTPB", 0.10, 0.20, "binder")
    page._append_variable_row("IPDI", 0.005, 0.03, "binder")
    page._append_variable_row("AP", 0.60, 0.85, "solids")
    page._append_variable_row("AL", 0.05, 0.25, "solids")

    page.table_group_rules.setRowCount(0)
    page._append_default_group_rule()
    page.table_group_rules.item(0, 0).setText("binder")
    fixed_type = page.table_group_rules.cellWidget(0, 1)
    assert isinstance(fixed_type, QComboBox)
    fixed_type.setCurrentText("fixed_proportion")

    page._append_default_group_rule()
    page.table_group_rules.item(1, 0).setText("solids")
    sum_type = page.table_group_rules.cellWidget(1, 1)
    assert isinstance(sum_type, QComboBox)
    sum_type.setCurrentText("sum_to_total")
    page.table_group_rules.item(1, 2).setText("0.84")
    page.table_group_rules.item(1, 3).setText("0.84")

    problem = page._collect_problem()

    assert len(problem.fixed_proportion_groups) == 1
    assert len(problem.sum_to_total_groups) == 1
    assert problem.fixed_proportion_groups[0].members == ["HTPB", "IPDI"]
    assert abs(problem.fixed_proportion_groups[0].ratios[0] - 0.15) < 1e-12
    assert abs(problem.fixed_proportion_groups[0].ratios[1] - 0.0175) < 1e-12
    assert problem.sum_to_total_groups[0].members == ["AP", "AL"]
    assert problem.sum_to_total_groups[0].minimum_total == 0.84
    assert problem.sum_to_total_groups[0].maximum_total == 0.84


def test_collect_problem_rejects_label_without_rule(page):
    page.table_vars.setRowCount(0)
    page._append_variable_row("AP", 0.60, 0.85, "solids")
    page._append_variable_row("AL", 0.05, 0.25, "solids")
    page.table_group_rules.setRowCount(0)

    with pytest.raises(ValueError, match="Missing group rule definitions"):
        _ = page._collect_problem()


def test_progress_update_handles_no_complete_trial(page):
    page.progress.setRange(0, 100)
    page._on_progress(
        {
            "trial": 0,
            "best_value": None,
            "status_kind": "solver_error",
            "status_reason": "Exit solve did not converge (1).",
        }
    )

    assert "no complete trial yet" in page.progress_label.text()


def test_progress_update_live_history_appends_point(page):
    page._live_history = []
    page._on_progress(
        {
            "trial": 3,
            "best_value": 7.123,
            "status_kind": "complete",
            "status_reason": "",
        }
    )

    assert page._live_history == [(3, 7.123)]
