"""GUI tests for formulation-first optimizer page behaviors."""

from __future__ import annotations

import json
import os

import pytest

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication, QCheckBox, QComboBox, QScrollArea

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
        self.applied_payload = None

    def composition_snapshot(self):
        return self._payload

    def apply_composition_snapshot(self, payload):
        self.applied_payload = payload


class _DummyEngineDock:
    def __init__(self):
        self.cleared = False

    def clear_previous_results(self):
        self.cleared = True


class _DummyStatusBar:
    def __init__(self):
        self.last_message = ""

    def showMessage(self, message: str, _timeout: int = 0):
        self.last_message = message


class _DummyMainWindow:
    def __init__(self):
        self.page_simulator = _DummySimulatorPage()
        self.engine_dock = _DummyEngineDock()
        self._status_bar = _DummyStatusBar()
        self.focus_calls = []
        self.optimizer_focus_calls = 0

    def _focus_simulator(self, simulator_tab_index: int | None = None):
        self.focus_calls.append(simulator_tab_index)

    def statusBar(self):
        return self._status_bar

    def _focus_optimizer(self):
        self.optimizer_focus_calls += 1


@pytest.fixture
def page(qapp):
    """Create an OptimizerPage with minimal dependencies."""
    _ = qapp
    return OptimizerPage(_DummyMainWindow(), prop_db=None)


def test_load_from_simulator_adds_group_label_column_entries(page):
    page.load_from_simulator()

    assert page.table_vars.columnCount() == 5
    assert page.table_vars.rowCount() == 3
    # Column 3 is the Pinned checkbox widget, not a QTableWidgetItem
    assert page.table_vars.item(0, 3) is None
    # Column 4 is Group Labels — empty by default after preload
    assert page.table_vars.item(0, 4).text() == ""


def test_optimizer_page_uses_scroll_area(page):
    assert len(page.findChildren(QScrollArea)) >= 1
    assert page.canvas_best.minimumHeight() >= 200


def test_optimizer_page_has_no_inline_config_buttons(page):
    assert not hasattr(page, "btn_save_config")
    assert not hasattr(page, "btn_load_config")
    assert callable(page.save_config_dialog)
    assert callable(page.load_config_dialog)


def test_optimizer_page_uses_right_panel_run_controls(page):
    assert page.btn_start is page.config_panel.btn_start
    assert page.btn_cancel is page.config_panel.btn_cancel
    assert page.btn_apply is page.config_panel.btn_apply
    assert page.progress is page.config_panel.progress_bar
    assert page.progress_label is page.config_panel.progress_label


def test_collect_problem_compiles_label_rules(page):
    page.table_vars.setRowCount(0)
    # Values are in percentages (0–100) as displayed in the GUI
    page._append_variable_row("HTPB", 10.0, 20.0, "binder")
    page._append_variable_row("IPDI", 0.5, 3.0, "binder")
    page._append_variable_row("AP", 60.0, 85.0, "solids")
    page._append_variable_row("AL", 5.0, 25.0, "solids")

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
    # Group totals also in percentages
    page.table_group_rules.item(1, 2).setText("84")
    page.table_group_rules.item(1, 3).setText("84")

    problem = page._collect_problem()

    assert len(problem.fixed_proportion_groups) == 1
    assert len(problem.sum_to_total_groups) == 1
    assert problem.fixed_proportion_groups[0].members == ["HTPB", "IPDI"]
    # Ratios derived from midpoints in 0–1: HTPB=0.15, IPDI=0.0175
    assert abs(problem.fixed_proportion_groups[0].ratios[0] - 0.15) < 1e-12
    assert abs(problem.fixed_proportion_groups[0].ratios[1] - 0.0175) < 1e-12
    assert problem.sum_to_total_groups[0].members == ["AP", "AL"]
    # Internal problem stores 0–1 fractions (84% → 0.84)
    assert abs(problem.sum_to_total_groups[0].minimum_total - 0.84) < 1e-12
    assert abs(problem.sum_to_total_groups[0].maximum_total - 0.84) < 1e-12
    # No closure concept — problem has no closure_ingredient_id field
    assert not hasattr(problem, "closure_ingredient_id")


def test_collect_problem_rejects_label_without_rule(page):
    page.table_vars.setRowCount(0)
    page._append_variable_row("AP", 60.0, 85.0, "solids")
    page._append_variable_row("AL", 5.0, 25.0, "solids")
    page.table_group_rules.setRowCount(0)

    with pytest.raises(ValueError, match="Missing group rule definitions"):
        _ = page._collect_problem()


def test_progress_update_handles_failed_start(page):
    page.progress.setRange(0, 4)
    page.progress.setValue(0)
    page._run_total_starts = 4
    page._on_progress(
        {
            "start": 0,
            "converged": False,
            "best_value": None,
            "isp": None,
            "n_starts": 4,
        }
    )

    assert "failed" in page.progress_label.text()
    assert page.progress.value() == 0


def test_progress_partial_update_does_not_advance_bar(page):
    page.progress.setRange(0, 4)
    page.progress.setValue(0)
    page._run_total_starts = 4
    page._on_progress(
        {
            "start": 0,
            "partial": True,
            "converged": False,
            "start_trace": [(0, 5.0)],
            "best_value": None,
            "n_starts": 4,
        }
    )

    assert page.progress.value() == 0


def test_progress_update_live_history_appends_point(page):
    page._live_history = []
    page._stage_history = {1: {}, 2: {}}
    page._stage_trace_enabled = {1: {}, 2: {}}
    page._on_progress(
        {
            "start": 3,
            "converged": True,
            "objective_value": 7.05,
            "start_trace": [(0, 6.2), (1, 6.8), (2, 7.05)],
            "best_value": 7.123,
            "isp": 233.5,
            "n_starts": 4,
        }
    )

    assert page._live_history == [(3, 7.123)]
    assert page._stage_history[1][3] == [(0, 6.2), (1, 6.8), (2, 7.05)]
    assert 3 in page._stage_trace_checks[1]


def test_start_trace_toggle_updates_visibility(page):
    page._stage_history = {
        1: {0: [(0, 6.8), (1, 7.1)], 1: [(0, 6.9)]},
        2: {},
    }
    page._stage_trace_enabled = {1: {0: True, 1: True}, 2: {}}
    page._rebuild_start_trace_toggles(1)

    toggle = page._stage_trace_checks[1][0]
    assert isinstance(toggle, QCheckBox)
    toggle.setChecked(False)

    assert page._stage_trace_enabled[1][0] is False


def test_apply_best_to_simulator_outputs_percentages_with_three_decimals(page):
    page._latest_best_composition = {
        "ALUMINUM_PURE_CRYSTALINE": 0.150000,
        "AMMONIUM_PERCHLORATE": 0.532452,
        "BISMUTH_TRIOXIDE": 0.157548,
        "MDI_143L": 0.040000,
        "R45": 0.120000,
    }

    page.apply_best_to_simulator()

    payload = page.main_window.page_simulator.applied_payload
    assert payload is not None
    total = sum(float(c["mass_fraction"]) for c in payload["components"])
    assert abs(total - 100.0) < 1e-12
    for component in payload["components"]:
        scaled = float(component["mass_fraction"]) * 1000.0
        assert abs(scaled - round(scaled)) < 1e-12


def test_load_optimizer_config_focuses_optimizer_tab(page, monkeypatch, tmp_path):
    config_path = tmp_path / "demo.prop-opt.json"
    config = {
        "problem": {
            "variables": [
                {"ingredient_id": "AP", "minimum": 60.0, "maximum": 80.0},
                {"ingredient_id": "AL", "minimum": 5.0, "maximum": 20.0},
            ],
            "fixed_proportion_groups": [],
            "sum_to_total_groups": [],
        },
        "objective": {"isp_variant": "isp_actual", "rho_exponent": 0.25},
        "operating_point": {"shifting": True},
        "run": {
            "n_starts": 4,
            "max_iter_per_start": 10,
            "n_workers": 0,
            "seed": 42,
        },
    }
    config_path.write_text(json.dumps(config), encoding="utf-8")

    monkeypatch.setattr(
        "prometheus_equilibrium.gui.pages.optimizer.QFileDialog.getOpenFileName",
        lambda *args, **kwargs: (
            str(config_path),
            "Optimizer Config (*.prop-opt.json)",
        ),
    )

    page.load_config_dialog()

    assert page.main_window.optimizer_focus_calls == 1
