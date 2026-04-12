"""GUI tests for SimulatorPage formulation-change invalidation behavior."""

from __future__ import annotations

import os

import pytest

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication, QComboBox

from prometheus_equilibrium.gui.pages.simulator import SimulatorPage


@pytest.fixture(scope="module")
def qapp():
    """Provide a headless-capable QApplication for widget tests."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


class _DummyEngineDock:
    def __init__(self):
        self.clear_calls = 0
        self.pc_mode_combo = QComboBox()
        self.pc_mode_combo.addItems(["Single Value", "Sweep Range"])

    def clear_previous_results(self):
        self.clear_calls += 1

    def update_actual_of(self):
        pass


class _DummyMainWindow:
    def __init__(self):
        self.engine_dock = _DummyEngineDock()


class _DummyPropDB:
    ingredient_ids = ["RP_1", "OXYGEN_LIQUID"]

    def find_ingredient(self, ingredient_id):
        if ingredient_id == "OXYGEN_LIQUID":
            return {
                "name": "Oxygen (L)",
                "roles": ["oxidizer"],
                "density": 1140.0,
            }
        return {"name": "RP-1", "roles": ["fuel"], "density": 810.0}

    def search_items(self):
        return [
            {"id": "RP_1", "name": "RP-1"},
            {"id": "OXYGEN_LIQUID", "name": "Oxygen (L)"},
        ]


@pytest.fixture
def simulator(qapp):
    """Create a SimulatorPage with minimal dependencies."""
    _ = qapp
    return SimulatorPage(_DummyMainWindow(), _DummyPropDB())


def test_biprop_mass_fraction_edit_clears_previous_results(simulator):
    engine = simulator.main_window.engine_dock
    baseline = engine.clear_calls

    w_inp = simulator.fuel_table.cellWidget(0, 1)
    w_inp.setText("0.25")

    assert engine.clear_calls > baseline


def test_of_ratio_edit_clears_previous_results(simulator):
    engine = simulator.main_window.engine_dock
    baseline = engine.clear_calls

    simulator.input_of_ratio.setValue(simulator.input_of_ratio.value() + 0.1)

    assert engine.clear_calls > baseline


def test_apply_snapshot_does_not_emit_incremental_clear_notifications(simulator):
    engine = simulator.main_window.engine_dock
    baseline = engine.clear_calls

    simulator.apply_composition_snapshot(
        {
            "schema_version": 1,
            "propellant_type": "bipropellant",
            "components": {
                "fuel": [{"name": "RP_1", "mass_fraction": 1.0}],
                "oxidizer": [{"name": "OXYGEN_LIQUID", "mass_fraction": 1.0}],
            },
            "of_ratio": {
                "mode": "Single Value",
                "value": 2.5,
                "sweep": {"min": "1.00", "max": "6.00", "steps": "11"},
            },
        }
    )

    assert engine.clear_calls == baseline
