import json
from pathlib import Path

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QActionGroup
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QListWidget,
    QMainWindow,
    QStackedWidget,
    QStatusBar,
    QWidget,
)

from prometheus_equilibrium.gui.pages.analysis import AnalysisPage
from prometheus_equilibrium.gui.pages.library import LibraryPage
from prometheus_equilibrium.gui.pages.optimizer import OptimizerPage
from prometheus_equilibrium.gui.pages.simulator import SimulatorPage


class PrometheusGUI(QMainWindow):
    def __init__(self, prop_db, spec_db):
        super().__init__()
        self.setWindowTitle("Prometheus | Modern Equilibrium Dashboard")
        self.resize(1400, 850)

        self.prop_db = prop_db
        self.spec_db = spec_db
        self.current_units = "SI"

        self.create_menu_bar()
        self.setStatusBar(QStatusBar(self))
        self.statusBar().showMessage("Prometheus Initialized - Ready")

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        self.nav_rail = QListWidget()
        self.nav_rail.setFixedWidth(100)
        self.nav_rail.setSpacing(10)
        self.nav_rail.setObjectName("NavRail")
        self.nav_rail.setIconSize(QSize(32, 32))
        self.nav_rail.addItem("Simulator")
        self.nav_rail.addItem("Optimizer")
        self.nav_rail.addItem("Analysis")
        self.nav_rail.addItem("Library")
        for i in range(self.nav_rail.count()):
            item = self.nav_rail.item(i)
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            item.setSizeHint(QSize(100, 80))

        self.nav_rail.currentRowChanged.connect(self.switch_nav_page)
        main_layout.addWidget(self.nav_rail)

        self.content_stack = QStackedWidget()
        main_layout.addWidget(self.content_stack)

        self.page_simulator = SimulatorPage(self, self.prop_db)
        self.page_optimizer = OptimizerPage(self, self.prop_db)
        self.page_analysis = AnalysisPage(self)
        self.page_library = LibraryPage(self, self.prop_db, self.spec_db)

        # Compat aliases used by library.py, optimizer.py, and change_units.
        self.engine_dock = self.page_simulator.config_panel
        self.optimizer_dock = self.page_optimizer.config_panel

        self.content_stack.addWidget(self.page_simulator)
        self.content_stack.addWidget(self.page_optimizer)
        self.content_stack.addWidget(self.page_analysis)
        self.content_stack.addWidget(self.page_library)

        self.action_save_optimizer_config.triggered.connect(
            self.page_optimizer.save_config_dialog
        )
        self.action_load_optimizer_config.triggered.connect(
            self.page_optimizer.load_config_dialog
        )

        self.nav_rail.setCurrentRow(0)

    def switch_nav_page(self, index: int) -> None:
        self.content_stack.setCurrentIndex(index)

    def _focus_simulator(self, simulator_tab_index: int | None = None):
        """Show the Simulator page and optionally select a simulator sub-tab."""
        self.nav_rail.setCurrentRow(0)
        if simulator_tab_index is not None:
            self.page_simulator.sim_tabs.setCurrentIndex(simulator_tab_index)

    def _focus_optimizer(self):
        """Show the Optimizer page."""
        self.nav_rail.setCurrentRow(1)

    def create_menu_bar(self):
        menubar = self.menuBar()
        menu_file = menubar.addMenu("File")

        menu_file.addAction("Load Composition...", self.load_composition)
        menu_file.addAction("Save Composition As...", self.save_composition)
        menu_file.addSeparator()
        self.action_load_optimizer_config = menu_file.addAction(
            "Load Optimizer Config..."
        )
        self.action_save_optimizer_config = menu_file.addAction(
            "Save Optimizer Config..."
        )
        menu_file.addSeparator()
        menu_file.addAction("Export Results...", self.export_results)

        menu_options = menubar.addMenu("Options")
        units_menu = menu_options.addMenu("Select Unit System")
        action_si = units_menu.addAction("SI (Default)")
        action_si.setCheckable(True)
        action_us = units_menu.addAction("US Customary")
        action_us.setCheckable(True)
        action_si.setChecked(True)

        unit_group = QActionGroup(self)
        unit_group.addAction(action_si)
        unit_group.addAction(action_us)
        unit_group.setExclusive(True)
        action_si.triggered.connect(lambda: self.change_units("SI"))
        action_us.triggered.connect(lambda: self.change_units("US"))

    @staticmethod
    def _safe_float(text: str, default: float = 0.0) -> float:
        """Parse float text safely, returning a default for blank/invalid input."""
        try:
            return float(text)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _is_prop_json_path(path: Path) -> bool:
        """Return True if path uses the canonical composition extension."""
        return [part.lower() for part in path.suffixes[-2:]] == [".prop", ".json"]

    @staticmethod
    def _format_pressure(value: float) -> str:
        """Format pressure values with fixed precision so decimals are preserved."""
        return f"{value:.6f}"

    def change_units(self, system: str) -> None:
        """Switch UI unit system and convert pressure fields in-place."""
        if system == self.current_units:
            return

        si_to_us = (self.current_units == "SI") and (system == "US")
        us_to_si = (self.current_units == "US") and (system == "SI")

        if si_to_us:
            pressure_factor = 1e6 / 6894.757
        elif us_to_si:
            pressure_factor = 6894.757 / 1e6
        else:
            pressure_factor = 1.0

        # Convert simulator dock fields
        ed = self.engine_dock
        pc = self._safe_float(ed.input_pc.text())
        pc_min = self._safe_float(ed.input_pc_min.text())
        pc_max = self._safe_float(ed.input_pc_max.text())
        ed_is_pressure = "Pressure" in ed.spec_combo.currentText()
        ed_exp_val = self._safe_float(ed.input_exp.text())
        ed_ambient = self._safe_float(ed.input_ambient.text())

        ed.input_pc.setText(self._format_pressure(pc * pressure_factor))
        ed.input_pc_min.setText(self._format_pressure(pc_min * pressure_factor))
        ed.input_pc_max.setText(self._format_pressure(pc_max * pressure_factor))
        ed.input_ambient.setText(self._format_pressure(ed_ambient * pressure_factor))
        if ed_is_pressure:
            ed.input_exp.setText(self._format_pressure(ed_exp_val * pressure_factor))

        # Convert optimizer dock fields
        od = self.optimizer_dock
        od_pc = self._safe_float(od.input_pc.text())
        od_is_pressure = "Pressure" in od.spec_combo.currentText()
        od_exp_val = self._safe_float(od.input_exp.text())
        od_ambient = self._safe_float(od.input_ambient.text())

        od.input_pc.setText(self._format_pressure(od_pc * pressure_factor))
        od.input_ambient.setText(self._format_pressure(od_ambient * pressure_factor))
        if od_is_pressure:
            od.input_exp.setText(self._format_pressure(od_exp_val * pressure_factor))

        self.current_units = system

        ed.refresh_pressure_labels()
        od.refresh_pressure_labels()
        if hasattr(ed, "refresh_report_for_units"):
            ed.refresh_report_for_units()

        self.statusBar().showMessage(f"Unit system changed to {system}", 3000)

    def load_composition(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Load Composition",
            "",
            "Propellant Files (*.prop.json);;All Files (*)",
        )
        if not file_name:
            return
        if not self._is_prop_json_path(Path(file_name)):
            self.statusBar().showMessage(
                "Load failed: composition files must use the .prop.json extension.",
                8000,
            )
            return

        try:
            with open(file_name, "r", encoding="utf-8") as f:
                payload = json.load(f)
            self.page_simulator.apply_composition_snapshot(payload)
            self.engine_dock.clear_previous_results()
            self._focus_simulator(self.page_simulator.current_mode_tab_index())
            self.statusBar().showMessage(
                f"Loaded composition: {Path(file_name).name}",
                4000,
            )
        except Exception as exc:
            self.statusBar().showMessage(f"Load failed: {exc}", 8000)

    def save_composition(self):
        file_name, _ = QFileDialog.getSaveFileName(
            self,
            "Save Composition As",
            "",
            "Propellant Files (*.prop.json);;All Files (*)",
        )
        if not file_name:
            return

        save_path = Path(file_name)
        if not self._is_prop_json_path(save_path):
            base = save_path
            while base.suffix:
                base = base.with_suffix("")
            save_path = Path(f"{base}.prop.json")

        try:
            payload = self.page_simulator.composition_snapshot()
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            self.statusBar().showMessage(
                f"Saved composition: {save_path.name}",
                4000,
            )
        except Exception as exc:
            self.statusBar().showMessage(f"Save failed: {exc}", 8000)

    def export_results(self):
        file_name, _ = QFileDialog.getSaveFileName(
            self,
            "Export Results",
            "",
            "CSV Files (*.csv);;All Files (*)",
        )
        if file_name:
            self.statusBar().showMessage(f"Exported: {file_name}", 3000)
