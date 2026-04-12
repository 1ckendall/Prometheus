import traceback

from loguru import logger
from PySide6.QtCore import QLocale, Qt, QThread, Signal
from PySide6.QtGui import QDoubleValidator, QIntValidator
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDockWidget,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from prometheus_equilibrium.equilibrium.performance import (
    PerformanceSolver,
    RocketPerformanceComparison,
)
from prometheus_equilibrium.equilibrium.problem import EquilibriumProblem, ProblemType
from prometheus_equilibrium.equilibrium.solver import GordonMcBrideSolver

_PA_PER_MPA = 1_000_000.0
_PA_PER_PSI = 6894.757


class SolverWorker(QThread):
    finished = Signal(object)  # dict payload

    def __init__(self, problem, solver):
        super().__init__()
        self.problem = problem
        self.solver = solver

    def run(self):
        try:
            logger.info(f"Starting solver on background thread...")
            solution = self.solver.solve(self.problem)
            self.finished.emit({"ok": True, "solution": solution})
        except Exception as e:
            logger.exception("Solver thread encountered an unhandled exception")
            self.finished.emit(
                {
                    "ok": False,
                    "message": str(e),
                    "traceback": traceback.format_exc(),
                }
            )


class PerformanceWorker(QThread):
    finished = Signal(object)  # dict payload
    progress = Signal(int, int, str)  # done, total, label

    def __init__(
        self,
        jobs,
        exit_value,
        is_pressure,
        ambient_pressure,
        solver,
        spec_db,
        sweep_axis="none",
        sweep_label="Run Index",
        compute_profile=False,
    ):
        super().__init__()
        self.jobs = jobs
        self.exit_value = exit_value
        self.is_pressure = is_pressure
        self.ambient_pressure = ambient_pressure
        self.sweep_axis = sweep_axis
        self.sweep_label = sweep_label
        self.compute_profile = compute_profile
        # Performance expansion can require condensed-phase partner swaps at low T.
        self.perf_solver = PerformanceSolver(solver, db=spec_db)

    def run(self):
        cases = []
        total = len(self.jobs)
        try:
            logger.info("Starting dual-mode performance solver on background thread...")
            pe_pa = self.exit_value if self.is_pressure else None
            area_ratio = self.exit_value if not self.is_pressure else None
            self.progress.emit(0, total, "Starting")
            for i, (sweep_value, problem) in enumerate(self.jobs, start=1):
                pair = self.perf_solver.solve_pair(
                    problem,
                    pe_pa=pe_pa,
                    area_ratio=area_ratio,
                    ambient_pressure=self.ambient_pressure,
                    compute_profile=self.compute_profile,
                )
                cases.append((sweep_value, pair))
                self.progress.emit(i, total, f"Case {i}/{total}")
            self.finished.emit(
                {
                    "ok": True,
                    "cases": cases,
                    "ambient_pressure": self.ambient_pressure,
                    "sweep_axis": self.sweep_axis,
                    "sweep_label": self.sweep_label,
                }
            )
        except Exception as e:
            logger.exception(
                "Performance solver thread encountered an unhandled exception"
            )
            self.finished.emit(
                {
                    "ok": False,
                    "message": str(e),
                    "traceback": traceback.format_exc(),
                    "cases": cases,
                    "ambient_pressure": self.ambient_pressure,
                    "sweep_axis": self.sweep_axis,
                    "sweep_label": self.sweep_label,
                }
            )


class EngineDock(QDockWidget):
    def __init__(self, main_window, title="Simulation Configuration"):
        super().__init__(title, main_window)
        self.main_window = main_window
        self.setAllowedAreas(Qt.RightDockWidgetArea)
        self.setFeatures(
            QDockWidget.DockWidgetFloatable | QDockWidget.DockWidgetMovable
        )
        self.setMinimumWidth(340)

        self.double_validator = QDoubleValidator()
        self.double_validator.setNotation(QDoubleValidator.StandardNotation)
        locale = QLocale(QLocale.English, QLocale.UnitedStates)
        locale.setNumberOptions(QLocale.RejectGroupSeparator)
        self.double_validator.setLocale(locale)
        self.int_validator = QIntValidator(2, 10000, self)

        dock_contents = QWidget()
        layout = QVBoxLayout(dock_contents)

        # 1. Operating Conditions
        group_conditions = QGroupBox("Operating Conditions")
        form_conditions = QFormLayout()

        self.lbl_pc = QLabel("Chamber Pressure (MPa)")
        self.pc_mode_combo = QComboBox()
        self.pc_mode_combo.addItems(["Single Value", "Sweep Range"])

        self.input_pc = QLineEdit("6.894757")

        pc_sweep_layout = QHBoxLayout()
        self.input_pc_min = QLineEdit("1.000000")
        self.input_pc_max = QLineEdit("10.000000")
        self.input_pc_steps = QLineEdit("10")

        for w in (self.input_pc_min, self.input_pc_max):
            w.setEnabled(False)
            w.setValidator(self.double_validator)
            w.setAlignment(Qt.AlignRight)
        self.input_pc_steps.setEnabled(False)
        self.input_pc_steps.setValidator(self.int_validator)
        self.input_pc_steps.setAlignment(Qt.AlignRight)

        pc_sweep_layout.addWidget(QLabel("Min:"))
        pc_sweep_layout.addWidget(self.input_pc_min)
        pc_sweep_layout.addWidget(QLabel("Max:"))
        pc_sweep_layout.addWidget(self.input_pc_max)
        pc_sweep_layout.addWidget(QLabel("Steps:"))
        pc_sweep_layout.addWidget(self.input_pc_steps)

        self.pc_mode_combo.currentTextChanged.connect(self.handle_pc_mode)

        self.spec_combo = QComboBox()
        self.spec_combo.addItems(["Exhaust Pressure", "Area Ratio (Ae/At)"])
        self.lbl_exp = QLabel("Exhaust Pressure (MPa)")
        self.input_exp = QLineEdit("0.101325")

        self.spec_combo.currentTextChanged.connect(self.update_exp_label)

        self.lbl_amb = QLabel("Ambient Pressure (MPa)")
        self.input_ambient = QLineEdit("0.101325")

        for widget in (self.input_pc, self.input_exp, self.input_ambient):
            widget.setAlignment(Qt.AlignRight)
            widget.setValidator(self.double_validator)

        form_conditions.addRow("Pc Mode:", self.pc_mode_combo)
        form_conditions.addRow(self.lbl_pc, self.input_pc)
        self.lbl_pc_bounds = QLabel("Pc Bounds (MPa):")
        form_conditions.addRow(self.lbl_pc_bounds, pc_sweep_layout)
        form_conditions.addRow("Expansion Spec.", self.spec_combo)
        form_conditions.addRow(self.lbl_exp, self.input_exp)
        form_conditions.addRow(self.lbl_amb, self.input_ambient)
        group_conditions.setLayout(form_conditions)
        layout.addWidget(group_conditions)

        # 1.5 Species Database Selection
        group_db = QGroupBox("Species Database Selection")
        db_layout = QGridLayout()
        self.check_nasa7 = QCheckBox("NASA-7")
        self.check_nasa9 = QCheckBox("NASA-9")
        self.check_janaf = QCheckBox("JANAF")
        self.check_afcesic = QCheckBox("AFCESIC")
        self.check_terra = QCheckBox("TERRA")

        # Set defaults: NASA-7, NASA-9, TERRA enabled. JANAF, AFCESIC disabled.
        self.check_nasa7.setChecked(True)
        self.check_nasa9.setChecked(True)
        self.check_janaf.setChecked(False)
        self.check_afcesic.setChecked(False)
        self.check_terra.setChecked(True)

        # Connect signals to update species explorer list when databases change
        for cb in [
            self.check_nasa7,
            self.check_nasa9,
            self.check_janaf,
            self.check_afcesic,
            self.check_terra,
        ]:
            cb.stateChanged.connect(self.on_db_selection_changed)

        db_layout.addWidget(self.check_nasa7, 0, 0)
        db_layout.addWidget(self.check_nasa9, 0, 1)
        db_layout.addWidget(self.check_janaf, 1, 0)
        db_layout.addWidget(self.check_afcesic, 1, 1)
        db_layout.addWidget(self.check_terra, 2, 0)
        group_db.setLayout(db_layout)
        layout.addWidget(group_db)

        # 2. Performance Results
        group_results = QGroupBox("Solver Results (Nominal)")
        results_form = QFormLayout()
        self.res_isp = QLabel("---")
        self.res_isp_frozen = QLabel("---")
        self.res_cstar = QLabel("---")
        self.res_cstar_frozen = QLabel("---")
        self.res_tc = QLabel("---")
        for w in (
            self.res_isp,
            self.res_isp_frozen,
            self.res_cstar,
            self.res_cstar_frozen,
            self.res_tc,
        ):
            w.setStyleSheet("font-weight: bold; color: #2a82da; font-size: 14px;")

        results_form.addRow("Shifting / Frozen", QLabel(""))
        results_form.addRow("Isp (actual, s):", self.res_isp)
        results_form.addRow("Isp (actual, frozen, s):", self.res_isp_frozen)
        results_form.addRow("C* (m/s):", self.res_cstar)
        results_form.addRow("C* (frozen, m/s):", self.res_cstar_frozen)
        results_form.addRow("Chamber Temp (K):", self.res_tc)
        group_results.setLayout(results_form)
        layout.addWidget(group_results)

        # 2.5 Solver Options
        group_solver_opts = QGroupBox("Solver Options")
        solver_opts_layout = QVBoxLayout()
        self.check_nozzle_profile = QCheckBox("Compute Nozzle Profile (15 pts)")
        self.check_nozzle_profile.setChecked(False)
        self.check_nozzle_profile.setToolTip(
            "Compute 15 intermediate states from chamber to exit for expansion plots. "
            "Disable to speed up each calculation point."
        )
        solver_opts_layout.addWidget(self.check_nozzle_profile)
        group_solver_opts.setLayout(solver_opts_layout)
        layout.addWidget(group_solver_opts)

        # 3. Actions
        btn_layout = QHBoxLayout()
        self.btn_calculate = QPushButton("Calculate")
        self.btn_calculate.setStyleSheet(
            "background-color: #2a82da; font-weight: bold; height: 40px;"
        )
        self.btn_calculate.clicked.connect(self.on_calculate)

        btn_layout.addWidget(self.btn_calculate, stretch=2)
        layout.addLayout(btn_layout)

        self.run_progress_label = QLabel("")
        self.run_progress_bar = QProgressBar()
        self.run_progress_bar.setTextVisible(True)
        self.run_progress_label.hide()
        self.run_progress_bar.hide()
        layout.addWidget(self.run_progress_label)
        layout.addWidget(self.run_progress_bar)

        layout.addStretch()
        self.setWidget(dock_contents)

        # Ensure pressure-related labels are initialised consistently.
        self.refresh_pressure_labels()

        # Solver components
        self.solver = GordonMcBrideSolver(capture_history=True)
        self.worker = None
        self._last_report_payload = None

    def on_db_selection_changed(self, state):
        # Notify species explorer to refresh its list
        if hasattr(self.main_window, "page_library"):
            self.main_window.page_library.refresh_species_list()

    def get_enabled_databases(self):
        dbs = []
        if self.check_nasa7.isChecked():
            dbs.append("NASA-7")
        if self.check_nasa9.isChecked():
            dbs.append("NASA-9")
        if self.check_janaf.isChecked():
            dbs.append("JANAF")
        if self.check_afcesic.isChecked():
            dbs.append("AFCESIC")
        if self.check_terra.isChecked():
            dbs.append("TERRA")
        return dbs

    def handle_pc_mode(self, text):
        is_sweep = text == "Sweep Range"
        self.input_pc.setEnabled(not is_sweep)
        self.input_pc_min.setEnabled(is_sweep)
        self.input_pc_max.setEnabled(is_sweep)
        self.input_pc_steps.setEnabled(is_sweep)

        # Only one sweep may be active at a time: if Pc sweep is enabled,
        # force O/F back to single-value mode.
        if is_sweep and hasattr(self.main_window, "page_simulator"):
            sim_page = self.main_window.page_simulator
            if sim_page.of_mode_combo.currentText() == "Sweep Range":
                sim_page.of_mode_combo.blockSignals(True)
                sim_page.of_mode_combo.setCurrentText("Single Value")
                sim_page.of_mode_combo.blockSignals(False)
                sim_page.handle_of_mode("Single Value")

    def refresh_pressure_labels(self):
        """Refresh pressure-related labels based on the active unit system."""
        is_si = self.main_window.current_units == "SI"
        p_unit = "MPa" if is_si else "PSI"
        self.lbl_pc.setText(f"Chamber Pressure ({p_unit})")
        self.lbl_pc_bounds.setText(f"Pc Bounds ({p_unit}):")
        self.lbl_amb.setText(f"Ambient Pressure ({p_unit})")
        if "Area Ratio" in self.spec_combo.currentText():
            self.lbl_exp.setText("Area Ratio")
        else:
            self.lbl_exp.setText(f"Exhaust Pressure ({p_unit})")

    def update_exp_label(self, text):
        if "Area Ratio" in text:
            self.lbl_exp.setText("Area Ratio")
        else:
            self.lbl_exp.setText(
                f"Exhaust Pressure ({'MPa' if self.main_window.current_units == 'SI' else 'PSI'})"
            )

    def _pressure_input_to_pa(self, value: float) -> float:
        """Convert a pressure input value from current UI units to Pa."""
        if self.main_window.current_units == "US":
            return value * _PA_PER_PSI
        return value * _PA_PER_MPA

    @staticmethod
    def _build_sweep_values(
        v_min: float, v_max: float, steps: int, axis: str
    ) -> list[float]:
        """Build exactly ``steps`` sweep values including both endpoints."""
        if steps < 2:
            raise ValueError(f"{axis} sweep steps must be >= 2.")
        if v_max < v_min:
            raise ValueError(f"{axis} sweep max must be >= min.")
        step = (v_max - v_min) / (steps - 1)
        return [v_min + i * step for i in range(steps)]

    def update_actual_of(self):
        # Forward to simulator page if needed, or update here
        pass

    def _pressure_display_unit(self) -> str:
        return "MPa" if self.main_window.current_units == "SI" else "PSI"

    def _temperature_display_unit(self) -> str:
        return "K" if self.main_window.current_units == "SI" else "F"

    def _pressure_to_display(self, p_pa: float) -> float:
        if self.main_window.current_units == "SI":
            return p_pa / _PA_PER_MPA
        return p_pa / _PA_PER_PSI

    def _temperature_to_display(self, t_k: float) -> float:
        if self.main_window.current_units == "SI":
            return t_k
        return (t_k * 9.0 / 5.0) - 459.67

    def _build_performance_report_text(self, payload: dict) -> str:
        cases = payload.get("cases", [])
        _, perf = cases[0]
        shifting = perf.shifting
        frozen = perf.frozen
        chamber = shifting.chamber

        p_unit = self._pressure_display_unit()
        t_unit = self._temperature_display_unit()
        spec_kind = payload.get("spec_kind", "pressure")
        if spec_kind == "pressure":
            expansion_spec = f"Pe target: {self._pressure_to_display(shifting.exit.pressure):.6f} {p_unit}"
        else:
            expansion_spec = f"Ae/At target: {payload.get('expansion_target', shifting.area_ratio):.3f}"

        def _state_row(mode, sol):
            return (
                f"{mode:<10}"
                f"{self._temperature_to_display(sol.temperature):>11.2f}"
                f"{self._pressure_to_display(sol.pressure):>12.6f}"
                f"{sol.gamma:>9.4f}"
                f"{sol.gas_mean_molar_mass * 1000:>12.3f}"
            )

        def _perf_row(mode, perf_result):
            return (
                f"{mode:<10}"
                f"{perf_result.isp_actual:>10.2f}"
                f"{perf_result.isp_vac:>10.2f}"
                f"{perf_result.isp_sl:>10.2f}"
                f"{perf_result.cstar:>10.1f}"
                f"{perf_result.area_ratio:>9.3f}"
                f"{perf_result.pressure_ratio:>9.2f}"
                f"{self._temperature_to_display(perf_result.throat.temperature):>11.2f}"
                f"{self._temperature_to_display(perf_result.exit.temperature):>11.2f}"
            )

        lines = [
            "=== Rocket Performance Report ===",
            "",
            "[Simulation Configuration]",
            f"Unit System: {self.main_window.current_units}",
            f"Temperature Unit: {t_unit}",
            f"Pressure Unit: {p_unit}",
            f"Ambient pressure : {self._pressure_to_display(perf.ambient_pressure):.6f} {p_unit}",
            f"Expansion target : {expansion_spec}",
            "",
            "[Shared Chamber State]",
            f"Mode        T({t_unit})     P({p_unit})    gamma   M(g/mol)",
            "-----------------------------------------------------------",
            _state_row("Chamber", chamber),
            "",
            "[Exit State Comparison]",
            f"Mode        T({t_unit})     P({p_unit})    gamma   M(g/mol)",
            "-----------------------------------------------------------",
            _state_row("Frozen", frozen.exit),
            _state_row("Shifting", shifting.exit),
            "",
            "[Performance Comparison]",
            f"Mode       Isp(act)  Isp(vac)   Isp(SL)        C*    Ae/At    Pc/Pe    T* ({t_unit})    Te ({t_unit})",
            "-------------------------------------------------------------------------------------------------------",
            _perf_row("Frozen", frozen),
            _perf_row("Shifting", shifting),
            "",
        ]

        def _species_column(header, sol):
            rows = [
                "",
                f"[{header}]",
                f"{'Species':<27} {'Moles':>12}",
                "----------------------------------------",
            ]
            for sp, n in sorted(
                zip(sol.mixture.species, sol.mixture.moles),
                key=lambda x: x[1],
                reverse=True,
            ):
                if n > 1e-6:
                    rows.append(f"{str(sp):<27} {n:>12.4e}")
            return rows

        def _species_table(header, chamber_sol, throat_sol, exit_sol):
            def moles_dict(sol):
                return {str(sp): n for sp, n in zip(sol.mixture.species, sol.mixture.moles)}

            ch_d = moles_dict(chamber_sol)
            th_d = moles_dict(throat_sol)
            ex_d = moles_dict(exit_sol)

            all_species = {
                sp
                for d in (ch_d, th_d, ex_d)
                for sp, n in d.items()
                if n > 1e-6
            }
            sorted_species = sorted(all_species, key=lambda s: ch_d.get(s, 0.0), reverse=True)

            rows = [
                "",
                f"[{header}]",
                f"{'Species':<27} {'Chamber':>12} {'Throat':>12} {'Exit':>12}",
                "-------------------------------------------------------------------",
            ]
            for sp_name in sorted_species:
                ch = ch_d.get(sp_name, 0.0)
                th = th_d.get(sp_name, 0.0)
                ex = ex_d.get(sp_name, 0.0)
                rows.append(f"{sp_name:<27} {ch:>12.4e} {th:>12.4e} {ex:>12.4e}")
            return rows

        lines.extend(_species_column("Frozen Species Moles", chamber))
        lines.extend(_species_table("Shifting Species Moles", chamber, shifting.throat, shifting.exit))

        lines.extend(
            [
                "",
                "[Machine Data | format=prometheus-report-v1]",
                f"unit_system={self.main_window.current_units}",
                f"temperature_unit={t_unit}",
                f"pressure_unit={p_unit}",
                f"ambient_pressure={self._pressure_to_display(perf.ambient_pressure):.10g}",
                f"chamber_temperature={self._temperature_to_display(chamber.temperature):.10g}",
                f"chamber_pressure={self._pressure_to_display(chamber.pressure):.10g}",
                f"frozen_exit_temperature={self._temperature_to_display(frozen.exit.temperature):.10g}",
                f"frozen_exit_pressure={self._pressure_to_display(frozen.exit.pressure):.10g}",
                f"shifting_exit_temperature={self._temperature_to_display(shifting.exit.temperature):.10g}",
                f"shifting_exit_pressure={self._pressure_to_display(shifting.exit.pressure):.10g}",
                f"isp_actual_frozen={frozen.isp_actual:.10g}",
                f"isp_actual_shifting={shifting.isp_actual:.10g}",
                f"cstar_frozen={frozen.cstar:.10g}",
                f"cstar_shifting={shifting.cstar:.10g}",
                f"area_ratio_frozen={frozen.area_ratio:.10g}",
                f"area_ratio_shifting={shifting.area_ratio:.10g}",
                f"pressure_ratio_frozen={frozen.pressure_ratio:.10g}",
                f"pressure_ratio_shifting={shifting.pressure_ratio:.10g}",
            ]
        )
        return "\n".join(lines)

    def _build_equilibrium_report_text(self, sol) -> str:
        t_unit = self._temperature_display_unit()
        p_unit = self._pressure_display_unit()
        t_disp = self._temperature_to_display(sol.temperature)
        p_disp = self._pressure_to_display(sol.pressure)
        cp_unit = "J/mol.K"
        lines = [
            "=== Equilibrium Solution Report ===",
            "",
            "[State]",
            f"Unit System: {self.main_window.current_units}",
            f"Temperature Unit: {t_unit}",
            f"Pressure Unit: {p_unit}",
            f"Converged: {sol.converged} in {sol.iterations} iterations",
            f"Temperature: {t_disp:.2f} {t_unit}",
            f"Pressure: {p_disp:.6f} {p_unit}",
            f"Mean Molar Mass: {sol.gas_mean_molar_mass * 1000:.3f} g/mol",
            f"Gamma (cp/cv): {sol.gamma:.4f}",
            f"Cp: {sol.cp:.2f} {cp_unit}",
            "",
            "[Species Moles]",
        ]
        species_moles = list(zip(sol.mixture.species, sol.mixture.moles))
        for sp, n in sorted(species_moles, key=lambda x: x[1], reverse=True):
            if n > 1e-6:
                lines.append(f"{str(sp):<25}: {n:.4e}")

        lines.extend(
            [
                "",
                "[Machine Data | format=prometheus-report-v1]",
                f"unit_system={self.main_window.current_units}",
                f"temperature_unit={t_unit}",
                f"pressure_unit={p_unit}",
                f"temperature={t_disp:.10g}",
                f"pressure={p_disp:.10g}",
                f"mean_molar_mass_g_per_mol={sol.gas_mean_molar_mass * 1000:.10g}",
                f"gamma={sol.gamma:.10g}",
                f"cp={sol.cp:.10g}",
            ]
        )
        return "\n".join(lines)

    def refresh_report_for_units(self) -> None:
        """Re-render the latest successful report with the active unit system."""
        payload = self._last_report_payload
        if not payload:
            return
        report_type = payload.get("type")
        if report_type == "performance":
            self.main_window.page_analysis.results_text.setText(
                self._build_performance_report_text(payload)
            )
        elif report_type == "equilibrium":
            self.main_window.page_analysis.results_text.setText(
                self._build_equilibrium_report_text(payload["solution"])
            )

    def _render_error_report(self, title: str, message: str, tb: str = "") -> None:
        report = [
            f"=== {title} ===",
            "",
            f"Message: {message}",
        ]
        if tb:
            report.extend(["", "Traceback:", tb])
        self.main_window.page_analysis.results_text.setText("\n".join(report))

    def _start_run_progress(self, total_cases: int) -> None:
        self.run_progress_label.show()
        self.run_progress_bar.show()
        if total_cases <= 1:
            self.run_progress_label.setText("Calculating single point...")
            self.run_progress_bar.setRange(0, 0)
            self.run_progress_bar.setFormat("Working...")
        else:
            self.run_progress_label.setText(f"Running sweep: 0/{total_cases}")
            self.run_progress_bar.setRange(0, total_cases)
            self.run_progress_bar.setValue(0)
            self.run_progress_bar.setFormat("%v/%m")

    def _update_run_progress(self, done: int, total: int, label: str) -> None:
        if total <= 1:
            self.run_progress_label.setText("Calculating single point...")
            return
        self.run_progress_label.setText(f"Running sweep: {done}/{total}")
        self.run_progress_bar.setRange(0, total)
        self.run_progress_bar.setValue(done)
        self.run_progress_bar.setFormat(label if label else "%v/%m")

    def _finish_run_progress(self) -> None:
        self.run_progress_label.hide()
        self.run_progress_bar.hide()

    def on_calculate(self):
        logger.info("Calculate button clicked.")
        # 1. Gather Propellants
        sim_page = self.main_window.page_simulator

        jobs = []
        try:
            if sim_page.sim_tabs.currentIndex() == 0:
                # Bipropellant
                f_tot = float(sim_page.fuel_tot.text())
                o_tot = float(sim_page.ox_tot.text())

                if f_tot == 0 or o_tot == 0:
                    msg = "Cannot calculate: Missing fuel or oxidizer."
                    logger.warning(msg)
                    self.main_window.statusBar().showMessage(msg, 5000)
                    return

                of_values = []
                pc_inputs = []
                of_mode_sweep = sim_page.of_mode_combo.currentText() == "Sweep Range"
                pc_mode_sweep = self.pc_mode_combo.currentText() == "Sweep Range"

                if of_mode_sweep and pc_mode_sweep:
                    msg = "Select only one sweep mode at a time: O/F or Pc."
                    self.main_window.statusBar().showMessage(msg, 6000)
                    return

                if sim_page.of_mode_combo.currentText() == "Sweep Range":
                    of_min = float(sim_page.input_of_min.text())
                    of_max = float(sim_page.input_of_max.text())
                    of_steps = int(sim_page.input_of_steps.text())
                    of_values = self._build_sweep_values(
                        of_min, of_max, of_steps, "O/F"
                    )
                else:
                    of_values = [sim_page.input_of_ratio.value()]

                if self.pc_mode_combo.currentText() == "Sweep Range":
                    pc_min = float(self.input_pc_min.text())
                    pc_max = float(self.input_pc_max.text())
                    pc_steps = int(self.input_pc_steps.text())
                    pc_inputs = self._build_sweep_values(pc_min, pc_max, pc_steps, "Pc")
                else:
                    pc_inputs = [float(self.input_pc.text())]

                sweep_axis = "none"
                if len(of_values) > 1:
                    sweep_axis = "of"
                elif len(pc_inputs) > 1:
                    sweep_axis = "pc"

                if sweep_axis == "of":
                    case_inputs = [(of, pc_inputs[0], of) for of in of_values]
                elif sweep_axis == "pc":
                    case_inputs = [
                        (of_values[0], pc_input, pc_input) for pc_input in pc_inputs
                    ]
                else:
                    case_inputs = [(of_values[0], pc_inputs[0], None)]

                logger.info(
                    f"Bipropellant mode: O/F cases={of_values}, fuel_tot={f_tot}, ox_tot={o_tot}"
                )

                fuel_components = {}
                ox_components = {}

                # Iterate through fuel table
                for r in range(sim_page.fuel_table.rowCount()):
                    container = sim_page.fuel_table.cellWidget(r, 0)
                    combo = container.findChild(QComboBox) if container else None
                    w_inp = sim_page.fuel_table.cellWidget(r, 1)

                    name = (combo.currentData() or combo.currentText()) if combo else ""
                    w = float(w_inp.text()) if w_inp and w_inp.text() else 0.0
                    if w > 0 and name:
                        fuel_components[name] = fuel_components.get(name, 0.0) + w

                # Iterate through ox table
                for r in range(sim_page.ox_table.rowCount()):
                    container = sim_page.ox_table.cellWidget(r, 0)
                    combo = container.findChild(QComboBox) if container else None
                    w_inp = sim_page.ox_table.cellWidget(r, 1)

                    name = (combo.currentData() or combo.currentText()) if combo else ""
                    w = float(w_inp.text()) if w_inp and w_inp.text() else 0.0
                    if w > 0 and name:
                        ox_components[name] = ox_components.get(name, 0.0) + w

                for of, pc_input, sweep_value in case_inputs:
                    mix_dict = {}
                    f_scale = 1.0 / f_tot
                    o_scale = of / o_tot
                    for name, w in fuel_components.items():
                        mix_dict[name] = mix_dict.get(name, 0.0) + w * f_scale
                    for name, w in ox_components.items():
                        mix_dict[name] = mix_dict.get(name, 0.0) + w * o_scale

                    prop_db = self.main_window.prop_db
                    mixture = prop_db.mix(list(mix_dict.items()))

                    spec_db = self.main_window.spec_db
                    enabled_dbs = self.get_enabled_databases()
                    if not enabled_dbs:
                        msg = "Cannot calculate: No species databases selected."
                        logger.warning(msg)
                        self.main_window.statusBar().showMessage(msg, 5000)
                        return
                    products = spec_db.get_species(
                        mixture.elements, max_atoms=20, enabled_databases=enabled_dbs
                    )

                    pc_pa = self._pressure_input_to_pa(pc_input)

                    prob = EquilibriumProblem(
                        reactants=mixture.reactants,
                        products=products,
                        problem_type=ProblemType.HP,
                        constraint1=mixture.enthalpy,
                        constraint2=pc_pa,
                        t_init=3500.0,
                    )
                    prob.validate()
                    jobs.append((sweep_value, prob))

            else:
                # Solid
                logger.info("Solid propellant mode selected.")
                if self.pc_mode_combo.currentText() == "Sweep Range":
                    pc_min = float(self.input_pc_min.text())
                    pc_max = float(self.input_pc_max.text())
                    pc_steps = int(self.input_pc_steps.text())
                    pc_inputs = self._build_sweep_values(pc_min, pc_max, pc_steps, "Pc")
                    sweep_axis = "pc"
                else:
                    pc_inputs = [float(self.input_pc.text())]
                    sweep_axis = "none"

                mix_dict = {}
                for row in range(sim_page.solid_table.rowCount()):
                    container = sim_page.solid_table.cellWidget(row, 0)
                    combo = container.findChild(QComboBox) if container else None
                    w_inp = sim_page.solid_table.cellWidget(row, 1)

                    name = (combo.currentData() or combo.currentText()) if combo else ""
                    w = float(w_inp.text()) if w_inp and w_inp.text() else 0.0
                    if w > 0 and name:
                        mix_dict[name] = mix_dict.get(name, 0.0) + w

                if not mix_dict:
                    msg = "Cannot calculate: Propellant mixture is empty."
                    logger.warning(msg)
                    self.main_window.statusBar().showMessage(msg, 5000)
                    return

                prop_db = self.main_window.prop_db
                mixture = prop_db.mix(list(mix_dict.items()))

                spec_db = self.main_window.spec_db
                enabled_dbs = self.get_enabled_databases()
                if not enabled_dbs:
                    msg = "Cannot calculate: No species databases selected."
                    logger.warning(msg)
                    self.main_window.statusBar().showMessage(msg, 5000)
                    return
                products = spec_db.get_species(
                    mixture.elements, max_atoms=20, enabled_databases=enabled_dbs
                )

                for pc_input in pc_inputs:
                    pc_pa = self._pressure_input_to_pa(pc_input)
                    prob = EquilibriumProblem(
                        reactants=mixture.reactants,
                        products=products,
                        problem_type=ProblemType.HP,
                        constraint1=mixture.enthalpy,
                        constraint2=pc_pa,
                        t_init=3500.0,
                    )
                    prob.validate()
                    jobs.append((pc_input if sweep_axis == "pc" else None, prob))
        except Exception as e:
            logger.exception("Failed to gather propellant data from UI")
            self.main_window.statusBar().showMessage(f"UI Error: {e}", 5000)
            return

        if not jobs:
            self.main_window.statusBar().showMessage(
                "No valid simulation cases generated.", 5000
            )
            return

        # Sweep metadata for plotting/report labels.
        if sim_page.sim_tabs.currentIndex() == 0:
            if self.pc_mode_combo.currentText() == "Sweep Range":
                sweep_axis = "pc"
            elif sim_page.of_mode_combo.currentText() == "Sweep Range":
                sweep_axis = "of"
            else:
                sweep_axis = "none"
        else:
            sweep_axis = (
                "pc" if self.pc_mode_combo.currentText() == "Sweep Range" else "none"
            )

        if sweep_axis == "of":
            sweep_label = "O/F Ratio"
        elif sweep_axis == "pc":
            pressure_unit = "MPa" if self.main_window.current_units == "SI" else "PSI"
            sweep_label = f"Chamber Pressure ({pressure_unit})"
        else:
            sweep_label = "Run Index"

        logger.info(f"Final mixture components: {jobs}")

        # 5. Run Solver
        self.btn_calculate.setEnabled(False)
        self.btn_calculate.setText("Calculating...")

        # Performance solve (run both shifting and frozen in one pass)
        exit_val = float(self.input_exp.text())
        ambient_input = float(self.input_ambient.text())
        is_pressure = "Pressure" in self.spec_combo.currentText()

        if is_pressure:
            exit_val = self._pressure_input_to_pa(exit_val)
        ambient_pa = self._pressure_input_to_pa(ambient_input)

        self.worker = PerformanceWorker(
            jobs,
            exit_val,
            is_pressure,
            ambient_pa,
            self.solver,
            self.main_window.spec_db,
            sweep_axis=sweep_axis,
            sweep_label=sweep_label,
            compute_profile=self.check_nozzle_profile.isChecked(),
        )
        self._start_run_progress(len(jobs))
        self.worker.progress.connect(self._update_run_progress)
        self.worker.finished.connect(self.on_perf_finished)
        self.worker.start()

    def on_perf_finished(self, result):
        self._finish_run_progress()
        self.btn_calculate.setEnabled(True)
        self.btn_calculate.setText("Calculate")

        if isinstance(result, dict) and result.get("ok") is False:
            msg = result.get("message", "Unknown performance error")
            tb = result.get("traceback", "")
            self.res_tc.setText("ERROR")
            self.main_window.statusBar().showMessage(f"Performance Error: {msg}", 12000)
            self._render_error_report("Performance Solver Error", msg, tb)
            self._last_report_payload = None
            return

        if isinstance(result, str):
            self.res_tc.setText("ERROR")
            self.main_window.statusBar().showMessage(
                f"Performance Error: {result}", 12000
            )
            self._render_error_report("Performance Solver Error", result)
            self._last_report_payload = None
            return

        payload = result if isinstance(result, dict) else {"cases": [(None, result)]}
        cases = payload.get("cases", [])
        sweep_axis = payload.get("sweep_axis", "none")
        sweep_label = payload.get("sweep_label", "Run Index")
        if not cases:
            self.res_tc.setText("ERROR")
            self.main_window.statusBar().showMessage("No results returned.", 10000)
            self._render_error_report(
                "Performance Solver Error", "No results returned."
            )
            self._last_report_payload = None
            return

        _, perf = cases[0]
        if isinstance(perf, str):
            self.res_tc.setText("ERROR")
            self.main_window.statusBar().showMessage(
                f"Performance Error: {perf}", 12000
            )
            self._render_error_report("Performance Solver Error", perf)
            self._last_report_payload = None
            return

        perf: RocketPerformanceComparison
        shifting = perf.shifting
        frozen = perf.frozen

        if shifting.chamber.converged and frozen.chamber.converged:
            self.res_tc.setText(f"{shifting.chamber.temperature:.1f}")
            self.res_cstar.setText(f"{shifting.cstar:.1f}")
            self.res_cstar_frozen.setText(f"{frozen.cstar:.1f}")
            self.res_isp.setText(f"{shifting.isp_actual:.1f}")
            self.res_isp_frozen.setText(f"{frozen.isp_actual:.1f}")

            self.main_window.statusBar().showMessage(
                (
                    f"Converged. Isp(actual): shifting={shifting.isp_actual:.1f}s, "
                    f"frozen={frozen.isp_actual:.1f}s"
                ),
                5000,
            )

            chamber = shifting.chamber
            is_pressure_spec = "Pressure" in self.spec_combo.currentText()
            self._last_report_payload = {
                "type": "performance",
                "cases": cases,
                "sweep_axis": sweep_axis,
                "sweep_label": sweep_label,
                "spec_kind": "pressure" if is_pressure_spec else "area",
                "expansion_target": float(self.input_exp.text()),
            }
            self.main_window.page_analysis.results_text.setText(
                self._build_performance_report_text(self._last_report_payload)
            )
            self.main_window.page_analysis.update_convergence_plots(shifting.chamber)
            self.main_window.page_analysis.update_expansion_plots(perf)
            self.main_window.page_analysis.update_performance_plots(
                cases,
                sweep_axis=sweep_axis,
                sweep_label=sweep_label,
            )

        else:
            self.res_tc.setText("FAIL")
            fail_msg = "Performance solve failed to converge."
            self.main_window.statusBar().showMessage(fail_msg, 5000)
            self._render_error_report("Performance Solver Failure", fail_msg)
            self._last_report_payload = None

    def on_solve_finished(self, result):
        self._finish_run_progress()
        self.btn_calculate.setEnabled(True)
        self.btn_calculate.setText("Calculate")

        if isinstance(result, dict):
            if not result.get("ok", False):
                msg = result.get("message", "Unknown solver error")
                tb = result.get("traceback", "")
                self.res_tc.setText("ERROR")
                self.main_window.statusBar().showMessage(f"Solver Error: {msg}", 10000)
                self._render_error_report("Equilibrium Solver Error", msg, tb)
                self._last_report_payload = None
                return
            sol = result.get("solution")
        elif isinstance(result, str):
            self.res_tc.setText("ERROR")
            self.main_window.statusBar().showMessage(f"Solver Error: {result}", 10000)
            self._render_error_report("Equilibrium Solver Error", result)
            self._last_report_payload = None
            return
        else:
            sol = result

        # Success
        if sol.converged:
            self.res_tc.setText(f"{sol.temperature:.1f}")
            # Approximate c* (assuming frozen gamma for display)
            try:
                gamma = sol.gamma
                M = sol.gas_mean_molar_mass
                R = 8.314462
                T = sol.temperature

                # C* = sqrt(R*T/M) / GammaFunc
                from math import sqrt

                v_term = gamma * ((2 / (gamma + 1)) ** ((gamma + 1) / (gamma - 1)))
                cstar = sqrt(R * T / M) / sqrt(v_term)
                self.res_cstar.setText(f"{cstar:.1f}")
            except Exception:
                self.res_cstar.setText("N/A")

            self.main_window.statusBar().showMessage(
                f"Converged in {sol.iterations} iterations.", 5000
            )

            # Update report
            self._last_report_payload = {"type": "equilibrium", "solution": sol}
            self.main_window.page_analysis.results_text.setText(
                self._build_equilibrium_report_text(sol)
            )
            self.main_window.page_analysis.update_convergence_plots(sol)

        else:
            self.res_tc.setText("FAIL")
            fail_msg = "Solver failed to converge."
            self.main_window.statusBar().showMessage(fail_msg, 5000)
            self._render_error_report("Equilibrium Solver Failure", fail_msg)
            self._last_report_payload = None
