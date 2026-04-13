"""GUI page for Optuna-based propellant optimization."""

from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from prometheus_equilibrium.gui.widgets.graph_canvas import GraphCanvas
from prometheus_equilibrium.optimization import (
    FixedProportionGroup,
    ObjectiveSpec,
    OperatingPoint,
    OptimizationProblem,
    OptunaOptimizer,
    SumToTotalGroup,
    VariableBound,
)


@dataclass(frozen=True)
class _RunConfig:
    """Lightweight optimization execution config passed to worker."""

    n_trials: int
    timeout_seconds: int | None
    seed: int | None


class OptimizationWorker(QThread):
    """Background worker that runs a full Optuna study."""

    progress = Signal(object)
    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, optimizer: OptunaOptimizer, run_config: _RunConfig):
        super().__init__()
        self.optimizer = optimizer
        self.run_config = run_config
        self._cancel_requested = False

    def request_cancel(self) -> None:
        """Request cancellation at the next trial boundary."""
        self._cancel_requested = True

    def run(self) -> None:
        try:
            result = self.optimizer.optimize(
                n_trials=self.run_config.n_trials,
                timeout_seconds=self.run_config.timeout_seconds,
                seed=self.run_config.seed,
                progress_callback=self.progress.emit,
                should_stop=lambda: self._cancel_requested,
            )
            self.finished.emit(result)
        except Exception as exc:
            self.failed.emit(str(exc))


class OptimizerPage(QWidget):
    """Configuration and execution page for propellant optimization."""

    def __init__(self, main_window, prop_db):
        super().__init__()
        self.main_window = main_window
        self.prop_db = prop_db
        self.worker: OptimizationWorker | None = None
        self._latest_best_composition: dict[str, float] | None = None
        self._live_history: list[tuple[int, float]] = []
        self._baseline_mass_fraction: dict[str, float] = {}

        layout = QVBoxLayout(self)

        scroll_area = QScrollArea(self)
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        content = QWidget(scroll_area)
        content_layout = QVBoxLayout(content)
        content_layout.addWidget(self._build_controls_group())
        content_layout.addWidget(self._build_variables_group())
        content_layout.addWidget(self._build_group_rules_group())
        content_layout.addWidget(self._build_actions_group())
        content_layout.addWidget(self._build_results_group())
        content_layout.addStretch()

        scroll_area.setWidget(content)
        layout.addWidget(scroll_area)

    def _build_controls_group(self) -> QGroupBox:
        group = QGroupBox("Objective and Study Controls")
        form = QFormLayout(group)

        self.combo_isp_variant = QComboBox()
        self.combo_isp_variant.addItems(["isp_actual", "isp_vac", "isp_sl"])

        self.spin_rho_exp = QDoubleSpinBox()
        self.spin_rho_exp.setRange(0.0, 1.0)
        self.spin_rho_exp.setDecimals(3)
        self.spin_rho_exp.setSingleStep(0.05)
        self.spin_rho_exp.setValue(0.25)

        self.spin_trials = QSpinBox()
        self.spin_trials.setRange(1, 100000)
        self.spin_trials.setValue(64)

        self.spin_timeout = QSpinBox()
        self.spin_timeout.setRange(0, 86_400)
        self.spin_timeout.setValue(0)
        self.spin_timeout.setSuffix(" s (0 = none)")

        self.spin_seed = QSpinBox()
        self.spin_seed.setRange(0, 2_147_483_647)
        self.spin_seed.setValue(42)

        self.check_shifting = QCheckBox("Score shifting expansion")
        self.check_shifting.setChecked(True)

        self.combo_closure = QComboBox()
        self.combo_closure.addItem("(none)", None)

        form.addRow("Isp variant", self.combo_isp_variant)
        form.addRow("Density exponent n", self.spin_rho_exp)
        form.addRow("Trials", self.spin_trials)
        form.addRow("Timeout", self.spin_timeout)
        form.addRow("Seed", self.spin_seed)
        form.addRow("Expansion mode", self.check_shifting)
        form.addRow("Closure ingredient", self.combo_closure)
        return group

    def _build_variables_group(self) -> QGroupBox:
        group = QGroupBox("Variable Bounds")
        layout = QVBoxLayout(group)

        # Columns: 0=Ingredient, 1=Min (%), 2=Max (%), 3=Pinned, 4=Group Labels
        self.table_vars = QTableWidget(0, 5)
        self.table_vars.setHorizontalHeaderLabels(
            ["Ingredient", "Min (%)", "Max (%)", "Pinned", "Group Labels"]
        )
        self.table_vars.horizontalHeader().setStretchLastSection(True)
        self.table_vars.setMinimumHeight(220)

        btns = QHBoxLayout()
        self.btn_from_sim = QPushButton("Load from Simulator (Solid)")
        self.btn_from_sim.clicked.connect(self.load_from_simulator)
        self.btn_add_var = QPushButton("+ Add Variable")
        self.btn_add_var.clicked.connect(
            lambda: self._append_variable_row("", 0.0, 1.0, "")
        )
        self.btn_remove_var = QPushButton("Remove Selected")
        self.btn_remove_var.clicked.connect(self._remove_selected_variable)
        btns.addWidget(self.btn_from_sim)
        btns.addWidget(self.btn_add_var)
        btns.addWidget(self.btn_remove_var)

        layout.addWidget(self.table_vars)
        layout.addLayout(btns)
        return group

    def _build_group_rules_group(self) -> QGroupBox:
        group = QGroupBox("Group Rules")
        grid = QGridLayout(group)

        self.table_group_rules = QTableWidget(0, 4)
        self.table_group_rules.setHorizontalHeaderLabels(
            ["Group Label", "Type", "Min Total", "Max Total"]
        )
        self.table_group_rules.horizontalHeader().setStretchLastSection(True)
        self.table_group_rules.setMinimumHeight(180)

        self.btn_add_rule = QPushButton("+ Add Group Rule")
        self.btn_add_rule.clicked.connect(self._append_default_group_rule)
        self.btn_remove_rule = QPushButton("Remove Selected")
        self.btn_remove_rule.clicked.connect(self._remove_selected_rule)
        self.btn_group_help = QPushButton("Group Rule Helper...")
        self.btn_group_help.clicked.connect(self._open_group_rule_helper)

        grid.addWidget(
            QLabel(
                "Assign group labels in Variable Bounds (comma-separated for multiple groups), "
                "then define one rule per label. "
                "Fixed-proportion ratios are derived from the starting formulation. "
                "Pinned ingredients are held at their Min value and excluded from optimisation."
            ),
            0,
            0,
        )
        grid.addWidget(self.table_group_rules, 1, 0)
        actions = QHBoxLayout()
        actions.addWidget(self.btn_add_rule)
        actions.addWidget(self.btn_remove_rule)
        actions.addWidget(self.btn_group_help)
        actions.addStretch()
        grid.addLayout(actions, 2, 0)
        return group

    def _build_actions_group(self) -> QGroupBox:
        group = QGroupBox("Run")
        layout = QVBoxLayout(group)

        buttons = QHBoxLayout()
        self.btn_start = QPushButton("Start Optimization")
        self.btn_cancel = QPushButton("Cancel")
        self.btn_apply = QPushButton("Apply Best to Simulator")
        self.btn_cancel.setEnabled(False)
        self.btn_apply.setEnabled(False)

        self.btn_start.clicked.connect(self.start_optimization)
        self.btn_cancel.clicked.connect(self.cancel_optimization)
        self.btn_apply.clicked.connect(self.apply_best_to_simulator)

        buttons.addWidget(self.btn_start)
        buttons.addWidget(self.btn_cancel)
        buttons.addWidget(self.btn_apply)

        config_btns = QHBoxLayout()
        self.btn_save_config = QPushButton("Save Config...")
        self.btn_load_config = QPushButton("Load Config...")
        self.btn_save_config.clicked.connect(self._save_config)
        self.btn_load_config.clicked.connect(self._load_config)
        config_btns.addWidget(self.btn_save_config)
        config_btns.addWidget(self.btn_load_config)
        config_btns.addStretch()

        self.progress_label = QLabel("Idle")
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)

        layout.addLayout(buttons)
        layout.addLayout(config_btns)
        layout.addWidget(self.progress_label)
        layout.addWidget(self.progress)
        return group

    def _build_results_group(self) -> QGroupBox:
        group = QGroupBox("Optimization Results")
        layout = QVBoxLayout(group)

        self.canvas_best = GraphCanvas(
            self,
            "Best Objective (log scale objective)",
            "Trial",
            "Best log(Isp * rho^n)",
        )
        self.canvas_best.setMinimumHeight(260)
        self.output = QTextEdit()
        self.output.setReadOnly(True)
        self.output.setText("Run an optimization to populate results.")

        layout.addWidget(self.canvas_best)
        layout.addWidget(self.output)
        return group

    def _append_variable_row(
        self,
        ingredient_id: str,
        minimum: float,
        maximum: float,
        group_label: str,
        pinned: bool = False,
    ) -> None:
        row = self.table_vars.rowCount()
        self.table_vars.insertRow(row)
        self.table_vars.setItem(row, 0, QTableWidgetItem(ingredient_id))
        self.table_vars.setItem(row, 1, QTableWidgetItem(f"{minimum:.6f}"))
        self.table_vars.setItem(row, 2, QTableWidgetItem(f"{maximum:.6f}"))
        pin_chk = QCheckBox()
        pin_chk.setChecked(pinned)
        self.table_vars.setCellWidget(row, 3, pin_chk)
        self.table_vars.setItem(row, 4, QTableWidgetItem(group_label))
        self._refresh_closure_options()

    def _remove_selected_variable(self) -> None:
        """Remove the currently selected row from the variable bounds table."""
        row = self.table_vars.currentRow()
        if row >= 0:
            self.table_vars.removeRow(row)
            self._refresh_closure_options()

    def _remove_selected_rule(self) -> None:
        """Remove the currently selected row from the group rules table."""
        row = self.table_group_rules.currentRow()
        if row >= 0:
            self.table_group_rules.removeRow(row)

    def _refresh_closure_options(self) -> None:
        """Synchronize closure ingredient choices with variable-table ingredient IDs."""
        current = self.combo_closure.currentData()
        ingredient_ids = []
        for row in range(self.table_vars.rowCount()):
            item = self.table_vars.item(row, 0)
            if item is None:
                continue
            ingredient_id = item.text().strip()
            if ingredient_id:
                ingredient_ids.append(ingredient_id)

        self.combo_closure.blockSignals(True)
        self.combo_closure.clear()
        self.combo_closure.addItem("(none)", None)
        for ingredient_id in ingredient_ids:
            self.combo_closure.addItem(ingredient_id, ingredient_id)
        idx = self.combo_closure.findData(current)
        self.combo_closure.setCurrentIndex(idx if idx >= 0 else 0)
        self.combo_closure.blockSignals(False)

    def _append_default_group_rule(self) -> None:
        row = self.table_group_rules.rowCount()
        self.table_group_rules.insertRow(row)
        self.table_group_rules.setItem(row, 0, QTableWidgetItem("group"))
        combo = QComboBox()
        combo.addItems(["fixed_proportion", "sum_to_total"])
        self.table_group_rules.setCellWidget(row, 1, combo)
        self.table_group_rules.setItem(row, 2, QTableWidgetItem(""))
        self.table_group_rules.setItem(row, 3, QTableWidgetItem(""))

    def _open_group_rule_helper(self) -> None:
        """Open a lightweight helper for quickly adding one group rule row."""
        labels = sorted(
            {
                label.strip()
                for r in range(self.table_vars.rowCount())
                for raw in [
                    (
                        self.table_vars.item(r, 4).text()
                        if self.table_vars.item(r, 4)
                        else ""
                    )
                ]
                for label in raw.split(",")
                if label.strip()
            }
        )
        if not labels:
            QMessageBox.information(
                self,
                "No Labels",
                "Add group labels in Variable Bounds before using the helper.",
            )
            return

        dialog = QDialog(self)
        dialog.setWindowTitle("Group Rule Helper")
        form = QFormLayout(dialog)
        combo_label = QComboBox(dialog)
        combo_label.addItems(labels)
        combo_type = QComboBox(dialog)
        combo_type.addItems(["fixed_proportion", "sum_to_total"])
        edit_min = QLineEdit(dialog)
        edit_max = QLineEdit(dialog)
        form.addRow("Group label", combo_label)
        form.addRow("Rule type", combo_type)
        form.addRow("Min total", edit_min)
        form.addRow("Max total", edit_max)

        buttons = QHBoxLayout()
        btn_ok = QPushButton("Add")
        btn_cancel = QPushButton("Cancel")
        buttons.addWidget(btn_ok)
        buttons.addWidget(btn_cancel)
        form.addRow(buttons)

        def _accept() -> None:
            self._append_default_group_rule()
            row = self.table_group_rules.rowCount() - 1
            self.table_group_rules.item(row, 0).setText(combo_label.currentText())
            type_combo = self.table_group_rules.cellWidget(row, 1)
            if isinstance(type_combo, QComboBox):
                type_combo.setCurrentText(combo_type.currentText())
            self.table_group_rules.item(row, 2).setText(edit_min.text().strip())
            self.table_group_rules.item(row, 3).setText(edit_max.text().strip())
            dialog.accept()

        btn_ok.clicked.connect(_accept)
        btn_cancel.clicked.connect(dialog.reject)
        dialog.exec()

    def load_from_simulator(self) -> None:
        """Pre-populate optimization variables from the current solid formulation."""
        payload = self.main_window.page_simulator.composition_snapshot()
        if payload.get("propellant_type") != "solid":
            QMessageBox.warning(
                self,
                "Solid Mode Required",
                "Optimizer v1 preload supports solid/monopropellant mode only.",
            )
            return

        rows = payload.get("components", [])
        total = sum(float(r.get("mass_fraction", 0.0)) for r in rows)
        if total <= 0.0:
            QMessageBox.warning(
                self, "No Composition", "No positive mass fractions found."
            )
            return

        self.table_vars.setRowCount(0)
        self.table_group_rules.setRowCount(0)
        self._baseline_mass_fraction = {}
        for row in rows:
            ingredient_id = str(row.get("name", "")).strip()
            if not ingredient_id:
                continue
            center = float(row.get("mass_fraction", 0.0)) / total
            minimum = max(0.0, center * 0.7)
            maximum = min(1.0, center * 1.3)
            self._append_variable_row(ingredient_id, minimum, maximum, "")
            self._baseline_mass_fraction[ingredient_id] = center

        self._refresh_closure_options()
        if self.combo_closure.count() > 1 and self.combo_closure.currentData() is None:
            # Default closure to the first ingredient to avoid all-pruned runs.
            self.combo_closure.setCurrentIndex(1)

        self.output.setText(
            "Loaded optimization bounds from Simulator solid formulation. "
            "Adjust bounds/relations before starting."
        )

    def _collect_problem(self) -> OptimizationProblem:
        variables: list[VariableBound] = []
        members_by_label: dict[str, list[str]] = {}
        midpoint_by_ingredient: dict[str, float] = {}
        for row in range(self.table_vars.rowCount()):
            ingredient_item = self.table_vars.item(row, 0)
            min_item = self.table_vars.item(row, 1)
            max_item = self.table_vars.item(row, 2)
            pin_widget = self.table_vars.cellWidget(row, 3)
            label_item = self.table_vars.item(row, 4)
            if not ingredient_item or not min_item or not max_item:
                continue
            ingredient_id = ingredient_item.text().strip()
            if not ingredient_id:
                continue
            pinned = isinstance(pin_widget, QCheckBox) and pin_widget.isChecked()
            min_val = float(min_item.text())
            max_val = min_val if pinned else float(max_item.text())
            variables.append(
                VariableBound(
                    ingredient_id=ingredient_id,
                    minimum=min_val,
                    maximum=max_val,
                )
            )
            midpoint_by_ingredient[ingredient_id] = (min_val + max_val) * 0.5
            raw_label = label_item.text().strip() if label_item else ""
            for label in (l.strip() for l in raw_label.split(",") if l.strip()):
                members_by_label.setdefault(label, []).append(ingredient_id)

        fixed_groups: list[FixedProportionGroup] = []
        sum_groups: list[SumToTotalGroup] = []
        labels_with_rules: set[str] = set()
        for row in range(self.table_group_rules.rowCount()):
            label_item = self.table_group_rules.item(row, 0)
            ratios_item = self.table_group_rules.item(row, 2)
            total_item = self.table_group_rules.item(row, 3)
            if not label_item:
                continue
            group_label = label_item.text().strip()
            if not group_label:
                continue
            labels_with_rules.add(group_label)
            members = members_by_label.get(group_label, [])
            if len(members) < 2:
                raise ValueError(
                    f"Group label {group_label!r} must be assigned to at least two ingredients."
                )

            rule_type = "fixed_proportion"
            type_widget = self.table_group_rules.cellWidget(row, 1)
            if isinstance(type_widget, QComboBox):
                rule_type = type_widget.currentText().strip()

            if rule_type == "fixed_proportion":
                ratios: list[float] = []
                for member in members:
                    base = self._baseline_mass_fraction.get(member)
                    if base is None or base <= 0.0:
                        base = midpoint_by_ingredient.get(member, 0.0)
                    if base <= 0.0:
                        raise ValueError(
                            f"Group {group_label!r}: cannot derive ratio for {member!r}."
                        )
                    ratios.append(base)
                fixed_groups.append(
                    FixedProportionGroup(
                        group_id=group_label,
                        members=members,
                        ratios=ratios,
                    )
                )
                continue

            if rule_type == "sum_to_total":
                min_text = ratios_item.text().strip() if ratios_item else ""
                max_text = total_item.text().strip() if total_item else ""
                min_total = float(min_text) if min_text else None
                max_total = float(max_text) if max_text else None
                if min_total is None and max_total is None:
                    raise ValueError(
                        f"Group {group_label!r}: set min and/or max total for inequality constraint."
                    )
                sum_groups.append(
                    SumToTotalGroup(
                        group_id=group_label,
                        members=members,
                        minimum_total=min_total,
                        maximum_total=max_total,
                    )
                )
                continue

            raise ValueError(f"Group {group_label!r}: unknown rule type {rule_type!r}.")

        missing_rules = sorted(set(members_by_label) - labels_with_rules)
        if missing_rules:
            raise ValueError(
                "Missing group rule definitions for labels: " + ", ".join(missing_rules)
            )

        return OptimizationProblem(
            variables=variables,
            fixed_proportion_groups=fixed_groups,
            sum_to_total_groups=sum_groups,
            total_mass_fraction=1.0,
            closure_ingredient_id=self.combo_closure.currentData(),
        )

    def _collect_optimizer(self) -> tuple[OptunaOptimizer, _RunConfig]:
        objective = ObjectiveSpec(
            isp_variant=self.combo_isp_variant.currentText(),
            rho_exponent=float(self.spin_rho_exp.value()),
        )

        ed = self.main_window.engine_dock
        chamber_pressure_pa = ed._pressure_input_to_pa(float(ed.input_pc.text()))
        is_pressure = "Pressure" in ed.spec_combo.currentText()
        expansion_value = float(ed.input_exp.text())
        if is_pressure:
            expansion_value = ed._pressure_input_to_pa(expansion_value)
        ambient_pressure = ed._pressure_input_to_pa(float(ed.input_ambient.text()))

        operating_point = OperatingPoint(
            chamber_pressure_pa=chamber_pressure_pa,
            expansion_type="pressure" if is_pressure else "area_ratio",
            expansion_value=expansion_value,
            ambient_pressure_pa=ambient_pressure,
            shifting=self.check_shifting.isChecked(),
        )

        problem = self._collect_problem()
        run_cfg = _RunConfig(
            n_trials=int(self.spin_trials.value()),
            timeout_seconds=int(self.spin_timeout.value()) or None,
            seed=int(self.spin_seed.value()),
        )
        optimizer = OptunaOptimizer(
            problem=problem,
            objective=objective,
            operating_point=operating_point,
            prop_db=self.main_window.prop_db,
            spec_db=self.main_window.spec_db,
            solver=ed.solver,
            enabled_databases=ed.get_enabled_databases(),
            max_atoms=ed._max_atoms(),
        )
        return optimizer, run_cfg

    # ------------------------------------------------------------------
    # Config save / load
    # ------------------------------------------------------------------

    def _collect_config(self) -> dict:
        """Collect the full optimizer config as a JSON-serialisable dict.

        Returns:
            Config dict suitable for :func:`~prometheus_equilibrium.optimization.config.save_json`.

        Raises:
            Exception: If the current setup cannot be serialised (e.g. validation error).
        """
        optimizer, run_cfg = self._collect_optimizer()
        ed = self.main_window.engine_dock
        solver_type = "gmcb"
        if hasattr(ed, "solver_combo"):
            solver_type = ed.solver_combo.currentData() or "gmcb"

        from prometheus_equilibrium.optimization.config import dump_config

        return dump_config(
            problem=optimizer.problem,
            objective=optimizer.objective,
            operating_point=optimizer.operating_point,
            n_trials=run_cfg.n_trials,
            timeout_seconds=run_cfg.timeout_seconds,
            seed=run_cfg.seed,
            solver_type=solver_type,
            enabled_databases=ed.get_enabled_databases(),
            max_atoms=ed._max_atoms(),
        )

    def _save_config(self) -> None:
        """Prompt for a path and write the current setup to a JSON config file."""
        try:
            config = self._collect_config()
        except Exception as exc:
            QMessageBox.critical(self, "Cannot Save Config", str(exc))
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Save Optimizer Config", "", "JSON Files (*.json);;All Files (*)"
        )
        if not path:
            return

        from prometheus_equilibrium.optimization.config import save_json

        try:
            save_json(path, config)
        except OSError as exc:
            QMessageBox.critical(self, "Save Failed", str(exc))
            return

        self.output.setText(f"Config saved to:\n{path}")

    def _load_config(self) -> None:
        """Prompt for a config JSON file and populate the optimizer page from it."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Optimizer Config", "", "JSON Files (*.json);;All Files (*)"
        )
        if not path:
            return

        import json

        try:
            with open(path, encoding="utf-8") as f:
                config = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            QMessageBox.critical(self, "Load Failed", str(exc))
            return

        try:
            self._apply_config(config)
        except Exception as exc:
            QMessageBox.critical(self, "Config Error", str(exc))
            return

        self.output.setText(f"Config loaded from:\n{path}")

    def _apply_config(self, config: dict) -> None:
        """Populate the optimizer page from a config dict.

        The variable bounds table, group rules, objective, and run settings
        are all restored.  The engine-dock operating point and solver settings
        are left unchanged (edit them in the dock if needed, or use the
        headless runner to apply the full saved config).

        Args:
            config: Config dict as produced by
                :func:`~prometheus_equilibrium.optimization.config.dump_config`.
        """
        p = config.get("problem", {})

        # Build a map: ingredient_id -> sorted list of group_ids it belongs to,
        # so we can restore the Group Labels column.
        label_map: dict[str, list[str]] = {}
        for g in p.get("fixed_proportion_groups", []):
            for m in g["members"]:
                if g["group_id"] not in label_map.setdefault(m, []):
                    label_map[m].append(g["group_id"])
        for g in p.get("sum_to_total_groups", []):
            for m in g["members"]:
                if g["group_id"] not in label_map.setdefault(m, []):
                    label_map[m].append(g["group_id"])

        # --- Variables table ---
        self.table_vars.setRowCount(0)
        self._baseline_mass_fraction = {}
        for v in p.get("variables", []):
            ingredient_id = v["ingredient_id"]
            pinned = v.get("pinned", False)
            min_val = float(v["minimum"])
            max_val = min_val if pinned else float(v["maximum"])
            labels = ",".join(label_map.get(ingredient_id, []))
            self._append_variable_row(
                ingredient_id, min_val, max_val, labels, pinned=pinned
            )
            self._baseline_mass_fraction[ingredient_id] = (
                min_val if pinned else (min_val + max_val) / 2.0
            )

        # --- Group rules table ---
        self.table_group_rules.setRowCount(0)
        for g in p.get("fixed_proportion_groups", []):
            self._append_default_group_rule()
            row = self.table_group_rules.rowCount() - 1
            self.table_group_rules.item(row, 0).setText(g["group_id"])
            w = self.table_group_rules.cellWidget(row, 1)
            if isinstance(w, QComboBox):
                w.setCurrentText("fixed_proportion")
            # Min/Max total fields are unused for fixed-proportion
            self.table_group_rules.item(row, 2).setText("")
            self.table_group_rules.item(row, 3).setText("")

        for g in p.get("sum_to_total_groups", []):
            self._append_default_group_rule()
            row = self.table_group_rules.rowCount() - 1
            self.table_group_rules.item(row, 0).setText(g["group_id"])
            w = self.table_group_rules.cellWidget(row, 1)
            if isinstance(w, QComboBox):
                w.setCurrentText("sum_to_total")
            if "total" in g:
                t = str(g["total"])
                self.table_group_rules.item(row, 2).setText(t)
                self.table_group_rules.item(row, 3).setText(t)
            else:
                lo = g.get("minimum_total")
                hi = g.get("maximum_total")
                self.table_group_rules.item(row, 2).setText(
                    "" if lo is None else str(lo)
                )
                self.table_group_rules.item(row, 3).setText(
                    "" if hi is None else str(hi)
                )

        # --- Closure ---
        self._refresh_closure_options()
        closure_id = p.get("closure_ingredient_id")
        if closure_id:
            idx = self.combo_closure.findData(closure_id)
            if idx >= 0:
                self.combo_closure.setCurrentIndex(idx)

        # --- Objective ---
        obj = config.get("objective", {})
        isp_idx = self.combo_isp_variant.findText(obj.get("isp_variant", "isp_actual"))
        if isp_idx >= 0:
            self.combo_isp_variant.setCurrentIndex(isp_idx)
        self.spin_rho_exp.setValue(float(obj.get("rho_exponent", 0.0)))
        self.check_shifting.setChecked(
            bool(config.get("operating_point", {}).get("shifting", True))
        )

        # --- Run config ---
        run = config.get("run", {})
        self.spin_trials.setValue(int(run.get("n_trials", 64)))
        self.spin_timeout.setValue(int(run.get("timeout_seconds") or 0))
        self.spin_seed.setValue(int(run.get("seed") or 42))

    def start_optimization(self) -> None:
        """Launch optimization on a background thread."""
        try:
            optimizer, run_cfg = self._collect_optimizer()
        except Exception as exc:
            QMessageBox.critical(self, "Invalid Configuration", str(exc))
            return

        self.btn_start.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.btn_apply.setEnabled(False)
        self.progress_label.setText("Running optimization...")
        self.progress.setRange(0, run_cfg.n_trials)
        self.progress.setValue(0)
        self._latest_best_composition = None
        self._live_history = []
        self.output.setText("Running optimization...")
        self._reset_history_plot()

        self.worker = OptimizationWorker(optimizer, run_cfg)
        self.worker.progress.connect(self._on_progress)
        self.worker.finished.connect(self._on_finished)
        self.worker.failed.connect(self._on_failed)
        self.worker.start()

    def cancel_optimization(self) -> None:
        """Request graceful cancellation."""
        if self.worker is None:
            return
        self.worker.request_cancel()
        self.progress_label.setText("Cancellation requested; stopping at next trial...")

    def _on_progress(self, payload: dict) -> None:
        trial = int(payload.get("trial", 0))
        best = payload.get("best_value")
        status_kind = payload.get("status_kind", "")
        status_reason = payload.get("status_reason", "")
        self.progress.setValue(min(trial + 1, self.progress.maximum()))
        if best is None:
            self.progress_label.setText(
                f"Trial {trial + 1}: no complete trial yet"
                + (f" ({status_kind})" if status_kind else "")
            )
            return
        if not self._live_history or self._live_history[-1][0] != trial:
            self._live_history.append((trial, float(best)))
            self._plot_history(self._live_history)
        self.progress_label.setText(
            f"Trial {trial + 1}: best log FoM = {best:.6f}"
            + (f" ({status_kind}: {status_reason})" if status_reason else "")
        )

    def _on_finished(self, result) -> None:
        self.btn_start.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.btn_apply.setEnabled(True)
        self._latest_best_composition = dict(result.best_composition)

        self.progress_label.setText(
            f"Finished: {result.completed_trials} complete, {result.pruned_trials} pruned"
        )
        self.progress.setValue(self.progress.maximum())

        lines = [
            f"Best objective (Isp * rho^n): {result.best_objective:.6f}",
            f"Best Isp ({self.combo_isp_variant.currentText()}): {result.best_isp:.6f}",
            f"Best density: {result.best_density:.6f} kg/m^3",
            "",
            "Best composition:",
        ]
        for ingredient_id, value in sorted(result.best_composition.items()):
            lines.append(f"- {ingredient_id}: {value:.6f}")
        self.output.setText("\n".join(lines))
        self._live_history = list(result.trial_history)
        self._plot_history(result.trial_history)

    def _on_failed(self, message: str) -> None:
        self.btn_start.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.progress_label.setText("Optimization failed")
        self.output.setText(f"Optimization failed:\n{message}")

    def _reset_history_plot(self) -> None:
        ax = self.canvas_best.axes
        ax.clear()
        ax.set_title(self.canvas_best.title_text, color="white")
        ax.set_xlabel(self.canvas_best.xlabel_text, color="white")
        ax.set_ylabel(self.canvas_best.ylabel_text, color="white")
        ax.tick_params(colors="white")
        ax.grid(True, linestyle="--", alpha=0.3)
        self.canvas_best.draw()

    def _plot_history(self, history: list[tuple[int, float]]) -> None:
        self._reset_history_plot()
        if not history:
            return
        x = [t for t, _ in history]
        y = [v for _, v in history]
        self.canvas_best.axes.plot(x, y, "o-", color="#2a82da")
        self.canvas_best.draw()

    def apply_best_to_simulator(self) -> None:
        """Apply the current best composition into Simulator solid mode."""
        if not self._latest_best_composition:
            QMessageBox.information(self, "No Result", "Run optimization first.")
            return

        payload = {
            "schema_version": 1,
            "propellant_type": "solid",
            "components": [
                {"name": ingredient_id, "mass_fraction": value}
                for ingredient_id, value in sorted(
                    self._latest_best_composition.items()
                )
                if value > 0.0
            ],
        }

        self.main_window.page_simulator.apply_composition_snapshot(payload)
        self.main_window.engine_dock.clear_previous_results()
        self.main_window._focus_simulator(simulator_tab_index=1)
        self.main_window.statusBar().showMessage(
            "Applied optimizer best composition to Simulator (solid mode).", 5000
        )
