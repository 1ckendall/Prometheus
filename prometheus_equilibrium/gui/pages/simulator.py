from PySide6.QtCore import QLocale, Qt
from PySide6.QtGui import QDoubleValidator, QIntValidator
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QTableWidget,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from prometheus_equilibrium.gui.dialogs.database_search import DatabaseSearchDialog


class SimulatorPage(QWidget):
    def __init__(self, main_window, prop_db):
        super().__init__()
        self.main_window = main_window
        self.prop_db = prop_db
        self._suspend_formulation_notifications = True
        self.database_items = self.prop_db.ingredient_ids

        # Build {id -> display_name} map for combo population.
        # Display name is the human-readable "name" field; falls back to id.
        self._id_to_display: dict[str, str] = {}
        for ing_id in self.database_items:
            try:
                rec = self.prop_db.find_ingredient(ing_id)
                self._id_to_display[ing_id] = rec.get("name", ing_id)
            except Exception:
                self._id_to_display[ing_id] = ing_id

        # Categorize ingredients by role for filtering.
        # Each list contains (display_name, ingredient_id) pairs so combos can
        # call addItem(display, userData=id).
        self.oxidizer_items: list[tuple[str, str]] = [("", "")]
        self.fuel_items: list[tuple[str, str]] = [("", "")]
        self.all_items: list[tuple[str, str]] = [("", "")]

        for ing_id in self.database_items:
            display = self._id_to_display[ing_id]
            pair = (display, ing_id)
            self.all_items.append(pair)
            try:
                roles = self.prop_db.find_ingredient(ing_id).get("roles", [])
                if "oxidizer" in roles:
                    self.oxidizer_items.append(pair)
                else:
                    self.fuel_items.append(pair)
            except Exception:
                self.fuel_items.append(pair)

        self.double_validator = QDoubleValidator()
        self.double_validator.setNotation(QDoubleValidator.StandardNotation)
        self.double_validator.setLocale(QLocale(QLocale.English, QLocale.UnitedStates))
        self.int_validator = QIntValidator(2, 10000, self)

        layout = QVBoxLayout(self)
        self.sim_tabs = QTabWidget()

        self.tab_solid = QWidget()
        self.tab_biprop = QWidget()
        self.sim_tabs.addTab(self.tab_biprop, "Bipropellant (O/F)")
        self.sim_tabs.addTab(self.tab_solid, "Solid / Monopropellant")

        self.setup_bipropellant_tab()
        self.setup_formulation_tab()
        self.sim_tabs.currentChanged.connect(
            lambda _idx: self._notify_formulation_changed()
        )

        layout.addWidget(self.sim_tabs)
        self._suspend_formulation_notifications = False

    def _notify_formulation_changed(self) -> None:
        """Clear stale outputs when the formulation inputs are edited."""
        if self._suspend_formulation_notifications:
            return
        engine_dock = getattr(self.main_window, "engine_dock", None)
        if engine_dock is not None and hasattr(engine_dock, "clear_previous_results"):
            engine_dock.clear_previous_results()

    def setup_bipropellant_tab(self):
        main_layout = QVBoxLayout(self.tab_biprop)

        # Build tables
        self.fuel_table, self.fuel_tot, self.fuel_density_lbl = (
            self._build_biprop_table(
                main_layout, "Fuel Mixture", self.add_fuel_row, self.normalize_fuel
            )
        )
        self.ox_table, self.ox_tot, self.ox_density_lbl = self._build_biprop_table(
            main_layout, "Oxidizer Mixture", self.add_ox_row, self.normalize_ox
        )

        # Start with one empty row each
        self.add_fuel_row("", "1.0")
        self.add_ox_row("", "1.0")

        # Engine Mixture Ratio (O/F)
        group_of = QGroupBox("Engine Mixture Ratio (O/F)")
        form_of = QFormLayout()
        self.of_mode_combo = QComboBox()
        self.of_mode_combo.addItems(["Single Value", "Sweep Range"])
        self.of_mode_combo.currentTextChanged.connect(self.handle_of_mode)

        self.input_of_ratio = QDoubleSpinBox()
        self.input_of_ratio.setLocale(QLocale(QLocale.English, QLocale.UnitedStates))
        self.input_of_ratio.setRange(0.0, 1000.0)
        self.input_of_ratio.setDecimals(3)
        self.input_of_ratio.setSingleStep(0.1)
        self.input_of_ratio.setValue(2.50)
        self.input_of_ratio.setButtonSymbols(QDoubleSpinBox.UpDownArrows)

        of_sweep_layout = QHBoxLayout()
        self.input_of_min = QLineEdit("1.00")
        self.input_of_max = QLineEdit("6.00")
        self.input_of_steps = QLineEdit("11")
        for w in (self.input_of_min, self.input_of_max):
            w.setValidator(self.double_validator)
            w.setAlignment(Qt.AlignRight)
            w.setEnabled(False)
        self.input_of_steps.setValidator(self.int_validator)
        self.input_of_steps.setAlignment(Qt.AlignRight)
        self.input_of_steps.setEnabled(False)
        of_sweep_layout.addWidget(QLabel("Min:"))
        of_sweep_layout.addWidget(self.input_of_min)
        of_sweep_layout.addWidget(QLabel("Max:"))
        of_sweep_layout.addWidget(self.input_of_max)
        of_sweep_layout.addWidget(QLabel("Steps:"))
        of_sweep_layout.addWidget(self.input_of_steps)

        self.lbl_actual_of = QLabel("Actual Mass O/F: N/A")
        self.lbl_actual_of.setStyleSheet("color: #aaa; font-style: italic;")
        self.lbl_combined_density = QLabel("Combined Density: N/A")
        self.lbl_combined_density.setStyleSheet("color: #aaa; font-style: italic;")
        self.input_of_ratio.valueChanged.connect(
            lambda _: self._update_biprop_combined_density()
        )
        self.input_of_ratio.valueChanged.connect(
            lambda _: self._notify_formulation_changed()
        )
        self.input_of_min.editingFinished.connect(self._notify_formulation_changed)
        self.input_of_max.editingFinished.connect(self._notify_formulation_changed)
        self.input_of_steps.editingFinished.connect(self._notify_formulation_changed)
        form_of.addRow("Analysis Mode:", self.of_mode_combo)
        form_of.addRow("Nominal O/F Ratio:", self.input_of_ratio)
        form_of.addRow("O/F Bounds:", of_sweep_layout)
        form_of.addRow("", self.lbl_actual_of)
        form_of.addRow("", self.lbl_combined_density)
        group_of.setLayout(form_of)
        main_layout.addWidget(group_of)

        main_layout.addStretch()

    def handle_of_mode(self, text):
        """Enable/disable O/F single value and sweep inputs."""
        is_sweep = text == "Sweep Range"
        self.input_of_ratio.setEnabled(not is_sweep)
        self.input_of_min.setEnabled(is_sweep)
        self.input_of_max.setEnabled(is_sweep)
        self.input_of_steps.setEnabled(is_sweep)

        # Only one sweep may be active at a time: if O/F sweep is enabled,
        # force Pc sweep back to single-value mode.
        if is_sweep and hasattr(self.main_window, "engine_dock"):
            engine = self.main_window.engine_dock
            if engine.pc_mode_combo.currentText() == "Sweep Range":
                engine.pc_mode_combo.blockSignals(True)
                engine.pc_mode_combo.setCurrentText("Single Value")
                engine.pc_mode_combo.blockSignals(False)
                engine.handle_pc_mode("Single Value")
        self._notify_formulation_changed()

    def _build_biprop_table(self, parent_layout, title, add_func, norm_func):
        group = QGroupBox(title)
        layout = QVBoxLayout()
        table = QTableWidget(0, 3)
        table.setHorizontalHeaderLabels(["Component", "Mass Frac", ""])
        table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        table.setColumnWidth(1, 100)
        table.setColumnWidth(2, 40)

        ctrl = QHBoxLayout()
        btn_add = QPushButton("+ Add")
        btn_add.clicked.connect(lambda: add_func())
        btn_norm = QPushButton("Norm to 1.0")
        btn_norm.clicked.connect(norm_func)
        tot_disp = QLineEdit("0.00")
        tot_disp.setReadOnly(True)
        tot_disp.setFixedWidth(60)
        density_lbl = QLabel("ρ: N/A")
        density_lbl.setStyleSheet("color: #aaa; font-style: italic;")

        ctrl.addWidget(btn_add)
        ctrl.addWidget(btn_norm)
        ctrl.addStretch()
        ctrl.addWidget(density_lbl)
        ctrl.addWidget(QLabel("Total:"))
        ctrl.addWidget(tot_disp)

        layout.addWidget(table)
        layout.addLayout(ctrl)
        group.setLayout(layout)
        parent_layout.addWidget(group)
        return table, tot_disp, density_lbl

    def _select_combo_ingredient(self, combo: QComboBox, ingredient_id: str) -> None:
        """Set *combo* to the entry whose userData matches *ingredient_id*.

        Falls back to ``setCurrentText`` if no userData match is found (e.g.
        user-typed text or an ID from an older saved composition file).
        """
        idx = combo.findData(ingredient_id)
        if idx >= 0:
            combo.setCurrentIndex(idx)
        else:
            combo.setCurrentText(ingredient_id)

    def _add_biprop_row(self, table, ingredient_id, wt, update_func, items=None):
        row = table.rowCount()
        table.insertRow(row)

        if items is None:
            items = self.all_items

        # Container for combo + search button
        container = QWidget()
        c_lay = QHBoxLayout(container)
        c_lay.setContentsMargins(2, 2, 2, 2)
        c_lay.setSpacing(2)

        combo = QComboBox()
        combo.setEditable(True)
        for display, ing_id in items:
            combo.addItem(display, ing_id)
        self._select_combo_ingredient(combo, ingredient_id)
        combo.currentTextChanged.connect(update_func)

        search_btn = QPushButton("🔍")
        search_btn.setFixedWidth(25)
        search_btn.setToolTip("Search database...")
        search_btn.clicked.connect(lambda: self.open_database_search(combo, items))

        c_lay.addWidget(combo)
        c_lay.addWidget(search_btn)
        table.setCellWidget(row, 0, container)
        # Store a reference to the combo box on the row for easier solving
        table.setProperty(f"combo_{row}", combo)

        w_inp = QLineEdit(wt)
        w_inp.setValidator(self.double_validator)
        w_inp.textChanged.connect(update_func)
        w_inp.setAlignment(Qt.AlignRight)
        table.setCellWidget(row, 1, w_inp)

        del_btn = QPushButton("X")
        del_btn.setStyleSheet("background-color: #6a2c2c; color: white; border: none;")
        del_btn.clicked.connect(
            lambda: self._del_biprop_row(table, del_btn, update_func)
        )
        table.setCellWidget(row, 2, del_btn)
        update_func()

    def open_database_search(self, target_combo, items=None):
        if items is None:
            items = self.all_items
        # Convert (display, id) pairs to the dict form DatabaseSearchDialog expects.
        # Skip the blank placeholder entry.
        search_items = self.prop_db.search_items()
        # Filter to the subset relevant to this combo (fuel/ox/all).
        if items is not self.all_items:
            allowed_ids = {ing_id for _, ing_id in items if ing_id}
            search_items = [s for s in search_items if s["id"] in allowed_ids]
        dialog = DatabaseSearchDialog(search_items, self)
        if dialog.exec():
            selected = dialog.get_selected_item()  # returns the ingredient ID
            if selected:
                self._select_combo_ingredient(target_combo, selected)

    def _del_biprop_row(self, table, btn, update_func):
        for r in range(table.rowCount()):
            if table.cellWidget(r, 2) == btn:
                table.removeRow(r)
                update_func()
                break

    def add_fuel_row(self, name="", wt="0.00"):
        self._add_biprop_row(
            self.fuel_table, name, wt, self.update_fuel_totals, items=self.fuel_items
        )

    def add_ox_row(self, name="", wt="0.00"):
        self._add_biprop_row(
            self.ox_table, name, wt, self.update_ox_totals, items=self.oxidizer_items
        )

    def _compute_mixture_density(self, table):
        """Return mixture density [kg/m³] via the rule of mixtures, or None.

        Uses the volumetric rule of mixtures: ρ = 1 / Σ(wᵢ/ρᵢ) where wᵢ are
        the normalised mass fractions and ρᵢ are the ingredient densities from
        the propellant database.  Returns None if any named ingredient is
        missing a density value or is not found in the database.
        """
        entries = []
        for r in range(table.rowCount()):
            container = table.cellWidget(r, 0)
            combo = container.findChild(QComboBox) if container else None
            w_widget = table.cellWidget(r, 1)
            if not combo or not w_widget:
                continue
            ing_id = (combo.currentData() or combo.currentText()).strip()
            try:
                wt = float(w_widget.text())
            except ValueError:
                continue
            if not ing_id or wt <= 0:
                continue
            try:
                ing = self.prop_db.find_ingredient(ing_id)
                rho = float(ing.get("density", 0))
            except KeyError:
                return None
            if rho <= 0:
                return None
            entries.append((wt, rho))
        if not entries:
            return None
        total_wt = sum(w for w, _ in entries)
        if total_wt <= 0:
            return None
        return total_wt / sum(w / rho for w, rho in entries)

    def _update_biprop_combined_density(self):
        """Recompute and display the combined propellant density at the current O/F."""
        if not hasattr(self, "input_of_ratio") or not hasattr(
            self, "lbl_combined_density"
        ):
            return
        rho_fuel = self._compute_mixture_density(self.fuel_table)
        rho_ox = self._compute_mixture_density(self.ox_table)
        of = self.input_of_ratio.value()
        if rho_fuel is not None and rho_ox is not None and of > 0:
            # Mass fractions: fuel = 1/(1+OF), ox = OF/(1+OF)
            wf = 1.0 / (1.0 + of)
            wo = of / (1.0 + of)
            rho_combined = 1.0 / (wf / rho_fuel + wo / rho_ox)
            self.lbl_combined_density.setText(
                f"Combined Density: {rho_combined / 1000:.3f} g/cc"
            )
        else:
            self.lbl_combined_density.setText("Combined Density: N/A")

    def _update_biprop_totals(self, table, tot_disp, density_lbl):
        total = 0.0
        for r in range(table.rowCount()):
            try:
                w_widget = table.cellWidget(r, 1)
                if w_widget:
                    total += float(w_widget.text())
            except:
                pass
        tot_disp.setText(f"{total:.3f}")
        rho = self._compute_mixture_density(table)
        density_lbl.setText(
            f"ρ: {rho / 1000:.3f} g/cc" if rho is not None else "ρ: N/A"
        )
        self._update_biprop_combined_density()
        # Notify engine dock to update O/F display
        if hasattr(self.main_window, "engine_dock"):
            self.main_window.engine_dock.update_actual_of()

    def update_fuel_totals(self):
        self._update_biprop_totals(
            self.fuel_table, self.fuel_tot, self.fuel_density_lbl
        )
        self._notify_formulation_changed()

    def update_ox_totals(self):
        self._update_biprop_totals(self.ox_table, self.ox_tot, self.ox_density_lbl)
        self._notify_formulation_changed()

    def normalize_fuel(self):
        self._norm_biprop(self.fuel_table, self.update_fuel_totals)

    def normalize_ox(self):
        self._norm_biprop(self.ox_table, self.update_ox_totals)

    def _norm_biprop(self, table, update_func):
        try:
            tot = sum(
                float(table.cellWidget(r, 1).text()) for r in range(table.rowCount())
            )
            if tot > 0:
                for r in range(table.rowCount()):
                    val = float(table.cellWidget(r, 1).text())
                    table.cellWidget(r, 1).setText(f"{val / tot:.3f}")
            update_func()
        except:
            pass

    def setup_formulation_tab(self):
        main_layout = QVBoxLayout(self.tab_solid)
        self.solid_table = QTableWidget(0, 3)
        self.solid_table.setHorizontalHeaderLabels(["Ingredient", "Mass Frac", ""])
        header = self.solid_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        self.solid_table.setColumnWidth(1, 100)
        self.solid_table.setColumnWidth(2, 40)

        main_layout.addWidget(QLabel("Propellant Ingredient List"))
        main_layout.addWidget(self.solid_table)

        ctrl_layout = QHBoxLayout()
        btn_add = QPushButton("+ Add Ingredient")
        btn_add.clicked.connect(lambda: self.add_solid_row())
        btn_norm = QPushButton("Normalize to 100%")
        btn_norm.clicked.connect(self.normalize_solid)
        btn_clear = QPushButton("Clear Table")
        btn_clear.clicked.connect(self.clear_solid)

        ctrl_layout.addWidget(btn_add)
        ctrl_layout.addWidget(btn_norm)
        ctrl_layout.addWidget(btn_clear)
        self.solid_total_display = QLineEdit("0.00")
        self.solid_total_display.setReadOnly(True)
        self.solid_total_display.setFixedWidth(80)
        self.solid_density_lbl = QLabel("ρ: N/A")
        self.solid_density_lbl.setStyleSheet("color: #aaa; font-style: italic;")
        ctrl_layout.addStretch()
        ctrl_layout.addWidget(self.solid_density_lbl)
        ctrl_layout.addWidget(QLabel("Total Wt:"))
        ctrl_layout.addWidget(self.solid_total_display)
        main_layout.addLayout(ctrl_layout)

        # Start empty
        self.add_solid_row("", "0.0")

    def add_solid_row(self, ingredient_id="", wt="0.00"):
        row = self.solid_table.rowCount()
        self.solid_table.insertRow(row)

        container = QWidget()
        c_lay = QHBoxLayout(container)
        c_lay.setContentsMargins(2, 2, 2, 2)
        c_lay.setSpacing(2)

        combo = QComboBox()
        combo.setEditable(True)
        for display, ing_id in self.all_items:
            combo.addItem(display, ing_id)
        self._select_combo_ingredient(combo, ingredient_id)
        combo.currentTextChanged.connect(self.update_solid_totals)

        search_btn = QPushButton("🔍")
        search_btn.setFixedWidth(25)
        search_btn.setToolTip("Search database...")
        search_btn.clicked.connect(
            lambda: self.open_database_search(combo, self.all_items)
        )

        c_lay.addWidget(combo)
        c_lay.addWidget(search_btn)
        self.solid_table.setCellWidget(row, 0, container)
        self.solid_table.setProperty(f"combo_{row}", combo)

        w_inp = QLineEdit(wt)
        w_inp.setValidator(self.double_validator)
        w_inp.textChanged.connect(self.update_solid_totals)
        w_inp.setAlignment(Qt.AlignRight)
        self.solid_table.setCellWidget(row, 1, w_inp)

        del_btn = QPushButton("X")
        del_btn.setStyleSheet(
            "background-color: #6a2c2c; color: white; border: none; font-weight: bold;"
        )
        del_btn.clicked.connect(lambda: self.delete_solid_row(del_btn))
        self.solid_table.setCellWidget(row, 2, del_btn)

        self.update_solid_totals()

    def delete_solid_row(self, btn):
        for row in range(self.solid_table.rowCount()):
            if self.solid_table.cellWidget(row, 2) == btn:
                self.solid_table.removeRow(row)
                self.update_solid_totals()
                break

    def update_solid_totals(self):
        total = 0.0
        for row in range(self.solid_table.rowCount()):
            w = self.solid_table.cellWidget(row, 1)
            try:
                total += float(w.text())
            except:
                pass
        self.solid_total_display.setText(f"{total:.2f}")
        rho = self._compute_mixture_density(self.solid_table)
        self.solid_density_lbl.setText(
            f"ρ: {rho / 1000:.3f} g/cc" if rho is not None else "ρ: N/A"
        )
        self._notify_formulation_changed()

    def normalize_solid(self):
        total = float(self.solid_total_display.text())
        if total > 0:
            for r in range(self.solid_table.rowCount()):
                w = self.solid_table.cellWidget(r, 1)
                try:
                    w.setText(f"{(float(w.text()) / total) * 100:.2f}")
                except:
                    pass

    def clear_solid(self):
        self.solid_table.setRowCount(0)
        self.update_solid_totals()

    def _table_rows(self, table):
        """Return non-empty (name, mass_fraction) rows from a composition table."""
        rows = []
        for r in range(table.rowCount()):
            container = table.cellWidget(r, 0)
            combo = container.findChild(QComboBox) if container else None
            w_widget = table.cellWidget(r, 1)
            ing_id = (
                (combo.currentData() or combo.currentText()).strip() if combo else ""
            )
            weight_text = w_widget.text().strip() if w_widget else "0"
            try:
                weight = float(weight_text)
            except ValueError:
                continue
            if not ing_id or weight <= 0.0:
                continue
            rows.append({"name": ing_id, "mass_fraction": weight})
        return rows

    def _load_rows(self, table, add_row_fn, rows):
        """Replace table contents with saved rows using the provided row-adder."""
        table.setRowCount(0)
        for row in rows:
            add_row_fn(str(row.get("name", "")), str(row.get("mass_fraction", 0.0)))

    def composition_snapshot(self):
        """Return a JSON-serializable snapshot of the currently visible setup."""
        if self.sim_tabs.currentIndex() == 0:
            return {
                "schema_version": 1,
                "propellant_type": "bipropellant",
                "components": {
                    "fuel": self._table_rows(self.fuel_table),
                    "oxidizer": self._table_rows(self.ox_table),
                },
                "of_ratio": {
                    "mode": self.of_mode_combo.currentText(),
                    "value": float(self.input_of_ratio.value()),
                    "sweep": {
                        "min": self.input_of_min.text().strip(),
                        "max": self.input_of_max.text().strip(),
                        "steps": self.input_of_steps.text().strip(),
                    },
                },
            }

        return {
            "schema_version": 1,
            "propellant_type": "solid",
            "components": self._table_rows(self.solid_table),
        }

    def current_mode_tab_index(self):
        """Return the current simulator tab index from active UI state."""
        return self.sim_tabs.currentIndex()

    def apply_composition_snapshot(self, payload):
        """Apply a composition snapshot loaded from disk.

        Args:
            payload: Parsed composition JSON object.

        Raises:
            ValueError: If payload is missing required keys or has invalid mode.
        """
        self._suspend_formulation_notifications = True
        try:
            if not isinstance(payload, dict):
                raise ValueError("Composition file must be a JSON object.")

            propellant_type = payload.get("propellant_type")
            if propellant_type not in ("bipropellant", "solid"):
                raise ValueError(
                    "Composition propellant_type must be 'bipropellant' or 'solid'."
                )

            if propellant_type == "bipropellant":
                components = payload.get("components")
                of_ratio = payload.get("of_ratio")

                if not isinstance(components, dict):
                    raise ValueError(
                        "Missing 'components' section for bipropellant file."
                    )

                self._load_rows(
                    self.fuel_table, self.add_fuel_row, components.get("fuel", [])
                )
                self._load_rows(
                    self.ox_table, self.add_ox_row, components.get("oxidizer", [])
                )
                if self.fuel_table.rowCount() == 0:
                    self.add_fuel_row("", "0.0")
                if self.ox_table.rowCount() == 0:
                    self.add_ox_row("", "0.0")

                sweep = of_ratio.get("sweep", {}) if isinstance(of_ratio, dict) else {}
                mode_text = "Single Value"
                ratio_value = 2.5
                of_min = "1.00"
                of_max = "6.00"
                of_steps = "11"
                if isinstance(of_ratio, dict):
                    mode_text = str(of_ratio.get("mode", "Single Value"))
                    ratio_value = float(of_ratio.get("value", 2.5))
                    of_min = str(sweep.get("min", "1.00"))
                    of_max = str(sweep.get("max", "6.00"))
                    of_steps = str(sweep.get("steps", "11"))

                self.of_mode_combo.setCurrentText(mode_text)
                self.input_of_ratio.setValue(ratio_value)
                self.input_of_min.setText(of_min)
                self.input_of_max.setText(of_max)
                self.input_of_steps.setText(of_steps)
                self.handle_of_mode(self.of_mode_combo.currentText())
                self.sim_tabs.setCurrentIndex(0)
                self.update_fuel_totals()
                self.update_ox_totals()
                return

            components = payload.get("components")
            if not isinstance(components, list):
                raise ValueError("Missing 'components' list for solid propellant file.")
            self._load_rows(self.solid_table, self.add_solid_row, components)
            if self.solid_table.rowCount() == 0:
                self.add_solid_row("", "0.0")
            self.sim_tabs.setCurrentIndex(1)
            self.update_solid_totals()
        finally:
            self._suspend_formulation_notifications = False
