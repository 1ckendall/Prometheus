import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from prometheus_equilibrium.gui.dialogs.add_propellant import AddPropellantDialog
from prometheus_equilibrium.gui.dialogs.database_search import DatabaseSearchDialog
from prometheus_equilibrium.gui.widgets.graph_canvas import GraphCanvas


class LibraryPage(QWidget):
    def __init__(self, main_window, prop_db, spec_db):
        super().__init__()
        self.main_window = main_window
        self.prop_db = prop_db
        self.spec_db = spec_db

        layout = QVBoxLayout(self)
        self.library_tabs = QTabWidget()

        self.tab_database = QWidget()
        self.tab_species = QWidget()
        self.library_tabs.addTab(self.tab_database, "Propellant Database")
        self.library_tabs.addTab(self.tab_species, "Species Explorer")

        self.setup_database_tab()
        self.setup_species_tab()

        layout.addWidget(self.library_tabs)

    def setup_database_tab(self):
        layout = QVBoxLayout(self.tab_database)

        # Filter layout
        filter_lay = QHBoxLayout()
        filter_lay.addWidget(QLabel("Filter:"))
        self.db_filter = QLineEdit()
        self.db_filter.setPlaceholderText("Search ingredients...")
        self.db_filter.textChanged.connect(self.filter_db_table)
        filter_lay.addWidget(self.db_filter)
        layout.addLayout(filter_lay)

        self.db_table = QTableWidget(0, 8)
        self.db_table.setHorizontalHeaderLabels(
            [
                "Name",
                "ID",
                "CAS",
                "Aliases",
                "Roles",
                "Density (g/cc)",
                "dHf298 (J/mol)",
                "Source",
            ]
        )
        self.db_table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.db_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.db_table)

        btn = QPushButton("Add New Propellant Ingredient")
        btn.clicked.connect(self.open_add_propellant_dialog)
        layout.addWidget(btn)

        # Populate table
        for prop_id in self.prop_db.ingredient_ids:
            p_data = self.prop_db.find_ingredient(prop_id)
            row_idx = self.db_table.rowCount()
            self.db_table.insertRow(row_idx)

            density_kgm3 = p_data.get("density", 0.0)
            density_str = (
                f"{float(density_kgm3) / 1000:.3f}"
                if density_kgm3 and float(density_kgm3) > 0
                else "—"
            )
            dhf298 = p_data.get("dHf298", "")
            dhf298_str = f"{float(dhf298):.1f}" if dhf298 != "" else "—"
            roles = ", ".join(p_data.get("roles", []))
            cas = p_data.get("cas", "")
            aliases = ", ".join(p_data.get("aliases", []))
            name = p_data.get("name", prop_id)
            source = p_data.get("source", "")

            # Store all searchable text in column 0 tooltip so filter_db_table
            # can search hidden metadata without extra hidden columns.
            cols = [name, prop_id, cas, aliases, roles, density_str, dhf298_str, source]
            for col_idx, text in enumerate(cols):
                item = QTableWidgetItem(str(text))
                item.setFlags(item.flags() ^ Qt.ItemIsEditable)
                self.db_table.setItem(row_idx, col_idx, item)

    def filter_db_table(self, text):
        for r in range(self.db_table.rowCount()):
            match = False
            for c in range(self.db_table.columnCount()):
                item = self.db_table.item(r, c)
                if item and text.lower() in item.text().lower():
                    match = True
                    break
            self.db_table.setRowHidden(r, not match)

    def open_add_propellant_dialog(self):
        dialog = AddPropellantDialog(self)
        if dialog.exec():
            data = dialog.get_data()
            row_idx = self.db_table.rowCount()
            self.db_table.insertRow(row_idx)
            cols = [
                data["id"],
                data["name"],
                data["phase"],
                data["roles"],
                data["density"],
                data["dhf298"],
            ]
            for col_idx, text in enumerate(cols):
                self.db_table.setItem(row_idx, col_idx, QTableWidgetItem(text))
            self.main_window.statusBar().showMessage(
                f"Added {data['id']} to database (Memory Only)", 3000
            )

    def setup_species_tab(self):
        layout = QHBoxLayout(self.tab_species)
        left = QVBoxLayout()
        control_group = QGroupBox("Species Selection")
        c_lay = QFormLayout()

        self.species_combo = QComboBox()
        self.species_combo.setEditable(True)

        # Connect automatic plot update
        self.species_combo.currentTextChanged.connect(self.update_species_plots)

        # Search button
        search_btn = QPushButton("🔍")
        search_btn.setFixedWidth(30)
        search_btn.setToolTip("Search species database...")
        search_btn.clicked.connect(self.open_species_search)

        # Combo + Search layout
        species_search_layout = QHBoxLayout()
        species_search_layout.addWidget(self.species_combo)
        species_search_layout.addWidget(search_btn)

        c_lay.addRow("Species:", species_search_layout)

        self.lbl_species_count = QLabel("Available: 0")
        self.lbl_species_count.setStyleSheet("color: #aaaaaa; font-style: italic;")
        c_lay.addRow("", self.lbl_species_count)

        control_group.setLayout(c_lay)
        left.addWidget(control_group)

        prop_group = QGroupBox("Key Properties")
        prop_layout = QFormLayout()
        self.lbl_elements = QLabel("N/A")
        self.lbl_state = QLabel("N/A")
        self.lbl_molar_mass = QLabel("N/A")
        self.lbl_data_source = QLabel("N/A")

        prop_layout.addRow("Elements:", self.lbl_elements)
        prop_layout.addRow("State (S/L/G):", self.lbl_state)
        prop_layout.addRow("Molar Mass (kg/mol):", self.lbl_molar_mass)
        prop_layout.addRow("Data Source:", self.lbl_data_source)
        prop_group.setLayout(prop_layout)
        left.addWidget(prop_group)

        left.addStretch()
        layout.addLayout(left, 1)

        graph_panel = QGridLayout()
        self.canvas_cp = GraphCanvas(self, "Cp vs T", "T (K)", "J/mol·K")
        self.canvas_h = GraphCanvas(self, "H vs T", "T (K)", "J/mol")
        self.canvas_s = GraphCanvas(self, "S vs T", "T (K)", "J/mol·K")
        self.canvas_g = GraphCanvas(self, "G vs T", "T (K)", "J/mol")

        graph_panel.addWidget(self.canvas_cp, 0, 0)
        graph_panel.addWidget(self.canvas_h, 0, 1)
        graph_panel.addWidget(self.canvas_s, 1, 0)
        graph_panel.addWidget(self.canvas_g, 1, 1)
        layout.addLayout(graph_panel, 3)

        # Populate and plot default
        self.refresh_species_list()
        if self.species_combo.count() > 0:
            self.species_combo.setCurrentText("H2O_G")  # default target
            self.update_species_plots()

    def refresh_species_list(self):
        """Update species combo box based on enabled databases in EngineDock."""
        enabled_dbs = []
        if hasattr(self.main_window, "engine_dock"):
            enabled_dbs = self.main_window.engine_dock.get_enabled_databases()

        # Filter all species by enabled databases
        filtered_ids = []
        # If no databases are enabled, filtered_ids stays empty.
        if enabled_dbs:
            for sp_id, sp in self.spec_db.species.items():
                if sp.source in enabled_dbs:
                    filtered_ids.append(sp_id)

        filtered_ids.sort()

        # Update combo box without triggering plots multiple times
        self.species_combo.blockSignals(True)
        current = self.species_combo.currentText()
        self.species_combo.clear()
        self.species_combo.addItems(filtered_ids)

        # If the previously selected species is still available, keep it
        if current in filtered_ids:
            self.species_combo.setCurrentText(current)
        else:
            self.species_combo.setCurrentText("")

        self.species_combo.blockSignals(False)

        # Update count label
        if hasattr(self, "lbl_species_count"):
            self.lbl_species_count.setText(f"Available: {len(filtered_ids)}")

        # Manually trigger plot update to handle clear state if needed
        self.update_species_plots()

    def open_species_search(self):
        # Get current items from combo
        items = [
            self.species_combo.itemText(i) for i in range(self.species_combo.count())
        ]
        dialog = DatabaseSearchDialog(items, self)
        if dialog.exec():
            selected = dialog.get_selected_item()
            if selected:
                self.species_combo.setCurrentText(selected)

    def update_species_plots(self):
        sp_id = self.species_combo.currentText()
        if not sp_id or sp_id not in self.spec_db.species:
            # Clear labels
            self.lbl_elements.setText("N/A")
            self.lbl_state.setText("N/A")
            self.lbl_molar_mass.setText("N/A")
            self.lbl_data_source.setText("N/A")
            # Clear plots
            for canvas in [self.canvas_cp, self.canvas_h, self.canvas_s, self.canvas_g]:
                canvas.axes.clear()
                canvas.axes.set_title(canvas.title_text, color="white")
                canvas.axes.set_xlabel(canvas.xlabel_text, color="white")
                canvas.axes.set_ylabel(canvas.ylabel_text, color="white")
                canvas.axes.grid(True, linestyle="--", alpha=0.3)
                canvas.axes.tick_params(colors="white")
                canvas.draw()
            return

        sp = self.spec_db.species[sp_id]

        # Update labels
        self.lbl_elements.setText(sp.formula)
        self.lbl_state.setText(sp.state)
        self.lbl_molar_mass.setText(f"{sp.molar_mass():.4f}")
        self.lbl_data_source.setText(sp.source)

        # Determine temperature range to plot
        t_min = 298.15
        t_max = 5000.0

        if hasattr(sp, "temperatures"):
            t_min = min(sp.temperatures)
            t_max = max(sp.temperatures)
        elif hasattr(sp, "T_low"):
            t_min = sp.T_low
            t_max = sp.T_high

        T = np.linspace(t_min, t_max, 500)

        # Calculate properties
        Cp = sp.specific_heat_capacity(T)
        H = sp.enthalpy(T)
        S = sp.entropy(T)
        G = sp.gibbs_free_energy(T)

        # Plot
        self._plot_on_canvas(self.canvas_cp, T, Cp)
        self._plot_on_canvas(self.canvas_h, T, H)
        self._plot_on_canvas(self.canvas_s, T, S)
        self._plot_on_canvas(self.canvas_g, T, G)

    def _plot_on_canvas(self, canvas, X, Y):
        canvas.axes.clear()

        # Restore titles and styling from the canvas attributes
        canvas.axes.set_title(canvas.title_text, color="white")
        canvas.axes.set_xlabel(canvas.xlabel_text, color="white")
        canvas.axes.set_ylabel(canvas.ylabel_text, color="white")
        canvas.axes.grid(True, linestyle="--", alpha=0.3)
        canvas.axes.tick_params(colors="white")

        canvas.axes.plot(X, Y, color="#2a82da")
        canvas.draw()
