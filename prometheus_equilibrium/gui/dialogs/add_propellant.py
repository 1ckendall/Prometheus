from PySide6.QtGui import QDoubleValidator
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLineEdit,
    QTextEdit,
)


class AddPropellantDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Add New Propellant Ingredient")
        self.resize(450, 550)

        layout = QFormLayout(self)
        double_val = QDoubleValidator()
        double_val.setNotation(QDoubleValidator.StandardNotation)

        self.inp_id = QLineEdit()
        self.inp_name = QLineEdit()
        self.inp_cas = QLineEdit()
        self.inp_phase = QComboBox()
        self.inp_phase.addItems(["S", "L", "G"])
        self.inp_roles = QLineEdit()
        self.inp_roles.setPlaceholderText('e.g., "fuel", "binder"')
        self.inp_thermo_id = QLineEdit()
        self.inp_thermo_id.setPlaceholderText("e.g., H4ClNO4_C")
        self.inp_elements = QLineEdit()
        self.inp_elements.setPlaceholderText("e.g., {C = 4, H = 6.074, O = 0.074}")
        self.inp_molar_mass = QLineEdit()
        self.inp_molar_mass.setValidator(double_val)
        self.inp_dhf298 = QLineEdit()
        self.inp_dhf298.setValidator(double_val)
        self.inp_cp = QLineEdit()
        self.inp_cp.setValidator(double_val)
        self.inp_density = QLineEdit()
        self.inp_density.setValidator(double_val)
        self.inp_tsupply = QLineEdit("298.15")
        self.inp_tsupply.setValidator(double_val)
        self.inp_notes = QTextEdit()
        self.inp_notes.setMaximumHeight(60)

        layout.addRow("ID:", self.inp_id)
        layout.addRow("Name:", self.inp_name)
        layout.addRow("CAS:", self.inp_cas)
        layout.addRow("Phase:", self.inp_phase)
        layout.addRow("Roles (comma separated):", self.inp_roles)
        layout.addRow("Thermo ID (optional):", self.inp_thermo_id)
        layout.addRow("Elements (optional):", self.inp_elements)
        layout.addRow("Molar Mass [g/mol] (optional):", self.inp_molar_mass)
        layout.addRow("dHf298 [J/mol] (optional):", self.inp_dhf298)
        layout.addRow("Cp [J/mol/K] (optional):", self.inp_cp)
        layout.addRow("Density [kg/m³]:", self.inp_density)
        layout.addRow("T Supply [K]:", self.inp_tsupply)
        layout.addRow("Notes:", self.inp_notes)

        self.button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        layout.addRow(self.button_box)

    def get_data(self):
        return {
            "id": self.inp_id.text(),
            "name": self.inp_name.text(),
            "phase": self.inp_phase.currentText(),
            "roles": self.inp_roles.text(),
            "density": self.inp_density.text(),
            "dhf298": self.inp_dhf298.text(),
        }
