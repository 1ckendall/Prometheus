from PySide6.QtWidgets import (QDialog, QDialogButtonBox, QLineEdit,
                               QListWidget, QVBoxLayout)


class DatabaseSearchDialog(QDialog):
    def __init__(self, database_items, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Search Thermodynamic Database")
        self.resize(350, 400)

        self.database_items = database_items

        layout = QVBoxLayout(self)
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Type to filter propellants...")
        self.search_input.textChanged.connect(self.filter_list)
        layout.addWidget(self.search_input)

        self.list_widget = QListWidget()
        self.list_widget.addItems(self.database_items)
        self.list_widget.itemDoubleClicked.connect(self.accept)
        layout.addWidget(self.list_widget)

        self.button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        layout.addWidget(self.button_box)

    def filter_list(self, text):
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            item.setHidden(text.lower() not in item.text().lower())

    def get_selected_item(self):
        selected = self.list_widget.selectedItems()
        if selected:
            return selected[0].text()
        return None
