import re

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QVBoxLayout,
)


def _query_to_hill(query: str) -> str | None:
    """Try to parse *query* as a chemical formula and return its Hill-canonical form.

    Accepts standard notation (``Al2O3``) and explicit-1 notation (``Al1F3``).
    Returns ``None`` if the query contains characters that cannot be part of a
    formula (spaces, ``=``, digits at the start, etc.) so that plain text
    searches are never disrupted.

    Examples::

        _query_to_hill("Al2O3")   # "Al2O3"
        _query_to_hill("O3Al2")   # "Al2O3"
        _query_to_hill("Al1F3")   # "AlF3"
        _query_to_hill("F3Al1")   # "AlF3"
        _query_to_hill("AP")      # None  (single-letter run — treat as text)
        _query_to_hill("ammonium")# None
    """
    # Must start with an uppercase letter (element symbol) and contain only
    # element symbols (upper + optional lower) and digits.
    if not re.match(r'^[A-Z]', query):
        return None
    if re.search(r'[^A-Za-z0-9]', query):
        return None

    tokens = re.findall(r'([A-Z][a-z]?)(\d*)', query)
    if not tokens:
        return None

    counts: dict[str, int] = {}
    for sym, num in tokens:
        if not sym:
            continue
        counts[sym] = counts.get(sym, 0) + (int(num) if num else 1)

    if not counts:
        return None

    order: list[str] = []
    for sym in ("C", "H"):
        if sym in counts:
            order.append(sym)
    for sym in sorted(counts):
        if sym not in ("C", "H"):
            order.append(sym)

    # Omit count of 1, so Al1F3 and AlF3 both canonicalise to "AlF3"
    return "".join(
        sym if counts[sym] == 1 else f"{sym}{counts[sym]}" for sym in order
    ).lower()


class DatabaseSearchDialog(QDialog):
    """Search dialog for propellant ingredients or thermodynamic species.

    Accepts either a plain list of strings (species IDs — backward-compatible
    with the Species Explorer) or a list of dicts produced by
    :py:meth:`PropellantDatabase.search_items` (propellant ingredients).

    Dict form keys:
        id (str): The unique key returned by :py:meth:`get_selected_item`.
        display (str): Human-readable label shown in the list.
        search_text (str): Pre-lowercased, space-joined concatenation of all
            searchable fields (ID, name, CAS, aliases).
    """

    def __init__(self, items, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Search Database")
        self.resize(420, 500)

        layout = QVBoxLayout(self)

        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText(
            "Search by name, ID, CAS, alias, or formula (e.g. Al2O3, O3Al2)…"
        )
        self.search_input.textChanged.connect(self._filter)
        layout.addWidget(self.search_input)

        self.list_widget = QListWidget()
        for item in items:
            if isinstance(item, str):
                lw = QListWidgetItem(item)
                lw.setData(Qt.UserRole, item)           # id
                lw.setData(Qt.UserRole + 1, item.lower())  # search_text
            else:
                lw = QListWidgetItem(item["display"])
                lw.setData(Qt.UserRole, item["id"])
                lw.setData(Qt.UserRole + 1, item["search_text"])
            self.list_widget.addItem(lw)

        self.list_widget.itemDoubleClicked.connect(self.accept)
        layout.addWidget(self.list_widget)

        self.button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        layout.addWidget(self.button_box)

    def _filter(self, text):
        text_lower = text.lower()
        # Try to interpret the query as a formula and canonicalise it so that
        # order-independent matches work (e.g. "O3Al2" finds "Al2O3").
        canonical = _query_to_hill(text)
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            search_text = item.data(Qt.UserRole + 1)
            match = text_lower in search_text or (
                canonical is not None and canonical in search_text
            )
            item.setHidden(not match)

    def get_selected_item(self):
        """Return the ID (``Qt.UserRole`` data) of the selected list entry."""
        selected = self.list_widget.selectedItems()
        if selected:
            return selected[0].data(Qt.UserRole)
        return None
