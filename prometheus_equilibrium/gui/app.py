import sys
from pathlib import Path

import matplotlib

matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
from loguru import logger
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication

from prometheus_equilibrium.equilibrium.species import SpeciesDatabase
from prometheus_equilibrium.gui.main_window import ProPepUI
from prometheus_equilibrium.propellants import PropellantDatabase


def apply_modern_style(app: QApplication):
    app.setStyle("Fusion")
    dark_palette = QPalette()
    dark_palette.setColor(QPalette.Window, QColor(40, 40, 40))
    dark_palette.setColor(QPalette.WindowText, Qt.white)
    dark_palette.setColor(QPalette.Base, QColor(25, 25, 25))
    dark_palette.setColor(QPalette.AlternateBase, QColor(40, 40, 40))
    dark_palette.setColor(QPalette.ToolTipBase, Qt.white)
    dark_palette.setColor(QPalette.ToolTipText, Qt.white)
    dark_palette.setColor(QPalette.Text, Qt.white)
    dark_palette.setColor(QPalette.Button, QColor(53, 53, 53))
    dark_palette.setColor(QPalette.ButtonText, Qt.white)
    dark_palette.setColor(QPalette.BrightText, Qt.red)
    dark_palette.setColor(QPalette.Link, QColor(42, 130, 218))
    dark_palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
    dark_palette.setColor(QPalette.HighlightedText, Qt.black)
    app.setPalette(dark_palette)


def main():
    # Configure Loguru: WARNING by default; set PROMETHEUS_LOG_LEVEL to override.
    logger.remove()
    logger.add(
        sys.stderr,
        level=__import__("os").environ.get("PROMETHEUS_LOG_LEVEL", "WARNING"),
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    )
    logger.info("Starting Prometheus GUI...")

    app = QApplication(sys.argv)
    apply_modern_style(app)

    # Apply dark theme to all matplotlib plots to match UI
    plt.style.use("dark_background")

    # Resolve data paths relative to this file so the GUI works regardless of
    # the working directory and on case-sensitive filesystems.
    _pkg = Path(__file__).resolve().parent.parent
    _thermo = _pkg / "thermo_data"
    _propellants = _pkg / "propellants"

    print("Loading Species Database...")
    spec_db = SpeciesDatabase(
        nasa7_path=str(_thermo / "nasa7.json"),
        nasa9_path=str(_thermo / "nasa9.json"),
        janaf_path=str(_thermo / "janaf.csv"),
        afcesic_path=str(_thermo / "afcesic.json"),
        terra_path=str(_thermo / "terra.json"),
    )
    # Standard load settings: JANAF excluded for stability (see tests/benchmark.py).
    spec_db.load(
        include_nasa7=True,
        include_nasa9=True,
        include_afcesic=True,
        include_terra=True,
        include_janaf=False,
    )

    print("Loading Propellant Database...")
    prop_db = PropellantDatabase(
        str(_propellants / "propellants.toml"), species_db=spec_db
    )
    prop_db.load()

    window = ProPepUI(prop_db, spec_db)
    window.show()
    return app.exec()
