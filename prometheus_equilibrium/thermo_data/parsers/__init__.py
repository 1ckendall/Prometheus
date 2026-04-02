"""Thermodynamic data parsers.

Each parser returns a dict (NASA-7/9) or list-of-rows (JANAF) ready for the
compiler to merge or write directly.

Typical use::

    from prometheus_equilibrium.thermo_data.parsers import Burcat7Parser, Burcat9Parser, CEAParser, JANAFParser

    db7  = Burcat7Parser().parse("raw/burcat7.thr")
    db9  = Burcat9Parser().parse("raw/burcat9.thr")
    cea  = CEAParser().parse("raw/cea_thermo.inp")
    rows = JANAFParser().parse("raw/JANAF.jnf")
"""

from ._common import NASA7Parser, NASA9Parser
from .burcat import Burcat7Parser, Burcat9Parser
from .cea import CEAParser
from .janaf import JANAFParser
from .shomate import ShomateParser
from .terra import TERRAParser

__all__ = [
    "NASA7Parser",
    "NASA9Parser",
    "Burcat7Parser",
    "Burcat9Parser",
    "CEAParser",
    "JANAFParser",
    "ShomateParser",
    "TERRAParser",
]
