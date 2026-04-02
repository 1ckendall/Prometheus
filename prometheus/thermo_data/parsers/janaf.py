"""Parser for JANAF thermochemical table data (.jnf format).

Produces rows in the janaf.csv schema used by ``SpeciesDatabase._load_janaf``.

Raw source file: ``JANAF.jnf``

File format
-----------
Each species block begins with a header line::

    {HillFormula} {phase_code}

followed immediately by a fixed column-header line, then one CSV data row
per temperature point::

    T,Cp,S,[G-H(Tr)]/T,H-H(Tr),Delta_fH,Delta_fG,log(Kf)
    0.0,0.0,0.0,inf,-6.197,904.858,,
    100.0,20.786,...

The next species block starts at the next line that begins with an uppercase
letter and does not start with ``T,``.
"""

from __future__ import annotations

import json
import logging
from typing import Dict, Iterator, List, Tuple

from ._common import canonical_id, hill_formula, parse_janaf_header

log = logging.getLogger(__name__)

# Column header that appears after every species header line
_COL_HEADER = "T,Cp,S,"

# CSV column names for the output file — must match SpeciesDatabase._load_janaf
JANAF_COLUMNS = [
    "id",
    "elements",
    "phase",
    "T_K",
    "Cp_J_molK",
    "S_J_molK",
    "GH_T_J_molK",
    "H_H298_kJ_mol",
    "dHf_kJ_mol",
    "dGf_kJ_mol",
    "log_Kf",
]


class JANAFParser:
    """Parse a JANAF.jnf file into janaf.csv rows.

    Usage::

        rows = JANAFParser().parse("raw/JANAF.jnf")
        # rows is a list of lists; first row is the header

    The first row of the return value is always ``JANAF_COLUMNS``.
    """

    def parse(self, path: str, source: str = "") -> List[List[str]]:
        """Parse *path* and return all rows (header + data) as string lists.

        Each data row has 11 fields matching :data:`JANAF_COLUMNS`.  Empty
        fields are represented as empty strings.

        Args:
            path:   Path to the JANAF.jnf file.
            source: Human-readable label for this data source (e.g. ``"JANAF-4th-Ed"``).
                    Logged at INFO level; not embedded in CSV rows (schema is fixed).

        Returns:
            List of rows; first row is the column-header list.
        """
        if source:
            log.info("JANAF source: %s", source)
        rows: List[List[str]] = [JANAF_COLUMNS]
        n_species = n_rows = 0

        for elements, phase, data_lines in self._iter_species(path):
            if not elements:
                continue

            sp_id = canonical_id(elements, phase)
            elem_str = json.dumps(elements, ensure_ascii=False)

            for line in data_lines:
                fields = [f.strip() for f in line.split(",")]
                if not fields or not fields[0]:
                    continue
                try:
                    float(fields[0])  # first field must be a temperature
                except ValueError:
                    continue

                # Pad / trim to exactly 8 data columns
                data = fields[:8] + [""] * max(0, 8 - len(fields))
                rows.append([sp_id, elem_str, phase] + data)
                n_rows += 1

            n_species += 1

        log.info("JANAF: parsed %d species, %d data rows.", n_species, n_rows)
        return rows

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _iter_species(
        path: str,
    ) -> Iterator[Tuple[Dict[str, float], str, List[str]]]:
        """Yield (elements, phase, data_lines) for each species block."""
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()

        current_elements: Dict[str, float] = {}
        current_phase = "G"
        current_data: List[str] = []
        in_species = False

        for raw in lines:
            line = raw.rstrip("\n")

            # Detect a new species header: starts with uppercase letter,
            # is NOT the column-header row, and contains a phase code token.
            first_char = line[:1]
            if first_char.isupper() and not line.startswith(_COL_HEADER):
                # Flush previous species
                if in_species and current_data:
                    yield current_elements, current_phase, current_data

                try:
                    elements, phase = parse_janaf_header(line)
                except Exception as exc:
                    log.debug("Cannot parse JANAF header %r: %s", line, exc)
                    in_species = False
                    current_data = []
                    continue

                current_elements = elements
                current_phase = phase
                current_data = []
                in_species = True
                continue

            # Skip the fixed column-header row
            if line.startswith(_COL_HEADER):
                continue

            # Data row
            if in_species and line.strip():
                current_data.append(line)

        # Flush last species
        if in_species and current_data:
            yield current_elements, current_phase, current_data
