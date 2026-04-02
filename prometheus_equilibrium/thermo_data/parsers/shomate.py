"""Loader/validator for the hand-authored NIST Shomate database (shomate.json).

Unlike the other parsers, Shomate data has no machine-readable raw source
format — coefficients are looked up manually from the NIST WebBook
(https://webbook.nist.gov) and the F coefficient is computed by hand to
encode ΔfH°(298.15 K).  This module therefore acts as:

1. A **loader** that reads and validates an existing ``shomate.json``.
2. A **template helper** that prints a ready-to-paste JSON stub for a new
   species, to be filled in from NIST WebBook data.

Adding a new Shomate species
----------------------------
1. Look up A–E and G on the NIST Chemistry WebBook (Shomate equation tab).
2. Note ΔfH°(298.15 K) from JANAF or NIST-JANAF tables (kJ/mol).
3. Compute F so that ``H(298.15 K) = ΔfH°``::

       t = 0.29815          # T / 1000 at 298.15 K
       polynomial_sum = A*t + B*t²/2 + C*t³/3 + D*t⁴/4 - E/t
       F = delta_f_H298_kJ_mol - polynomial_sum

4. Verify Cp and S at 298.15 K against literature.
5. Paste the output of :meth:`ShomateParser.template` into ``shomate.json``.

shomate.json schema
-------------------
Keys starting with ``_`` (e.g. ``_schema``) are metadata and are skipped.
Each species entry::

    "<ID>": {
        "elements": {"Bi": 2, "O": 3},
        "phase":    "S",
        "alias":    "Bi2O3",          // optional
        "_note":    "...",             // optional verification note
        "segments": [
            {
                "t_low":        298.0,
                "t_high":       1097.0,
                "coefficients": [A, B, C, D, E, F, G, H]
            }
        ]
    }

``H`` (the 8th coefficient) is the NIST standard-enthalpy reference value
and is stored for documentation purposes; it is not used in any calculation.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from ._common import canonical_id, hill_formula

log = logging.getLogger(__name__)


class ShomateParser:
    """Load and validate a ``shomate.json`` file.

    This is a passthrough loader: the JSON is already in the compiled schema
    and requires no format translation.  The parser validates required fields
    and normalises optional ones.

    Usage::

        db = ShomateParser().parse("shomate.json")
        # db is a Dict[str, dict] ready for ThermoCompiler
    """

    # Required fields for each species record
    _REQUIRED = {"elements", "phase", "segments"}
    # Required fields for each segment
    _SEG_REQUIRED = {"t_low", "t_high", "coefficients"}

    def parse(self, path: str) -> Dict[str, dict]:
        """Load *path* and return a validated dict keyed by species ID.

        Entries whose keys start with ``_`` are skipped (metadata).
        Species that fail validation are logged and excluded.

        Returns:
            ``{sp_id: record}`` in the shomate.json schema.
        """
        with open(path, "r", encoding="utf-8") as f:
            raw: Dict[str, Any] = json.load(f)

        db: Dict[str, dict] = {}
        for sp_id, rec in raw.items():
            if sp_id.startswith("_"):
                continue
            errors = self._validate(sp_id, rec)
            if errors:
                for err in errors:
                    log.warning("Shomate %s: %s", sp_id, err)
                continue
            db[sp_id] = rec

        log.info("Shomate: loaded %d species from %s.", len(db), path)
        return db

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate(self, sp_id: str, rec: dict) -> List[str]:
        errors: List[str] = []

        missing = self._REQUIRED - rec.keys()
        if missing:
            errors.append(f"missing fields: {missing}")
            return errors  # can't go further

        segs = rec.get("segments", [])
        if not segs:
            errors.append("no segments")
        for k, seg in enumerate(segs):
            seg_missing = self._SEG_REQUIRED - seg.keys()
            if seg_missing:
                errors.append(f"segment {k}: missing {seg_missing}")
                continue
            coeffs = seg["coefficients"]
            if len(coeffs) not in (7, 8):
                errors.append(
                    f"segment {k}: expected 7 or 8 coefficients, got {len(coeffs)}"
                )

        return errors

    # ------------------------------------------------------------------
    # Template helper
    # ------------------------------------------------------------------

    @staticmethod
    def template(
        elements: Dict[str, float],
        phase: str,
        t_low: float = 298.0,
        t_high: float = 1000.0,
        alias: Optional[str] = None,
        delta_f_H298_kJ_mol: Optional[float] = None,
    ) -> str:
        """Return a JSON stub for a new Shomate species, ready to paste into shomate.json.

        Args:
            elements:              Element dict, e.g. ``{"Bi": 2, "O": 3}``.
            phase:                 Phase character: ``"G"``, ``"L"``, or ``"S"``.
            t_low:                 Lower temperature bound (K), default 298.0.
            t_high:                Upper temperature bound (K), default 1000.0.
            alias:                 Human-readable name (optional).
            delta_f_H298_kJ_mol:   ΔfH°(298.15 K) in kJ/mol (optional, for the note).

        Returns:
            Formatted JSON string suitable for copy-pasting into ``shomate.json``.

        Example::

            print(ShomateParser.template({"Al": 2, "O": 3}, "S", alias="Al2O3"))
        """
        sp_id = canonical_id(elements, phase)
        name = alias or hill_formula(elements)

        note_lines = [
            "Fill in A–G from NIST WebBook (Shomate equation tab).",
            "Compute F = delta_f_H298_kJ_mol - (A*t + B*t²/2 + C*t³/3 + D*t⁴/4 - E/t)  at t=0.29815.",
            "Verify Cp and S at 298.15 K against JANAF/NIST-JANAF.",
        ]
        if delta_f_H298_kJ_mol is not None:
            note_lines.append(f"Target: delta_f_H298 = {delta_f_H298_kJ_mol} kJ/mol")

        stub = {
            sp_id: {
                "elements": elements,
                "phase": phase,
                "alias": name,
                "_note": "  ".join(note_lines),
                "segments": [
                    {
                        "t_low": t_low,
                        "t_high": t_high,
                        "coefficients": ["A", "B", "C", "D", "E", "F", "G", "H"],
                    }
                ],
            }
        }
        return json.dumps(stub, indent=2, ensure_ascii=False)
