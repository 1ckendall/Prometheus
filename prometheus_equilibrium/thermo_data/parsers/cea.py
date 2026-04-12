"""Parser for NASA CEA thermo.inp (NASA TP-2001-210959 format).

Produces the nasa9.json schema, identical to :class:`.burcat.Burcat9Parser`
output.  The three-line interval format is **shared** with the Burcat9 parser
via :func:`._common.parse_nasa9_interval` — the only differences from
``burcat9.thr`` are file-level framing:

* A ``thermo`` keyword and global T-bounds line precede the species data.
* Comment lines start with ``!``.
* The species section ends at ``END PRODUCTS``.

Only the PRODUCTS section is parsed; the REACTANTS section contains
fuel/oxidiser entries with no polynomial data.

Raw source file: ``cea_thermo.inp``
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

from ._common import (
    NASA9Parser,
    canonical_id,
    dedup_id,
    fw,
    hill_formula,
    normalise_element,
    parse_nasa9_descriptor,
    parse_nasa9_interval,
    to_float,
)

log = logging.getLogger(__name__)


class CEAParser(NASA9Parser):
    """Parse a CEA ``thermo.inp`` file into nasa9.json schema.

    Per-species structure (identical field layout to Burcat9):

    * **Line 1** — species name (cols 0–17), rest is reference/comment
    * **Line 2** — descriptor: n_intervals, elements, condensed flag
      (parsed by :func:`._common.parse_nasa9_descriptor`)
    * **Per interval** — 3 lines parsed by :func:`._common.parse_nasa9_interval`
    """

    def parse(self, path: str, source: str = "") -> Dict[str, dict]:
        """Parse *path* and return a dict keyed by canonical species ID.

        Args:
            path:   Path to the CEA thermo.inp file.
            source: Human-readable label for this data source (e.g. ``"CEA-NRL2002"``).
                    Stored in each record under the ``"source"`` key.
        """
        species_list, stats = self._parse_file(path, source=source)
        db = {sp["id"]: sp for sp in species_list}
        log.info("CEA: parsed %d species.", len(db))
        for key, val in stats.items():
            if val:
                log.info("  %-36s %d", key, val)
        return db

    # ------------------------------------------------------------------

    @staticmethod
    def _parse_file(path: str, source: str = "") -> Tuple[List[dict], Dict[str, int]]:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            lines = [ln.rstrip("\n") for ln in f]
        n = len(lines)

        # Skip file-level comments to the 'thermo' keyword.
        i = 0
        while i < n and not lines[i].strip().lower().startswith("thermo"):
            i += 1
        i += 2  # skip 'thermo' and global T-bounds line

        species: List[dict] = []
        used_ids: Dict[str, int] = {}
        stats: Dict[str, int] = {
            "parsed": 0,
            "skipped_pseudo_element": 0,
            "skipped_empty_elements": 0,
            "skipped_zero_intervals": 0,
            "skipped_parse_error": 0,
            "collisions_disambiguated": 0,
        }

        while i < n:
            line = lines[i]

            if line.startswith("END PRODUCTS"):
                break
            if line.startswith("!") or not line.strip():
                i += 1
                continue

            # Line 1: species name
            cea_name = fw(line, 0, 18).strip()

            # Line 2 (descriptor) — reuse shared parser
            i += 1
            if i >= n:
                break

            try:
                n_intervals, elements, phase = parse_nasa9_descriptor(
                    lines[i], cea_name
                )
            except ValueError:
                i += 1
                continue

            # CEA pseudo-element check: normalise_element returns None for
            # IO/IN/IC/IH.  Re-validate elements through normalise_element
            # so that any CEA-specific pseudo-elements are caught.
            clean_elements: Dict[str, float] = {}
            skip = False
            for raw_sym, count in list(elements.items()):
                if raw_sym == "e-":
                    clean_elements["e-"] = count
                    continue
                canon: Optional[str] = normalise_element(raw_sym)
                if canon is None:
                    skip = True
                    break
                clean_elements[canon] = count

            if skip:
                stats["skipped_pseudo_element"] += 1
                i += 1 + n_intervals * 3
                continue
            if not clean_elements:
                stats["skipped_empty_elements"] += 1
                i += 1 + n_intervals * 3
                continue

            if n_intervals == 0:
                stats["skipped_zero_intervals"] += 1
                i += 2
                continue

            # Intervals — identical layout to Burcat9, shared parser
            segments: List[dict] = []
            parse_ok = True
            base_i = i + 1

            for seg_idx in range(n_intervals):
                hi = base_i + seg_idx * 3
                if hi + 2 >= n:
                    parse_ok = False
                    break
                try:
                    seg = parse_nasa9_interval(lines[hi], lines[hi + 1], lines[hi + 2])
                except (ValueError, IndexError):
                    parse_ok = False
                    break
                segments.append(seg)

            i += 1 + n_intervals * 3

            if not parse_ok or not segments:
                stats["skipped_parse_error"] += 1
                continue

            sp_id = dedup_id(canonical_id(clean_elements, phase), used_ids)
            if used_ids.get(canonical_id(clean_elements, phase), 0) > 1:
                stats["collisions_disambiguated"] += 1
                log.debug("Polymorph: alias=%s -> id=%s", cea_name, sp_id)

            species.append(
                {
                    "id": sp_id,
                    "name": hill_formula(clean_elements),
                    "alias": cea_name,
                    "phase": phase,
                    "elements": clean_elements,
                    "format": "NASA-9",
                    "source": source,
                    "segments": segments,
                }
            )
            stats["parsed"] += 1

        return species, stats
