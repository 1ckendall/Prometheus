"""Parsers for BURCAT NASA-7 and NASA-9 thermodynamic data files (.thr).

Both parsers produce dicts keyed by canonical Hill-order species ID
(``{HillFormula}_{Phase}``) in the same JSON schemas loaded by
``SpeciesDatabase._load_nasa7`` and ``SpeciesDatabase._load_nasa9``.

The NASA-9 three-line interval format is **identical** to the CEA format;
:func:`parse_nasa9_descriptor` and :func:`parse_nasa9_interval` from
``_common`` are shared with :mod:`.cea`.

Raw source files: ``burcat7.thr`` (NASA-7), ``burcat9.thr`` (NASA-9).
Maintained by A. Burcat and B. Ruscic —
http://garfield.chem.elte.hu/Burcat/burcat.html
"""

from __future__ import annotations

import logging
import re
from typing import Dict, List

from ._common import (
    VALID_ELEMENTS,
    NASA7Parser,
    NASA9Parser,
    canonical_id,
    dedup_id,
    fw,
    hill_formula,
    parse_nasa9_descriptor,
    parse_nasa9_interval,
    to_float,
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Non-ground-state species filter (Burcat9 only)
# ---------------------------------------------------------------------------

# The BURCAT database includes electronic excited states (e.g. "O2 singlet",
# "CO2 triplet") and cyclic structural isomers (e.g. "NO2 cyclo N(OO)") that
# have the same Hill formula and phase as their ground-state counterparts.
# If these appear before the true ground-state entry in the file — or if the
# ground state is absent from burcat9.thr and only present in nasa7 — they
# would take the primary canonical ID (e.g. O2_G) and shadow the correct data.
#
# Detection rules applied to the raw card-1 line:
#   1. Contains "singlet" or "triplet" as a whole word → electronic excited state.
#   2. Contains " cyclo " (space-delimited, i.e. used as a structural qualifier
#      after a molecular formula) → cyclic structural isomer of a normally
#      non-cyclic species.  This pattern distinguishes "NO2  cyclo N(OO)" from
#      "cyclopentadienyl" (where "cyclo" is part of the species name itself).
_NON_GROUND_STATE_RE = re.compile(
    r"\bsinglet\b|\btriplet\b|\s+cyclo\s",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Shared numeric helpers (NASA-7 only; NASA-9 uses _common)
# ---------------------------------------------------------------------------

_FLOAT_RE = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][+-]?\d+)?")


def _norm_num(s: str) -> str:
    s = s.replace("D", "E").replace("d", "e").replace(",", "")
    return re.sub(r"[Ee]\s*([+-]?)\s*(\d+)", r"E\1\2", s)


def _first_float(s: str) -> float:
    s = _norm_num(s)
    m = _FLOAT_RE.search(s)
    if not m:
        raise ValueError(f"no float found in {s!r}")
    return float(m.group(0))


def _floats_in(s: str) -> List[float]:
    return [float(m.group(0)) for m in _FLOAT_RE.finditer(_norm_num(s))]


def _coeff_row_5x15(line: str) -> List[float]:
    """Parse five 15-char coefficient fields (NASA-7 layout)."""
    out = []
    for a, b in [(0, 15), (15, 30), (30, 45), (45, 60), (60, 75)]:
        raw = line[a:b] if b <= len(line) else ""
        tok = raw.strip().replace(",", "").replace("D", "E").replace("d", "e")
        tok = re.sub(r"[Ee]\s*([+-]?)\s*(\d+)", r"E\1\2", tok)
        if not tok or tok.upper() == "N/A":
            out.append(0.0)
            continue
        try:
            out.append(float(tok))
        except ValueError:
            m = _FLOAT_RE.search(tok)
            out.append(float(m.group(0)) if m else 0.0)
    return out


# ---------------------------------------------------------------------------
# NASA-7 element field parser (Burcat-specific fixed-width formula)
# ---------------------------------------------------------------------------

_HDR_FORMULA = (24, 44)
_HDR_PHASE = (44, 45)
_HDR_TLOW = (48, 57)
_HDR_THIGH = (57, 66)
_HDR_TMID = (66, 75)


def _parse_nasa7_elements(formula20: str) -> Dict[str, float]:
    elems: Dict[str, float] = {}
    for i in range(0, 20, 5):
        unit = formula20[i : i + 5].strip()
        if not unit:
            continue
        j = 0
        while j < len(unit) and unit[j].isalpha():
            j += 1
        sym_raw = unit[:j]
        num_raw = unit[j:].strip().rstrip(".")
        if not sym_raw or not num_raw:
            continue
        sym = sym_raw[0].upper() + sym_raw[1:].lower()
        if sym == "E":
            sym = "e-"
        try:
            qty = (
                float(int(num_raw))
                if num_raw.lstrip("+-").isdigit()
                else float(num_raw)
            )
        except ValueError:
            continue
        if sym in VALID_ELEMENTS or sym == "e-":
            elems[sym] = elems.get(sym, 0.0) + qty
    return elems


def _scan_overflow_elements(header: str) -> Dict[str, float]:
    tail = header[75:]
    if not tail.strip():
        return {}
    toks = tail.replace(".", " ").split()
    out: Dict[str, float] = {}
    k = 0
    while k + 1 < len(toks):
        sym_raw, qty_raw = toks[k], toks[k + 1]
        sym = sym_raw[0].upper() + sym_raw[1:].lower() if sym_raw else ""
        if sym == "E":
            sym = "e-"
        if sym in VALID_ELEMENTS or sym == "e-":
            try:
                out[sym] = out.get(sym, 0.0) + float(int(qty_raw))
                k += 2
                continue
            except ValueError:
                pass
        k += 1
    return out


# ---------------------------------------------------------------------------
# Burcat7Parser
# ---------------------------------------------------------------------------


class Burcat7Parser(NASA7Parser):
    """Parse a BURCAT NASA-7 .thr file into nasa7.json schema.

    Each 4-line block encodes one species:

    * Line 1: name, 20-char formula field, phase char, temperature bounds
    * Line 2: high-T coefficients 1–5
    * Line 3: high-T coefficients 6–7, then low-T coefficients 1–3
    * Line 4: low-T coefficients 4–7
    """

    def parse(self, path: str, source: str = "") -> Dict[str, dict]:
        """Parse *path* and return a dict keyed by canonical species ID.

        Args:
            path:   Path to the burcat7.thr file.
            source: Human-readable label for this data source (e.g. ``"Burcat-2024"``).
                    Stored in each record under the ``"source"`` key so the origin
                    is visible in the compiled JSON and in conflict messages.

        Schema::

            {
              "id": "CO2_G",
              "name": "CO2",
              "alias": "<original label>",
              "phase": "G",
              "elements": {"C": 1.0, "O": 2.0},
              "format": "NASA-7",
              "source": "<source label>",
              "hf298_j_mol": null,
              "t_low": 200.0, "t_mid": 1000.0, "t_high": 6000.0,
              "coeffs": {"low": [...7], "high": [...7]}
            }
        """
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            lines = [ln.rstrip("\n") for ln in f]

        db: Dict[str, dict] = {}
        seen: Dict[str, int] = {}
        n_parsed = n_skipped = 0

        i = 0
        while i + 3 < len(lines):
            h1, h2, h3, h4 = lines[i], lines[i + 1], lines[i + 2], lines[i + 3]
            i += 4

            if not h1.strip():
                continue

            try:
                t_low = _first_float(fw(h1, *_HDR_TLOW))
                t_high = _first_float(fw(h1, *_HDR_THIGH))
                t_mid = _first_float(fw(h1, *_HDR_TMID))
            except ValueError:
                nums = _floats_in(h1)
                if len(nums) < 3:
                    n_skipped += 1
                    continue
                t_low, t_high, t_mid = nums[-3], nums[-2], nums[-1]

            elements = _parse_nasa7_elements(fw(h1, *_HDR_FORMULA))
            elements.update(_scan_overflow_elements(h1))
            if not elements:
                n_skipped += 1
                continue

            phase_raw = fw(h1, *_HDR_PHASE).strip()
            _PHASE_MAP = {"G": "G", "S": "S", "L": "L", "C": "L", "B": "L"}
            phase = _PHASE_MAP.get(phase_raw.upper(), "G")

            # Coefficient layout across lines 2-4:
            # h2: high[0..4]
            # h3: high[5], high[6], low[0..2]
            # h4: low[3..6], (unused col 60-75)
            try:
                row2 = _coeff_row_5x15(h2)
                row3 = _coeff_row_5x15(h3)
                row4 = _coeff_row_5x15(h4)
            except Exception:
                n_skipped += 1
                continue

            high_coeffs = row2[:5] + row3[:2]
            low_coeffs = row3[2:5] + row4[:4]

            if len(high_coeffs) != 7 or len(low_coeffs) != 7:
                n_skipped += 1
                continue

            alias = fw(h1, 0, 18).strip() or "UNKNOWN"
            sp_id = dedup_id(canonical_id(elements, phase), seen)

            db[sp_id] = {
                "id": sp_id,
                "name": hill_formula(elements),
                "alias": alias,
                "phase": phase,
                "elements": elements,
                "format": "NASA-7",
                "source": source,
                "hf298_j_mol": None,
                "t_low": t_low,
                "t_mid": t_mid,
                "t_high": t_high,
                "coeffs": {"low": low_coeffs, "high": high_coeffs},
            }
            n_parsed += 1

        log.info("Burcat7: parsed %d species, skipped %d blocks.", n_parsed, n_skipped)
        return db


# ---------------------------------------------------------------------------
# Burcat9Parser
# ---------------------------------------------------------------------------


class Burcat9Parser(NASA9Parser):
    """Parse a BURCAT NASA-9 .thr file into nasa9.json schema.

    The per-species structure mirrors the CEA format (card-2 + n × 3-line
    intervals).  The only file-level differences from CEA are:

    * No ``thermo`` header or ``END PRODUCTS`` terminator.
    * Blank lines separate species; non-letter lines are skipped.

    The interval parsing delegates entirely to :func:`parse_nasa9_interval`
    from ``_common``, which is also used by :class:`.cea.CEAParser`.
    """

    def parse(self, path: str, source: str = "") -> Dict[str, dict]:
        """Parse *path* and return a dict keyed by canonical species ID.

        Args:
            path:   Path to the burcat9.thr file.
            source: Human-readable label for this data source (e.g. ``"Burcat-2024"``).
                    Stored in each record under the ``"source"`` key.

        Schema identical to :class:`CEAParser` output::

            {
              "id": "CO2_G",
              "name": "CO2",
              "alias": "<card-1 label>",
              "phase": "G",
              "elements": {...},
              "format": "NASA-9",
              "source": "<source label>",
              "segments": [...]
            }
        """
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            lines = [ln.rstrip("\n") for ln in f]

        db: Dict[str, dict] = {}
        seen: Dict[str, int] = {}
        n_parsed = n_skipped = 0
        i, n = 0, len(lines)

        while i < n:
            # Skip blank lines
            while i < n and not lines[i].strip():
                i += 1
            if i >= n:
                break
            # Skip lines that contain no letters (numeric separators)
            while i < n and not re.search(r"[a-zA-Z]", lines[i]):
                i += 1
            if i >= n:
                break

            # Card 1: alias / name
            raw_card1 = lines[i]
            alias = re.sub(r"[^A-Za-z0-9()+\-\.]", "", raw_card1[:15]) or "Unnamed"
            i += 1

            # Card 2: n_intervals, elements, phase
            if i >= n:
                break
            try:
                n_intervals, elements, phase = parse_nasa9_descriptor(lines[i], alias)
            except ValueError:
                i += 1
                n_skipped += 1
                continue
            i += 1

            if not elements:
                i += n_intervals * 3
                n_skipped += 1
                continue

            # Skip electronic excited states and cyclic structural isomers that
            # would shadow the ground-state entry under the same canonical ID.
            if _NON_GROUND_STATE_RE.search(raw_card1):
                i += n_intervals * 3
                n_skipped += 1
                log.debug("Skipping non-ground-state species: %s", alias)
                continue

            # Intervals (shared logic with CEAParser)
            segments: List[dict] = []
            parse_ok = True

            for _ in range(n_intervals):
                if i + 2 >= n:
                    parse_ok = False
                    break
                try:
                    seg = parse_nasa9_interval(lines[i], lines[i + 1], lines[i + 2])
                except (ValueError, IndexError) as exc:
                    log.debug("Bad interval for %s: %s", alias, exc)
                    parse_ok = False
                    break
                segments.append(seg)
                i += 3

            if not parse_ok or not segments:
                n_skipped += 1
                log.debug("Skipping %s: no valid segments.", alias)
                continue

            sp_id = dedup_id(canonical_id(elements, phase), seen)
            db[sp_id] = {
                "id": sp_id,
                "name": hill_formula(elements),
                "alias": alias,
                "phase": phase,
                "elements": elements,
                "format": "NASA-9",
                "source": source,
                "segments": segments,
            }
            n_parsed += 1

        log.info("Burcat9: parsed %d species, skipped %d.", n_parsed, n_skipped)
        return db
