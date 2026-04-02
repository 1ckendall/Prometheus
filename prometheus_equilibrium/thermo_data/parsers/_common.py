"""Shared utilities used by all thermodynamic data parsers.

Abstract base classes
---------------------
:class:`NASA7Parser`
    Any parser that produces a ``nasa7.json``-schema dict.
    Currently: :class:`.burcat.Burcat7Parser`.

:class:`NASA9Parser`
    Any parser that produces a ``nasa9.json``-schema dict.
    Currently: :class:`.burcat.Burcat9Parser`, :class:`.cea.CEAParser`.
    The three-line interval format (:func:`parse_nasa9_interval`) is identical
    across all NASA-9 sources; only the file-level framing differs.

Both ABCs share a ``parse(path, source="") -> Dict[str, dict]`` contract so
:class:`.compiler.ThermoCompiler` can accept any conforming parser without
knowing the concrete class.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Re-exported symbols so callers only need to import from _common
# ---------------------------------------------------------------------------
__all__ = [
    # ABCs
    "NASA7Parser",
    "NASA9Parser",
    # helpers
    "fw",
    "to_float",
    "VALID_ELEMENTS",
    "normalise_element",
    "hill_formula",
    "canonical_id",
    "dedup_id",
    "phase_from_cond_flag",
    "parse_janaf_header",
    "parse_nasa9_elements",
    "rescue_nasa9_elements",
    "parse_nasa9_descriptor",
    "parse_nasa9_interval",
]


# ---------------------------------------------------------------------------
# Abstract base classes
# ---------------------------------------------------------------------------


class NASA7Parser(ABC):
    """Base class for all NASA-7 polynomial parsers.

    Subclasses must implement :meth:`parse`, which reads a source file and
    returns a ``nasa7.json``-schema dict::

        {
          "<ID>": {
            "id": "CO2_G",
            "name": "CO2",
            "alias": "...",
            "phase": "G",
            "elements": {"C": 1.0, "O": 2.0},
            "format": "NASA-7",
            "source": "<label>",
            "hf298_j_mol": null,
            "t_low": 200.0, "t_mid": 1000.0, "t_high": 6000.0,
            "coeffs": {"low": [...7], "high": [...7]}
          }
        }
    """

    @abstractmethod
    def parse(self, path: str, source: str = "") -> Dict[str, dict]:
        """Parse *path* and return a dict keyed by canonical species ID.

        Args:
            path:   Path to the source file.
            source: Human-readable label embedded in every record under ``"source"``.
        """


class NASA9Parser(ABC):
    """Base class for all NASA-9 polynomial parsers.

    All NASA-9 source formats share the same three-line interval layout
    (see :func:`parse_nasa9_interval`); only file-level framing differs.

    Subclasses must implement :meth:`parse`, which returns a
    ``nasa9.json``-schema dict::

        {
          "<ID>": {
            "id": "CO2_G",
            "name": "CO2",
            "alias": "...",
            "phase": "G",
            "elements": {"C": 1.0, "O": 2.0},
            "format": "NASA-9",
            "source": "<label>",
            "segments": [
              {"t_low": ..., "t_high": ...,
               "exponents": [...7], "coeffs": [...7],
               "b1": ..., "b2": ...},
              ...
            ]
          }
        }
    """

    @abstractmethod
    def parse(self, path: str, source: str = "") -> Dict[str, dict]:
        """Parse *path* and return a dict keyed by canonical species ID.

        Args:
            path:   Path to the source file.
            source: Human-readable label embedded in every record under ``"source"``.
        """


# ---------------------------------------------------------------------------
# Low-level text helpers
# ---------------------------------------------------------------------------


def fw(line: str, a: int, b: int) -> str:
    """Return fixed-width slice line[a:b], or '' when the line is too short."""
    return line[a:b] if a < len(line) else ""


def to_float(s: str, default: float = 0.0) -> float:
    """Parse a float, handling Fortran D/d exponent notation."""
    try:
        return float(s.strip().replace("D", "E").replace("d", "e"))
    except (ValueError, AttributeError):
        return default


# ---------------------------------------------------------------------------
# Element set
# ---------------------------------------------------------------------------

VALID_ELEMENTS: frozenset = frozenset(
    {
        "H",
        "He",
        "Li",
        "Be",
        "B",
        "C",
        "N",
        "O",
        "F",
        "Ne",
        "Na",
        "Mg",
        "Al",
        "Si",
        "P",
        "S",
        "Cl",
        "Ar",
        "K",
        "Ca",
        "Sc",
        "Ti",
        "V",
        "Cr",
        "Mn",
        "Fe",
        "Co",
        "Ni",
        "Cu",
        "Zn",
        "Ga",
        "Ge",
        "As",
        "Se",
        "Br",
        "Kr",
        "Rb",
        "Sr",
        "Y",
        "Zr",
        "Nb",
        "Mo",
        "Tc",
        "Ru",
        "Rh",
        "Pd",
        "Ag",
        "Cd",
        "In",
        "Sn",
        "Sb",
        "Te",
        "I",
        "Xe",
        "Cs",
        "Ba",
        "La",
        "Ce",
        "Pr",
        "Nd",
        "Pm",
        "Sm",
        "Eu",
        "Gd",
        "Tb",
        "Dy",
        "Ho",
        "Er",
        "Tm",
        "Yb",
        "Lu",
        "Hf",
        "Ta",
        "W",
        "Re",
        "Os",
        "Ir",
        "Pt",
        "Au",
        "Hg",
        "Tl",
        "Pb",
        "Bi",
        "Po",
        "At",
        "Rn",
        "Fr",
        "Ra",
        "Ac",
        "Th",
        "Pa",
        "U",
        "Np",
        "Pu",
        "Am",
        "Cm",
        "Bk",
        "Cf",
        "Es",
        "Fm",
        "Md",
        "No",
        "Lr",
        "D",
        "T",  # deuterium, tritium
    }
)

# CEA two-char pseudo-element codes to skip entirely
_PSEUDO_ELEMENTS: frozenset = frozenset({"IO", "IN", "IC", "IH"})


def normalise_element(raw: str) -> Optional[str]:
    """Map a raw 2-char element symbol (e.g. from CEA) to canonical form.

    Returns None for pseudo-elements, empty strings, or unknown symbols.
    Maps 'E' / 'e' → 'e-' (electron).
    """
    s = raw.strip()
    if not s:
        return None
    if s.upper() in _PSEUDO_ELEMENTS:
        return None
    if s.upper() == "E":
        return "e-"
    canon = s[0].upper() + s[1:].lower()
    return canon if canon in VALID_ELEMENTS else None


# ---------------------------------------------------------------------------
# Canonical species ID helpers
# ---------------------------------------------------------------------------


def hill_formula(elements: Dict[str, float]) -> str:
    """Return Hill-order formula string with charge suffix.

    Hill order: C first, H second, then alphabetical.
    Charge encoded via electron count: e_count = -1 → charge +1 → suffix "1+".

    Args:
        elements: Mapping of element symbol → stoichiometric count.
            Use key ``"e-"`` for electrons; negative count = cation.

    Returns:
        Hill-order formula string, e.g. ``"H2O"``, ``"CO2"``, ``"Al1+"``.
    """
    chem = {k: v for k, v in elements.items() if k != "e-"}
    e_count = elements.get("e-", 0.0)

    order: List[str] = []
    for sym in ("C", "H"):
        if sym in chem:
            order.append(sym)
    for sym in sorted(chem.keys()):
        if sym not in ("C", "H"):
            order.append(sym)

    parts: List[str] = []
    for sym in order:
        qty = chem[sym]
        n: int | float = int(round(qty)) if abs(qty - round(qty)) < 1e-9 else qty
        parts.append(sym if n == 1 else f"{sym}{n}")

    charge = -e_count
    if abs(charge) > 1e-9:
        c_int = int(round(abs(charge)))
        sign = "+" if charge > 0 else "-"
        parts.append(f"{c_int}{sign}" if c_int > 1 else sign)

    return "".join(parts) or "e-"


def canonical_id(elements: Dict[str, float], phase: str) -> str:
    """Return canonical species ID ``{HillFormula}_{Phase}``."""
    return f"{hill_formula(elements)}_{phase.upper()}"


def dedup_id(candidate: str, seen: Dict[str, int]) -> str:
    """Return a unique ID, appending ``_2``, ``_3``, … on collisions.

    Args:
        candidate: The desired ID.
        seen: Mutable dict mapping base ID → collision count (updated in-place).

    Returns:
        The unique ID to use.
    """
    count = seen.get(candidate, 0)
    seen[candidate] = count + 1
    return candidate if count == 0 else f"{candidate}_{count + 1}"


def phase_from_cond_flag(cond_flag: str) -> str:
    """Map a CEA condensed-flag character to G / L / S."""
    c = str(cond_flag).strip()
    if c in ("", "0"):
        return "G"
    if c == "2":
        return "L"
    return "S"


# ---------------------------------------------------------------------------
# JANAF species name parser
# ---------------------------------------------------------------------------

# JANAF.jnf formula format: {ElementSymbol1}{count1}{ElementSymbol2}{count2}...{charge?}
# e.g. "Al1Br3", "Al1Cl1+", "H2O1", "e-"
_JANAF_ELEM_RE = re.compile(r"([A-Z][a-z]?)(\d+(?:\.\d+)?)")


def parse_janaf_header(line: str) -> Tuple[Dict[str, float], str]:
    """Parse a JANAF.jnf species header line into (elements, phase).

    Header format: ``{formula} {phase_code}``

    Phase codes: ``g`` → G, ``l`` → L, ``cr`` / ``cr,l`` / ``s`` → S.

    Args:
        line: A header line such as ``"Al1Br3 cr"`` or ``"H2O1 g"``.

    Returns:
        Tuple of (elements dict, phase char ``"G"``/``"L"``/``"S"``).
    """
    parts = line.strip().split(None, 1)
    formula_part = parts[0] if parts else ""
    phase_raw = parts[1].strip() if len(parts) > 1 else "g"

    # Phase
    pl = phase_raw.lower()
    if "cr" in pl or pl in ("s", "solid", "c"):
        phase = "S"
    elif pl in ("l", "liquid"):
        phase = "L"
    else:
        phase = "G"

    # Special case: bare electron
    if formula_part.strip() == "e-":
        return {"e-": 1.0}, phase

    # Extract trailing charge: single '+' or '-' after the last element/count
    e_count = 0.0
    formula = formula_part
    if formula.endswith("+"):
        e_count = -1.0  # cation = lost one electron
        formula = formula[:-1]
    elif formula.endswith("-"):
        e_count = 1.0  # anion = gained one electron
        formula = formula[:-1]

    elements: Dict[str, float] = {}
    for m in _JANAF_ELEM_RE.finditer(formula):
        sym = m.group(1)
        count = float(m.group(2))
        if sym in VALID_ELEMENTS:
            elements[sym] = elements.get(sym, 0.0) + count

    if abs(e_count) > 1e-12:
        elements["e-"] = e_count

    return elements, phase


# ---------------------------------------------------------------------------
# Shared NASA-9 card-2 / descriptor parsing
# ---------------------------------------------------------------------------


def parse_nasa9_elements(line: str) -> Dict[str, float]:
    """Parse 5 × (A2 symbol + F6.2 count) at cols 10+ from a NASA-9 card-2 line.

    Used by both Burcat9 and CEA format files — the field layout is identical.
    """
    elems: Dict[str, float] = {}
    base = 10
    for k in range(5):
        a = base + k * 8
        sym_raw = fw(line, a, a + 2).strip()
        qty_raw = fw(line, a + 2, a + 8).strip()
        if not sym_raw:
            continue
        sym = sym_raw.capitalize()
        if sym == "E":
            sym = "e-"
        qty = to_float(qty_raw, 0.0)
        if abs(qty) < 1e-12:
            continue
        if sym in VALID_ELEMENTS or sym == "e-":
            elems[sym] = elems.get(sym, 0.0) + qty
    return elems


def rescue_nasa9_elements(line: str) -> Dict[str, float]:
    """Fallback element parser shifted left by one column (off-by-one recovery)."""
    elems: Dict[str, float] = {}
    base = 9
    for k in range(5):
        a = base + k * 8
        sym_raw = fw(line, a, a + 2).strip()
        qty_raw = fw(line, a + 2, a + 8).strip()
        if not sym_raw:
            continue
        sym = sym_raw.capitalize()
        if sym == "E":
            sym = "e-"
        qty = to_float(qty_raw, 0.0)
        if abs(qty) < 1e-12:
            continue
        if sym in VALID_ELEMENTS or sym == "e-":
            elems[sym] = elems.get(sym, 0.0) + qty
    return elems


def parse_nasa9_descriptor(line: str) -> Tuple[int, Dict[str, float], str]:
    """Parse a NASA-9 card-2 / CEA descriptor line.

    The format is shared between Burcat9 and CEA source files:

    * cols 0–1  : n_intervals (integer, may be preceded by whitespace)
    * cols 10+  : 5 × (A2 element symbol + F6.2 count)
    * col  50   : condensed/phase flag (``"0"``/``""`` → G, ``"2"`` → L, else → S)

    Args:
        line: The descriptor / card-2 line.

    Returns:
        Tuple ``(n_intervals, elements, phase)``.

    Raises:
        ValueError: if n_intervals cannot be found.
    """
    m = re.match(r"\s*(\d+)", line[:6])
    if not m:
        raise ValueError(f"Cannot parse n_intervals from {line[:6]!r}")
    n_intervals = int(m.group(1))

    elements = parse_nasa9_elements(line)
    if not elements:
        elements = rescue_nasa9_elements(line)

    phase = phase_from_cond_flag(fw(line, 51, 52))
    return n_intervals, elements, phase


# ---------------------------------------------------------------------------
# Shared NASA-9 three-line interval parser
# ---------------------------------------------------------------------------


def parse_nasa9_interval(hdr: str, line_a: str, line_b: str) -> dict:
    """Parse one 3-line NASA-9 temperature interval block.

    This layout is **identical** across both the Burcat9 ``.thr`` and the CEA
    ``thermo.inp`` formats::

        hdr   :  t_low (0:11)  t_high (11:22)  7 exponents (23:58, F5.1 each)
        line_a:  a1–a5  (5 × E16.9D, cols 0:80)
        line_b:  a6, a7 (2 × E16.9D, cols 0:32)  blank (32:48)
                 b1, b2 (2 × E16.9D, cols 48:64 and 64:80)

    Args:
        hdr:    Interval header line.
        line_a: First coefficient line.
        line_b: Second coefficient line (also holds b1 and b2).

    Returns:
        ``{"t_low", "t_high", "exponents", "coeffs", "b1", "b2"}``

    Raises:
        ValueError: if temperatures are NaN or fields cannot be parsed.
    """
    t_low = to_float(fw(hdr, 0, 11))
    t_high = to_float(fw(hdr, 11, 22))
    if t_low != t_low or t_high != t_high:  # NaN guard
        raise ValueError(f"NaN temperatures in interval header: {hdr!r}")

    exps = [to_float(fw(hdr, 23 + k * 5, 28 + k * 5)) for k in range(7)]

    coeffs = [to_float(fw(line_a, k * 16, (k + 1) * 16)) for k in range(5)]
    coeffs += [to_float(fw(line_b, k * 16, (k + 1) * 16)) for k in range(2)]

    b1 = to_float(fw(line_b, 48, 64))
    b2 = to_float(fw(line_b, 64, 80))

    return {
        "t_low": t_low,
        "t_high": t_high,
        "exponents": exps,
        "coeffs": coeffs,
        "b1": b1,
        "b2": b2,
    }
