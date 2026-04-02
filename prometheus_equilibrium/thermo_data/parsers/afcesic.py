"""Parser for the AFCESIC thermodynamic database.

AFCESIC is a Soviet/Russian condensed-phase thermodynamic database developed
at the Institute of Physical Chemistry (IPhCh RAS).  The name is an acronym
from the Russian for "Automated Physico-Chemical Equilibrium System with
Information about Condensed Substances".

Binary format
-------------
Three DAT files, each holding one phase group:

    THERGG.DAT  –  gas-phase species   (label ``GG``)
    THERII.DAT  –  liquid-phase species (label ``II``)
    THERSS.DAT  –  solid-phase species  (label ``SS``)

Each file begins with a 2-byte little-endian ``uint16`` element-count header,
possibly followed by a species-count ``uint16`` and a contiguous array of
96-byte species records.

Record layout (all fields little-endian)
-----------------------------------------
Bytes  0–23   Four element slots, each 6 bytes:
                 0–1  two-char ASCII element symbol
                 2–5  float32 stoichiometric count
Bytes 24–25   uint16 charge
Bytes 26–27   uint16 number of elements
Bytes 28–29   uint16 element-table reference index
Bytes 30–33   float32 RF  – enthalpy integration constant (kcal/mol / 1000)
Bytes 34–37   float32 (unused padding / second constant)
Bytes 38–49   12-char ASCII species name
Bytes 50–69   5 × float32  high-T Cp polynomial coefficients (θ = T/1000)
Bytes 70–89   5 × float32  low-T Cp polynomial coefficients
Bytes 90–95   3 × uint16   catalog ID, unused, year code

Cp polynomial
-------------
Cp/R = a₁ + a₂·θ + a₃·θ² + a₄·θ³ + a₅·θ⁴     θ = T / 1000 K

Units are cal/(mol·K); RF and CH are in the same caloric units.
"""

from __future__ import annotations

import logging
import math
import struct
from pathlib import Path
from typing import Any, Optional

from ._common import canonical_id, normalise_element

log = logging.getLogger(__name__)

_RECORD_SIZE = 96
_EXCLUDED_ISOTOPE_SYMBOLS: frozenset[str] = frozenset({"D", "T"})

_DAT_FILES = {
    "GG": "THERGG.DAT",
    "II": "THERII.DAT",
    "SS": "THERSS.DAT",
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _parse_charge(raw_u16: int) -> int:
    """Convert AFCESIC uint16 charge to signed integer.

    Args:
        raw_u16: Raw uint16 value.

    Returns:
        Signed integer charge.
    """
    if raw_u16 == 0:
        return 0
    if raw_u16 == 1:
        return 1
    if raw_u16 == 65535:
        return -1
    if raw_u16 > 32767:
        return raw_u16 - 65536
    return raw_u16


def _phase_letter(name: str, file_label: str) -> str:
    """Infer phase from AFCESIC name and source file.

    Args:
        name: Species alias/name from AFCESIC record.
        file_label: ``GG``, ``II``, or ``SS``.

    Returns:
        ``G``, ``L``, or ``S``.
    """
    upper = name.upper()
    if "(L)" in upper:
        return "L"
    if "(C)" in upper or "(S)" in upper or "(CR)" in upper:
        return "S"
    for tag in ("ALPHA", "BETA", "GAMMA", "DELTA", "KAPPA", "GRAPHITE"):
        if tag in upper:
            return "S"
    if file_label == "SS":
        return "S"
    return "G"


def _parse_afcesic_record(rec: bytes, file_label: str) -> Optional[dict[str, Any]]:
    """Parse one AFCESIC binary record.

    Args:
        rec: Raw 96-byte record.
        file_label: ``GG``, ``II``, or ``SS``.

    Returns:
        Parsed record dict or ``None`` if invalid/empty.
    """
    if len(rec) != _RECORD_SIZE:
        return None

    name = rec[38:50].decode("ascii", errors="replace").strip()
    if not name or not name[0].isalnum():
        return None

    charge_u16 = struct.unpack_from("<H", rec, 24)[0]
    n_el = struct.unpack_from("<H", rec, 26)[0]
    el_ref = struct.unpack_from("<H", rec, 28)[0]

    max_slots = min(n_el, 4) if n_el > 0 else 4
    elements: dict[str, float] = {}
    for j in range(max_slots):
        off = j * 6
        sym_raw = rec[off : off + 2].decode("ascii", errors="ignore").strip()
        if not sym_raw:
            continue
        sym = normalise_element(sym_raw)
        if sym is None:
            continue

        cnt = struct.unpack_from("<f", rec, off + 2)[0]
        if not math.isfinite(cnt) or cnt <= 0:
            continue
        elements[sym] = round(float(cnt), 4)

    if not elements:
        return None

    rf = struct.unpack_from("<f", rec, 30)[0]
    ch = struct.unpack_from("<f", rec, 34)[0]
    hi_coeffs = list(struct.unpack_from("<5f", rec, 50))
    lo_coeffs = list(struct.unpack_from("<5f", rec, 70))

    if any(not math.isfinite(v) for v in [rf, ch, *hi_coeffs, *lo_coeffs]):
        return None

    charge = _parse_charge(charge_u16)
    if charge != 0 and not (len(elements) == 1 and "e-" in elements):
        elements["e-"] = float(-charge)

    trail = list(struct.unpack_from("<3H", rec, 90))

    return {
        "name": name,
        "elements": elements,
        "phase": _phase_letter(name, file_label),
        "n_el": n_el,
        "el_ref": el_ref,
        "rf": float(rf),
        "ch": float(ch),
        "high_coefficients": [float(c) for c in hi_coeffs],
        "low_coefficients": [float(c) for c in lo_coeffs],
        "catalog_id": trail[0],
        "year_code": trail[2],
    }


def _iter_afcesic_records(path: Path, file_label: str) -> list[dict[str, Any]]:
    """Read and parse all species records from one AFCESIC DAT file.

    Args:
        path: Path to the DAT file.
        file_label: ``GG``, ``II``, or ``SS``.

    Returns:
        Parsed records.
    """
    data = path.read_bytes()
    total = len(data)

    n_elem_hdr = struct.unpack_from("<H", data, 0)[0]
    table_end = 2 + n_elem_hdr * 6

    if table_end + 2 <= total:
        n_species_cand = struct.unpack_from("<H", data, table_end)[0]
        rec_start_cand = table_end + 2
        remainder = total - rec_start_cand
        if (
            n_species_cand > 0
            and remainder > 0
            and remainder % _RECORD_SIZE == 0
            and remainder // _RECORD_SIZE == n_species_cand
        ):
            n_species = n_species_cand
            rec_start = rec_start_cand
        else:
            n_species = n_elem_hdr
            rec_start = 2
    else:
        n_species = n_elem_hdr
        rec_start = 2

    out: list[dict[str, Any]] = []
    for i in range(n_species):
        off = rec_start + i * _RECORD_SIZE
        if off + _RECORD_SIZE > total:
            break
        parsed = _parse_afcesic_record(data[off : off + _RECORD_SIZE], file_label)
        if parsed is not None:
            out.append(parsed)
    return out


def _format_float(val: float) -> str:
    """Format a float for the raw dump output.

    Args:
        val: Numeric value.

    Returns:
        Fixed-width scientific notation string.
    """
    try:
        return f"{val:>12.5e}"
    except (TypeError, ValueError):
        return f"{str(val):>12}"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def collect_afcesic_species(raw_dir: Path) -> dict[str, dict[str, Any]]:
    """Parse AFCESIC DAT files into canonical species records.

    Reads ``THERGG.DAT``, ``THERII.DAT``, and ``THERSS.DAT`` from *raw_dir*,
    deduplicates by canonical ID (first occurrence wins), and filters out
    isotope-labelled species (D, T).

    Args:
        raw_dir: Directory containing the three THER*.DAT files.

    Returns:
        Canonical AFCESIC records keyed by Prometheus species ID, sorted
        alphabetically.  Each value contains: ``elements``, ``phase``,
        ``name``, ``rf``, ``ch``, ``high_coefficients``, ``low_coefficients``.
    """
    all_species: dict[str, dict[str, Any]] = {}
    totals: dict[str, int] = {}
    skipped_isotopes = 0

    for label, filename in _DAT_FILES.items():
        file_path = raw_dir / filename
        if not file_path.exists():
            log.warning("Missing %s, skipping.", file_path)
            continue
        records = _iter_afcesic_records(file_path, label)
        totals[label] = len(records)
        for sp in records:
            if any(sym in _EXCLUDED_ISOTOPE_SYMBOLS for sym in sp["elements"]):
                skipped_isotopes += 1
                continue
            sp_id = (
                f"e-_{sp['phase']}"
                if list(sp["elements"].keys()) == ["e-"]
                else canonical_id(sp["elements"], sp["phase"])
            )
            if sp_id in all_species:
                continue
            all_species[sp_id] = {
                "elements": sp["elements"],
                "phase": sp["phase"],
                "name": sp["name"],
                "rf": sp["rf"],
                "ch": sp["ch"],
                "high_coefficients": sp["high_coefficients"],
                "low_coefficients": sp["low_coefficients"],
            }

    log.info("Parsed %d AFCESIC species from DAT files", len(all_species))
    if skipped_isotopes:
        log.info(
            "Skipped %d AFCESIC isotope species containing %s",
            skipped_isotopes,
            sorted(_EXCLUDED_ISOTOPE_SYMBOLS),
        )
    for label, count in totals.items():
        log.info("  THER%s.DAT: %d records", label, count)
    return dict(sorted(all_species.items()))


def dump_afcesic_raw(raw_dir: Path, out_path: Path) -> None:
    """Write a human-readable transcription of all raw AFCESIC binary records.

    Args:
        raw_dir: Directory containing the three THER*.DAT files.
        out_path: Output text file path.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as out_file:
        out_file.write("AFCESIC RAW BINARY TRANSCRIPTION\n")
        out_file.write("=" * 120 + "\n\n")

        for label, filename in _DAT_FILES.items():
            file_path = raw_dir / filename
            if not file_path.exists():
                log.warning("Missing %s, skipping.", file_path)
                continue

            data = file_path.read_bytes()
            total = len(data)
            n_elem_hdr = struct.unpack_from("<H", data, 0)[0]
            table_end = 2 + n_elem_hdr * 6

            if table_end + 2 <= total:
                n_species_cand = struct.unpack_from("<H", data, table_end)[0]
                rec_start_cand = table_end + 2
                remainder = total - rec_start_cand
                if (
                    n_species_cand > 0
                    and remainder > 0
                    and remainder % _RECORD_SIZE == 0
                    and remainder // _RECORD_SIZE == n_species_cand
                ):
                    n_species = n_species_cand
                    rec_start = rec_start_cand
                else:
                    n_species = n_elem_hdr
                    rec_start = 2
            else:
                n_species = n_elem_hdr
                rec_start = 2

            out_file.write(f"=== {file_path.name} ===\n")
            for idx in range(n_species):
                off = rec_start + idx * _RECORD_SIZE
                if off + _RECORD_SIZE > total:
                    break
                rec = data[off : off + _RECORD_SIZE]
                name = rec[38:50].decode("ascii", errors="replace").strip() or "[EMPTY]"
                charge_u16 = struct.unpack_from("<H", rec, 24)[0]
                n_el = struct.unpack_from("<H", rec, 26)[0]
                el_ref = struct.unpack_from("<H", rec, 28)[0]
                rf = struct.unpack_from("<f", rec, 30)[0]
                ch = struct.unpack_from("<f", rec, 34)[0]
                hi_coeffs = struct.unpack_from("<5f", rec, 50)
                lo_coeffs = struct.unpack_from("<5f", rec, 70)

                slots = []
                for j in range(4):
                    slot_off = j * 6
                    sym = rec[slot_off : slot_off + 2].decode("ascii", errors="replace").strip() or "--"
                    cnt = struct.unpack_from("<f", rec, slot_off + 2)[0]
                    slots.append(f"{sym}:{cnt:<6.2f}")

                out_file.write(
                    f"[{label} | {idx:04d}] NAME: {name:<12} | CHARGE: {charge_u16:<5} | N_EL: {n_el:<2} | REF: {el_ref:<4}\n"
                )
                out_file.write(f"    ELEMENTS : {' | '.join(slots)}\n")
                out_file.write(
                    f"    CONSTANTS: RF = {_format_float(rf)} | CH = {_format_float(ch)}\n"
                )
                out_file.write(
                    f"    HIGH_CP  : [{', '.join(_format_float(c) for c in hi_coeffs)}]\n"
                )
                out_file.write(
                    f"    LOW_CP   : [{', '.join(_format_float(c) for c in lo_coeffs)}]\n"
                )
                out_file.write(f"    HEX DUMP : {rec.hex(' ').upper()}\n")
                out_file.write("-" * 120 + "\n")

    log.info("Wrote AFCESIC raw dump to %s", out_path)
