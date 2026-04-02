#!/usr/bin/env python3
"""Build `propellants.toml` from legacy `PEPCODED.DAF`.

The DAF record layout is fixed-width and documented in-file as:
`A1,I5,A30,6(I3,A2),F5.0,F6.0`.

This tool parses each ingredient record, computes SI values, assigns roles using
chemistry-first heuristics, and writes a deterministic TOML ingredient list.
"""

from __future__ import annotations

import argparse
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from prometheus.core.constants import ELEMENTS_MOLAR_MASSES
from prometheus.thermo_data.parsers._common import normalise_element

log = logging.getLogger(__name__)

# Unit conversions
_CAL_TO_J = 4.184
_LB_PER_IN3_TO_KG_PER_M3 = 27679.9047102

_EXCLUDED_ISOTOPE_SYMBOLS = {"D", "T"}
_METAL_ELEMENTS = {
    "Al",
    "Mg",
    "Be",
    "Li",
    "Na",
    "K",
    "Ca",
    "Ti",
    "Zr",
    "B",
}
_OXIDIZER_NAME_HINTS = (
    "NITRATE",
    "PERCHLORATE",
    "CHLORATE",
    "PEROXIDE",
    "NITROGEN TETROXIDE",
    "NITRIC ACID",
    "FLUORINE",
    "OXYGEN",
)
_FUEL_NAME_HINTS = (
    "HYDRAZINE",
    "BORANE",
    "ALCOHOL",
    "METHANE",
    "ETHANE",
    "PROPANE",
    "ACETYLENE",
    "PARAFFIN",
    "KEROSENE",
    "AMINE",
)
_BINDER_NAME_HINTS = ("HTPB", "PBAN", "POLYBUTADIENE", "BINDER")
_PLASTICIZER_NAME_HINTS = ("DIOCTYL", "CITRATE", "ADIPATE", "PLASTICIZER")
_CATALYST_NAME_HINTS = ("FERROCENE", "CHROMIUM OXIDE", "IRON OXIDE", "COPPER OXIDE")


@dataclass
class PepIngredient:
    """Parsed PEP ingredient record."""

    seq: int
    flags: str
    name: str
    elements: Dict[str, float]
    heat_cal_per_g: float
    density_lb_per_in3: float


@dataclass
class TomlIngredient:
    """Prepared TOML ingredient record."""

    id: str
    name: str
    elements: Dict[str, float]
    molar_mass: float
    dHf298: float
    density: float
    roles: List[str]
    pep_line: int
    pep_flags: str


def _parse_fixed_record(line: str) -> PepIngredient | None:
    """Parse a single 80-char DAF ingredient line.

    Args:
        line: Raw line from `PEPCODED.DAF`.

    Returns:
        Parsed ingredient, or None for headers/separators/non-data lines.
    """
    raw = line.rstrip("\n")
    if not raw.strip() or raw.startswith("*") or raw.startswith("+"):
        return None

    body = raw.rstrip("]").ljust(80)

    seq_text = body[2:8].strip()
    if not seq_text.isdigit():
        return None

    seq = int(seq_text)
    flags = body[:2].strip()
    name = body[8:38].strip()

    elements: Dict[str, float] = {}
    # DAF records include one spacer after the 30-char name field.
    pairs = body[39:69]
    for i in range(0, 30, 5):
        chunk = pairs[i : i + 5]
        count_text = chunk[:3].strip()
        symbol_text = chunk[3:5]
        if not count_text:
            continue
        try:
            count = float(int(count_text))
        except ValueError:
            continue
        if count <= 0:
            continue
        symbol = normalise_element(symbol_text)
        if symbol is None:
            continue
        elements[symbol] = elements.get(symbol, 0.0) + count

    hf_text = body[69:74].strip()
    density_text = body[74:80].strip()

    try:
        heat_cal_per_g = float(hf_text) if hf_text else 0.0
    except ValueError:
        heat_cal_per_g = 0.0
    try:
        density_lb_per_in3 = float(density_text) if density_text else 0.0
    except ValueError:
        density_lb_per_in3 = 0.0

    return PepIngredient(
        seq=seq,
        flags=flags,
        name=name,
        elements=elements,
        heat_cal_per_g=heat_cal_per_g,
        density_lb_per_in3=density_lb_per_in3,
    )


def _molar_mass_g_mol(elements: Dict[str, float]) -> float:
    """Compute molar mass from elemental composition."""
    return sum(ELEMENTS_MOLAR_MASSES[sym] * count for sym, count in elements.items())


def _slugify(name: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", name.upper()).strip("_")
    slug = re.sub(r"_+", "_", slug)
    return slug or "INGREDIENT"


def _assign_roles(name: str, elements: Dict[str, float], flags: str) -> List[str]:
    """Assign roles using simple chemistry and name-hint heuristics."""
    roles: List[str] = []
    upper_name = name.upper()

    total_atoms = sum(elements.values()) or 1.0
    o = elements.get("O", 0.0)
    f = elements.get("F", 0.0)
    cl = elements.get("Cl", 0.0)
    br = elements.get("Br", 0.0)
    c = elements.get("C", 0.0)
    h = elements.get("H", 0.0)
    b = elements.get("B", 0.0)
    si = elements.get("Si", 0.0)
    s = elements.get("S", 0.0)
    p = elements.get("P", 0.0)

    metal_atoms = sum(elements.get(sym, 0.0) for sym in _METAL_ELEMENTS)

    oxidizer_strength = (o + 1.25 * f + 1.10 * cl + 0.90 * br) / total_atoms
    reducing_demand = (
        2.0 * c
        + 0.5 * h
        + 1.5 * b
        + 2.0 * si
        + 2.0 * s
        + 2.5 * p
        + 1.5 * elements.get("Al", 0.0)
        + 1.0 * elements.get("Mg", 0.0)
        + 1.0 * elements.get("Be", 0.0)
    )
    oxidizer_supply = o + 1.5 * f + 1.5 * cl + 1.0 * br

    has_oxidizer_hint = any(hint in upper_name for hint in _OXIDIZER_NAME_HINTS)
    has_fuel_hint = any(hint in upper_name for hint in _FUEL_NAME_HINTS)

    if oxidizer_strength >= 0.45 or has_oxidizer_hint or oxidizer_supply >= 0.95 * reducing_demand:
        roles.append("oxidizer")

    if reducing_demand > 1.10 * max(oxidizer_supply, 1e-9) or has_fuel_hint:
        roles.append("fuel")

    # Metals are both energetic fuels and useful for downstream filtering.
    if metal_atoms / total_atoms >= 0.55 or upper_name.startswith(("ALUMINUM", "MAGNESIUM", "BORON")):
        roles.append("metal")
        if "fuel" not in roles:
            roles.append("fuel")

    if any(hint in upper_name for hint in _BINDER_NAME_HINTS):
        roles.append("binder")
    if any(hint in upper_name for hint in _PLASTICIZER_NAME_HINTS):
        roles.append("plasticizer")
    if any(hint in upper_name for hint in _CATALYST_NAME_HINTS):
        roles.append("catalyst")

    # Flag 'A' in this legacy file is mostly heated-air entries; classify inert.
    if flags == "A" and "oxidizer" not in roles and "fuel" not in roles:
        roles.append("inert")

    if not roles:
        roles.append("inert")

    order = ["oxidizer", "fuel", "metal", "binder", "plasticizer", "catalyst", "inert"]
    dedup = []
    seen = set()
    for role in order:
        if role in roles and role not in seen:
            dedup.append(role)
            seen.add(role)
    return dedup


def _convert_record(rec: PepIngredient, used_ids: set[str]) -> TomlIngredient | None:
    """Convert parsed PEP ingredient to TOML ingredient format."""
    if not rec.elements:
        return None
    if any(sym in _EXCLUDED_ISOTOPE_SYMBOLS for sym in rec.elements):
        return None

    mm = _molar_mass_g_mol(rec.elements)
    if mm <= 0.0:
        return None

    base = _slugify(rec.name)
    candidate = base
    if candidate in used_ids:
        candidate = f"{base}_{rec.seq:04d}"
    suffix = 2
    while candidate in used_ids:
        candidate = f"{base}_{rec.seq:04d}_{suffix}"
        suffix += 1
    used_ids.add(candidate)

    d_hf_j_mol = rec.heat_cal_per_g * _CAL_TO_J * mm
    density_kg_m3 = rec.density_lb_per_in3 * _LB_PER_IN3_TO_KG_PER_M3

    return TomlIngredient(
        id=candidate,
        name=rec.name,
        elements={k: float(v) for k, v in sorted(rec.elements.items())},
        molar_mass=round(mm, 6),
        dHf298=round(d_hf_j_mol, 6),
        density=round(density_kg_m3, 6),
        roles=_assign_roles(rec.name, rec.elements, rec.flags),
        pep_line=rec.seq,
        pep_flags=rec.flags,
    )


def parse_daf(path: Path) -> List[PepIngredient]:
    """Parse all ingredient records from `PEPCODED.DAF`.

    Args:
        path: Input DAF path.

    Returns:
        Parsed ingredient records.
    """
    out: List[PepIngredient] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        rec = _parse_fixed_record(line)
        if rec is not None:
            out.append(rec)
    return out


def build_propellants_toml(daf_path: Path, out_path: Path) -> int:
    """Build `propellants.toml` from `PEPCODED.DAF`.

    Args:
        daf_path: Input DAF path.
        out_path: Output TOML path.

    Returns:
        Number of emitted `[[ingredient]]` records.
    """
    parsed = parse_daf(daf_path)
    used_ids: set[str] = set()
    converted: List[TomlIngredient] = []

    for rec in parsed:
        out = _convert_record(rec, used_ids)
        if out is not None:
            converted.append(out)

    lines: List[str] = []
    lines.append('schema_version = "1"')
    lines.append("")

    for rec in converted:
        lines.append("[[ingredient]]")
        lines.append(f'id = "{rec.id}"')
        lines.append(f'name = "{rec.name.replace("\\", "\\\\").replace("\"", "\\\"")}"')

        parts = ", ".join(f"{k} = {v:.6f}" for k, v in rec.elements.items())
        lines.append(f"elements = {{{parts}}}")
        lines.append(f"molar_mass = {rec.molar_mass:.6f}")
        lines.append(f"dHf298 = {rec.dHf298:.6f}")
        lines.append(f"density = {rec.density:.6f}")
        role_text = ", ".join(f'"{r}"' for r in rec.roles)
        lines.append(f"roles = [{role_text}]")
        lines.append(f"pep_line = {rec.pep_line}")
        lines.append(f'pep_flags = "{rec.pep_flags}"')
        lines.append("")

    out_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return len(converted)


def _build_parser() -> argparse.ArgumentParser:
    """Build CLI parser.

    Returns:
        Configured parser.
    """
    root = Path(__file__).resolve().parents[1]
    prop_dir = root / "propellants"

    parser = argparse.ArgumentParser(
        prog="prometheus-build-propellants",
        description="Generate propellants.toml from PEPCODED.DAF",
    )
    parser.add_argument("--src", default=str(prop_dir / "PEPCODED.DAF"))
    parser.add_argument("--out", default=str(prop_dir / "propellants.toml"))
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser


def main() -> None:
    """CLI entrypoint."""
    parser = _build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    count = build_propellants_toml(Path(args.src), Path(args.out))
    log.info("Wrote %d ingredients to %s", count, args.out)


if __name__ == "__main__":
    main()

