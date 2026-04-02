"""Parser for the TERRA thermodynamic database.

Developed by Boris Georgievich Trusov at Bauman Moscow State Technical University.
This parser decodes the binary 'terra.bas' (metadata) and 'terra_a.bas' (coefficients)
files and converts them into the Prometheus Shomate JSON format.

Temperature scaling: x = T / 10000
G*(x) = f1 + f2*ln(x) + f3*x^-2 + f4*x^-1 + f5*x + f6*x^2 + f7*x^3
"""

import json
import logging
import math
import os
import struct
from typing import Any, Dict, List, Optional, Tuple

from ._common import canonical_id, dedup_id

log = logging.getLogger(__name__)

ELEMENTS_BY_Z = {
    0: "e-",
    1: "H",
    2: "He",
    3: "Li",
    4: "Be",
    5: "B",
    6: "C",
    7: "N",
    8: "O",
    9: "F",
    10: "Ne",
    11: "Na",
    12: "Mg",
    13: "Al",
    14: "Si",
    15: "P",
    16: "S",
    17: "Cl",
    18: "Ar",
    19: "K",
    20: "Ca",
    21: "Sc",
    22: "Ti",
    23: "V",
    24: "Cr",
    25: "Mn",
    26: "Fe",
    27: "Co",
    28: "Ni",
    29: "Cu",
    30: "Zn",
    31: "Ga",
    32: "Ge",
    33: "As",
    34: "Se",
    35: "Br",
    36: "Kr",
    37: "Rb",
    38: "Sr",
    39: "Y",
    40: "Zr",
    41: "Nb",
    42: "Mo",
    43: "Tc",
    44: "Ru",
    45: "Rh",
    46: "Pd",
    47: "Ag",
    48: "Cd",
    49: "In",
    50: "Sn",
    51: "Sb",
    52: "Te",
    53: "I",
    54: "Xe",
    55: "Cs",
    56: "Ba",
    57: "La",
    58: "Ce",
    59: "Pr",
    60: "Nd",
    61: "Pm",
    62: "Sm",
    63: "Eu",
    64: "Gd",
    65: "Tb",
    66: "Dy",
    67: "Ho",
    68: "Er",
    69: "Tm",
    70: "Yb",
    71: "Lu",
    72: "Hf",
    73: "Ta",
    74: "W",
    75: "Re",
    76: "Os",
    77: "Ir",
    78: "Pt",
    79: "Au",
    80: "Hg",
    81: "Tl",
    82: "Pb",
    83: "Bi",
    84: "Po",
    85: "At",
    86: "Rn",
    87: "Fr",
    88: "Ra",
    89: "Ac",
    90: "Th",
    91: "Pa",
    92: "U",
    93: "Np",
    94: "Pu",
}


class TERRAParser:
    def parse(self, bas_path: str, a_bas_path: str) -> Dict[str, Any]:
        with open(bas_path, "rb") as f:
            meta_data = f.read()
        with open(a_bas_path, "rb") as f:
            poly_data = f.read()

        entries = self._parse_metadata(meta_data)
        page_map = self._map_pages(entries)
        substances_data = self._extract_intervals(page_map, poly_data)

        # Dynamically discover reference states for all elements
        ref_h298_shifted = self._discover_reference_states(substances_data)
        self.ref_h298_shifted = ref_h298_shifted  # Store for external calibration use

        return self._convert_to_shomate(substances_data, ref_h298_shifted)

    def _discover_reference_states(
        self, substances_data: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Identify the reference substance for each element and calculate its raw H(298.15)."""
        all_elements = set()
        for item in substances_data:
            all_elements.update(item["entry"]["elements"].keys())

        ref_h298_shifted = {}
        for sym in all_elements:
            # Find all substances consisting ONLY of this element (exclude multi-element ions)
            candidates = []
            for item in substances_data:
                el_map = item["entry"]["elements"]
                if len(el_map) == 1 and sym in el_map:
                    candidates.append(item)

            if not candidates:
                log.warning("No reference candidate found for element %s", sym)
                continue

            # Find the most stable state (lowest G_raw per atom at 298.15 K)
            best_item = None
            min_g_per_atom = float("inf")

            for cand in candidates:
                g_raw = self._calc_raw_terra_g(cand, 298.15)
                atom_count = cand["entry"]["elements"][sym]
                # Note: atom_count can be negative for electrons in ions,
                # but those were excluded by len(el_map)==1 unless sym=='e-'
                g_per_atom = g_raw / abs(atom_count)
                if g_per_atom < min_g_per_atom:
                    min_g_per_atom = g_per_atom
                    best_item = cand

            if best_item:
                h_raw = self._calc_raw_terra_h(best_item, 298.15)
                atom_count = best_item["entry"]["elements"][sym]
                ref_h298_shifted[sym] = h_raw / atom_count
                log.debug(
                    "Reference for %-3s: %-15s | H_raw/atom: %12.2f",
                    sym,
                    best_item["entry"]["name"],
                    ref_h298_shifted[sym],
                )

        return ref_h298_shifted

    def _parse_metadata(self, data: bytes) -> List[Dict[str, Any]]:
        entries = []
        for rec in range(len(data) // 120):
            for sub in range(2):
                off = rec * 120 + sub * 60
                chunk = data[off : off + 60]
                if len(chunk) < 60:
                    continue
                elements = {}
                for j in range(5):
                    z, count = chunk[j * 2], chunk[j * 2 + 1]
                    if count > 0:
                        elements[ELEMENTS_BY_Z.get(z, f"Z{z}")] = float(count)

                phase_code = chunk[10]
                name_len = chunk[12]
                name = (
                    chunk[13 : 13 + min(name_len, 19)]
                    .decode("cp1251", errors="replace")
                    .strip()
                )

                charge = (
                    1.0 if name.endswith("+") else (-1.0 if name.endswith("-") else 0.0)
                )
                if abs(charge) > 0.1:
                    elements["e-"] = elements.get("e-", 0.0) - charge

                entries.append(
                    {
                        "name": name,
                        "elements": elements,
                        "state": "G" if phase_code == 2 or phase_code >= 13 else "S",
                        "id": struct.unpack("<I", chunk[56:60])[0],
                        "enthalpy_meta": struct.unpack("<f", chunk[32:36])[0],
                        "num_intervals": chunk[52],
                    }
                )
        return entries

    def _map_pages(
        self, entries: List[Dict[str, Any]]
    ) -> Dict[int, List[Dict[str, Any]]]:
        page_map = {}
        for entry in entries:
            pg = (entry["id"] // 10000) - 1
            page_map.setdefault(pg, []).append(entry)
        for pg in page_map:
            page_map[pg].sort(key=lambda x: x["id"])
        return page_map

    def _extract_intervals(
        self, page_map: Dict[int, List[Dict[str, Any]]], poly_data: bytes
    ) -> List[Dict[str, Any]]:
        results = []
        for pg in range(256):
            if pg not in page_map:
                continue
            pg_start = pg * 2048
            if pg_start + 2048 > len(poly_data):
                continue
            page_intervals = []
            for s in range(56):
                off = pg_start + s * 36
                vals = struct.unpack("<9f", poly_data[off : off + 36])
                if vals[1] < 100.0:
                    break
                page_intervals.append(vals)
            items = []
            slot_idx = 0
            for entry in page_map[pg]:
                ni = entry["num_intervals"]
                ints = page_intervals[slot_idx : slot_idx + ni]
                slot_idx += ni
                items.append({"entry": entry, "intervals": ints})
            for i in range(len(items)):
                curr = items[i]
                if (
                    i + 1 < len(items)
                    and len(curr["intervals"]) == 2
                    and len(items[i + 1]["intervals"]) == 1
                ):
                    if (
                        abs(curr["intervals"][0][1] - items[i + 1]["intervals"][0][0])
                        < 1.0
                    ):
                        curr["intervals"].append(items[i + 1]["intervals"][0])
                if (
                    i + 1 < len(items)
                    and len(curr["intervals"]) == 1
                    and len(items[i + 1]["intervals"]) == 2
                ):
                    if (
                        abs(curr["intervals"][0][1] - items[i + 1]["intervals"][0][0])
                        < 1.0
                    ):
                        items[i + 1]["intervals"].insert(0, curr["intervals"][0])
            results.extend(items)
        return results

    def _calc_phi_star(self, item: Dict, T: float) -> float:
        x = T / 10000.0
        f = next(
            (seg[2:] for seg in item["intervals"] if seg[0] - 1.0 <= T <= seg[1] + 1.0),
            None,
        )
        if not f:
            if not item["intervals"]:
                return 0.0
            sorted_segs = sorted(item["intervals"], key=lambda s: s[0])
            f = sorted_segs[0][2:] if T < sorted_segs[0][0] else sorted_segs[-1][2:]
        # G*(x) = f1 + f2*ln(x) + f3*x^-2 + f4*x^-1 + f5*x + f6*x^2 + f7*x^3
        return (
            f[0]
            + f[1] * math.log(x)
            + f[2] * x**-2
            + f[3] * x**-1
            + f[4] * x
            + f[5] * x**2
            + f[6] * x**3
        )

    def _calc_raw_terra_g(self, item: Dict, T: float) -> float:
        phi = self._calc_phi_star(item, T)
        return item["entry"]["enthalpy_meta"] - T * phi

    def _calc_raw_terra_h(self, item: Dict, T: float) -> float:
        x = T / 10000.0
        f = next(
            (seg[2:] for seg in item["intervals"] if seg[0] - 1.0 <= T <= seg[1] + 1.0),
            None,
        )
        if not f:
            if not item["intervals"]:
                return item["entry"]["enthalpy_meta"]
            sorted_segs = sorted(item["intervals"], key=lambda s: s[0])
            f = sorted_segs[0][2:] if T < sorted_segs[0][0] else sorted_segs[-1][2:]
        # H_poly(x) = 10000 * (f2*x - 2*f3/x - f4 + f5*x^2 + 2*f6*x^3 + 3*f7*x^4)
        h_poly = 10000.0 * (
            f[1] * x
            - 2.0 * f[2] / x
            - f[3]
            + f[4] * x**2
            + 2.0 * f[5] * x**3
            + 3.0 * f[6] * x**4
        )
        return h_poly + item["entry"]["enthalpy_meta"]

    def _convert_to_shomate(
        self, substances_data: List[Dict[str, Any]], ref_h298_shifted: Dict[str, float]
    ) -> Dict[str, Any]:
        db = {}
        seen = {}
        for item in substances_data:
            entry = item["entry"]
            ints = item["intervals"]
            if not ints:
                continue
            sp_id = dedup_id(canonical_id(entry["elements"], entry["state"]), seen)

            # Stoichiometric shift at 298.15
            stoich_shift = sum(
                count * ref_h298_shifted.get(sym, 0.0)
                for sym, count in entry["elements"].items()
            )

            segments = []
            for v in sorted(ints, key=lambda x: x[0]):
                f = v[2:]
                A = f[1]
                B = 0.2 * f[4]
                C = 0.06 * f[5]
                D = 0.012 * f[6]
                E = 200.0 * f[2]
                F = (entry["enthalpy_meta"] - 10000.0 * f[3] - stoich_shift) / 1000.0
                G = f[0] + f[1] - f[1] * 2.302585  # ln(10) correction
                t_ref = 0.29815
                h_ref = (
                    A * t_ref
                    + B * t_ref**2 / 2.0
                    + C * t_ref**3 / 3.0
                    + D * t_ref**4 / 4.0
                    - E / t_ref
                    + F
                )
                segments.append(
                    {
                        "t_low": v[0],
                        "t_high": v[1],
                        "coefficients": [A, B, C, D, E, F, G, h_ref],
                    }
                )
            db[sp_id] = {
                "elements": entry["elements"],
                "phase": entry["state"],
                "alias": entry["name"],
                "segments": segments,
            }
        return db
