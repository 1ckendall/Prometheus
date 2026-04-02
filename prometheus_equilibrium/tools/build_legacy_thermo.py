#!/usr/bin/env python3
"""CLI for TERRA/AFCESIC parsing and AFCESIC calibration.

This consolidates one-off scripts into a single tool under ``prometheus.tools``.

Subcommands:
    terra
        Parse ``terra.bas`` + ``terra_a.bas`` to ``terra.json``.
    afcesic-dump
        Dump all AFCESIC binary records into a human-readable text file.
    afcesic-calibrate
        Parse and calibrate AFCESIC from DAT files to ``afcesic.json``.
    all
        Run ``terra`` + ``afcesic-calibrate`` in sequence.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Any

from prometheus_equilibrium.equilibrium.species import AFCESICCoeff, SpeciesDatabase
from prometheus_equilibrium.thermo_data.parsers.afcesic import (
    collect_afcesic_species,
    dump_afcesic_raw,
)
from prometheus_equilibrium.thermo_data.parsers.terra import TERRAParser

log = logging.getLogger(__name__)

_CAL_TO_J = 4.184
_DEFAULT_T_REF = 1000.0

_ROOT = Path(__file__).resolve().parents[1]
_THERMO_DIR = _ROOT / "thermo_data"
_RAW_DIR = _THERMO_DIR / "raw"


def build_terra_json(
    bas_path: Path,
    a_bas_path: Path,
    out_path: Path,
) -> None:
    """Parse TERRA binary files and write ``terra.json``.

    Args:
        bas_path: Path to ``terra.bas``.
        a_bas_path: Path to ``terra_a.bas``.
        out_path: Output JSON path.
    """
    parser = TERRAParser()
    db = parser.parse(str(bas_path), str(a_bas_path))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(db, f, indent=2, ensure_ascii=True)

    log.info("Wrote %d TERRA species to %s", len(db), out_path)


def calibrate_afcesic(
    raw_dir: Path,
    nasa9_path: Path,
    nasa7_path: Path,
    terra_bas_path: Path,
    terra_a_bas_path: Path,
    out_path: Path,
    t_ref: float = _DEFAULT_T_REF,
) -> None:
    """Calibrate AFCESIC RF/CH constants to NASA and TERRA references.

    Uses a hybrid strategy:
        1. If species exists in NASA-9/7, point-match both H and S at ``t_ref``.
        2. Otherwise, apply a TERRA stoichiometric enthalpy shift using the
           dynamically discovered element references from TERRA.

    Args:
        raw_dir: Directory containing THERGG/THERII/THERSS DAT files.
        nasa9_path: Path to NASA-9 JSON.
        nasa7_path: Path to NASA-7 JSON.
        terra_bas_path: Path to TERRA metadata binary.
        terra_a_bas_path: Path to TERRA coefficient binary.
        out_path: Output path for corrected AFCESIC JSON.
        t_ref: Match temperature in K for NASA-overlap species.

    Raises:
        RuntimeError: If any AFCESIC species cannot be calibrated because both
            NASA overlap and TERRA stoichiometric references are unavailable.
    """
    terra_parser = TERRAParser()
    _ = terra_parser.parse(str(terra_bas_path), str(terra_a_bas_path))
    ref_h298 = getattr(terra_parser, "ref_h298_shifted", {})
    afc_data = collect_afcesic_species(raw_dir)

    db_nasa9 = SpeciesDatabase(nasa7_path="", nasa9_path=str(nasa9_path), janaf_path="")
    db_nasa9.load(
        include_nasa7=False,
        include_nasa9=True,
        include_afcesic=False,
        include_janaf=False,
        include_shomate=False,
        include_terra=False,
    )

    db_nasa7 = SpeciesDatabase(nasa7_path=str(nasa7_path), nasa9_path="", janaf_path="")
    db_nasa7.load(
        include_nasa7=True,
        include_nasa9=False,
        include_afcesic=False,
        include_janaf=False,
        include_shomate=False,
        include_terra=False,
    )

    stats = {
        "nasa9_or_7": 0,
        "terra_stoich": 0,
    }
    out_species: dict[str, dict[str, Any]] = {}
    failures: list[str] = []

    for sp_id, rec in afc_data.items():
        phase = str(rec.get("phase", "G")).strip().upper()
        if phase.startswith("G"):
            state = "G"
        elif phase.startswith("L"):
            state = "L"
        else:
            state = "S"

        try:
            afc_sp = AFCESICCoeff(
                rec["elements"],
                state,
                (300.0, 1200.0, 6000.0),
                tuple(rec["low_coefficients"]),
                tuple(rec["high_coefficients"]),
                rec["rf"],
                rec["ch"],
                rec.get("phase"),
            )
        except Exception as exc:
            failures.append(f"{sp_id}: invalid AFCESIC record ({exc})")
            continue

        out_rec = dict(rec)
        nasa_sp = db_nasa9.species.get(sp_id) or db_nasa7.species.get(sp_id)

        if nasa_sp is not None:
            h_afc = afc_sp.enthalpy(t_ref)
            h_nasa = nasa_sp.enthalpy(t_ref)
            s_afc = afc_sp.entropy(t_ref)
            s_nasa = nasa_sp.entropy(t_ref)

            if all(math.isfinite(v) for v in (h_afc, h_nasa, s_afc, s_nasa)):
                delta_rf = (h_nasa - h_afc) / _CAL_TO_J
                delta_ch = (s_nasa - s_afc) / _CAL_TO_J
                out_rec["rf"] = out_rec["rf"] + delta_rf
                out_rec["ch"] = out_rec["ch"] + delta_ch
                out_rec["calibration_source"] = "NASA"
                out_rec["delta_rf"] = delta_rf
                out_rec["delta_ch"] = delta_ch
                stats["nasa9_or_7"] += 1
                out_species[sp_id] = out_rec
                continue

        stoich_shift = 0.0
        missing_symbols: list[str] = []
        for symbol, count in out_rec.get("elements", {}).items():
            if symbol not in ref_h298:
                missing_symbols.append(symbol)
                continue
            stoich_shift += float(count) * float(ref_h298[symbol])

        if missing_symbols:
            failures.append(
                f"{sp_id}: missing TERRA elemental references for {sorted(set(missing_symbols))}"
            )
            continue

        delta_rf = -stoich_shift / _CAL_TO_J
        out_rec["rf"] = out_rec["rf"] + delta_rf
        out_rec["calibration_source"] = "TERRA_STOICH"
        out_rec["delta_rf"] = delta_rf
        out_rec["delta_ch"] = 0.0
        stats["terra_stoich"] += 1

        out_species[sp_id] = out_rec

    if failures:
        sample = "\n  - " + "\n  - ".join(failures[:20])
        extra = "" if len(failures) <= 20 else f"\n  ... and {len(failures) - 20} more"
        raise RuntimeError(
            "AFCESIC calibration failed because calibration references are incomplete."
            f"\nOffending species ({len(failures)}):{sample}{extra}"
        )

    output = {
        "_schema": {
            "description": "AFCESIC thermodynamic database (native cal units)",
            "units": {
                "Cp": "cal/(mol*K)",
                "RF": "kcal/mol / 1000 (enthalpy integration constant)",
                "CH": "cal/(mol*K) (entropy integration constant)",
                "T": "K",
            },
            "theta": "T / 1000",
            "low_T_range": "300.0-1000.0 K",
            "high_T_range": "1000.0-6000.0 K",
        },
        **dict(sorted(out_species.items())),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=True)

    log.info("Wrote calibrated AFCESIC database to %s", out_path)
    log.info("  NASA-matched: %d", stats["nasa9_or_7"])
    log.info("  TERRA-shifted: %d", stats["terra_stoich"])


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser.

    Returns:
        Configured argument parser.
    """
    parser = argparse.ArgumentParser(
        prog="prometheus-build-legacy",
        description="Build and calibrate TERRA/AFCESIC thermodynamic databases.",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable DEBUG logs."
    )

    sub = parser.add_subparsers(dest="command", required=True)

    terra = sub.add_parser(
        "terra", help="Build terra.json from terra.bas + terra_a.bas"
    )
    terra.add_argument("--bas", default=str(_RAW_DIR / "terra.bas"))
    terra.add_argument("--a-bas", default=str(_RAW_DIR / "terra_a.bas"), dest="a_bas")
    terra.add_argument("--out", default=str(_THERMO_DIR / "terra.json"))

    afc_dump = sub.add_parser("afcesic-dump", help="Dump AFCESIC DAT binaries to text")
    afc_dump.add_argument("--raw-dir", default=str(_RAW_DIR), dest="raw_dir")
    afc_dump.add_argument("--out", default=str(_THERMO_DIR / "afcesic_raw_dump.txt"))

    afc_cal = sub.add_parser(
        "afcesic-calibrate",
        help="Build afcesic.json (always calibrated; no unshifted output)",
    )
    afc_cal.add_argument("--raw-dir", default=str(_RAW_DIR), dest="raw_dir")
    afc_cal.add_argument("--nasa9", default=str(_THERMO_DIR / "nasa9.json"))
    afc_cal.add_argument("--nasa7", default=str(_THERMO_DIR / "nasa7.json"))
    afc_cal.add_argument(
        "--terra-bas", default=str(_RAW_DIR / "terra.bas"), dest="terra_bas"
    )
    afc_cal.add_argument(
        "--terra-a-bas", default=str(_RAW_DIR / "terra_a.bas"), dest="terra_a_bas"
    )
    afc_cal.add_argument("--out", default=str(_THERMO_DIR / "afcesic.json"))
    afc_cal.add_argument("--t-ref", type=float, default=_DEFAULT_T_REF, dest="t_ref")

    all_cmd = sub.add_parser("all", help="Run terra + afcesic-calibrate")
    all_cmd.add_argument("--raw-dir", default=str(_RAW_DIR), dest="raw_dir")
    all_cmd.add_argument("--thermo-dir", default=str(_THERMO_DIR), dest="thermo_dir")
    all_cmd.add_argument("--t-ref", type=float, default=_DEFAULT_T_REF, dest="t_ref")

    return parser


def main() -> None:
    """Run the CLI entrypoint."""
    parser = _build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    if args.command == "terra":
        build_terra_json(Path(args.bas), Path(args.a_bas), Path(args.out))
        return

    if args.command == "afcesic-dump":
        dump_afcesic_raw(Path(args.raw_dir), Path(args.out))
        return

    if args.command == "afcesic-calibrate":
        calibrate_afcesic(
            raw_dir=Path(args.raw_dir),
            nasa9_path=Path(args.nasa9),
            nasa7_path=Path(args.nasa7),
            terra_bas_path=Path(args.terra_bas),
            terra_a_bas_path=Path(args.terra_a_bas),
            out_path=Path(args.out),
            t_ref=float(args.t_ref),
        )
        return

    thermo_dir = Path(args.thermo_dir)
    raw_dir = Path(args.raw_dir)

    terra_json = thermo_dir / "terra.json"
    afc_corrected = thermo_dir / "afcesic.json"

    build_terra_json(raw_dir / "terra.bas", raw_dir / "terra_a.bas", terra_json)
    calibrate_afcesic(
        raw_dir=raw_dir,
        nasa9_path=thermo_dir / "nasa9.json",
        nasa7_path=thermo_dir / "nasa7.json",
        terra_bas_path=raw_dir / "terra.bas",
        terra_a_bas_path=raw_dir / "terra_a.bas",
        out_path=afc_corrected,
        t_ref=float(args.t_ref),
    )


if __name__ == "__main__":
    main()
