#!/usr/bin/env python3
"""Build all Prometheus thermodynamic databases in one command.

This script orchestrates both database pipelines:

1. Core thermo compiler (NASA-7, NASA-9, JANAF).
2. Legacy binary translators/calibration (TERRA, AFCESIC).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from prometheus_equilibrium.thermo_data.compiler import ThermoCompiler
from prometheus_equilibrium.tools.build_legacy_thermo import (
    build_terra_json,
    calibrate_afcesic,
)

log = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_RAW_DIR = _ROOT / "thermo_data" / "raw"
_DEFAULT_OUT_DIR = _ROOT / "thermo_data"


def build_all_thermo_databases(
    raw_dir: Path,
    out_dir: Path,
    mode: str,
    burcat_source: str,
    cea_source: str,
    janaf_source: str,
    t_ref: float,
) -> None:
    """Build all thermo database outputs.

    Args:
        raw_dir: Directory containing raw source/binary files.
        out_dir: Output directory for generated thermo databases.
        mode: Merge mode for NASA/JANAF compiler.
        burcat_source: Source label for Burcat-derived records.
        cea_source: Source label for CEA-derived NASA-9 records.
        janaf_source: Source label for JANAF records.
        t_ref: Calibration match temperature in Kelvin for AFCESIC-NASA matches.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build NASA-7/NASA-9/JANAF from text sources.
    compiler = ThermoCompiler()
    compiler.compile_all(
        str(raw_dir),
        str(out_dir),
        burcat_source=burcat_source,
        cea_source=cea_source,
        janaf_source=janaf_source,
        mode=mode,
    )

    # Build TERRA translation.
    build_terra_json(
        bas_path=raw_dir / "terra.bas",
        a_bas_path=raw_dir / "terra_a.bas",
        out_path=out_dir / "terra.json",
    )

    # Parse and calibrate AFCESIC onto the project reference scale.
    calibrate_afcesic(
        raw_dir=raw_dir,
        nasa9_path=out_dir / "nasa9.json",
        nasa7_path=out_dir / "nasa7.json",
        terra_bas_path=raw_dir / "terra.bas",
        terra_a_bas_path=raw_dir / "terra_a.bas",
        out_path=out_dir / "afcesic.json",
        t_ref=t_ref,
    )

    log.info("Completed full thermo database build in %s", out_dir)


def _build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser.

    Returns:
        Configured argument parser.
    """
    parser = argparse.ArgumentParser(
        prog="prometheus-build-all-thermo",
        description="Build all Prometheus thermodynamic databases.",
    )
    parser.add_argument("--raw-dir", default=str(_DEFAULT_RAW_DIR), dest="raw_dir")
    parser.add_argument("--out-dir", default=str(_DEFAULT_OUT_DIR), dest="out_dir")
    parser.add_argument(
        "--mode",
        choices=["overwrite", "append", "interactive"],
        default="overwrite",
        help="Merge mode for NASA/JANAF compiler outputs.",
    )
    parser.add_argument("--burcat-source", default="Burcat", dest="burcat_source")
    parser.add_argument("--cea-source", default="CEA", dest="cea_source")
    parser.add_argument("--janaf-source", default="JANAF", dest="janaf_source")
    parser.add_argument("--t-ref", type=float, default=1000.0, dest="t_ref")
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable DEBUG logs."
    )
    return parser


def main() -> None:
    """Run the build-all thermo command."""
    parser = _build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    build_all_thermo_databases(
        raw_dir=Path(args.raw_dir),
        out_dir=Path(args.out_dir),
        mode=args.mode,
        burcat_source=args.burcat_source,
        cea_source=args.cea_source,
        janaf_source=args.janaf_source,
        t_ref=float(args.t_ref),
    )


if __name__ == "__main__":
    main()
