"""Compare TERRA on/off with shifting/frozen nozzle expansion and emit debug logs.

This script runs four cases for a representative AP/Al/HTPB motor setup:
1) TERRA ON + SHIFTING
2) TERRA ON + FROZEN
3) TERRA OFF + SHIFTING
4) TERRA OFF + FROZEN

Each case writes a dedicated detailed log file and the script writes summary
artifacts under a timestamped output directory.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from loguru import logger

from prometheus_equilibrium.equilibrium.performance import PerformanceSolver
from prometheus_equilibrium.equilibrium.problem import EquilibriumProblem, ProblemType
from prometheus_equilibrium.equilibrium.solver import GordonMcBrideSolver
from prometheus_equilibrium.equilibrium.species import SpeciesDatabase
from prometheus_equilibrium.propellants.loader import PropellantDatabase

PSI_TO_PA = 6894.757
G0 = 9.80665


def _make_problem(
    include_terra: bool, chamber_pressure_pa: float
) -> tuple[SpeciesDatabase, EquilibriumProblem]:
    db = SpeciesDatabase()
    db.load(include_janaf=False, include_terra=include_terra)

    repo_root = Path(__file__).resolve().parents[1]
    prop_db = PropellantDatabase(
        str(repo_root / "prometheus_equilibrium" / "propellants" / "propellants.toml"),
        species_db=db,
    )
    prop_db.load()

    components = [
        ("AMMONIUM_PERCHLORATE", 0.68),
        ("ALUMINUM_PURE_CRYSTALINE", 0.18),
        ("HTPB_R_45HT", 0.14),
    ]
    mixture = prop_db.mix(components)
    products = db.get_species(set(mixture.elements), max_atoms=20)

    problem = EquilibriumProblem(
        reactants=mixture.reactants,
        products=products,
        problem_type=ProblemType.HP,
        constraint1=mixture.enthalpy,
        constraint2=chamber_pressure_pa,
        t_init=3500.0,
    )
    return db, problem


def _configure_case_logger(log_path: Path) -> None:
    logger.remove()
    logger.add(str(log_path), level="DEBUG", enqueue=False)
    logger.add(sys.stderr, level="INFO")


def _collect_metrics(result, ambient_pressure: float) -> Dict[str, float]:
    dh = result.chamber.total_enthalpy - result.exit.total_enthalpy
    v_exit = math.sqrt(max(0.0, 2.0 * dh))
    if result.exit.density * max(v_exit, 1e-30) > 0.0:
        v_pressure = (result.exit.pressure - ambient_pressure) / (
            result.exit.density * v_exit
        )
    else:
        v_pressure = 0.0

    return {
        "cstar_m_s": result.cstar,
        "isp_actual_s": result.isp_actual,
        "isp_vac_s": result.isp_vac,
        "isp_sl_s": result.isp_sl,
        "chamber_t_k": result.chamber.temperature,
        "throat_t_k": result.throat.temperature,
        "exit_t_k": result.exit.temperature,
        "chamber_mgas_g_mol": result.chamber.gas_mean_molar_mass * 1000.0,
        "exit_mgas_g_mol": result.exit.gas_mean_molar_mass * 1000.0,
        "delta_h_j_per_kg": dh,
        "v_exit_m_s": v_exit,
        "v_pressure_m_s": v_pressure,
        "isp_from_decomposition_s": (v_exit + v_pressure) / G0,
    }


def _run_case(
    out_dir: Path,
    include_terra: bool,
    shifting: bool,
    pc_pa: float,
    pe_pa: float,
    ambient_pa: float,
    sp_entropy_mode: str,
) -> Dict[str, Any]:
    case_name = (
        f"terra_{'on' if include_terra else 'off'}"
        f"_{'shifting' if shifting else 'frozen'}"
    )
    log_path = out_dir / f"{case_name}.log"
    _configure_case_logger(log_path)

    logger.info("Starting case {}", case_name)
    logger.info(
        "Inputs: Pc={:.3f} Pa, Pe={:.3f} Pa, Pa={:.3f} Pa, sp_entropy_mode={}",
        pc_pa,
        pe_pa,
        ambient_pa,
        sp_entropy_mode,
    )

    db, problem = _make_problem(include_terra=include_terra, chamber_pressure_pa=pc_pa)
    solver = GordonMcBrideSolver(max_iterations=120)
    perf = PerformanceSolver(solver=solver, db=db, sp_entropy_mode=sp_entropy_mode)
    result = perf.solve(
        problem,
        pe_pa=pe_pa,
        shifting=shifting,
        ambient_pressure=ambient_pa,
    )

    metrics = _collect_metrics(result, ambient_pressure=ambient_pa)
    logger.info("Case {} complete", case_name)
    for key, value in metrics.items():
        logger.info("{} = {:.8g}", key, value)

    if result.profile:
        for idx, point in enumerate(result.profile):
            mach = getattr(point, "mach_number", float("nan"))
            logger.debug(
                "profile[{idx:02d}] P={p:.6e} T={t:.2f} Mgas={m:.6f} Mach={mach:.6f}",
                idx=idx,
                p=point.pressure,
                t=point.temperature,
                m=point.gas_mean_molar_mass * 1000.0,
                mach=mach,
            )

    return {
        "case": case_name,
        "include_terra": include_terra,
        "shifting": shifting,
        "sp_entropy_mode": sp_entropy_mode,
        **metrics,
        "log_file": str(log_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Debug TERRA on/off and shifting/frozen nozzle performance cases."
    )
    parser.add_argument(
        "--pc-psi", type=float, default=1000.0, help="Chamber pressure [psi]."
    )
    parser.add_argument(
        "--pe-pa", type=float, default=101325.0, help="Exit pressure [Pa]."
    )
    parser.add_argument(
        "--ambient-pa",
        type=float,
        default=101325.0,
        help="Ambient pressure for actual Isp [Pa].",
    )
    parser.add_argument(
        "--sp-entropy-mode",
        choices=["total", "total_normalized"],
        default="total_normalized",
        help="Shifting SP entropy mode used by PerformanceSolver.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("tests") / "artifacts" / "nozzle_debug",
        help="Root directory where timestamped debug outputs are written.",
    )
    args = parser.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.output_root / stamp
    out_dir.mkdir(parents=True, exist_ok=True)

    pc_pa = args.pc_psi * PSI_TO_PA
    cases = [
        (True, True),
        (True, False),
        (False, True),
        (False, False),
    ]

    rows: List[Dict[str, Any]] = []
    for include_terra, shifting in cases:
        rows.append(
            _run_case(
                out_dir=out_dir,
                include_terra=include_terra,
                shifting=shifting,
                pc_pa=pc_pa,
                pe_pa=args.pe_pa,
                ambient_pa=args.ambient_pa,
                sp_entropy_mode=args.sp_entropy_mode,
            )
        )

    summary_json = out_dir / "summary.json"
    summary_csv = out_dir / "summary.csv"
    summary_txt = out_dir / "summary.txt"

    with summary_json.open("w", encoding="ascii") as f:
        json.dump(rows, f, indent=2)

    csv_fields = list(rows[0].keys())
    with summary_csv.open("w", newline="", encoding="ascii") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        f"Output directory: {out_dir}",
        "",
        "case                          Isp(actual)  Isp(vac)   Exit T [K]  Exit Mgas [g/mol]",
        "-------------------------------------------------------------------------------",
    ]
    for row in rows:
        lines.append(
            f"{row['case']:<29} {row['isp_actual_s']:>10.3f} {row['isp_vac_s']:>10.3f}"
            f" {row['exit_t_k']:>11.2f} {row['exit_mgas_g_mol']:>17.3f}"
        )
    summary_txt.write_text("\n".join(lines) + "\n", encoding="ascii")

    logger.remove()
    logger.add(sys.stderr, level="INFO")
    logger.info("Wrote debug outputs to {}", out_dir)
    logger.info("Summary: {}", summary_txt)


if __name__ == "__main__":
    main()
