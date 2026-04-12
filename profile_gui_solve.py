"""
profile_gui_solve.py — Profile PerformanceSolver.solve_pair as the GUI executes it.

Replicates the exact call chain used by PerformanceWorker.run() for the KNSB
solid propellant loaded from KNSB.prop.

Usage:
    uv run python profile_gui_solve.py              # wall-clock timing + max_atoms sweep
    uv run python profile_gui_solve.py --cprofile   # cProfile top-40 by cumtime
    uv run python profile_gui_solve.py --snakeviz   # cProfile + open snakeviz browser
"""

from __future__ import annotations

import argparse
import cProfile
import io
import pstats
import time
from pathlib import Path

from loguru import logger

logger.remove()

from prometheus_equilibrium.equilibrium.performance import PerformanceSolver
from prometheus_equilibrium.equilibrium.problem import EquilibriumProblem, ProblemType
from prometheus_equilibrium.equilibrium.solver import (
    GordonMcBrideSolver,
    HybridSolver,
    MajorSpeciesSolver,
)
from prometheus_equilibrium.equilibrium.species import SpeciesDatabase
from prometheus_equilibrium.propellants import PropellantDatabase

# ---------------------------------------------------------------------------
# Configuration — mirrors GUI defaults for KNSB.prop
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent
_THERMO_DIR = _REPO_ROOT / "prometheus_equilibrium" / "thermo_data"
_PROP_TOML = _REPO_ROOT / "prometheus_equilibrium" / "propellants" / "propellants.toml"

# KNSB composition (from KNSB.prop)
COMPONENTS = [
    ("POTASSIUM_NITRATE", 65.0),
    ("SORBITOL", 35.0),
]

# Operating conditions (GUI defaults)
PC_MPA = 6.894757
PE_MPA = 0.101325
AMBIENT_MPA = 0.101325

# Solver to profile ("gmcb", "hybrid", "mss")
SOLVER_KEY = "gmcb"

# Feature flags
COMPUTE_PROFILE = False
CAPTURE_HISTORY = False

# max_atoms for the profiled run (GUI default is 6)
MAX_ATOMS = 6

_PA_PER_MPA = 1_000_000.0


def _build_db():
    db = SpeciesDatabase(
        nasa7_path=str(_THERMO_DIR / "nasa7.json"),
        nasa9_path=str(_THERMO_DIR / "nasa9.json"),
        janaf_path=str(_THERMO_DIR / "janaf.csv"),
        terra_path=str(_THERMO_DIR / "terra.json"),
        afcesic_path=str(_THERMO_DIR / "afcesic.json"),
    )
    db.load(include_janaf=False)
    return db


def _build_solver():
    if SOLVER_KEY == "hybrid":
        return HybridSolver(capture_history=CAPTURE_HISTORY)
    if SOLVER_KEY == "mss":
        return MajorSpeciesSolver(capture_history=CAPTURE_HISTORY)
    return GordonMcBrideSolver(capture_history=CAPTURE_HISTORY)


def _build_mixture(db):
    prop_db = PropellantDatabase(str(_PROP_TOML), species_db=db)
    prop_db.load()
    return prop_db.mix(COMPONENTS)


def _build_problem(db, mixture, max_atoms):
    products = db.get_species(mixture.elements, max_atoms=max_atoms)
    return EquilibriumProblem(
        reactants=mixture.reactants,
        products=products,
        problem_type=ProblemType.HP,
        constraint1=mixture.enthalpy,
        constraint2=PC_MPA * _PA_PER_MPA,
        t_init=3500.0,
    ), len(products)


def _run_solve_pair(db, solver, problem):
    perf = PerformanceSolver(solver, db=db)
    return perf.solve_pair(
        problem,
        pe_pa=PE_MPA * _PA_PER_MPA,
        ambient_pressure=AMBIENT_MPA * _PA_PER_MPA,
        compute_profile=COMPUTE_PROFILE,
    )


def run_timed(db, solver, problem, label="solve_pair") -> float:
    t0 = time.perf_counter()
    result = _run_solve_pair(db, solver, problem)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    s = result.shifting
    f = result.frozen
    print(
        f"  {label:<16} {elapsed_ms:7.1f} ms  |"
        f"  Isp_shift={s.isp_actual:.2f} s"
        f"  Isp_froz={f.isp_actual:.2f} s"
        f"  Tc={s.chamber.temperature:.1f} K"
        f"  Ae/At={s.area_ratio:.3f}"
    )
    return elapsed_ms


def run_cprofile(db, solver, problem):
    pr = cProfile.Profile()
    pr.enable()
    _run_solve_pair(db, solver, problem)
    pr.disable()

    buf = io.StringIO()
    ps = pstats.Stats(pr, stream=buf).sort_stats("cumulative")
    ps.print_stats(40)
    print(buf.getvalue())

    pr.dump_stats("profile_gui_solve.prof")
    print("Profile written to profile_gui_solve.prof")
    print("View with:  uv run snakeviz profile_gui_solve.prof")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cprofile", action="store_true")
    parser.add_argument("--snakeviz", action="store_true")
    args = parser.parse_args()

    print(f"Solver   : {SOLVER_KEY}  (capture_history={CAPTURE_HISTORY})")
    print(
        f"Propellant: KNSB  Pc={PC_MPA} MPa  Pe={PE_MPA} MPa  profile={COMPUTE_PROFILE}"
    )

    print("\nLoading species database...", end="", flush=True)
    db = _build_db()
    print(f" {len(db.species)} species loaded")

    solver = _build_solver()

    print("\nBuilding mixture...", end="", flush=True)
    mixture = _build_mixture(db)
    print(f"  elements={sorted(mixture.elements)}")

    problem, n_species = _build_problem(db, mixture, MAX_ATOMS)
    print(f"\n  Products (max_atoms={MAX_ATOMS}): {n_species} species")

    if args.cprofile or args.snakeviz:
        print(f"\nRunning cProfile (max_atoms={MAX_ATOMS}, {n_species} species)...")
        run_timed(db, solver, problem, "warm-up")
        run_cprofile(db, solver, problem)
        if args.snakeviz:
            import subprocess

            subprocess.Popen(["uv", "run", "snakeviz", "profile_gui_solve.prof"])
        return

    print(f"\nTiming (max_atoms={MAX_ATOMS}, {n_species} species):")
    run_timed(db, solver, problem, "warm-up")
    times = [run_timed(db, solver, problem, f"run {i+1}") for i in range(3)]
    print(f"\n  mean={sum(times)/3:.0f} ms  min={min(times):.0f} ms")


if __name__ == "__main__":
    main()
