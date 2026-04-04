"""
Diagnostic script: Verify that shifting equilibrium Isp is consistent
regardless of whether the TERRA database is loaded.

Run with:  uv run python tests/diagnose_shifting_condensed.py
"""

import math
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore", category=RuntimeWarning)

from prometheus_equilibrium.equilibrium.performance import PerformanceSolver
from prometheus_equilibrium.equilibrium.problem import EquilibriumProblem, ProblemType
from prometheus_equilibrium.equilibrium.solver import GordonMcBrideSolver
from prometheus_equilibrium.equilibrium.species import SpeciesDatabase
from prometheus_equilibrium.propellants.loader import PropellantDatabase

REPO_ROOT = Path(__file__).resolve().parents[1]
PSI_TO_PA = 6894.757


def _t_range(sp):
    if hasattr(sp, "T_low") and hasattr(sp, "T_high"):
        return (sp.T_low, sp.T_high)
    if hasattr(sp, "temperatures") and len(sp.temperatures) >= 2:
        return (sp.temperatures[0], sp.temperatures[-1])
    return (None, None)


def run_performance(include_terra: bool):
    label = "WITH TERRA" if include_terra else "WITHOUT TERRA"
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")

    db = SpeciesDatabase()
    db.load(include_janaf=False, include_terra=include_terra, include_afcesic=False)

    prop_db = PropellantDatabase(
        str(REPO_ROOT / "prometheus_equilibrium" / "propellants" / "propellants.toml"),
        species_db=db,
    )
    prop_db.load()

    mixture = prop_db.mix([
        ("AMMONIUM_PERCHLORATE", 0.68),
        ("ALUMINUM_PURE_CRYSTALINE", 0.18),
        ("HTPB_R_45HT", 0.14),
    ])

    products = db.get_species(mixture.elements, max_atoms=20)
    problem = EquilibriumProblem(
        reactants=mixture.reactants,
        products=products,
        problem_type=ProblemType.HP,
        constraint1=mixture.enthalpy,
        constraint2=1000.0 * PSI_TO_PA,
        t_init=3500.0,
    )

    solver = GordonMcBrideSolver(max_iterations=120)
    perf = PerformanceSolver(solver, db=db)

    # Count Al2O3 condensed species in products
    al2o3_count = sum(
        1 for sp in products
        if sp.condensed == 1
        and sp.elements.get("Al", 0) == 2
        and sp.elements.get("O", 0) == 3
    )
    print(f"  Al2O3 condensed species in products: {al2o3_count}")

    results = {}
    for mode in ("shifting", "frozen"):
        try:
            result = perf.solve(problem, pe_pa=101325.0, shifting=(mode == "shifting"))
            mbar = result.exit.gas_mean_molar_mass * 1000
            print(f"\n  {mode.upper()}:")
            print(f"    Chamber T = {result.chamber.temperature:.2f} K")
            print(f"    Exit T    = {result.exit.temperature:.2f} K")
            print(f"    Exit Mbar    = {mbar:.3f} g/mol")
            print(f"    Isp_vac   = {result.isp_vac:.2f} m/s")
            print(f"    Converged = {result.exit.converged}")

            # Check for condensed species in exit
            n_cond = sum(
                1 for sp, n in zip(result.exit.mixture.species, result.exit.mixture.moles)
                if sp.condensed == 1 and n > 1e-10
            )
            print(f"    Active condensed species at exit: {n_cond}")

            results[mode] = result
        except Exception as e:
            print(f"\n  {mode.upper()}: FAILED — {e}")
            results[mode] = None

    return results


if __name__ == "__main__":
    r_no_terra = run_performance(include_terra=False)
    r_with_terra = run_performance(include_terra=True)

    print(f"\n{'='*60}")
    print("  COMPARISON")
    print(f"{'='*60}")

    for mode in ("shifting", "frozen"):
        r1 = r_no_terra.get(mode)
        r2 = r_with_terra.get(mode)
        if r1 and r2:
            isp_diff = abs(r1.isp_vac - r2.isp_vac)
            isp_pct = isp_diff / r2.isp_vac * 100
            mbar1 = r1.exit.gas_mean_molar_mass * 1000
            mbar2 = r2.exit.gas_mean_molar_mass * 1000
            print(f"\n  {mode.upper()}:")
            print(f"    Isp_vac: {r1.isp_vac:.2f} vs {r2.isp_vac:.2f} m/s "
                  f"(diff={isp_diff:.2f}, {isp_pct:.2f}%)")
            print(f"    Exit Mbar: {mbar1:.3f} vs {mbar2:.3f} g/mol")
            print(f"    Exit T:  {r1.exit.temperature:.2f} vs {r2.exit.temperature:.2f} K")
            if isp_pct < 2.0:
                print(f"    STATUS: OK (< 2% difference)")
            else:
                print(f"    STATUS: CONCERN (> 2% difference)")
