"""
ch4_o2_worked_example.py — Detailed CH4/O2 solve trace for documentation.

Runs an HP equilibrium for methane/oxygen at stoichiometric O/F=4.0, 1000 psia
with both MajorSpeciesSolver and GordonMcBrideSolver.  Captures the full
convergence history and prints a structured trace suitable for use as a
documentation worked example.

Run from repo root:
    uv run python docs/scripts/ch4_o2_worked_example.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

from loguru import logger

# ---------------------------------------------------------------------------
# Silence loguru's default stderr handler; we configure our own below.
# ---------------------------------------------------------------------------
logger.remove()
_log_records: list[dict] = []


def _capture_handler(message):
    record = message.record
    _log_records.append(
        {
            "solver": record["extra"].get("solver", "?"),
            "level": record["level"].name,
            "message": record["message"],
        }
    )


logger.add(
    _capture_handler,
    level="DEBUG",
    format="{message}",
    filter=lambda r: "solver" in r["extra"],
)

# ---------------------------------------------------------------------------
# Prometheus imports
# ---------------------------------------------------------------------------
_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO))

from prometheus_equilibrium.equilibrium.problem import EquilibriumProblem, ProblemType
from prometheus_equilibrium.equilibrium.solver import (
    GordonMcBrideSolver,
    MajorSpeciesSolver,
)
from prometheus_equilibrium.equilibrium.species import SpeciesDatabase

# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------
_THERMO = _REPO / "prometheus_equilibrium" / "thermo_data"
db = SpeciesDatabase(
    nasa7_path=str(_THERMO / "nasa7.json"),
    nasa9_path=str(_THERMO / "nasa9.json"),
    janaf_path=str(_THERMO / "janaf.csv"),
)
db.load(include_janaf=False)

# ---------------------------------------------------------------------------
# Problem definition:  CH4 / O2,  O/F = 4.0 (mass),  P = 1000 psia ≈ 6.895 MPa
#
# Stoichiometry:  CH4 + 2 O2 → CO2 + 2 H2O   (complete combustion)
# O/F = 4.0 (mass):  1 mol CH4 × 16 g/mol,  n_O2 × 32 g/mol = 4 × 16 g
#   → n_O2 = 64/32 = 2.0 mol   (exactly stoichiometric)
# ---------------------------------------------------------------------------
P_PA = 1000.0 * 6894.757  # 1000 psia → Pa

ch4 = db.find("CH4")
o2 = db.find("O2")
T_REF = 298.15  # K

n_ch4 = 1.0
n_o2 = 2.0  # stoichiometric

H0 = ch4.enthalpy(T_REF) * n_ch4 + o2.enthalpy(T_REF) * n_o2

products = db.get_species({"C", "H", "O"}, max_atoms=20)

problem = EquilibriumProblem(
    reactants={ch4: n_ch4, o2: n_o2},
    products=products,
    problem_type=ProblemType.HP,
    constraint1=H0,
    constraint2=P_PA,
    t_init=3500.0,
)

print("=" * 72)
print("CH4 / O2 — HP Equilibrium   O/F=4.0 (mass),  P=1000 psia")
print(f"  CH4 enthalpy at {T_REF} K:  {ch4.enthalpy(T_REF):.2f} J/mol")
print(f"  O2  enthalpy at {T_REF} K:  {o2.enthalpy(T_REF):.2f} J/mol")
print(f"  Total H0 (reactants):        {H0:.2f} J")
print(f"  Pressure:                    {P_PA:.0f} Pa  ({P_PA/1e6:.4f} MPa)")
print(f"  Initial T guess:             {problem.t_init:.0f} K")
print(f"  Number of candidate species: {len(products)}")
print("=" * 72)

# ---------------------------------------------------------------------------
# Run each solver and capture the convergence history
# ---------------------------------------------------------------------------


def run_solver(name: str, solver):
    global _log_records
    _log_records = []

    sol = solver.solve(problem)
    logs = list(_log_records)  # copy before next run clears
    return sol, logs


major = MajorSpeciesSolver(capture_history=True)
gmcb = GordonMcBrideSolver(capture_history=True)

sol_major, logs_major = run_solver("MajorSpeciesSolver", major)
sol_gmcb, logs_gmcb = run_solver("GordonMcBrideSolver", gmcb)


# ---------------------------------------------------------------------------
# Pretty-print convergence history
# ---------------------------------------------------------------------------


def print_history(solver_name: str, sol, logs: list[dict]):
    print()
    print(f"{'-'*72}")
    print(f"  {solver_name}")
    print(f"{'-'*72}")
    print(f"  Converged : {sol.converged}")
    print(f"  Iterations: {sol.iterations}")
    print(f"  T_final   : {sol.temperature:.4f} K")
    print(f"  P         : {sol.pressure/1e6:.4f} MPa")
    print(f"  M̄ (gas)   : {sol.gas_mean_molar_mass*1000:.4f} g/mol")
    print(f"  γ (frozen): {sol.gamma:.5f}")
    print(f"  Cp        : {sol.cp:.4f} J/(mol·K)")
    print(f"  a (frozen): {sol.speed_of_sound:.2f} m/s")
    print(f"  El. balance error: {sol.element_balance_error:.3e}")
    print()
    print(f"  {'':4s}  Major species at equilibrium (xⱼ ≥ 0.01%):")
    for sp, x in sol.major_species(threshold=1e-4).items():
        print(f"    {sp:20s}  {x:.6f}")

    print()
    print("  Outer temperature-search history (every outer step):")
    print(f"  {'Step':>4}  {'T [K]':>10}  {'|ΔlnT|':>12}  Major species snapshot")
    print(f"  {'-'*4}  {'-'*10}  {'-'*12}  {'-'*30}")
    if sol.history:
        for i, step in enumerate(sol.history):
            top = sorted(step.mole_fractions.items(), key=lambda kv: -kv[1])[:3]
            top_str = "  ".join(f"{k}={v:.4f}" for k, v in top)
            print(
                f"  {i:>4d}  {step.temperature:>10.3f}  {step.max_residual:>12.4e}  {top_str}"
            )

    print()
    print("  Solver debug log (first 25 lines + last 5):")
    subset = logs[:25] + (["  ..."] if len(logs) > 30 else []) + logs[-5:]
    for entry in subset:
        if isinstance(entry, str):
            print(entry)
        else:
            print(f"  [{entry['solver']:14s}] {entry['message']}")


print_history("MajorSpeciesSolver", sol_major, logs_major)
print_history("GordonMcBrideSolver", sol_gmcb, logs_gmcb)

# ---------------------------------------------------------------------------
# Side-by-side comparison
# ---------------------------------------------------------------------------
print()
print("=" * 72)
print("  Side-by-side comparison")
print("=" * 72)
print(f"  {'Property':<24}  {'Major-Species':>16}  {'G-McBride':>16}")
print(f"  {'-'*24}  {'-'*16}  {'-'*16}")
rows = [
    ("T [K]", f"{sol_major.temperature:.4f}", f"{sol_gmcb.temperature:.4f}"),
    (
        "M̄_gas [g/mol]",
        f"{sol_major.gas_mean_molar_mass*1000:.4f}",
        f"{sol_gmcb.gas_mean_molar_mass*1000:.4f}",
    ),
    ("γ (frozen)", f"{sol_major.gamma:.5f}", f"{sol_gmcb.gamma:.5f}"),
    ("Cp [J/(mol·K)]", f"{sol_major.cp:.4f}", f"{sol_gmcb.cp:.4f}"),
    ("a [m/s]", f"{sol_major.speed_of_sound:.2f}", f"{sol_gmcb.speed_of_sound:.2f}"),
    ("iterations", str(sol_major.iterations), str(sol_gmcb.iterations)),
    ("converged", str(sol_major.converged), str(sol_gmcb.converged)),
    (
        "el. balance error",
        f"{sol_major.element_balance_error:.2e}",
        f"{sol_gmcb.element_balance_error:.2e}",
    ),
]
for label, v_maj, v_gmc in rows:
    print(f"  {label:<24}  {v_maj:>16}  {v_gmc:>16}")

print()

# ---------------------------------------------------------------------------
# Detailed inner-iteration trace for MajorSpeciesSolver
# (extract from captured debug logs)
# ---------------------------------------------------------------------------
print("=" * 72)
print("  MajorSpeciesSolver inner-iteration log (from loguru debug output)")
print("=" * 72)
for entry in logs_major:
    if isinstance(entry, dict):
        print(f"  {entry['message']}")
print()

print("=" * 72)
print("  GordonMcBrideSolver inner-iteration log")
print("=" * 72)
for entry in logs_gmcb:
    if isinstance(entry, dict):
        print(f"  {entry['message']}")
print()
