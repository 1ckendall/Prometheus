"""
compare_solvers.py — Speed and robustness comparison: G-McB vs Hybrid vs MajorSpecies.

Tests H₂/O₂ and CH₄/O₂ across a grid of O/F ratios and pressures and reports:
  - Iteration count per solver
  - Wall-clock time per case
  - Convergence rate
  - Temperature accuracy relative to GordonMcBrideSolver ground truth

Run with:
    uv run python compare_solvers.py
"""

from __future__ import annotations

import math
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
from loguru import logger

logger.remove()  # suppress all loguru output during benchmark

from prometheus_equilibrium.equilibrium.problem import EquilibriumProblem, ProblemType
from prometheus_equilibrium.equilibrium.solver import (
    GordonMcBrideSolver,
    HybridSolver,
    MajorSpeciesSolver,
)
from prometheus_equilibrium.equilibrium.species import SpeciesDatabase

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent
_THERMO_DIR = _REPO_ROOT / "prometheus_equilibrium" / "thermo_data"

_DB = SpeciesDatabase(
    nasa7_path=str(_THERMO_DIR / "nasa7.json"),
    nasa9_path=str(_THERMO_DIR / "nasa9.json"),
    janaf_path=str(_THERMO_DIR / "janaf.csv"),
)
_DB.load(include_janaf=False)

SOLVERS = {
    "G-McB": GordonMcBrideSolver(),
    "Hybrid": HybridSolver(),
    "MajorSpecies": MajorSpeciesSolver(),
}

# Separate timing for the seed TP step vs G-McB main in Hybrid
_SEED_ONLY = MajorSpeciesSolver(max_iterations=10, tolerance=5e-6)

PSIA_TO_PA = 6894.757

# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------


@dataclass
class Case:
    name: str
    fuel_formula: str
    ox_formula: str
    elements: set
    of: float
    pressure_pa: float


def _build_cases() -> List[Case]:
    cases = []
    of_h2 = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0, 16.0, 20.0]
    pressures = [50.0, 200.0, 500.0, 1000.0, 3000.0]
    for of in of_h2:
        for p in pressures:
            cases.append(Case("H2/O2", "H2_G", "O2_G", {"H", "O"}, of, p * PSIA_TO_PA))
    of_ch4 = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0]
    for of in of_ch4:
        for p in pressures:
            cases.append(
                Case("CH4/O2", "CH4_G", "O2_G", {"C", "H", "O"}, of, p * PSIA_TO_PA)
            )
    return cases


def _make_problem(case: Case) -> Optional[EquilibriumProblem]:
    try:
        fuel = _DB[case.fuel_formula]
        ox = _DB[case.ox_formula]
    except KeyError:
        return None

    mw_fuel = fuel.molar_mass()
    mw_ox = ox.molar_mass()
    # Convert O/F mass ratio to mole amounts: 1 mol fuel, N mol ox
    n_ox = (case.of * mw_fuel) / mw_ox

    products = _DB.get_species(case.elements, max_atoms=20)
    H0 = fuel.enthalpy(298.15) * 1.0 + ox.enthalpy(298.15) * n_ox

    return EquilibriumProblem(
        reactants={fuel: 1.0, ox: n_ox},
        products=products,
        problem_type=ProblemType.HP,
        constraint1=H0,
        constraint2=case.pressure_pa,
        t_init=3500.0,
    )


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------


@dataclass
class Result:
    solver: str
    case_name: str
    of: float
    pressure_psia: float
    converged: bool
    iters: int
    time_ms: float
    temperature: float


def _run_all(cases: List[Case], n_warmup: int = 2) -> List[Result]:
    results: List[Result] = []
    problems = [_make_problem(c) for c in cases]

    # Warm up JIT / import paths
    for _ in range(n_warmup):
        for name, solver in SOLVERS.items():
            p = problems[0]
            if p is not None:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    solver.solve(p, log_failure=False)

    for case, prob in zip(cases, problems):
        if prob is None:
            continue
        for name, solver in SOLVERS.items():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                t0 = time.perf_counter()
                sol = solver.solve(prob, log_failure=False)
                elapsed = (time.perf_counter() - t0) * 1000.0

            results.append(
                Result(
                    solver=name,
                    case_name=case.name,
                    of=case.of,
                    pressure_psia=case.pressure_pa / PSIA_TO_PA,
                    converged=sol.converged,
                    iters=sol.iterations,
                    time_ms=elapsed,
                    temperature=sol.temperature if sol.converged else float("nan"),
                )
            )

    return results


# ---------------------------------------------------------------------------
# Statistics and reporting
# ---------------------------------------------------------------------------


def _stats(vals: List[float]):
    """Return (mean, std, min, max) ignoring NaNs."""
    arr = [v for v in vals if math.isfinite(v)]
    if not arr:
        return float("nan"), float("nan"), float("nan"), float("nan")
    a = np.array(arr)
    return float(a.mean()), float(a.std()), float(a.min()), float(a.max())


def _print_summary(results: List[Result]) -> None:
    solver_names = list(SOLVERS.keys())
    propellant_systems = sorted({r.case_name for r in results})

    # Per-solver aggregates
    print("\n" + "=" * 70)
    print("OVERALL SUMMARY (all cases)")
    print("=" * 70)
    hdr = f"{'Solver':<16} {'Converged':>9} {'AvgIter':>8} {'AvgTime(ms)':>12} {'MaxTime(ms)':>12}"
    print(hdr)
    print("-" * 70)

    gmcb_temps: dict = {}  # case_key → G-McB temperature (ground truth)
    for r in results:
        if r.solver == "G-McB" and r.converged:
            gmcb_temps[(r.case_name, r.of, r.pressure_psia)] = r.temperature

    for sname in solver_names:
        rs = [r for r in results if r.solver == sname]
        n_conv = sum(1 for r in rs if r.converged)
        iter_mean, _, _, _ = _stats([r.iters for r in rs if r.converged])
        t_mean, _, _, t_max = _stats([r.time_ms for r in rs])
        print(
            f"{sname:<16} {n_conv:>5}/{len(rs):<4} {iter_mean:>8.1f} {t_mean:>12.2f} {t_max:>12.2f}"
        )

    # Temperature accuracy vs G-McB
    print("\n" + "=" * 70)
    print("TEMPERATURE ACCURACY vs G-McB (converged cases, |dT| [K])")
    print("=" * 70)
    print(f"{'Solver':<16} {'MeanErr':>9} {'MaxErr':>9} {'RMSErr':>9}")
    print("-" * 50)
    for sname in solver_names:
        if sname == "G-McB":
            continue
        dts = []
        for r in results:
            if r.solver != sname or not r.converged:
                continue
            key = (r.case_name, r.of, r.pressure_psia)
            ref = gmcb_temps.get(key)
            if ref is not None and math.isfinite(ref):
                dts.append(abs(r.temperature - ref))
        if not dts:
            continue
        a = np.array(dts)
        print(
            f"{sname:<16} {a.mean():>9.4f} {a.max():>9.4f} {float(np.sqrt((a**2).mean())):>9.4f}"
        )

    # Per-propellant breakdown
    for prop in propellant_systems:
        print(f"\n{'='*70}")
        print(f"PROPELLANT: {prop}")
        print(f"{'='*70}")
        print(
            f"{'Solver':<16} {'Converged':>9} {'AvgIter':>8} {'SeedIter*':>10} {'AvgTime(ms)':>12}"
        )
        print("-" * 70)
        for sname in solver_names:
            rs = [r for r in results if r.solver == sname and r.case_name == prop]
            n_conv = sum(1 for r in rs if r.converged)
            iter_mean, _, _, _ = _stats([r.iters for r in rs if r.converged])
            t_mean, _, _, _ = _stats([r.time_ms for r in rs])
            print(
                f"{sname:<16} {n_conv:>5}/{len(rs):<4} {iter_mean:>8.1f} {'':>10} {t_mean:>12.2f}"
            )
        print("* Hybrid 'AvgIter' = seed TP iters + G-McB iters combined")

    # Iteration breakdown: Hybrid seed cost vs G-McB saving
    print(f"\n{'='*70}")
    print("ITERATION BREAKDOWN: Hybrid vs G-McB")
    print("  Hybrid total iters = MSS-TP iters + G-McB-main iters")
    print("  Reduction = G-McB cold iters - Hybrid G-McB-main iters")
    print(f"{'='*70}")
    hybrid_iters = {
        (r.case_name, r.of, r.pressure_psia): r.iters
        for r in results
        if r.solver == "Hybrid" and r.converged
    }
    gmcb_iters = {
        (r.case_name, r.of, r.pressure_psia): r.iters
        for r in results
        if r.solver == "G-McB" and r.converged
    }
    savings = []
    for key in set(hybrid_iters) & set(gmcb_iters):
        savings.append(gmcb_iters[key] - hybrid_iters[key])
    if savings:
        s = np.array(savings)
        print(
            f"  G-McB cold iters saved by seeding: mean={s.mean():.1f}  "
            f"min={s.min():.0f}  max={s.max():.0f}"
        )
        print(f"  (negative = Hybrid used MORE iterations total; " f"positive = fewer)")

    # Timing speedup
    print(f"\n{'='*70}")
    print("WALL-CLOCK SPEEDUP: Hybrid vs G-McB (median over all cases)")
    print(f"{'='*70}")
    hybrid_times = {
        (r.case_name, r.of, r.pressure_psia): r.time_ms
        for r in results
        if r.solver == "Hybrid"
    }
    gmcb_times = {
        (r.case_name, r.of, r.pressure_psia): r.time_ms
        for r in results
        if r.solver == "G-McB"
    }
    ratios = []
    for key in set(hybrid_times) & set(gmcb_times):
        g = gmcb_times[key]
        h = hybrid_times[key]
        if g > 0:
            ratios.append(g / h)
    if ratios:
        r_arr = np.array(ratios)
        print(
            f"  G-McB time / Hybrid time: "
            f"median={float(np.median(r_arr)):.3f}x  "
            f"mean={r_arr.mean():.3f}x  "
            f"min={r_arr.min():.3f}x  max={r_arr.max():.3f}x"
        )
        print("  (>1.0 means Hybrid is faster; <1.0 means G-McB is faster)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _show_t_correction_quality() -> None:
    """Show how well the Newton T correction approximates T_adiabatic."""
    print("\n" + "=" * 70)
    print("T CORRECTION QUALITY: T_est vs T_adiabatic (G-McB reference)")
    print("  T_est = T_init - (H_seed(T_init) - H0) / Cp_seed(T_init)")
    print("=" * 70)
    print(
        f"{'System':<10} {'O/F':>5} {'T_init':>7} {'T_est':>7} {'T_GMcB':>7} {'T_err_K':>8} {'GMcB_iters':>11}"
    )
    print("-" * 65)

    test_cases = [
        ("H2/O2", "H2_G", "O2_G", {"H", "O"}, 6.0, 1000 * PSIA_TO_PA),
        ("H2/O2", "H2_G", "O2_G", {"H", "O"}, 2.0, 1000 * PSIA_TO_PA),
        ("H2/O2", "H2_G", "O2_G", {"H", "O"}, 20.0, 1000 * PSIA_TO_PA),
        ("CH4/O2", "CH4_G", "O2_G", {"C", "H", "O"}, 3.5, 1000 * PSIA_TO_PA),
        ("CH4/O2", "CH4_G", "O2_G", {"C", "H", "O"}, 1.5, 1000 * PSIA_TO_PA),
    ]

    gmcb = GordonMcBrideSolver()
    seed = MajorSpeciesSolver(max_iterations=10, tolerance=5e-6)
    import warnings

    for sys_name, fuel_f, ox_f, elems, of, P in test_cases:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                fuel = _DB[fuel_f]
                ox = _DB[ox_f]
            except KeyError:
                continue
            n_ox = (of * fuel.molar_mass()) / ox.molar_mass()
            H0 = fuel.enthalpy(298.15) + ox.enthalpy(298.15) * n_ox
            products = _DB.get_species(elems, max_atoms=20)
            T_init = 3500.0

            prob = EquilibriumProblem(
                reactants={fuel: 1.0, ox: n_ox},
                products=products,
                problem_type=ProblemType.HP,
                constraint1=H0,
                constraint2=P,
                t_init=T_init,
            )
            tp_prob = EquilibriumProblem(
                reactants={fuel: 1.0, ox: n_ox},
                products=products,
                problem_type=ProblemType.TP,
                constraint1=T_init,
                constraint2=P,
                t_init=T_init,
            )

            seed_sol = seed.solve(tp_prob, log_failure=False)
            if float(seed_sol.mixture.total_gas_moles) > 0:
                f = seed_sol.mixture.total_enthalpy(T_init) - H0
                fp = seed_sol.mixture.total_cp(T_init)
                dT = -f / fp if abs(fp) > 1e-30 else 0.0
                dT = max(-1500.0, min(1500.0, dT))
                T_est = max(200.0, min(6000.0, T_init + dT))
            else:
                T_est = T_init

            ref_sol = gmcb.solve(prob, log_failure=False)
            T_ref = ref_sol.temperature

            # G-McB with T_est as t_init + composition seed
            seeded_prob = EquilibriumProblem(
                reactants={fuel: 1.0, ox: n_ox},
                products=products,
                problem_type=ProblemType.HP,
                constraint1=H0,
                constraint2=P,
                t_init=T_est,
            )
            seeded_sol = gmcb.solve(
                seeded_prob, guess=seed_sol.mixture, log_failure=False
            )

            print(
                f"{sys_name:<10} {of:>5.1f} {T_init:>7.0f} {T_est:>7.1f} "
                f"{T_ref:>7.1f} {abs(T_est - T_ref):>8.2f} {seeded_sol.iterations:>11}"
            )


if __name__ == "__main__":
    import sys

    _show_t_correction_quality()

    print("\nBuilding test cases...")
    cases = _build_cases()
    print(
        f"  {len(cases)} cases x {len(SOLVERS)} solvers = {len(cases)*len(SOLVERS)} solves"
    )

    print("Running benchmark (2 warm-up passes)...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results = _run_all(cases, n_warmup=2)

    _print_summary(results)

    # Per-case detail for first propellant system at one pressure
    print(f"\n{'='*70}")
    print("PER-CASE DETAIL: H2/O2 @ 1000 psia")
    print(f"{'='*70}")
    target_p = 1000.0
    detail = [
        r
        for r in results
        if r.case_name == "H2/O2" and abs(r.pressure_psia - target_p) < 1.0
    ]
    print(
        f"{'Solver':<16} {'O/F':>6} {'Iters':>6} {'Time(ms)':>10} {'T(K)':>8} {'Conv':>5}"
    )
    print("-" * 60)
    for r in sorted(detail, key=lambda x: (x.solver, x.of)):
        t_str = f"{r.temperature:.1f}" if math.isfinite(r.temperature) else " n/c "
        print(
            f"{r.solver:<16} {r.of:>6.1f} {r.iters:>6} {r.time_ms:>10.2f} {t_str:>8} {'Y' if r.converged else 'N':>5}"
        )
