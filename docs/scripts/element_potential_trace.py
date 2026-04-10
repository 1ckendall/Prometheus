"""
element_potential_trace.py — Detailed element-potential trace for documentation.

Runs two HP cases and captures π (element potentials), the Newton matrix,
and species-level detail at every iteration step:

  1. H2/O2  stoichiometric (2:1 mol), P=1000 psia — S=2 (H, O)
  2. CH4/O2 stoichiometric (1:2 mol), P=1000 psia — S=3 (C, H, O)

For each case the MajorSpeciesSolver is patched to print:
  - The assembled (S+1)x(S+1) Jacobian G and RHS at iteration 0 and 1
  - π (element potentials) at each inner iteration
  - The top 5 species mole fractions at each inner iteration
  - The minor-species formula applied to OH/HO/H/O at iteration 0 and final

Run with:
    PYTHONIOENCODING=utf-8 uv run python docs/scripts/element_potential_trace.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np

_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO))

from prometheus_equilibrium.core.constants import REFERENCE_PRESSURE as _P_REF
from prometheus_equilibrium.core.constants import UNIVERSAL_GAS_CONSTANT as _R
from prometheus_equilibrium.equilibrium.element_matrix import ElementMatrix
from prometheus_equilibrium.equilibrium.mixture import Mixture
from prometheus_equilibrium.equilibrium.problem import EquilibriumProblem, ProblemType
from prometheus_equilibrium.equilibrium.solver import (
    GordonMcBrideSolver,
    MajorSpeciesSolver,
)
from prometheus_equilibrium.equilibrium.solution import ConvergenceStep
from prometheus_equilibrium.equilibrium.species import SpeciesDatabase

# ── Database ──────────────────────────────────────────────────────────────────
_THERMO = _REPO / "prometheus_equilibrium" / "thermo_data"
db = SpeciesDatabase(
    nasa7_path=str(_THERMO / "nasa7.json"),
    nasa9_path=str(_THERMO / "nasa9.json"),
    janaf_path=str(_THERMO / "janaf.csv"),
)
db.load(include_janaf=False)

P_PA = 1000.0 * 6894.757  # 1000 psia
T_REF = 298.15


# ── Instrumented subclass ─────────────────────────────────────────────────────

class TracingMajorSpeciesSolver(MajorSpeciesSolver):
    """Subclass that prints full internal state at each inner iteration."""

    def __init__(self, label: str, max_T_outer: int = 1):
        super().__init__(capture_history=True)
        self.label = label
        self.max_T_outer = max_T_outer   # stop after this many outer T steps
        self._outer_count = 0

    def _tp_equilibrium(
        self,
        mixture: Mixture,
        element_matrix: ElementMatrix,
        b0: np.ndarray,
        T: float,
        P: float,
    ):
        """Override to instrument the inner Newton loop."""
        em = element_matrix
        S = em.n_elements
        pi = np.zeros(S)
        n_gas = mixture.n_gas
        A_gas = em.gas_rows()
        history: List[ConvergenceStep] = []

        self._last_failure_reason = None
        self._last_step_norm = float("inf")

        self._outer_count += 1
        if self._outer_count > self.max_T_outer:
            # delegate to parent without tracing
            return super()._tp_equilibrium(mixture, element_matrix, b0, T, P)

        sep = "─" * 72
        print(f"\n{sep}")
        print(f"  [{self.label}] Inner Newton solve at T = {T:.2f} K")
        print(f"  Elements: {em.elements}   S = {S}")
        print(f"  b0 (target gram-atoms): {dict(zip(em.elements, b0))}")
        print(f"  Gas species in mixture: {n_gas}")
        print(sep)

        n_inner = 0
        _tp_converged = False

        for n_inner in range(self.max_iterations):
            n_gas_arr = mixture.gas_moles()
            n_gas_total = float(n_gas_arr.sum())
            n_total = mixture.total_moles

            mu_gas = GordonMcBrideSolver._reduced_chemical_potentials(
                mixture.species[:n_gas], n_gas_arr, n_gas_total, T, P
            )

            basis_indices, _ = em.select_basis(mixture.moles)

            _n_basis_floor = math.exp(
                math.log(1e-8) + math.log(max(n_gas_total, 1e-300))
            )
            for _bid in basis_indices:
                if mixture.moles[_bid] <= 0.0:
                    mixture.moles[_bid] = _n_basis_floor
            n_gas_arr = mixture.gas_moles()
            n_gas_total = float(n_gas_arr.sum())

            major_nonbasis, minor_all = self._split_major_minor(
                mixture.moles, basis_indices
            )

            major_mask = np.zeros(n_gas, dtype=bool)
            for i in basis_indices:
                if i < n_gas:
                    major_mask[i] = True
            for i in major_nonbasis:
                if i < n_gas:
                    major_mask[i] = True
            major_gas_indices = np.flatnonzero(major_mask)
            minor_gas_indices = [
                i for i in minor_all if i < n_gas and mixture.species[i].condensed == 0
            ]

            element_res = em.element_residuals(mixture.moles, b0)

            # ── Print iteration header ──────────────────────────────────────
            print(f"\n  --- Inner iteration {n_inner} ---")
            print(f"  n_gas_total = {n_gas_total:.6f}  n_total = {n_total:.6f}")
            print(f"  Basis species: {[mixture.species[i].formula for i in basis_indices]}")
            print(f"  Major non-basis gas: {[mixture.species[i].formula for i in major_gas_indices if i not in set(basis_indices)]}")
            print(f"  Minor gas count: {len(minor_gas_indices)}")
            print()

            # Top 8 species by mole amount
            top_idx = sorted(range(n_gas), key=lambda i: -mixture.moles[i])[:8]
            print(f"  {'Species':20s}  {'n_j':>12}  {'x_j':>10}  {'mu_j(g/RT)':>12}")
            print(f"  {'-'*20}  {'-'*12}  {'-'*10}  {'-'*12}")
            for i in top_idx:
                sp = mixture.species[i]
                xj = mixture.moles[i] / max(n_gas_total, 1e-300)
                print(f"  {sp.formula:20s}  {mixture.moles[i]:12.6f}  {xj:10.6f}  {mu_gas[i]:12.6f}")

            # ── Assemble Jacobian ───────────────────────────────────────────
            active_cnd_local = self._active_condensed_indices(mixture)
            nc = len(active_cnd_local)
            A_maj = A_gas[major_gas_indices, :]
            n_maj = n_gas_arr[major_gas_indices]
            mu_maj = mu_gas[major_gas_indices]
            n_maj_total = float(n_maj.sum())
            A_cnd_act = np.zeros((0, S))
            mu_cnd = np.zeros(0)

            size = S + nc + 1
            G = np.zeros((size, size + 1))
            idx_n = S + nc

            G[:S, :S] = A_maj.T @ (n_maj[:, None] * A_maj)
            G[:S, idx_n] = A_maj.T @ n_maj
            G[idx_n, :S] = A_maj.T @ n_maj
            G[idx_n, idx_n] = n_maj_total - n_total
            G[:S, -1] = A_maj.T @ (n_maj * mu_maj) + element_res
            G[idx_n, -1] = n_total - n_maj_total + float(n_maj @ mu_maj)

            print(f"\n  Newton matrix G (rows = equations, last col = RHS)")
            print(f"  Size: {size} x {size+1}  (S={S}, nc={nc})")
            el_labels = list(em.elements) + ["n"]
            rhs_labels = list(em.elements) + ["n_tot"]
            header = "  " + f"{'':12s}" + "".join(f"  {L:>12s}" for L in el_labels) + f"  {'RHS':>12s}"
            print(header)
            print("  " + "-" * (14 + 14*size + 14))
            for row_i, row_label in enumerate(rhs_labels):
                row_str = "  " + f"{row_label:12s}" + "".join(f"  {G[row_i,col]:12.5g}" for col in range(size)) + f"  {G[row_i,-1]:12.5g}"
                print(row_str)

            # ── Solve ───────────────────────────────────────────────────────
            try:
                delta_x = np.linalg.solve(G[:, :-1], G[:, -1])
            except np.linalg.LinAlgError:
                print("  [SINGULAR — breaking]")
                break

            pi = delta_x[:S]
            delta_ln_n = delta_x[idx_n]

            print(f"\n  Solution: pi = {dict(zip(em.elements, pi))}")
            print(f"  delta_ln(n) = {delta_ln_n:.6f}")

            # ── Show minor species formula ──────────────────────────────────
            if minor_gas_indices:
                print(f"\n  Minor species update (from pi):")
                ln_n_gas = math.log(max(n_gas_total, 1e-300))
                ln_P_ratio = math.log(P / _P_REF)
                A = em.matrix
                shown = 0
                for idx in minor_gas_indices:
                    sp = mixture.species[idx]
                    g0 = sp.reduced_gibbs(T)
                    atom_pi = float(A[idx, :] @ pi)
                    ln_n_eq = atom_pi - g0 - ln_P_ratio + ln_n_gas
                    n_eq = math.exp(min(ln_n_eq, 700.0))
                    formula_parts = " + ".join(
                        f"{v:.0f}*pi_{k}" if v != 1 else f"pi_{k}"
                        for k, v in zip(em.elements, A[idx, :])
                        if abs(v) > 0
                    )
                    print(
                        f"    {sp.formula:12s}: ln(n) = {formula_parts} - g°/RT - ln(P/P°) + ln(n)"
                        f"  = {atom_pi:.4f} - {g0:.4f} - {ln_P_ratio:.4f} + {ln_n_gas:.4f}"
                        f"  = {ln_n_eq:.4f}  =>  n = {n_eq:.4e}"
                    )
                    shown += 1
                    if shown >= 5:
                        print(f"    ... ({len(minor_gas_indices)-5} more minor species)")
                        break

            # ── Apply update (delegate to parent helpers) ───────────────────
            lam = self._apply_damped_update(
                mixture=mixture,
                major_gas_indices=major_gas_indices,
                active_cnd_local=active_cnd_local,
                A_maj=A_maj,
                mu_maj=mu_maj,
                n_gas_arr=n_gas_arr,
                n_gas_total=n_gas_total,
                pi=pi,
                delta_n_cnd=np.zeros(0),
                delta_ln_n=delta_ln_n,
            )

            n_gas_updated = float(mixture.gas_moles().sum())
            self._update_minor_from_potentials(
                minor_gas_indices, mixture, em, pi, n_gas_updated, T, P
            )

            changed = self._manage_condensed_phases(mixture, em, pi, T)

            delta_ln_nj_maj = A_maj @ pi + delta_ln_n - mu_maj
            element_res2 = em.element_residuals(mixture.moles, b0)
            _el_max = float(np.max(np.abs(element_res2))) if len(element_res2) else 0.0

            _step_norm = max(
                max(
                    (abs(mixture.moles[idx] * d_i) / max(float(mixture.gas_moles().sum()), 1e-300)
                     for idx, d_i in zip(major_gas_indices, delta_ln_nj_maj)
                     if mixture.moles[idx] > 0.0),
                    default=0.0,
                ),
                abs(float(delta_ln_n)),
            )
            self._last_step_norm = float(_step_norm)

            print(f"\n  After update: el_res_max = {_el_max:.3e}  step_norm = {_step_norm:.3e}  lam = {lam:.4f}")

            if self._check_convergence(
                mixture, delta_ln_nj_maj, major_gas_indices, delta_ln_n, element_res2
            ):
                print(f"\n  CONVERGED at iteration {n_inner}  (step_norm={_step_norm:.3e} < tol={self.tolerance})")
                _tp_converged = True
                break

            if n_inner >= 4:
                print(f"\n  [Trace limit reached — delegating remaining iterations to base class]")
                # Continue with parent (non-printing) for the rest
                result = super()._tp_equilibrium(mixture, em, b0, T, P)
                return result

        # Final exact update
        _A_ex = em.matrix
        _n_gas_now = float(mixture.gas_moles().sum())
        _ln_n_ex = math.log(max(_n_gas_now, 1e-300))
        _ln_P_ex = math.log(P / _P_REF)
        _LOG_CONC_TOL = math.log(1e-8)
        for _j in range(mixture.n_gas):
            _sp = mixture.species[_j]
            _ln_eq = float(_A_ex[_j, :] @ pi) - _sp.reduced_gibbs(T) - _ln_P_ex + _ln_n_ex
            _ln_eq = min(_ln_eq, 700.0)
            if _ln_eq - _ln_n_ex <= _LOG_CONC_TOL:
                mixture.moles[_j] = 0.0
            else:
                mixture.moles[_j] = math.exp(_ln_eq)

        return mixture, pi, n_inner + 1, _tp_converged, history


# ── Build and run a case ───────────────────────────────────────────────────────

def run_case(
    label: str,
    fuel_formula: str,
    ox_formula: str,
    n_fuel: float,
    n_ox: float,
    elements: set,
):
    fuel = db.find(fuel_formula)
    ox = db.find(ox_formula)
    H0 = fuel.enthalpy(T_REF) * n_fuel + ox.enthalpy(T_REF) * n_ox
    products = db.get_species(elements, max_atoms=20)

    problem = EquilibriumProblem(
        reactants={fuel: n_fuel, ox: n_ox},
        products=products,
        problem_type=ProblemType.HP,
        constraint1=H0,
        constraint2=P_PA,
        t_init=3500.0,
    )

    banner = "=" * 72
    print(f"\n{banner}")
    print(f"  {label}")
    print(f"  {fuel_formula} ({n_fuel} mol) + {ox_formula} ({n_ox} mol)")
    print(f"  H0_reactants = {H0:.2f} J   P = {P_PA:.0f} Pa")
    print(f"  Product species pool: {len(products)}")
    print(banner)

    solver = TracingMajorSpeciesSolver(label=label, max_T_outer=1)
    import warnings
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        sol = solver.solve(problem)

    print(f"\n{banner}")
    print(f"  FINAL RESULT — {label}")
    print(f"  T = {sol.temperature:.4f} K   converged = {sol.converged}   iters = {sol.iterations}")
    print(f"  pi (element potentials at convergence):")
    for k, pi_k in zip(sol.mixture.species[0].elements.keys() if False else
                       # recover elements from the lagrange multipliers shape
                       [f"el_{i}" for i in range(len(sol.lagrange_multipliers))],
                       sol.lagrange_multipliers):
        print(f"    pi[{k}] = {pi_k:.6f}")
    print(f"  Major species (x >= 0.1%):")
    for sp, x in sol.major_species(threshold=1e-3).items():
        print(f"    {sp:20s}  {x:.6f}")
    print()


# ── Run both cases ─────────────────────────────────────────────────────────────

import warnings

print("\n" + "=" * 72)
print("  ELEMENT POTENTIAL TRACE — H2/O2 and CH4/O2")
print("=" * 72)

with warnings.catch_warnings(record=True):
    warnings.simplefilter("always")
    run_case(
        label="H2/O2  O/F=8.0 stoichiometric",
        fuel_formula="H2",
        ox_formula="O2",
        n_fuel=2.0,
        n_ox=1.0,
        elements={"H", "O"},
    )

with warnings.catch_warnings(record=True):
    warnings.simplefilter("always")
    run_case(
        label="CH4/O2  O/F=4.0 stoichiometric",
        fuel_formula="CH4",
        ox_formula="O2",
        n_fuel=1.0,
        n_ox=2.0,
        elements={"C", "H", "O"},
    )
