"""
Equilibrium solvers for Prometheus.

Three concrete solvers are provided, all implementing the same
:py:class:`EquilibriumSolver` interface.  They share the same inputs
(:py:class:`~Prometheus.equilibrium.problem.EquilibriumProblem`) and outputs
(:py:class:`~Prometheus.equilibrium.solution.EquilibriumSolution`) and differ
only in the numerical algorithm used internally.

Solver summary
--------------

MajorSpeciesSolver  *(recommended — default)*
    Compressed Newton on major species combined with an analytical minor-
    species update and a PEP-style outer temperature search.  Combines the
    quadratic convergence of Newton's method with PEP's small matrix size
    and guaranteed temperature convergence.  See :class:`MajorSpeciesSolver`.

PEPSolver  *(reference implementation)*
    Pure Villars-Browne reaction-adjustment method as described in the
    original NWC Propellant Evaluation Program (Cruise, NWC TP 6037, 1979).
    Converges linearly; useful as a simpler reference and for validating
    the more complex Major-Species solver.  See :class:`PEPSolver`.

GordonMcBrideSolver  *(comparison / validation)*
    Modified Lagrange multiplier method from NASA RP-1311 (Gordon &
    McBride, 1994).  The algorithm used by CEA.  Kept for direct comparison
    against RocketCEA output.  See :class:`GordonMcBrideSolver`.

Algorithm relationships
-----------------------
All three solvers minimise the same objective — Gibbs free energy subject to
element-balance constraints — and find the same solution.  They differ in
how they parameterise and solve the resulting nonlinear system:

+------------------+-------------------+------------------+-----------------+
| Feature          | G-McB             | PEP              | Major-Species   |
+==================+===================+==================+=================+
| Core variables   | π (Lagrange mult) | nⱼ directly      | π for major     |
+------------------+-------------------+------------------+-----------------+
| Matrix size      | (S+cnd+1)²        | S×S              | S×S             |
+------------------+-------------------+------------------+-----------------+
| Minor species    | log-space Newton  | Kᵢ/Qᵢ ratio      | analytic from π |
+------------------+-------------------+------------------+-----------------+
| Convergence rate | quadratic         | linear           | quadratic       |
+------------------+-------------------+------------------+-----------------+
| T convergence    | no guarantee      | interval halving | interval halving|
+------------------+-------------------+------------------+-----------------+
| Condensed phases | restart loop      | phase deletion   | phase deletion  |
+------------------+-------------------+------------------+-----------------+

where S = number of chemical elements, cnd = number of active condensed
phases.  For H₂/O₂ combustion (S=2, no condensed phases) the Major-Species and PEP
matrix is always 2×2 regardless of the number of candidate product species.

Shared infrastructure
---------------------
PEP and Major-Species solvers both use:

* :py:meth:`ElementMatrix.select_basis` — Browne's optimised basis selection
* :py:meth:`ElementMatrix.reaction_coefficients` — ν = C·B⁻¹
* :py:meth:`_ReactionAdjustmentBase._temperature_search` — Newton + interval halving

The common base class :class:`_ReactionAdjustmentBase` holds this shared
logic so it is not duplicated.
"""

# Prometheus: An open-source combustion equilibrium solver in Python.
# Copyright (C) 2026 Charles Kendall
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import numpy as np
from loguru import logger

from prometheus_equilibrium.core.constants import REFERENCE_PRESSURE as _P_REF
from prometheus_equilibrium.core.constants import UNIVERSAL_GAS_CONSTANT as _R
from prometheus_equilibrium.equilibrium.diagnostics import NonConvergenceReason
from prometheus_equilibrium.equilibrium.element_matrix import ElementMatrix
from prometheus_equilibrium.equilibrium.mixture import Mixture
from prometheus_equilibrium.equilibrium.problem import EquilibriumProblem, ProblemType
from prometheus_equilibrium.equilibrium.solution import (
    ConvergenceStep,
    EquilibriumSolution,
)

# Species mole fraction floor: below this relative to n_gas the species is
# considered trace and its ln(nⱼ) update may push it to exactly zero.
_CONC_TOL = 1e-8
_LOG_CONC_TOL = math.log(_CONC_TOL)  # ≈ −18.42


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class EquilibriumSolver(ABC):
    """Abstract base class for all equilibrium solvers.

    Concrete implementations must override :py:meth:`solve`.  The interface
    is solver-agnostic: any of the three concrete solvers can be swapped in
    without changing calling code.

    Parameters
    ----------
    max_iterations : int
        Maximum inner (composition) iterations before declaring non-convergence.
    tolerance : float
        Convergence threshold for species mole-fraction residuals.
    """

    def __init__(
        self,
        max_iterations: int = 50,
        tolerance: float = 5e-6,
        capture_history: bool = False,
        history_stride: int = 1,
    ) -> None:
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.capture_history = capture_history
        self.history_stride = max(1, int(history_stride))

    def _should_capture_history(self, iteration_index: int) -> bool:
        """Return True when the current step should be stored in history."""
        return self.capture_history and (iteration_index % self.history_stride == 0)

    @staticmethod
    def _history_snapshot(
        mixture: Mixture, n_gas_total: Optional[float] = None
    ) -> dict:
        """Build a gas-normalized mole-fraction snapshot for history storage."""
        denom = max(
            float(n_gas_total) if n_gas_total is not None else mixture.total_gas_moles,
            1e-10,
        )
        return {
            sp.formula: n / denom
            for sp, n in zip(mixture.species, mixture.moles)
            if n > 0.0
        }

    @abstractmethod
    def solve(
        self,
        problem: EquilibriumProblem,
        guess: Optional[Mixture] = None,
        log_failure: bool = True,
    ) -> EquilibriumSolution:
        """Run the equilibrium calculation and return the converged state.

        Parameters
        ----------
        problem : EquilibriumProblem
            Fully specified problem.  Call ``problem.validate()`` before
            passing to catch ill-posed inputs early.
        guess : Mixture, optional
            Initial composition guess.  If provided, the solver uses these
            mole amounts as a starting point instead of the default
            monatomic initialization.

        Returns
        -------
        EquilibriumSolution
            The converged state.  Always check ``.converged`` before using
            results.
        """
        ...

    # ------------------------------------------------------------------
    # Shared condensed-phase helpers (used by G-McB and Major-Species solvers)
    # ------------------------------------------------------------------

    @staticmethod
    def _active_condensed_indices(mixture: "Mixture") -> List[int]:
        """Return local indices (within condensed sub-array) of active condensed species.

        "Local" means indexed within the condensed sub-array: 0 = first
        condensed species in the mixture, regardless of gas count.
        """
        return [i for i, n in enumerate(mixture.condensed_moles()) if n > 0.0]

    @staticmethod
    def _manage_condensed_phases(
        mixture: "Mixture",
        em: "ElementMatrix",
        pi: np.ndarray,
        T: float,
        g0_arr: Optional[np.ndarray] = None,
    ) -> bool:
        """Activate or deactivate condensed phases based on element potentials.

        **Removal:** any active condensed species with nⱼ < 0 is clamped to 0
        and the changed flag is set (nⱼ = 0 exactly means already inactive).

        **Inclusion:** for each inactive condensed species c, compute
        ``g°c/RT − Σₖ πₖ·aₖc``.  If negative, including it would lower the
        Gibbs energy; activate the most favourable candidate.

        Args:
            mixture: Modified in-place.
            em: Element matrix.
            pi: Current element potentials from the Newton solve, shape (S,).
            T: Current temperature [K].
            g0_arr: Optional precomputed g°ⱼ/RT for all species (indexed by
                mixture position). When supplied the per-species thermo call
                is skipped for condensed species.

        Returns:
            True if the condensed phase set changed (caller should restart
            the convergence counter).
        """
        changed = False
        n_gas = mixture.n_gas
        A_cnd = em.condensed_rows()  # (n_cnd_all, S)

        # --- removal: clamp negative moles to zero ---
        for i in range(mixture.n_condensed):
            global_idx = n_gas + i
            if mixture.moles[global_idx] < 0.0:
                mixture.moles[global_idx] = 0.0
                changed = True

        # --- inclusion: activate most favourable inactive condensed ---
        best_val = 0.0
        best_i = -1
        for i in range(mixture.n_condensed):
            global_idx = n_gas + i
            if mixture.moles[global_idx] > 0.0:
                continue  # already active
            sp = mixture.species[global_idx]
            mu_c = g0_arr[global_idx] if g0_arr is not None else sp.reduced_gibbs(T)
            assigned = float(A_cnd[i, :] @ pi)
            val = mu_c - assigned
            if val < best_val:
                best_val = val
                best_i = i

        if best_i >= 0:
            mixture.moles[n_gas + best_i] = 1e-6 * max(
                float(mixture.gas_moles().sum()), 1e-10
            )
            changed = True

        return changed

    @staticmethod
    def _refresh_thermo_species_set(
        mixture: Mixture,
        species_pool: List,
        active_elements: List[str],
        T: float,
    ) -> Tuple[Mixture, ElementMatrix]:
        """Rebuild active species to include all and only thermo-valid species at T.

        Species with non-finite reduced Gibbs energy at the current temperature
        are excluded for this Newton step. Species that become valid as T changes
        are reintroduced with zero moles and can then grow through the Newton
        update.
        """
        valid_species = [
            sp for sp in species_pool if math.isfinite(sp.reduced_gibbs(T))
        ]
        if not valid_species:
            raise RuntimeError(
                f"No product species have valid thermodynamic data at T={T:.2f} K."
            )

        prev_moles = {sp: float(n) for sp, n in zip(mixture.species, mixture.moles)}
        new_moles = np.array(
            [prev_moles.get(sp, 0.0) for sp in valid_species], dtype=float
        )

        # Ensure at least one gas species has positive moles for log terms.
        n_gas = sum(1 for sp in valid_species if sp.condensed == 0)
        if n_gas == 0:
            raise RuntimeError(f"No gas-phase species are thermo-valid at T={T:.2f} K.")
        if float(np.sum(new_moles[:n_gas])) <= 0.0:
            new_moles[:n_gas] = max(1e-12, 1.0 / n_gas)

        refreshed = Mixture(valid_species, new_moles)
        em = ElementMatrix(refreshed.species, active_elements)
        return refreshed, em


# ---------------------------------------------------------------------------
# Shared base for PEP and Major-Species (reaction-adjustment family)
# ---------------------------------------------------------------------------


class _ReactionAdjustmentBase(EquilibriumSolver, ABC):
    """Shared infrastructure for PEP and major-species solvers.

    Both solvers use Browne's optimised basis selection and an outer
    temperature search with a Newton / interval-halving strategy.  This
    class holds that shared logic so it is not duplicated.

    Subclasses must implement :py:meth:`_tp_equilibrium`, which performs
    a single isothermal-isobaric (TP) equilibrium iteration to convergence
    and is called repeatedly by the outer temperature loop.
    """

    def __init__(
        self,
        max_iterations: int = 50,
        tolerance: float = 5e-6,
        minor_threshold: float = 1e-2,
        capture_history: bool = False,
        history_stride: int = 1,
    ) -> None:
        super().__init__(max_iterations, tolerance, capture_history, history_stride)
        # Species whose mole amount is less than (minor_threshold × smallest
        # basis mole amount) are classified as "minor" and handled separately.
        self.minor_threshold = minor_threshold

    # ------------------------------------------------------------------
    # Shared: initialisation
    # ------------------------------------------------------------------

    def _initialise(
        self, problem: EquilibriumProblem
    ) -> Tuple[Mixture, ElementMatrix, np.ndarray, float, EquilibriumProblem]:
        """Validate the problem and build the starting state.

        Returns
        -------
        mixture : Mixture
            Starting-guess composition from ``problem.initial_mixture()``.
        element_matrix : ElementMatrix
            Built from the product species list (reduced to independent elements).
        b0 : np.ndarray, shape (n_elements,)
            Target element abundances from the reactant composition.
        T : float
            Initial temperature estimate.
        problem : EquilibriumProblem
            Potentially filtered problem (species with no valid data at T_init
            are removed).
        """
        problem.validate()

        # Remove product species whose thermodynamic data do not cover T_init.
        valid_products = [
            sp
            for sp in problem.products
            if math.isfinite(sp.reduced_gibbs(problem.t_init))
        ]
        if len(valid_products) < len(problem.products):
            import warnings

            warnings.warn(
                f"{len(problem.products) - len(valid_products)} product species "
                f"removed because their thermodynamic data do not cover "
                f"T_init = {problem.t_init:.1f} K.",
                RuntimeWarning,
                stacklevel=3,
            )
            problem = EquilibriumProblem(
                reactants=problem.reactants,
                products=valid_products,
                problem_type=problem.problem_type,
                constraint1=problem.constraint1,
                constraint2=problem.constraint2,
                pressure=problem.pressure,
                t_init=problem.t_init,
            )

        mixture = problem.initial_mixture()
        em = ElementMatrix.from_mixture(mixture)
        em = em.reduced()

        # Further restrict to elements that have positive reactant abundance.
        # Elements with b₀[k] = 0 (e.g. "e-" charge balance in neutral
        # combustion) are dropped.  Species that are entirely composed of
        # such dropped elements (e.g. free electron {"e-": 1.0}) would be
        # thermodynamically unconstrained in the reduced system — they could
        # grow without bound — so they are removed from the mixture.
        b0_full = problem.b0_array(em.elements)
        active_elements = [el for el, b in zip(em.elements, b0_full) if b > 0.0]
        if len(active_elements) < em.n_elements:
            # Drop species that use ANY element outside the active set (e.g.
            # ionic species like OH+ contain "e-" which is not in active_elements).
            # Without the charge-balance constraint these species are
            # unconstrained and can grow to non-physical values.
            active_set = set(active_elements)
            species_mask = [
                all(el in active_set for el in sp.elements.keys())
                for sp in mixture.species
            ]
            if not all(species_mask):
                active_species = [
                    sp for sp, ok in zip(mixture.species, species_mask) if ok
                ]
                active_moles = mixture.moles[np.array(species_mask)]
                mixture = Mixture(active_species, active_moles)

            em = em.reduced(active_elements)
            # Rebuild em to match the (possibly filtered) mixture
            em = ElementMatrix(mixture.species, active_elements)

        b0 = problem.b0_array(em.elements)
        T = problem.t_init
        return mixture, em, b0, T, problem

    # ------------------------------------------------------------------
    # Shared: temperature search (outer loop)
    # ------------------------------------------------------------------

    def _temperature_search(
        self,
        problem: EquilibriumProblem,
        mixture: Mixture,
        element_matrix: ElementMatrix,
        b0: np.ndarray,
        T_init: float,
    ) -> Tuple[Mixture, float, np.ndarray, int, bool, List[ConvergenceStep]]:
        """Find the temperature satisfying the HP or SP energy constraint.
        ...
        Returns
        -------
        mixture : Mixture
        T : float
        pi : np.ndarray
        n_outer : int
        converged_outer : bool
        history : List[ConvergenceStep]
        """
        constraint = problem.constraint1  # H₀ [J] for HP;  S₀ [J/K] for SP
        sp_use_gas_entropy = (
            problem.problem_type == ProblemType.SP
            and getattr(problem, "sp_entropy_mode", "total") == "gas"
        )
        P = problem.constraint2
        is_hp = problem.problem_type == ProblemType.HP

        T = T_init
        # Narrow the bisection bracket around T_init so that a warm-started
        # nozzle expansion (where T_init ≈ previous T) converges in far fewer
        # outer iterations.  For HP problems the true T may exceed T_init (e.g.
        # low initial guess for a hot flame), so only the lower bound is
        # tightened.  For SP problems (isentropic nozzle expansion) T decreases
        # monotonically, so the upper bound is tightened too.
        T_lo = max(200.0, T_init * 0.4)
        T_hi = min(6000.0, T_init * 1.2) if not is_hp else 6000.0
        pi = np.zeros(element_matrix.n_elements)
        history: List[ConvergenceStep] = []

        _log_ts = logger.bind(solver=type(self).__name__)
        n_outer = 0
        converged_outer = False
        mixture_prev = mixture.copy()
        pi_prev = pi.copy()
        for n_outer in range(50):
            # Converge composition at current T.
            # Save mixture state in case T is out-of-range for some species and
            # we need to roll back before narrowing the bracket.
            mixture_prev = mixture.copy()
            pi_prev = pi.copy()
            mixture, pi, _, _inner_conv, _inner_history = self._tp_equilibrium(
                mixture, element_matrix, b0, T, P
            )

            # Energy residual f(T) and its derivative f'(T)
            if is_hp:
                f = mixture.total_enthalpy(T) - constraint
                fp = mixture.total_cp(T)
            else:
                if sp_use_gas_entropy:
                    f = mixture.total_gas_entropy(T, P) - constraint
                    fp = mixture.total_gas_cp(T) / max(T, 1.0)
                else:
                    f = mixture.total_entropy(T, P) - constraint
                    fp = mixture.total_cp(T) / max(T, 1.0)

            _dln_T = abs(f) / (T * abs(fp)) if abs(fp) > 1e-30 else float("inf")

            # Record outer step
            if self._should_capture_history(n_outer):
                history.append(
                    ConvergenceStep(
                        temperature=T,
                        max_residual=_dln_T,
                        mole_fractions=self._history_snapshot(mixture),
                    )
                )

            _log_ts.debug(
                "outer={:2d} T={:9.3f} f={:.4e} fp={:.4e} dlnT={:.4e} [T_lo={:.1f} T_hi={:.1f}] inner_conv={}",
                n_outer,
                T,
                f,
                fp,
                _dln_T,
                T_lo,
                T_hi,
                _inner_conv,
            )
            # Convergence: |Δln T| = |f| / (T·|f'|) < 1e-4, analogous to G-McB
            if abs(fp) > 1e-30 and abs(f) < 1e-4 * T * abs(fp):
                converged_outer = True
                break
            if T_hi - T_lo < 0.01:
                converged_outer = True
                break

            # Update bounds based on sign of residual.
            # NaN means T is above the valid range of some product species.
            # Roll back the corrupted mixture state and tighten the upper bound.
            if math.isnan(f):
                mixture = mixture_prev
                pi = pi_prev
                T_hi = T
            elif f > 0.0:
                T_hi = T
            else:
                T_lo = T

            # Newton step, with interval-halving fallback.
            T_mid = 0.5 * (T_lo + T_hi)
            bracket = T_hi - T_lo
            if abs(fp) > 1e-30:
                T_newton = T - f / fp
            else:
                T_newton = T_mid

            if (
                T_newton <= T_lo
                or T_newton >= T_hi
                or (T_newton - T_lo) < 0.05 * bracket
                or (T_hi - T_newton) < 0.05 * bracket
            ):
                T = T_mid
            else:
                T = T_newton

        # Correct T for the residual energy error without re-solving composition.
        #
        # After the outer loop declares convergence at |dlnT| < 1e-4 the
        # HP/SP residual f = Σ nⱼ Hⱼ(T) − H₀ can still be O(500 J/kg)
        # (~0.05 %).  This is within the declared tolerance but propagates
        # through the isentropic nozzle expansion to give up to ~0.6 % Isp
        # error for some CH4/O2 conditions.
        #
        # A single Newton correction ΔT = −f / (dH/dT) = −f / Cp applied
        # directly to T (without re-solving the composition) removes the
        # residual to near machine precision: by the mean-value theorem,
        # Σ nⱼ Hⱼ(T + ΔT) ≈ Σ nⱼ Hⱼ(T) + Cp·ΔT = H(T) − f = H₀.  The
        # composition is unchanged so the correction is exact to O(ΔT²).
        if converged_outer and not math.isnan(f) and abs(fp) > 1e-30:
            dT_corr = -f / fp
            # Guard: only apply if the correction is small (sanity check)
            if abs(dT_corr) < 1.0:
                T = T + dT_corr

        return mixture, T, pi, n_outer + 1, converged_outer, history

    # ------------------------------------------------------------------
    # Shared: major/minor species split
    # ------------------------------------------------------------------

    def _split_major_minor(
        self,
        moles: np.ndarray,
        basis_indices: list,
    ) -> Tuple[list, list]:
        """Partition non-basis species into major and minor.

        A non-basis species is *minor* if its mole amount is less than
        ``self.minor_threshold`` times the smallest basis mole amount
        (Cruise NWC TP 6037 — "two orders of magnitude smaller").

        Parameters
        ----------
        moles : np.ndarray
        basis_indices : list of int

        Returns
        -------
        major_indices : list of int
            Indices of major non-basis species.
        minor_indices : list of int
            Indices of minor non-basis species.
        """
        basis_set = set(basis_indices)
        positive_basis = [moles[i] for i in basis_indices if moles[i] > 0.0]
        basis_min = min(positive_basis) if positive_basis else 1e-30
        threshold = self.minor_threshold * basis_min

        major: list = []
        minor: list = []
        for i in range(len(moles)):
            if i in basis_set:
                continue
            if moles[i] >= threshold:
                major.append(i)
            else:
                minor.append(i)
        return major, minor

    # ------------------------------------------------------------------
    # Subclasses must implement TP equilibrium
    # ------------------------------------------------------------------

    @abstractmethod
    def _tp_equilibrium(
        self,
        mixture: Mixture,
        element_matrix: ElementMatrix,
        b0: np.ndarray,
        T: float,
        P: float,
    ) -> Tuple[Mixture, np.ndarray, int, bool, List[ConvergenceStep]]:
        """Solve isothermal-isobaric (TP) equilibrium to convergence.

        Called repeatedly by :py:meth:`_temperature_search` at each
        temperature guess.  Must preserve element balance (A^T·n = b₀)
        throughout.

        Parameters
        ----------
        mixture : Mixture
            Starting composition; modified in-place and returned.
        element_matrix : ElementMatrix
        b0 : np.ndarray
            Target element abundances.
        T : float
            Fixed temperature [K].
        P : float
            Fixed pressure [Pa].

        Returns
        -------
        mixture : Mixture
            Converged composition.
        pi : np.ndarray, shape (n_elements,)
            Converged element potentials (Lagrange multipliers).
            PEPSolver may return zeros here if it does not compute π.
        n_inner : int
            Number of inner iterations taken.
        """
        ...


# ---------------------------------------------------------------------------
# PEP solver — pure Villars-Browne reaction adjustment
# ---------------------------------------------------------------------------


class PEPSolver(_ReactionAdjustmentBase):
    ...

    def solve(
        self,
        problem: EquilibriumProblem,
        guess: Optional[Mixture] = None,
        log_failure: bool = True,
    ) -> EquilibriumSolution:
        """Run the equilibrium iteration.

        Args:
            problem: Fully specified equilibrium problem.
            guess: Optional initial composition guess.

        Returns:
            EquilibriumSolution with the converged composition.
        """
        mixture, em, b0, T, problem = self._initialise(problem)
        if guess is not None:
            new_moles = np.zeros(len(mixture.species))
            guess_map = {sp: n for sp, n in zip(guess.species, guess.moles)}
            for i, sp in enumerate(mixture.species):
                new_moles[i] = guess_map.get(sp, 0.0)

            _b_guess = em.element_abundances(new_moles)
            if np.allclose(_b_guess, b0, rtol=1e-3):
                mixture.moles = new_moles

        P = problem.constraint2
        pi = np.zeros(em.n_elements)
        history = []

        self._last_failure_reason = None
        if problem.problem_type.fixed_temperature:
            mixture, pi, n_iter, converged, history = self._tp_equilibrium(
                mixture, em, b0, T, P
            )
        else:
            mixture, T, pi, n_iter, converged, history = self._temperature_search(
                problem, mixture, em, b0, T
            )

        residuals = em.element_residuals(mixture.moles, b0)
        element_balance_error = (
            float(np.max(np.abs(residuals))) if residuals.size else 0.0
        )
        failure_reason = None
        if not converged:
            failure_reason = (
                self._last_failure_reason
                if self._last_failure_reason is not None
                else NonConvergenceReason.MAX_ITERATIONS_REACHED
            )
        last_step_norm = 0.0 if converged else float("inf")

        return EquilibriumSolution(
            mixture=mixture,
            temperature=T,
            pressure=P,
            converged=converged,
            iterations=n_iter,
            residuals=residuals,
            lagrange_multipliers=pi,
            history=history if self.capture_history else None,
            failure_reason=failure_reason,
            element_balance_error=element_balance_error,
            last_step_norm=last_step_norm,
        )

    def _tp_equilibrium(
        self,
        mixture: Mixture,
        element_matrix: ElementMatrix,
        b0: np.ndarray,
        T: float,
        P: float,
    ) -> Tuple[Mixture, np.ndarray, int, bool, List[ConvergenceStep]]:
        """PEP inner loop: serial reaction-adjustment at fixed T and P.

        Faithful reimplementation of the TWITCH/EQUIL algorithm from the
        original Fortran PEP (Cruise, NWC TP 6037).

        Structure (matching the Fortran):

        1. Select Browne basis **once** (DEFIOJ), compute reaction
           coefficients and equilibrium constants **once** (REACT).
        2. Outer loop (up to 20 iterations, indexed by JC):
           a. One full sweep over all non-basis species (TWITCH with JQ=0).
           b. If not converged, up to 3 more major-only sweeps (JQ=1).
           c. VQQ damping is tied to JC, NOT to the sweep count.
        3. Within each sweep, if a stoichiometric update drives a basis
           species below the corrected non-basis species, perform a
           TABLO pivot (basis swap) and recompute ν and ln_K.

        Args:
            mixture: Starting composition; modified in-place and returned.
            element_matrix: Element matrix (reduced to independent elements).
            b0: Target element abundances.
            T: Fixed temperature [K].
            P: Fixed pressure [Pa].

        Returns:
            Tuple of (mixture, pi, n_iters, converged, history).
        """
        em = element_matrix
        S = em.n_elements
        N = len(mixture.species)
        pi = np.zeros(S)
        ln_P_ratio = math.log(P / _P_REF)
        # FLOOR: minimum mole value (matches Fortran FLOOR = W27/10^(8+KR(5)))
        FLOOR = 1e-12
        history: List[ConvergenceStep] = []

        # --- Monatomic initialization (NWC TP 6037 §"Numerical Examples") ---
        # "Set the monatomic gases to the desired gram-atom amounts and the
        # rest of the species to zero."  This gives select_basis the
        # monatomics as the initial basis, from which TWITCH efficiently
        # builds up molecular species.
        _b_init = em.element_abundances(mixture.moles)
        _b_max = float(np.max(np.abs(b0))) + 1e-300
        if float(np.max(np.abs(_b_init - b0))) / _b_max > 0.05:
            self._monatomic_init(mixture, em, b0, FLOOR)

        _log_pep = logger.bind(solver="PEP")
        self._last_failure_reason = None

        # --- One-time basis selection (DEFIOJ) ---
        try:
            basis_indices, nonbasis_indices = em.select_basis(mixture.moles)
        except RuntimeError:
            self._last_failure_reason = NonConvergenceReason.NO_BASIS_FOUND
            _log_pep.info("exit converged=False iters=0 T={:.1f} (no basis)", T)
            return mixture, pi, 0, False, history

        # Floor zero-mole gas basis species
        for bid in basis_indices:
            if mixture.moles[bid] <= 0.0 and mixture.species[bid].condensed == 0:
                mixture.moles[bid] = FLOOR

        # --- One-time reaction setup (REACT) ---
        # LL flags: 0 = basis, 9 = active non-basis (major), 8 = minor,
        #           -1 = inactive (out-of-range condensed).  Matches Fortran.
        LL = [0] * N
        for i in nonbasis_indices:
            LL[i] = 9

        nu = em.reaction_coefficients(basis_indices)
        g_all = np.array([sp.reduced_gibbs(T) for sp in mixture.species])
        g_basis = g_all[np.array(basis_indices)]
        ln_K = self._equilibrium_constants(nu, g_all, g_basis)

        is_gas_basis = np.array(
            [mixture.species[j].condensed == 0 for j in basis_indices],
            dtype=bool,
        )

        # Number of gas-phase species (matches Fortran IG)
        IG = sum(1 for sp in mixture.species if sp.condensed == 0)

        converged = False
        total_sweeps = 0
        phase_changed_counter = 0

        # --- Outer loop (Fortran: DO 22 JC=1,20) ---
        n_outer_max = 20
        for JC in range(1, n_outer_max + 1):
            VQQ = max(0.05, 0.5 - (JC - 1) / 20.0)

            # Up to 4 sweeps per outer iteration:
            # sweep 0 = full (all non-basis), sweeps 1-3 = major only (LL==9)
            for sweep_pass in range(4):
                major_only = sweep_pass > 0

                # --- Refresh gas totals ---
                n_gas_total = sum(
                    mixture.moles[i]
                    for i in range(N)
                    if mixture.species[i].condensed == 0 and mixture.moles[i] > 0.0
                )
                n_gas_total = max(n_gas_total, 1e-300)
                ln_P_offset = ln_P_ratio - math.log(n_gas_total)

                # Record history before each full sweep
                if not major_only:
                    element_res = em.element_residuals(mixture.moles, b0)
                    _el_res_max = (
                        float(np.max(np.abs(element_res))) if len(element_res) else 0.0
                    )

                    if self._should_capture_history(len(history)):
                        history.append(
                            ConvergenceStep(
                                temperature=T,
                                max_residual=_el_res_max,
                                mole_fractions=self._history_snapshot(
                                    mixture, n_gas_total=n_gas_total
                                ),
                            )
                        )

                any_large_step = False  # tracks Fortran SLITE(4) flag

                # --- Serial sweep (TWITCH) ---
                for j in range(N):
                    if LL[j] <= 0:
                        continue  # basis or inactive
                    if major_only and LL[j] != 9:
                        continue  # skip minor in major-only passes

                    # Fortran line 1752: LL(J) = 1 at start of processing.
                    # If species converges this sweep, LL stays 1 and is
                    # skipped in subsequent major-only passes.
                    prev_LL = LL[j]
                    LL[j] = 1

                    n_j = mixture.moles[j]
                    is_gas = mixture.species[j].condensed == 0

                    # Compute TWID and step bounds (SETUP)
                    twid, sum_gas_nu = self._compute_twid(
                        nu[j],
                        basis_indices,
                        is_gas_basis,
                        mixture.moles,
                        ln_P_offset,
                    )
                    XMIN, XMAX = self._step_bounds(
                        nu[j],
                        basis_indices,
                        mixture.moles,
                        n_j,
                        not is_gas,
                    )

                    # Handle negative mole counts
                    if n_j <= 0.0:
                        DX = -1.001 * n_j + FLOOR
                        DX = max(VQQ * XMIN, min(VQQ * XMAX, DX))
                        self._apply_stoich_update(
                            mixture,
                            j,
                            DX,
                            nu,
                            basis_indices,
                            S,
                            FLOOR,
                        )
                        any_large_step = True
                        continue

                    VA = ln_K[j] - twid  # Fortran: VA = VLNK(J) - TWID(X)

                    if not is_gas:
                        # --- Condensed species ---
                        if abs(VA) < 8e-5:
                            continue  # converged, LL stays 1
                        if n_j <= FLOOR and VA >= 0.0:
                            continue  # inactive trace condensed, LL stays 1
                        # Major condensed correction
                        VB = sum(
                            nu[j, kk] ** 2
                            / max(mixture.moles[basis_indices[kk]], 1e-300)
                            for kk in range(S)
                            if is_gas_basis[kk] and abs(nu[j, kk]) > 1e-15
                        )
                        if abs(VB) < 1e-30:
                            VB = 1e-7
                            VQ = 0.999999
                        else:
                            VQ = VQQ
                        DX = -VA / VB
                        LL[j] = 9  # Fortran line 1808
                        DX = max(VQ * XMIN, min(VQ * XMAX, DX))
                        if abs(DX) < 1e-4 * n_j:
                            continue  # small step → converged
                        if n_j + DX > FLOOR:
                            any_large_step = True
                    else:
                        # --- Gas species ---
                        if VA + ln_P_offset >= 5.0:
                            # Minor gas: set directly from equilibrium
                            dvamvn = -(VA + ln_P_offset)
                            dvamvn = max(dvamvn, -100.0)
                            V = math.exp(dvamvn)

                            # Check if species is truly negligible
                            XMMM = min(
                                -XMIN if XMIN != -math.inf else 1e30,
                                XMAX if XMAX != math.inf else 1e30,
                            )
                            if XMMM > 1e-300:
                                if n_j / XMMM < 0.01:
                                    XMAX = 0.011 * XMMM
                                if (V + n_j) / XMMM > 0.01:
                                    # Not really minor — treat as major
                                    pass
                                else:
                                    V = max(V, FLOOR)
                                    # Minor species tolerance
                                    if abs(1.0 - n_j / V) < 8e-4 if V > 0 else False:
                                        continue
                                    DX = V - n_j
                                    LL[j] = 8
                                    mixture.moles[j] = V
                                    # stoichiometric update on basis
                                    # Minor species: GO TO 82 in Fortran, skips SLITE(4)
                                    self._apply_basis_update_and_tablo(
                                        mixture,
                                        j,
                                        DX,
                                        nu,
                                        basis_indices,
                                        LL,
                                        S,
                                        N,
                                        IG,
                                        em,
                                        g_all,
                                        ln_K,
                                        is_gas_basis,
                                        T,
                                    )
                                    continue
                            else:
                                V = max(V, FLOOR)
                                if abs(1.0 - n_j / V) < 8e-4 if V > 0 else False:
                                    continue
                                DX = V - n_j
                                LL[j] = 8
                                mixture.moles[j] = V
                                self._apply_basis_update_and_tablo(
                                    mixture,
                                    j,
                                    DX,
                                    nu,
                                    basis_indices,
                                    LL,
                                    S,
                                    N,
                                    IG,
                                    em,
                                    g_all,
                                    ln_K,
                                    is_gas_basis,
                                    T,
                                )
                                continue

                        # Major gas: full residual
                        VA = VA + math.log(max(n_j, 1e-300)) + ln_P_offset
                        if abs(VA) < 8e-5:
                            continue  # converged for this species
                        LL[j] = 9
                        VB = 1.0 / max(n_j, 1e-300)
                        for kk in range(S):
                            if is_gas_basis[kk] and abs(nu[j, kk]) > 1e-15:
                                VB += nu[j, kk] ** 2 / max(
                                    mixture.moles[basis_indices[kk]], 1e-300
                                )
                        if abs(VB) < 1e-30:
                            VB = 1e-7
                            VQ = 0.999999
                        else:
                            VQ = VQQ
                        DX = -VA / VB
                        DX = max(-VQ * n_j, DX)
                        DX = max(VQ * XMIN, min(VQ * XMAX, DX))
                        if abs(DX) < 1e-4 * n_j:
                            continue  # relative step small → converged
                        # Fortran SLITE(4): only flag if species will
                        # remain above FLOOR after correction
                        if n_j + DX > FLOOR:
                            any_large_step = True

                    # Apply stoichiometric update + TABLO if needed
                    mixture.moles[j] = max(FLOOR, n_j + DX)
                    self._apply_basis_update_and_tablo(
                        mixture,
                        j,
                        DX,
                        nu,
                        basis_indices,
                        LL,
                        S,
                        N,
                        IG,
                        em,
                        g_all,
                        ln_K,
                        is_gas_basis,
                        T,
                    )

                total_sweeps += 1

                # Count how many species triggered large step
                _n_large = sum(
                    1 for j in range(N) if LL[j] > 0 and mixture.moles[j] > FLOOR * 10
                )
                _log_pep.debug(
                    "JC={:2d} pass={} T={:.1f} VQQ={:.3f} any_large={} n_active={}",
                    JC,
                    sweep_pass,
                    T,
                    VQQ,
                    any_large_step,
                    _n_large,
                )

                if not any_large_step:
                    # All species converged this sweep
                    if sweep_pass == 0:
                        # Even the full sweep found no large steps → converged
                        converged = True
                    break  # exit sweep loop (move to next JC or finish)

            if converged:
                break

        # --- Recover π from converged basis potentials ---
        n_gas_total = sum(
            mixture.moles[i]
            for i in range(N)
            if mixture.species[i].condensed == 0 and mixture.moles[i] > 0.0
        )
        n_gas_total = max(n_gas_total, 1e-300)
        ln_n_gas_final = math.log(n_gas_total)
        B_mat = em.basis_matrix(basis_indices)
        mu_basis_vec = np.array(
            [
                (
                    (
                        g_all[bi]
                        + math.log(max(mixture.moles[bi], 1e-300))
                        - ln_n_gas_final
                        + ln_P_ratio
                    )
                    if mixture.species[bi].condensed == 0
                    else g_all[bi]
                )
                for bi in basis_indices
            ]
        )
        try:
            pi = np.linalg.solve(B_mat, mu_basis_vec)
        except np.linalg.LinAlgError:
            pi = np.zeros(S)

        # --- Condensed phase management ---
        changed = self._manage_condensed_phases(mixture, em, pi, T)
        if changed:
            phase_changed_counter += 1

        _log_pep.info(
            "exit converged={} sweeps={} T={:.1f}", converged, total_sweeps, T
        )
        if not converged and self._last_failure_reason is None:
            self._last_failure_reason = NonConvergenceReason.MAX_ITERATIONS_REACHED
        return mixture, pi, total_sweeps, converged, history

    def _apply_stoich_update(
        self,
        mixture: Mixture,
        j: int,
        DX: float,
        nu: np.ndarray,
        basis_indices: List[int],
        S: int,
        FLOOR: float,
    ) -> None:
        """Apply stoichiometric correction without TABLO pivot check."""
        for kk in range(S):
            if abs(nu[j, kk]) > 1e-15:
                k = basis_indices[kk]
                mixture.moles[k] -= nu[j, kk] * DX
                if mixture.moles[k] < FLOOR:
                    mixture.moles[k] = FLOOR

    def _apply_basis_update_and_tablo(
        self,
        mixture: Mixture,
        j: int,
        DX: float,
        nu: np.ndarray,
        basis_indices: List[int],
        LL: List[int],
        S: int,
        N: int,
        IG: int,
        em: ElementMatrix,
        g_all: np.ndarray,
        ln_K: np.ndarray,
        is_gas_basis: np.ndarray,
        T: float,
    ) -> None:
        """Apply stoichiometric update on basis and TABLO pivot if needed.

        Matches the Fortran TWITCH lines 1815-1829 + TABLO call.  If any
        basis species K is driven below 99% of the updated species J's
        moles, swap them via a tableau pivot and recompute ν and ln_K.

        Args:
            mixture: Current mixture (modified in-place).
            j: Index of non-basis species being corrected.
            DX: Step size applied to species j.
            nu: Reaction coefficient matrix (modified in-place on pivot).
            basis_indices: Current basis indices (modified in-place on pivot).
            LL: Species status flags (modified in-place on pivot).
            S: Number of elements (basis size).
            N: Total number of species.
            IG: Number of gas-phase species.
            em: Element matrix.
            g_all: Reduced Gibbs energies for all species.
            ln_K: Equilibrium constants (modified in-place on pivot).
            is_gas_basis: Gas-phase mask for basis (modified in-place on pivot).
            T: Temperature (unused but kept for interface consistency).
        """
        VC = 0.99 * mixture.moles[j]  # threshold for TABLO trigger
        kick = False
        pivot_kk = -1
        pivot_k = -1
        VD = math.inf

        for kk in range(S):
            if abs(nu[j, kk]) < 1e-15:
                continue
            k = basis_indices[kk]
            mixture.moles[k] -= nu[j, kk] * DX
            if mixture.moles[k] < 1e-300:
                mixture.moles[k] = 1e-300
            if mixture.moles[k] < VC:
                if kick and mixture.moles[k] > VD:
                    continue
                VD = mixture.moles[k]
                kick = True
                pivot_kk = kk
                pivot_k = k

        if not kick:
            return

        # --- TABLO pivot: swap basis_indices[pivot_kk] with species j ---
        self._tablo_pivot(
            j,
            pivot_k,
            pivot_kk,
            nu,
            basis_indices,
            LL,
            S,
            N,
            em,
            g_all,
            ln_K,
            is_gas_basis,
            mixture,
        )

    @staticmethod
    def _tablo_pivot(
        j_entering: int,
        k_leaving: int,
        kk: int,
        nu: np.ndarray,
        basis_indices: List[int],
        LL: List[int],
        S: int,
        N: int,
        em: ElementMatrix,
        g_all: np.ndarray,
        ln_K: np.ndarray,
        is_gas_basis: np.ndarray,
        mixture: Mixture,
    ) -> None:
        """Tableau pivot: swap species j_entering into the basis at position kk.

        Implements the Fortran TABLO subroutine: updates ν for all species
        via row operations, swaps the basis member, recomputes ln_K, and
        updates the LL flags.

        Args:
            j_entering: Species index entering the basis.
            k_leaving: Species index leaving the basis.
            kk: Position in basis_indices being swapped.
            nu: Reaction coefficient matrix, shape (N, S). Modified in-place.
            basis_indices: Basis species indices. Modified in-place.
            LL: Species status flags. Modified in-place.
            S: Number of elements.
            N: Total number of species.
            em: Element matrix.
            g_all: Reduced Gibbs energies.
            ln_K: Equilibrium constants. Modified in-place.
            is_gas_basis: Gas-phase mask for basis. Modified in-place.
            mixture: Current mixture.
        """
        pivot_val = nu[j_entering, kk]
        if abs(pivot_val) < 1e-15:
            return  # degenerate pivot, skip

        # Row operations on ν for all other species (Fortran lines 820-827)
        for L in range(N):
            if LL[L] < 0:
                continue
            if L == j_entering:
                continue
            if abs(nu[L, kk]) < 1e-5:
                continue
            VA = -nu[L, kk] / pivot_val
            for m in range(S):
                nu[L, m] += VA * nu[j_entering, m]
            nu[L, kk] = -VA
            # Clean near-zero entries
            for m in range(S):
                if abs(nu[L, m]) < 1e-5:
                    nu[L, m] = 0.0

        # Zero out the entering species' row and set identity at pivot column
        for m in range(S):
            nu[j_entering, m] = 0.0
        nu[j_entering, kk] = 1.0

        # Swap basis member
        basis_indices[kk] = j_entering
        LL[j_entering] = 0  # now basis
        LL[k_leaving] = 9  # now non-basis (major)

        # Update is_gas_basis
        is_gas_basis[kk] = mixture.species[j_entering].condensed == 0

        # Recompute ln_K (Fortran: CALL REACT(TE) after TABLO)
        g_basis = g_all[np.array(basis_indices)]
        ln_K[:] = g_all - nu @ g_basis

    # ------------------------------------------------------------------
    # PEP-specific helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _monatomic_init(
        mixture: Mixture,
        em: ElementMatrix,
        b0: np.ndarray,
        FLOOR: float,
    ) -> None:
        """Initialize moles using monatomic species (NWC TP 6037 §examples).

        "Set the monatomic gases to the desired gram-atom amounts and the
        rest of the species to zero."  For each element, find the
        monatomic gas species (e.g. H for hydrogen, O for oxygen) and
        assign it that element's gram-atom abundance from b0.  All other
        species are set to FLOOR.

        If no monatomic gas exists for an element, fall back to the
        simplest (fewest total atoms) gas species that contains only
        that element, and set its moles so that it provides the correct
        gram-atom amount.

        Args:
            mixture: Mixture whose moles are modified in-place.
            em: Element matrix (already reduced to active elements).
            b0: Target element abundances, shape (n_elements,).
            FLOOR: Minimum mole value for trace species.
        """
        elements = em.elements  # list of element names
        N = len(mixture.species)

        # Set all species to FLOOR initially
        for i in range(N):
            mixture.moles[i] = FLOOR

        # For each element, find the best monatomic (or simplest) gas species
        assigned = set()
        for k, el in enumerate(elements):
            best_idx = -1
            best_atoms = math.inf
            for i in range(N):
                sp = mixture.species[i]
                if sp.condensed != 0:
                    continue
                if i in assigned:
                    continue
                sp_els = sp.elements
                # Prefer species that contain ONLY this element
                non_el_atoms = sum(v for e, v in sp_els.items() if e != el)
                if non_el_atoms > 0:
                    continue
                el_count = sp_els.get(el, 0)
                if el_count <= 0:
                    continue
                total_atoms = sum(abs(v) for v in sp_els.values())
                # Prefer monatomic (total_atoms == 1), then simplest
                if total_atoms < best_atoms:
                    best_atoms = total_atoms
                    best_idx = i

            if best_idx >= 0:
                sp = mixture.species[best_idx]
                el_count = sp.elements.get(el, 1.0)
                mixture.moles[best_idx] = max(b0[k] / el_count, FLOOR)
                assigned.add(best_idx)

    @staticmethod
    def _equilibrium_constants(
        nu: np.ndarray,
        g_all: np.ndarray,
        g_basis: np.ndarray,
    ) -> np.ndarray:
        """Compute ln Kᵢ for all species.

        ``ln Kᵢ = g°ᵢ/RT − Σⱼ ν[i,j]·g°_basis_j/RT``

        Args:
            nu: Reaction-coefficient matrix, shape (n_species, S).
            g_all: Reduced Gibbs energies g°/RT for all species, shape (n_species,).
            g_basis: Reduced Gibbs energies for basis species (in basis order), shape (S,).

        Returns:
            ln_K array, shape (n_species,).  Basis species entries are 0
            by construction (a species reacts with itself → ΔG = 0).
        """
        return g_all - nu @ g_basis

    @staticmethod
    def _compute_twid(
        nu_i: np.ndarray,
        basis_indices: List[int],
        is_gas_basis: np.ndarray,
        moles: np.ndarray,
        ln_P_offset: float,
    ) -> Tuple[float, float]:
        """Compute TWID (log activity quotient) for species i.

        ``TWID_i = Σ_{gas basis j} ν[i,j]·ln(n_j) + (Σ_{gas basis j} ν[i,j])·ln_P_offset``

        Only gas-phase basis species contribute; condensed basis carry no
        pressure or concentration term (Cruise NWC TP 6037).

        Args:
            nu_i: Row i of the reaction-coefficient matrix, shape (S,).
            basis_indices: Species indices of the S basis species.
            is_gas_basis: Boolean mask, True if basis species j is gas, shape (S,).
            moles: Full moles array for all species.
            ln_P_offset: ``ln(P/P°) − ln(n_gas_total)``.

        Returns:
            Tuple of (twid, sum_gas_nu) where sum_gas_nu = Σ_{gas j} ν[i,j].
        """
        twid = 0.0
        sum_gas_nu = 0.0
        for jj in range(len(basis_indices)):
            if is_gas_basis[jj]:
                nu_ij = nu_i[jj]
                sum_gas_nu += nu_ij
                n_j = moles[basis_indices[jj]]
                # Fortran TWID (line 989): IF (VNT(K) .LE. 0.) GO TO 1
                # Skip the log term entirely for non-positive basis moles.
                if n_j > 0.0:
                    twid += nu_ij * math.log(n_j)
        twid += sum_gas_nu * ln_P_offset
        return twid, sum_gas_nu

    @staticmethod
    def _step_bounds(
        nu_i: np.ndarray,
        basis_indices: List[int],
        moles: np.ndarray,
        n_i: float,
        is_condensed: bool,
    ) -> Tuple[float, float]:
        """Compute XMIN/XMAX step bounds ensuring all mole counts stay ≥ 0.

        For each basis species j:

        * ``ν[i,j] > 0``: n_j decreases by ν·Δζ → upper bound ``n_j/ν[i,j]``.
        * ``ν[i,j] < 0``: n_j increases (never goes negative for Δζ > 0),
          but for Δζ < 0 it could: lower bound ``n_j/ν[i,j]`` (negative).

        For gas species i itself: lower bound ``−nᵢ`` (can't go below zero).

        Args:
            nu_i: Row of reaction-coefficient matrix for species i, shape (S,).
            basis_indices: Species indices of basis species.
            moles: Full moles array.
            n_i: Current mole amount of species i.
            is_condensed: True if species i is condensed phase.

        Returns:
            Tuple of (XMIN, XMAX).
        """
        XMIN = -math.inf
        XMAX = math.inf
        for jj, j in enumerate(basis_indices):
            nu_ij = nu_i[jj]
            n_j = moles[j]
            if nu_ij > 1e-15:
                XMAX = min(XMAX, n_j / nu_ij)
            elif nu_ij < -1e-15:
                XMIN = max(XMIN, n_j / nu_ij)
        if not is_condensed:
            XMIN = max(XMIN, -n_i)
        return XMIN, XMAX


# ---------------------------------------------------------------------------
# Major-species Newton solver — compressed Newton + analytical minor species
# ---------------------------------------------------------------------------


class MajorSpeciesSolver(_ReactionAdjustmentBase):
    """Compressed Newton + analytical minor-species update.

    Combines the best features of the Gordon-McBride and PEP algorithms:

    * **PEP basis selection** keeps the Newton system small (S×S for major
      species only, where S = number of elements).
    * **Newton quadratic convergence** on the compressed system instead of
      PEP's linear serial corrections.
    * **Analytical minor-species update** from the converged element
      potentials — exact at any mole-fraction scale, no log-space artefacts.
    * **PEP outer temperature search** with interval-halving fallback.
    """

    def solve(
        self,
        problem: EquilibriumProblem,
        guess: Optional[Mixture] = None,
        log_failure: bool = True,
    ) -> EquilibriumSolution:
        """Run the major-species equilibrium iteration.

        Args:
            problem: Fully specified equilibrium problem.
            guess: Optional initial composition guess.

        Returns:
            EquilibriumSolution with the converged composition.
        """
        mixture, em, b0, T, problem = self._initialise(problem)
        if guess is not None:
            new_moles = np.zeros(len(mixture.species))
            guess_map = {sp: n for sp, n in zip(guess.species, guess.moles)}
            for i, sp in enumerate(mixture.species):
                new_moles[i] = guess_map.get(sp, 0.0)
            _b_guess = em.element_abundances(new_moles)
            if np.allclose(_b_guess, b0, rtol=1e-3):
                mixture.moles = new_moles

        P = problem.constraint2
        pi = np.zeros(em.n_elements)
        history = []

        self._last_failure_reason = None
        self._last_step_norm = float("inf")
        if problem.problem_type.fixed_temperature:
            mixture, pi, n_iter, converged, history = self._tp_equilibrium(
                mixture, em, b0, T, P
            )
        else:
            mixture, T, pi, n_iter, converged, history = self._temperature_search(
                problem, mixture, em, b0, T
            )

        residuals = em.element_residuals(mixture.moles, b0)
        element_balance_error = (
            float(np.max(np.abs(residuals))) if residuals.size else 0.0
        )
        failure_reason = None
        if not converged:
            failure_reason = (
                self._last_failure_reason
                if self._last_failure_reason is not None
                else NonConvergenceReason.MAX_ITERATIONS_REACHED
            )
        last_step_norm = float(self._last_step_norm)
        if converged and not math.isfinite(last_step_norm):
            last_step_norm = 0.0

        return EquilibriumSolution(
            mixture=mixture,
            temperature=T,
            pressure=P,
            converged=converged,
            iterations=n_iter,
            residuals=residuals,
            lagrange_multipliers=pi,
            history=history if self.capture_history else None,
            failure_reason=failure_reason,
            element_balance_error=element_balance_error,
            last_step_norm=last_step_norm,
        )

    def _tp_equilibrium(
        self,
        mixture: Mixture,
        element_matrix: ElementMatrix,
        b0: np.ndarray,
        T: float,
        P: float,
    ) -> Tuple[Mixture, np.ndarray, int, bool, List[ConvergenceStep]]:
        """Major-species inner loop: compressed Newton at fixed T and P.

        At each iteration:
          1. Select Browne basis and split non-basis gas into major/minor.
          2. Build compressed (S+nc+1)² Jacobian from major gas species only.
          3. Solve for [π, Δnc, Δln(n)].
          4. Compute Δln(nⱼ) for major gas and apply damped update.
          5. Set minor gas species analytically from π.
          6. Manage condensed phases.
          7. Test convergence.
        """
        em = element_matrix
        S = em.n_elements
        pi = np.zeros(S)
        phase_changed_counter = 0
        n_gas = mixture.n_gas
        A_gas = em.gas_rows()  # (n_gas, S) — fixed layout
        history: List[ConvergenceStep] = []

        # Precompute values that are constant throughout the inner loop.
        # g°/RT: T is fixed, so one evaluation per species suffices.
        g0_arr = np.array(
            [sp.reduced_gibbs(T) for sp in mixture.species], dtype=float
        )
        g0_gas = g0_arr[:n_gas]
        # ln(P/P°): P is constant for the entire problem.
        _ln_P_ratio = math.log(P / _P_REF)

        _log_hyb = logger.bind(solver="MajorSpecies")
        self._last_failure_reason = None
        self._last_step_norm = float("inf")
        n_inner = 0
        _tp_converged = False
        for n_inner in range(self.max_iterations):
            n_gas_arr = mixture.gas_moles()  # shape (n_gas,)
            n_gas_total = float(n_gas_arr.sum())
            n_total = mixture.total_moles

            # Reduced chemical potentials for all gas species
            mu_gas = GordonMcBrideSolver._reduced_chemical_potentials(
                mixture.species[:n_gas], n_gas_arr, n_gas_total, T, P,
                g0_arr=g0_gas,
                ln_P_ratio=_ln_P_ratio,
            )

            # --- 1. Basis selection + major/minor split ---
            basis_indices, _ = em.select_basis(mixture.moles)

            # Ensure all basis species have positive moles
            _n_basis_floor = math.exp(
                _LOG_CONC_TOL + math.log(max(n_gas_total, 1e-300))
            )
            for _bid in basis_indices:
                if mixture.moles[_bid] <= 0.0:
                    mixture.moles[_bid] = _n_basis_floor
            # Refresh n_gas_arr after any floor corrections.
            n_gas_arr = mixture.gas_moles()
            n_gas_total = float(n_gas_arr.sum())

            major_nonbasis, minor_all = self._split_major_minor(
                mixture.moles, basis_indices
            )

            # Major gas = basis gas ∪ major non-basis gas.
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

            # Record state before Newton update
            # Use current residuals for the error plot
            element_res = em.element_residuals(mixture.moles, b0)
            _el_res_max = (
                float(np.max(np.abs(element_res))) if len(element_res) else 0.0
            )

            if self._should_capture_history(n_inner):
                history.append(
                    ConvergenceStep(
                        temperature=T,
                        max_residual=_el_res_max,
                        mole_fractions=self._history_snapshot(
                            mixture, n_gas_total=n_gas_total
                        ),
                    )
                )

            # --- 2. Assemble compressed Jacobian ---
            active_cnd_local = self._active_condensed_indices(mixture)
            nc = len(active_cnd_local)

            A_maj = A_gas[major_gas_indices, :]  # (n_maj, S)
            n_maj = n_gas_arr[major_gas_indices]  # (n_maj,)
            mu_maj = mu_gas[major_gas_indices]  # (n_maj,)
            n_maj_total = float(n_maj.sum())

            A_cnd = em.condensed_rows()
            if nc > 0:
                A_cnd_act = A_cnd[active_cnd_local, :]  # (nc, S)
                cnd_global_idx = [n_gas + i for i in active_cnd_local]
                mu_cnd = g0_arr[cnd_global_idx]
            else:
                A_cnd_act = np.zeros((0, S))
                mu_cnd = np.zeros(0)

            size = S + nc + 1
            G = np.zeros((size, size + 1))  # [J | F]
            idx_n = S + nc  # column/row for Δln(n)

            # Element–element block (major gas only)
            G[:S, :S] = A_maj.T @ (n_maj[:, None] * A_maj)
            # Element–condensed and condensed–element
            if nc > 0:
                G[:S, S : S + nc] = A_cnd_act.T
                G[S : S + nc, :S] = A_cnd_act
            # Element–Δln(n) and total-moles–element (symmetric)
            G[:S, idx_n] = A_maj.T @ n_maj
            G[idx_n, :S] = A_maj.T @ n_maj
            # Total-moles–Δln(n)
            G[idx_n, idx_n] = n_maj_total - n_total

            # RHS: element balance rows
            G[:S, -1] = A_maj.T @ (n_maj * mu_maj) + element_res
            if nc > 0:
                G[S : S + nc, -1] = mu_cnd
            G[idx_n, -1] = n_total - n_maj_total + float(n_maj @ mu_maj)

            # --- 3. Solve ---
            try:
                delta_x = np.linalg.solve(G[:, :-1], G[:, -1])
            except np.linalg.LinAlgError:
                if self._manage_condensed_phases(mixture, em, pi, T, g0_arr=g0_arr):
                    phase_changed_counter += 1
                    if phase_changed_counter <= 3 * S:
                        continue
                self._last_failure_reason = NonConvergenceReason.SINGULAR_JACOBIAN
                break

            # --- 4. Extract unknowns and apply damped update ---
            pi = delta_x[:S]
            delta_n_cnd = delta_x[S : S + nc]
            delta_ln_n = delta_x[idx_n]

            lam = self._apply_damped_update(
                mixture=mixture,
                major_gas_indices=major_gas_indices,
                active_cnd_local=active_cnd_local,
                A_maj=A_maj,
                mu_maj=mu_maj,
                n_gas_arr=n_gas_arr,
                n_gas_total=n_gas_total,
                pi=pi,
                delta_n_cnd=delta_n_cnd,
                delta_ln_n=delta_ln_n,
            )

            # Gas update (log-space) for major species (for convergence check)
            delta_ln_nj_maj = A_maj @ pi + delta_ln_n - mu_maj

            # --- 5. Minor gas species: analytical update from π ---
            n_gas_updated = float(mixture.gas_moles().sum())
            self._update_minor_from_potentials(
                minor_gas_indices, mixture, em, pi, n_gas_updated, T, P,
                g0_arr=g0_arr,
                ln_P_ratio=_ln_P_ratio,
                gas_only=True,
            )

            # --- 6. Condensed phase management ---
            changed = self._manage_condensed_phases(mixture, em, pi, T, g0_arr=g0_arr)
            if changed:
                phase_changed_counter += 1
                if phase_changed_counter > 3 * S:
                    break
                continue

            # --- 7. Convergence test ---
            element_res = em.element_residuals(mixture.moles, b0)
            _el_res_max = (
                float(np.max(np.abs(element_res))) if len(element_res) else 0.0
            )
            _log_hyb.debug(
                "inner={:3d} T={:.1f} el_res={:.3e} dlnn={:.3e} n_maj={} n_min={}",
                n_inner,
                T,
                _el_res_max,
                delta_ln_n,
                len(major_gas_indices),
                len(minor_gas_indices),
            )

            # Vectorised convergence check (avoids repeated gas_moles().sum() calls).
            _n_gas_chk = float(mixture.gas_moles().sum())
            _n_maj_chk = mixture.moles[major_gas_indices]
            _valid = _n_maj_chk > 0.0
            if _valid.any():
                _step_maj = float(
                    np.max(np.abs(_n_maj_chk[_valid] * delta_ln_nj_maj[_valid]))
                ) / max(_n_gas_chk, 1e-300)
            else:
                _step_maj = 0.0
            _step_norm = max(_step_maj, abs(float(delta_ln_n)))
            self._last_step_norm = _step_norm
            if _step_norm < self.tolerance:
                _log_hyb.info("exit converged=True iters={} T={:.1f}", n_inner + 1, T)
                _tp_converged = True
                break

        else:
            _log_hyb.info("exit converged=False iters={} T={:.1f}", n_inner + 1, T)

        if not _tp_converged and self._last_failure_reason is None:
            if phase_changed_counter > 3 * S:
                self._last_failure_reason = NonConvergenceReason.CONDENSED_PHASE_CYCLING
            else:
                self._last_failure_reason = NonConvergenceReason.MAX_ITERATIONS_REACHED

        # Final exact update: set every gas species to its analytical
        # equilibrium value from the converged π.
        #
        # This undamped single-shot update is essential for outer-loop
        # convergence: when the inner loop starts from a flat composition
        # (e.g. first outer iteration) it may exit before fully converging,
        # and this step brings the composition to the correct vicinity for
        # the next outer T so subsequent inner calls converge quickly.
        #
        # The update can inflate total gas moles (Σ cⱼ ≠ 1 when π is not
        # exact), disturbing element balance.  A Newton correction pass
        # immediately after restores element conservation.
        _A_ex = em.matrix
        _n_gas_now = float(mixture.gas_moles().sum())
        _ln_n_ex = math.log(max(_n_gas_now, 1e-300))
        for _j in range(mixture.n_gas):
            _ln_eq = (
                float(_A_ex[_j, :] @ pi) - g0_arr[_j] - _ln_P_ratio + _ln_n_ex
            )
            _ln_eq = min(_ln_eq, 700.0)
            if _ln_eq - _ln_n_ex <= _LOG_CONC_TOL:
                mixture.moles[_j] = 0.0
            else:
                mixture.moles[_j] = math.exp(_ln_eq)

        # Element-balance correction: Newton steps to restore conservation.
        #
        # After the exact update the element abundances Aᵀ·n may deviate
        # from b₀ because Σⱼ cⱼ ≠ 1 at non-exact π.  We solve the S×S
        # linearised system
        #
        #   (A_gas^T diag(n_gas) A_gas) δπ = b₀ − Aᵀ·n
        #
        # and apply the correction n_j ← n_j · exp(Aⱼ · δπ) to gas
        # species only (condensed species are managed separately).
        # Iterate until the relative element-balance error is < 1e-8.
        _A_gas_ex = _A_ex[:n_gas]  # (n_gas, S)
        for _ in range(4):
            _b_curr = _A_ex.T @ mixture.moles  # includes condensed
            _b_err = b0 - _b_curr
            _rel_err = float(
                np.max(np.abs(_b_err) / np.maximum(np.abs(b0), 1e-30))
            )
            if _rel_err <= 1e-8:
                break
            _n_gas_upd = mixture.moles[:n_gas]
            _J_corr = _A_gas_ex.T @ (
                _A_gas_ex * np.maximum(_n_gas_upd, 1e-300)[:, None]
            )
            try:
                _d_pi_corr = np.linalg.solve(_J_corr, _b_err)
            except np.linalg.LinAlgError:
                break  # leave uncorrected if system is degenerate
            _ln_corr = _A_gas_ex @ _d_pi_corr  # (n_gas,)
            _active = _n_gas_upd > 0.0
            mixture.moles[:n_gas][_active] *= np.exp(_ln_corr[_active])

        return mixture, pi, n_inner + 1, _tp_converged, history

    # ------------------------------------------------------------------
    # Major-species-specific helpers
    # ------------------------------------------------------------------

    def _assemble_reduced_jacobian(
        self,
        mixture: Mixture,
        element_matrix: ElementMatrix,
        major_gas_indices: list,
        mu_gas: np.ndarray,
        b0: np.ndarray,
        T: float,
        P: float,
    ) -> np.ndarray:
        """Assemble the compressed augmented Newton matrix G = [J | F].

        Only major gas species participate in the element-balance and
        total-moles blocks.  Minor species are handled analytically by
        :py:meth:`_update_minor_from_potentials` after the Newton step.
        Active condensed phases are included explicitly (Δnc unknowns).

        Matrix structure (same as G-McB TP but restricted to major gas):

        Rows / columns:
            - S element-balance equations / Δπ (element potentials)
            - n_active_condensed condensed equilibrium equations / Δnc
            - 1 total-moles equation / Δln(n)

        Temperature is NOT in this system — it is handled by the outer
        :py:meth:`_temperature_search` loop.

        Args:
            mixture: Current composition.
            element_matrix: Element matrix (reduced to independent elements).
            major_gas_indices: Indices into the gas sub-array for major gas.
            mu_gas: Reduced chemical potentials for ALL gas species (shape n_gas).
                    Only the major-species subset is used in the Jacobian.
            b0: Target element abundances, shape (S,).
            T: Current temperature [K].
            P: Pressure [Pa].

        Returns:
            Augmented matrix G, shape (S+nc+1, S+nc+2). Last column is RHS.
        """
        em = element_matrix
        S = em.n_elements
        n_gas = mixture.n_gas
        n_gas_arr = mixture.gas_moles()
        n_total = mixture.total_moles
        A_gas = em.gas_rows()  # (n_gas, S)

        active_cnd_local = self._active_condensed_indices(mixture)
        nc = len(active_cnd_local)
        A_cnd = em.condensed_rows()

        A_maj = A_gas[major_gas_indices, :]
        n_maj = n_gas_arr[major_gas_indices]
        mu_maj = mu_gas[major_gas_indices]
        n_maj_total = float(n_maj.sum())

        if nc > 0:
            A_cnd_act = A_cnd[active_cnd_local, :]
            mu_cnd = np.array(
                [mixture.species[n_gas + i].reduced_gibbs(T) for i in active_cnd_local]
            )
        else:
            A_cnd_act = np.zeros((0, S))
            mu_cnd = np.zeros(0)

        size = S + nc + 1
        G = np.zeros((size, size + 1))
        idx_n = S + nc

        G[:S, :S] = A_maj.T @ (n_maj[:, None] * A_maj)
        if nc > 0:
            G[:S, S : S + nc] = A_cnd_act.T
            G[S : S + nc, :S] = A_cnd_act
        G[:S, idx_n] = A_maj.T @ n_maj
        G[idx_n, :S] = A_maj.T @ n_maj
        G[idx_n, idx_n] = n_maj_total - n_total

        current_ab = em.element_abundances(mixture.moles)
        G[:S, -1] = A_maj.T @ (n_maj * mu_maj) - current_ab + b0
        if nc > 0:
            G[S : S + nc, -1] = mu_cnd
        G[idx_n, -1] = n_total - n_maj_total + float(n_maj @ mu_maj)

        return G

    def _update_minor_from_potentials(
        self,
        minor_indices: list,
        mixture: Mixture,
        element_matrix: ElementMatrix,
        pi: np.ndarray,
        n_total: float,
        T: float,
        P: float,
        g0_arr: Optional[np.ndarray] = None,
        ln_P_ratio: Optional[float] = None,
        gas_only: bool = False,
    ) -> None:
        """Set minor species mole amounts analytically from element potentials.

        Once the element potentials π are known from the compressed Newton
        solve, every minor (trace) gas species is set explicitly::

            ln(nⱼ) = Σₖ πₖ·A[j,k] − μⱼ°(T)/RT − ln(P/P°) + ln(n)

        This is exact (not an approximation) and works stably at mole
        fractions as small as 10⁻³⁰ without any log-space iteration.

        Condensed minor species satisfy μⱼ°(T)/RT = Σₖ πₖ·A[j,k], so
        they are zero (or trace-deleted) if this equality does not hold.

        Parameters
        ----------
        minor_indices : list of int
        mixture : Mixture
            Updated in-place.
        element_matrix : ElementMatrix
        pi : np.ndarray, shape (n_elements,)
            Converged element potentials from the compressed Newton solve.
        n_total : float
            Current total gas moles Σ nⱼ (gas only), used in the ln(n) term.
        T : float
        P : float
        g0_arr : np.ndarray, optional
            Precomputed g°ⱼ/RT for all species (indexed by mixture position).
            When supplied the per-species thermo call is skipped.
        ln_P_ratio : float, optional
            Precomputed ln(P/P°). When supplied the log call is skipped.
        gas_only : bool
            When True, all indices in *minor_indices* are assumed to be
            gas-phase; the condensed-species filter step is skipped.
        """
        if not minor_indices:
            return

        A = element_matrix.matrix
        ln_n_gas = math.log(max(n_total, 1e-300))
        if ln_P_ratio is None:
            ln_P_ratio = math.log(P / _P_REF)

        idx = np.asarray(minor_indices, dtype=int)
        if idx.size == 0:
            return

        if gas_only:
            gas_idx = idx
        else:
            gas_idx = idx[
                np.array([mixture.species[i].condensed == 0 for i in idx], dtype=bool)
            ]
        if gas_idx.size == 0:
            return

        if g0_arr is not None:
            g0 = g0_arr[gas_idx]
        else:
            g0 = np.array(
                [mixture.species[i].reduced_gibbs(T) for i in gas_idx], dtype=float
            )
        ln_n_eq = A[gas_idx, :] @ pi - g0 - ln_P_ratio + ln_n_gas

        current = mixture.moles[gas_idx]
        current_ln = np.where(current > 0.0, np.log(current), _LOG_CONC_TOL + ln_n_gas)
        ln_nj = current_ln + np.clip(ln_n_eq - current_ln, -2.0, 2.0)

        active = (ln_nj - ln_n_gas) > _LOG_CONC_TOL
        mixture.moles[gas_idx[~active]] = 0.0
        mixture.moles[gas_idx[active]] = np.exp(ln_nj[active])

    def _apply_damped_update(
        self,
        mixture: Mixture,
        major_gas_indices: list,
        active_cnd_local: list,
        A_maj: np.ndarray,
        mu_maj: np.ndarray,
        n_gas_arr: np.ndarray,
        n_gas_total: float,
        pi: np.ndarray,
        delta_n_cnd: np.ndarray,
        delta_ln_n: float,
    ) -> float:
        """Apply a damped Newton step to major gas and condensed species.

        Uses the same damping strategy as GordonMcBrideSolver (_compute_damping):
        - λ₁: cap the maximum log-space step to 2.
        - λ₂: prevent trace species from being driven below the floor.

        Args:
            mixture: Modified in-place.
            major_gas_indices: Gas-array indices of major species.
            active_cnd_local: Local condensed indices of active phases.
            A_maj: Element matrix rows for major gas, shape (n_maj, S).
            mu_maj: Reduced chemical potentials for major gas, shape (n_maj,).
            n_gas_arr: Gas moles array, shape (n_gas,).
            n_gas_total: Total gas moles before update.
            pi: New element potentials from Newton solve, shape (S,).
            delta_n_cnd: Linear condensed corrections, shape (nc,).
            delta_ln_n: Log-total-gas-moles correction.

        Returns:
            Damping factor λ applied to the update.
        """
        n_gas = mixture.n_gas
        n_maj = n_gas_arr[major_gas_indices]
        delta_ln_nj_maj = A_maj @ pi + delta_ln_n - mu_maj

        ln_nj_maj = np.where(
            n_maj > 0,
            np.log(n_maj),
            _LOG_CONC_TOL + math.log(max(n_gas_total, 1e-300)),
        )
        lam = GordonMcBrideSolver._compute_damping(
            ln_nj_maj,
            math.log(max(n_gas_total, 1e-300)),
            delta_ln_nj_maj,
            delta_ln_n,
            0.0,
        )

        ln_n_new = math.log(max(n_gas_total, 1e-300)) + lam * delta_ln_n
        new_ln_maj = ln_nj_maj + lam * delta_ln_nj_maj
        active_maj = (new_ln_maj - ln_n_new) > _LOG_CONC_TOL
        mixture.moles[major_gas_indices[~active_maj]] = 0.0
        mixture.moles[major_gas_indices[active_maj]] = np.exp(new_ln_maj[active_maj])

        for local_idx, delta in zip(active_cnd_local, delta_n_cnd):
            global_idx = n_gas + local_idx
            mixture.moles[global_idx] = max(
                0.0, mixture.moles[global_idx] + lam * delta
            )

        return lam

    def _check_convergence(
        self,
        mixture: Mixture,
        delta_ln_nj_maj: np.ndarray,
        major_gas_indices: list,
        delta_ln_n: float,
        element_residuals: np.ndarray,
    ) -> bool:
        """Test convergence for the major gas species.

        Criteria (matching RP-1311 defaults):

        - Gas major: ``|nⱼ · Δln(nⱼ)| / n_gas < tolerance`` for all j.
        - Total moles: ``n_gas · |Δln(n)| / n_gas < tolerance``.

        Args:
            mixture: Current composition.
            delta_ln_nj_maj: Log-space corrections for major gas species.
            major_gas_indices: Gas-array indices of major species.
            delta_ln_n: Correction to ln(n_gas_total).
            element_residuals: Current b₀ − A^T·n (not used in the primary
                criterion but reserved for future tightening).

        Returns:
            True if all criteria pass.
        """
        n_gas_arr = mixture.gas_moles()
        n_gas_total = float(n_gas_arr.sum())
        tol = self.tolerance

        # Gas major: |nⱼ · Δln(nⱼ)| / n_gas < tol
        for idx, d_i in zip(major_gas_indices, delta_ln_nj_maj):
            n_i = mixture.moles[idx]
            if n_i > 0 and abs(n_i * d_i) / max(n_gas_total, 1e-300) > tol:
                return False

        # Total moles: n_gas · |Δln(n)| / n_gas = |Δln(n)| < tol
        if abs(delta_ln_n) > tol:
            return False

        return True


# ---------------------------------------------------------------------------
# Gordon-McBride solver — Lagrange multiplier / augmented Newton (RP-1311)
# ---------------------------------------------------------------------------


class GordonMcBrideSolver(EquilibriumSolver):
    """Gibbs free energy minimisation via modified Lagrange multipliers.

    Implements the full algorithm from NASA RP-1311 (Gordon & McBride, 1994),
    using `cpropep` and `CEA` as references.

    Algorithm overview
    ------------------
    The equilibrium condition for gas-phase species j is::

        μⱼ/RT = μⱼ°(T)/RT + ln(nⱼ/n_gas) + ln(P/P°) = Σₖ πₖ·A[j,k]

    where πₖ are dimensionless element potentials (Lagrange multipliers in
    units of RT).  At each Newton step the system is linearised, yielding a
    small augmented matrix whose unknowns are [π, Δnc, Δln(n), Δln(T)]:

    * **π** (shape S): solved fresh at each step — not accumulated.
    * **Δnc** (shape nc_active): linear corrections to condensed mole amounts.
    * **Δln(n)**: correction to total gas moles.
    * **Δln(T)**: temperature correction (HP/SP only).

    Matrix size is (S + nc_active + 1) for TP or (S + nc_active + 2) for
    HP/SP, where S = n_elements and nc_active = currently present condensed
    species.  Gas species are updated analytically from π after each solve.

    After each Newton step the condensed phase set is managed:

    * **Remove**: any condensed with nⱼ ≤ 0 is deactivated.
    * **Include**: any inactive condensed with μ°c/RT < Σₖ πₖ·aₖc is a
      candidate; the most favourable is activated.

    Temperature converges in the same Newton loop as composition (HP/SP),
    not in a separate outer loop.

    Args:
        max_iterations: Maximum Newton iterations before declaring
            non-convergence.
        tolerance: Convergence threshold: max ``abs(nⱼ·Δln(nⱼ)) / n_gas``
            (and similar criteria for condensed, total-moles, and temperature).
    """

    def __init__(
        self,
        max_iterations: int = 300,
        tolerance: float = 1e-4,
        capture_history: bool = False,
        history_stride: int = 1,
    ) -> None:
        super().__init__(
            max_iterations=max_iterations,
            tolerance=tolerance,
            capture_history=capture_history,
            history_stride=history_stride,
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def solve(
        self,
        problem: EquilibriumProblem,
        guess: Optional[Mixture] = None,
        log_failure: bool = True,
    ) -> EquilibriumSolution:
        """Run the Gordon-McBride iteration and return the converged state.

        Args:
            problem: Fully specified equilibrium problem.
            guess: Optional initial composition guess. If provided, its mole
                amounts are used as the starting point instead of the default
                monatomic initialization.

        Returns:
            EquilibriumSolution with the converged composition.
        """
        problem.validate()

        # Keep the full product pool and only gate species by thermo validity
        # at the CURRENT iteration temperature, not by a one-time T_init check.
        base_mixture = problem.initial_mixture()
        em_full = ElementMatrix.from_mixture(base_mixture).reduced()

        b0_full = problem.b0_array(em_full.elements)
        active_elements = [el for el, b in zip(em_full.elements, b0_full) if b > 0.0]
        active_set = set(active_elements)

        species_pool = [
            sp
            for sp in problem.products
            if all(el in active_set for el in sp.elements.keys())
        ]
        initial_moles = np.array(
            [
                next(
                    (
                        float(n)
                        for s, n in zip(base_mixture.species, base_mixture.moles)
                        if s is sp
                    ),
                    0.0,
                )
                for sp in species_pool
            ],
            dtype=float,
        )
        mixture = Mixture(species_pool, initial_moles)

        # TP must use the specified fixed temperature; HP/SP use t_init as a seed.
        T = (
            problem.constraint1
            if problem.problem_type.fixed_temperature
            else problem.t_init
        )
        P = problem.constraint2  # always pressure for TP / HP / SP

        mixture, em = self._refresh_thermo_species_set(
            mixture, species_pool, active_elements, T
        )
        b0 = problem.b0_array(em.elements)

        if guess is not None:
            # Map guess moles to the current mixture species list
            new_moles = np.zeros(len(mixture.species))
            guess_map = {sp: n for sp, n in zip(guess.species, guess.moles)}
            for i, sp in enumerate(mixture.species):
                new_moles[i] = guess_map.get(sp, 0.0)

            # Ensure the guess is somewhat close to b0 (to avoid singular Jacobian)
            _b_guess = em.element_abundances(new_moles)
            if np.allclose(_b_guess, b0, rtol=1e-3):
                mixture.moles = new_moles
            else:
                _log_g = logger.bind(solver="GordonMcBride")
                _log_g.warning(
                    "Guess composition does not match target element abundances. Falling back to monatomic start."
                )

        # Refresh once after optional guess import to ensure a valid active set.
        mixture, em = self._refresh_thermo_species_set(
            mixture, species_pool, active_elements, T
        )
        b0 = problem.b0_array(em.elements)

        pi = np.zeros(em.n_elements)  # element potentials (re-solved each step)
        converged = False
        failure_reason = None
        last_step_norm = float("inf")
        n_inner = 0
        S = em.n_elements  # constant throughout the loop
        is_tp = problem.problem_type == ProblemType.TP  # constant throughout

        # n_var: the previous iteration's n_gas_total.
        n_var = float(mixture.gas_moles().sum())
        history: List[ConvergenceStep] = []

        _log = logger.bind(solver="GordonMcBride")
        _log.info(
            "Starting solver: T_init={:.1f} K, P={:.2e} Pa, n_sp={}, S={}, type={}",
            T,
            P,
            len(mixture.species),
            S,
            problem.problem_type.name,
        )
        sp_use_gas_entropy = (
            problem.problem_type == ProblemType.SP
            and getattr(problem, "sp_entropy_mode", "total") == "gas"
        )

        for n_inner in range(self.max_iterations):
            mixture, em = self._refresh_thermo_species_set(
                mixture, species_pool, active_elements, T
            )
            b0 = problem.b0_array(em.elements)
            S = em.n_elements

            # ---- pre-compute shared quantities once per iteration ----
            n_gas = mixture.n_gas
            n_gas_arr = mixture.gas_moles()
            n_gas_total = float(n_gas_arr.sum())
            A_gas = em.gas_rows()

            if n_gas_total <= 0:
                failure_reason = NonConvergenceReason.GAS_MOLES_COLLAPSED
                _log.error("Gas moles collapsed to zero at iteration {}", n_inner)
                break

            # Always use the actual species sum for μ normalisation.
            mu_gas = self._reduced_chemical_potentials(
                mixture.species[:n_gas], n_gas_arr, n_gas_total, T, P
            )
            H_gas = (
                np.array([sp.reduced_enthalpy(T) for sp in mixture.species[:n_gas]])
                if not is_tp
                else np.zeros(n_gas)
            )

            # Record state before Newton update (snapshot of CURRENT iteration results)
            # max_residual: GordonMcBride uses nⱼ·Δln(nⱼ)/n_gas, but for the plot
            # we'll use the relative change from the PREVIOUS iteration's step.
            # Actually, the GUI wants log concentration vs iteration.
            if self._should_capture_history(n_inner):
                history.append(
                    ConvergenceStep(
                        temperature=T,
                        max_residual=0.0,  # Filled after solve
                        mole_fractions=self._history_snapshot(
                            mixture, n_gas_total=n_gas_total
                        ),
                    )
                )

            # ---- assemble and solve the Newton system ----
            G = self._assemble_jacobian(
                mixture,
                em,
                b0,
                T,
                P,
                problem.problem_type,
                sp_use_gas_entropy=sp_use_gas_entropy,
                mu_gas=mu_gas,
                H_gas=H_gas,
                n_var=n_var,
            )
            # Energy-balance RHS requires the constraint value (H₀ or S₀)
            # which is not available inside _assemble_jacobian.
            if not is_tp:
                active_cnd_local_pre = self._active_condensed_indices(mixture)
                self._fill_energy_rhs(
                    G,
                    mixture,
                    em,
                    active_cnd_local_pre,
                    T,
                    P,
                    problem.problem_type,
                    problem.constraint1,
                    sp_use_gas_entropy=sp_use_gas_entropy,
                    mu_gas=mu_gas,
                    H_gas=H_gas,
                )
            try:
                delta_x = np.linalg.solve(G[:, :-1], G[:, -1])
                if np.any(np.isnan(delta_x)):
                    raise np.linalg.LinAlgError("NaN encountered in Newton step.")
            except np.linalg.LinAlgError as e:
                _log.warning("Linear solver failed at iter {}: {}", n_inner, e)
                if self._manage_condensed_phases(mixture, em, pi, T):
                    _log.info("Attempting to recover by changing condensed phases...")
                    continue
                failure_reason = NonConvergenceReason.SINGULAR_JACOBIAN
                break  # nothing more we can do

            # ---- extract unknowns from delta_x ----
            active_cnd_local = self._active_condensed_indices(mixture)
            nc = len(active_cnd_local)

            pi = delta_x[:S]
            delta_n_cnd = delta_x[S : S + nc]
            delta_ln_n = delta_x[S + nc]
            delta_ln_T = delta_x[S + nc + 1] if not is_tp else 0.0

            # ---- gas species corrections (analytical from π) ----
            delta_ln_nj = A_gas @ pi + delta_ln_n - mu_gas + H_gas * delta_ln_T

            # ---- damping ----
            ln_n_total = math.log(max(n_gas_total, 1e-300))
            _floor = _LOG_CONC_TOL + ln_n_total
            ln_nj = np.where(
                n_gas_arr > 0, np.log(np.maximum(n_gas_arr, 1e-300)), _floor
            )
            lam = self._compute_damping(
                ln_nj,
                ln_n_total,
                delta_ln_nj,
                delta_ln_n,
                delta_ln_T,
            )

            # Update max_residual for the step we just took
            _max_crit = max(
                max(
                    (
                        abs(n_gas_arr[j] * delta_ln_nj[j]) / max(n_gas_total, 1e-300)
                        for j in range(n_gas)
                        if n_gas_arr[j] > 0
                    ),
                    default=0.0,
                ),
                abs(delta_ln_n),
                abs(delta_ln_T),
            )
            if self._should_capture_history(n_inner) and history:
                history[-1].max_residual = _max_crit
            last_step_norm = float(_max_crit)

            _log.debug(
                "iter={:3d} T={:9.2f} lam={:.4f} dlnT={:.3e} dlnn={:.3e}",
                n_inner,
                T,
                lam,
                delta_ln_T,
                delta_ln_n,
            )

            # Store n_gas_total BEFORE update so next iteration's Jacobian
            # sees G[idx_n, idx_n] ≠ 0 (n_gas diverges from n_var after update).
            n_var = n_gas_total

            # ---- apply update ----
            self._apply_update(
                mixture,
                active_cnd_local,
                lam,
                ln_nj,
                delta_ln_nj,
                delta_n_cnd,
                delta_ln_n,
                ln_n=ln_n_total,
            )
            if not is_tp:
                ln_T_new = math.log(T) + lam * delta_ln_T
                # Clamp temperature to [200, 6000] K
                T = math.exp(max(math.log(200.0), min(math.log(6000.0), ln_T_new)))

            # ---- convergence test ----
            if self._check_convergence(
                mixture,
                active_cnd_local,
                delta_ln_nj,
                delta_n_cnd,
                delta_ln_n,
                delta_ln_T,
            ):
                # Converged — now test whether condensed phases need changing.
                changed = self._manage_condensed_phases(mixture, em, pi, T)
                if not changed:
                    converged = True
                    break
                _log.info(
                    "Converged composition, but condensed phases changed. Restarting Newton..."
                )

        if converged:
            _log.info(
                "Convergence reached in {} iterations. Final T={:.2f} K", n_inner + 1, T
            )
        else:
            if log_failure:
                _log.error(
                    "Failed to converge within {} iterations. Final T={:.2f} K",
                    self.max_iterations,
                    T,
                )
            else:
                _log.debug(
                    "Exploratory solve did not converge within {} iterations. Final T={:.2f} K",
                    self.max_iterations,
                    T,
                )

        if not converged and failure_reason is None:
            failure_reason = NonConvergenceReason.MAX_ITERATIONS_REACHED

        residuals = em.element_residuals(mixture.moles, b0)
        element_balance_error = (
            float(np.max(np.abs(residuals))) if residuals.size else 0.0
        )
        return EquilibriumSolution(
            mixture=mixture,
            temperature=T,
            pressure=P,
            converged=converged,
            iterations=n_inner + 1,
            residuals=residuals,
            lagrange_multipliers=pi,
            history=history if self.capture_history else None,
            failure_reason=None if converged else failure_reason,
            element_balance_error=element_balance_error,
            last_step_norm=(
                last_step_norm if math.isfinite(last_step_norm) else float("inf")
            ),
        )

    # ------------------------------------------------------------------
    # Matrix assembly
    # ------------------------------------------------------------------

    def _assemble_jacobian(
        self,
        mixture: Mixture,
        em: ElementMatrix,
        b0: np.ndarray,
        T: float,
        P: float,
        problem_type: ProblemType,
        sp_use_gas_entropy: bool = False,
        mu_gas: np.ndarray = None,
        H_gas: np.ndarray = None,
        n_var: float = None,
    ) -> np.ndarray:
        """Build the augmented Newton matrix G = [J | F].

        Returns G with shape (size, size+1) where the last column is the
        RHS vector F.  The caller solves ``J · Δx = F`` via
        ``np.linalg.solve(G[:, :-1], G[:, -1])``.

        Args:
            mixture: Current composition estimate.
            em: Element matrix (already reduced to independent elements).
            b0: Target element abundances, shape (S,).
            T: Current temperature [K].
            P: Pressure [Pa].
            problem_type: TP, HP, or SP.
            mu_gas: Pre-computed reduced chemical potentials (optional).
            H_gas: Pre-computed reduced enthalpies (optional).
            n_var: Newton-tracked total gas moles (CEA ``soln%n``).  When
                provided, the diagonal ``G[idx_n, idx_n] = n_gas_total − n_var``
                and the RHS ``G[idx_n, -1] = n_var − n_gas_total + Σnⱼμⱼ``
                carry the residual between tracked and actual gas moles,
                providing a non-zero restoring term that stabilises the
                iteration.  If None, falls back to ``n_total`` (old behaviour).

        Returns:
            Augmented matrix of shape (size, size+1).
        """
        S = em.n_elements
        is_tp = problem_type == ProblemType.TP
        roff = 1 if is_tp else 2  # extra row/col for Δln(T)

        n_gas = mixture.n_gas
        n_gas_arr = mixture.gas_moles()  # shape (n_gas,)
        n_gas_total = float(n_gas_arr.sum())
        n_total = mixture.total_moles
        # n_ref: the "total gas moles" variable in the Newton system.
        # When n_var is tracked independently (CEA style), use it; otherwise
        # fall back to n_total so that the existing behaviour is preserved for
        # callers that do not pass n_var.
        n_ref = n_var if n_var is not None else n_total
        A_gas = em.gas_rows()  # (n_gas, S)
        A_cnd = em.condensed_rows()  # (n_cnd_all, S)

        active_local = self._active_condensed_indices(mixture)
        nc = len(active_local)
        A_cnd_act = A_cnd[active_local, :]  # (nc, S)
        n_cnd_act = mixture.condensed_moles()[active_local]  # (nc,)

        size = S + nc + roff
        G = np.zeros((size, size + 1))  # [J | F]

        idx_cnd = S
        idx_n = S + nc
        idx_T = S + nc + 1

        # Reduced chemical potentials for gas and active condensed.
        # Always normalise by the actual species sum (n_gas_total), not n_ref —
        # n_ref (n_var) is only used for the stability correction in G[idx_n, *].
        if mu_gas is None:
            mu_gas = self._reduced_chemical_potentials(
                mixture.species[:n_gas], n_gas_arr, n_gas_total, T, P
            )
        mu_cnd = np.array(
            [mixture.species[n_gas + i].reduced_gibbs(T) for i in active_local]
        )

        # ----------------------------------------------------------
        # TP common blocks (element balance + condensed eq + total n)
        # ----------------------------------------------------------

        # Element–element: Σⱼ_gas aₖⱼ·aₗⱼ·nⱼ  →  (Aᵀ·diag(n)·A)[k,l]
        G[:S, :S] = A_gas.T @ (n_gas_arr[:, None] * A_gas)

        # Element–condensed: aₖc  (stoichiometric coefficient)
        if nc > 0:
            G[:S, idx_cnd:idx_n] = A_cnd_act.T  # (S, nc)

        # Element–Δln(n): Σⱼ_gas aₖⱼ·nⱼ
        G[:S, idx_n] = A_gas.T @ n_gas_arr

        # Condensed–element (symmetric to element–condensed)
        if nc > 0:
            G[idx_cnd:idx_n, :S] = A_cnd_act  # (nc, S)

        # Total-moles–element (symmetric to element–Δln(n))
        G[idx_n, :S] = A_gas.T @ n_gas_arr

        # Total-moles–Δln(n):  −n_delta = sum(nj) − n_ref
        # CEA: ``n_delta = n - sum(nj_eff_g)``, ``G(r,c) = -n_delta`` (line 1465).
        # With n_var tracking (previous iteration's sum), n_ref ≠ n_gas_total
        # after the first step, giving a non-zero diagonal.
        G[idx_n, idx_n] = n_gas_total - n_ref

        # ----------------------------------------------------------
        # RHS vector F (last column of G)
        # ----------------------------------------------------------
        current_ab = em.element_abundances(mixture.moles)

        # Element rows: b_delta + Aᵀ·(nj·μ)  where b_delta = b₀ − current_ab.
        # CEA: ``G(r,c+1) = b_delta(i) + dot_product(tmp, mu_g)`` (line 1407).
        G[:S, -1] = A_gas.T @ (n_gas_arr * mu_gas) - current_ab + b0

        # Condensed rows: μ°c/RT  (target for Σ πₖ·aₖc)
        if nc > 0:
            G[idx_cnd:idx_n, -1] = mu_cnd

        # Total-moles row RHS:  n_delta + Σ nⱼ·μⱼ  where n_delta = n_ref − n_gas
        # CEA: ``G(r,c+1) = n_delta + dot_product(nj_linear, mu_g)`` (line 1474).
        G[idx_n, -1] = n_ref - n_gas_total + float(n_gas_arr @ mu_gas)

        # ----------------------------------------------------------
        # HP extension (Δln(T) column and energy-balance row)
        # ----------------------------------------------------------
        if problem_type == ProblemType.HP:
            if H_gas is None:
                H_gas = np.array(
                    [sp.reduced_enthalpy(T) for sp in mixture.species[:n_gas]]
                )
            H_cnd = np.array(
                [mixture.species[n_gas + i].reduced_enthalpy(T) for i in active_local]
            )
            Cp_gas_r = np.array(
                [sp.specific_heat_capacity(T) / _R for sp in mixture.species[:n_gas]]
            )
            Cp_cnd_r = np.array(
                [
                    mixture.species[n_gas + i].specific_heat_capacity(T) / _R
                    for i in active_local
                ]
            )

            nH_gas = n_gas_arr * H_gas  # nⱼ · H°ⱼ/RT

            # Extra column in element, condensed, total-moles rows
            G[:S, idx_T] = A_gas.T @ nH_gas
            if nc > 0:
                G[idx_cnd:idx_n, idx_T] = H_cnd
            G[idx_n, idx_T] = float(nH_gas.sum())

            # Energy-balance row (symmetric to extra column)
            G[idx_T, :S] = A_gas.T @ nH_gas
            if nc > 0:
                G[idx_T, idx_cnd:idx_n] = H_cnd
            G[idx_T, idx_n] = float(nH_gas.sum())
            # Diagonal: Σ nⱼ·(Cp/R + H²) for gas + Σ nⱼ·Cp/R for condensed
            G[idx_T, idx_T] = float(n_gas_arr @ (Cp_gas_r + H_gas**2)) + (
                float(n_cnd_act @ Cp_cnd_r) if nc > 0 else 0.0
            )

            # Energy RHS is filled by solve() via _fill_energy_rhs()
            # because the constraint value (H₀) is not available here.

        elif problem_type == ProblemType.SP:
            if H_gas is None:
                H_gas = np.array(
                    [sp.reduced_enthalpy(T) for sp in mixture.species[:n_gas]]
                )
            S_gas = np.array(
                [
                    sp.reduced_entropy(T)
                    - math.log(max(n_gas_arr[j], 1e-300) / max(n_gas_total, 1e-300))
                    - math.log(P / _P_REF)
                    for j, sp in enumerate(mixture.species[:n_gas])
                ]
            )
            S_cnd = np.array(
                [mixture.species[n_gas + i].reduced_entropy(T) for i in active_local]
            )
            H_cnd = np.array(
                [mixture.species[n_gas + i].reduced_enthalpy(T) for i in active_local]
            )
            Cp_gas_r = np.array(
                [sp.specific_heat_capacity(T) / _R for sp in mixture.species[:n_gas]]
            )
            Cp_cnd_r = np.array(
                [
                    mixture.species[n_gas + i].specific_heat_capacity(T) / _R
                    for i in active_local
                ]
            )

            if sp_use_gas_entropy and nc > 0:
                # Gas-only SP mode excludes condensed entropy/Cp from the
                # entropy constraint row to avoid source-dependent condensed
                # reference-state offsets during nozzle expansion.
                S_cnd = np.zeros_like(S_cnd)
                Cp_cnd_r = np.zeros_like(Cp_cnd_r)

            nS_gas = n_gas_arr * S_gas
            nH_gas = n_gas_arr * H_gas

            # Extra column in element, condensed, total-moles rows (H weighting)
            G[:S, idx_T] = A_gas.T @ nH_gas
            if nc > 0:
                G[idx_cnd:idx_n, idx_T] = H_cnd
            G[idx_n, idx_T] = float(nH_gas.sum())

            # Entropy-balance row (S weighting)
            G[idx_T, :S] = A_gas.T @ nS_gas
            if nc > 0:
                G[idx_T, idx_cnd:idx_n] = S_cnd
            G[idx_T, idx_n] = float(nS_gas.sum())
            # Diagonal: Σ nⱼ·(Cp/R + H·S) for gas + Σ nⱼ·Cp/R for condensed
            G[idx_T, idx_T] = float(n_gas_arr @ (Cp_gas_r + H_gas * S_gas)) + (
                float(n_cnd_act @ Cp_cnd_r) if nc > 0 else 0.0
            )

        return G

    def _fill_energy_rhs(
        self,
        G: np.ndarray,
        mixture: Mixture,
        em: ElementMatrix,
        active_local: List[int],
        T: float,
        P: float,
        problem_type: ProblemType,
        constraint_value: float,
        sp_use_gas_entropy: bool = False,
        mu_gas: np.ndarray = None,
        H_gas: np.ndarray = None,
    ) -> None:
        """Fill the energy-balance RHS entry in G in-place.

        Called by solve() after _assemble_jacobian() because the constraint
        value (H₀ or S₀) is not available inside _assemble_jacobian.

        Args:
            G: Augmented matrix produced by _assemble_jacobian (modified in-place).
            mixture: Current composition.
            em: Element matrix.
            active_local: Active condensed species local indices.
            T: Current temperature [K].
            P: Pressure [Pa].
            problem_type: HP or SP.
            constraint_value: H₀ [J] for HP; S₀ [J/K] for SP.
        """
        n_gas = mixture.n_gas
        n_gas_arr = mixture.gas_moles()
        n_gas_total = float(n_gas_arr.sum())
        nc = len(active_local)
        S = em.n_elements
        idx_T = S + nc + 1

        if mu_gas is None:
            mu_gas = self._reduced_chemical_potentials(
                mixture.species[:n_gas], n_gas_arr, n_gas_total, T, P
            )

        if problem_type == ProblemType.HP:
            if H_gas is None:
                H_gas = np.array(
                    [sp.reduced_enthalpy(T) for sp in mixture.species[:n_gas]]
                )
            H_cnd_all = np.array(
                [
                    mixture.species[n_gas + i].reduced_enthalpy(T)
                    for i in range(mixture.n_condensed)
                ]
            )
            product_H = float(n_gas_arr @ H_gas) + float(
                mixture.condensed_moles() @ H_cnd_all
            )
            # F_energy = H₀/(RT) − Σ nⱼ·H°ⱼ/RT + Σⱼ_gas nⱼ·(H°ⱼ/RT)·μⱼ
            G[idx_T, -1] = (
                constraint_value / (_R * T)
                - product_H
                + float(n_gas_arr @ (H_gas * mu_gas))
            )
        elif problem_type == ProblemType.SP:
            # Sⱼ_mix (dimensionless) for gas: S°ⱼ/R − ln(nⱼ/n_gas) − ln(P/P°)
            S_gas = np.array(
                [
                    sp.reduced_entropy(T)
                    - math.log(max(n_gas_arr[j], 1e-300) / max(n_gas_total, 1e-300))
                    - math.log(P / _P_REF)
                    for j, sp in enumerate(mixture.species[:n_gas])
                ]
            )
            S_cnd_all = np.array(
                [
                    mixture.species[n_gas + i].reduced_entropy(T)
                    for i in range(mixture.n_condensed)
                ]
            )
            # product_entropy (dimensionless) = Σⱼ_gas nⱼ·Sⱼ_mix (+ condensed S°ⱼ/R in total mode)
            product_S = float(n_gas_arr @ S_gas)
            if not sp_use_gas_entropy:
                product_S += float(mixture.condensed_moles() @ S_cnd_all)
            # F_entropy = S₀/R − product_entropy + n_total − n_gas + Σⱼ_gas nⱼ·Sⱼ_mix·μⱼ
            # (The n_total − n_gas term arises from the RP-1311 algebraic manipulation
            # of the entropy-balance linearisation; matches cpropep lines 575-588.)
            n_total = mixture.total_moles
            correction_term = (n_total - n_gas_total) if not sp_use_gas_entropy else 0.0
            G[idx_T, -1] = (
                constraint_value / _R
                - product_S
                + correction_term
                + float(n_gas_arr @ (S_gas * mu_gas))
            )

    # ------------------------------------------------------------------
    # Newton step helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _reduced_chemical_potentials(
        gas_species: list,
        n_gas_arr: np.ndarray,
        n_gas_total: float,
        T: float,
        P: float,
        g0_arr: Optional[np.ndarray] = None,
        ln_P_ratio: Optional[float] = None,
    ) -> np.ndarray:
        """Compute μⱼ = g°ⱼ/RT + ln(nⱼ/n_gas) + ln(P/P°) for gas species.

        Args:
            gas_species: List of gas-phase Species objects.
            n_gas_arr: Mole amounts for gas species, shape (n_gas,).
            n_gas_total: Total gas moles Σⱼ_gas nⱼ.
            T: Temperature [K].
            P: Pressure [Pa].
            g0_arr: Optional precomputed g°ⱼ/RT for all gas species. When
                supplied (e.g. from a caller that iterates at fixed T), the
                per-species thermo evaluation is skipped entirely.
            ln_P_ratio: Optional precomputed ln(P/P°). When supplied the
                log call is skipped.

        Returns:
            Reduced chemical potentials, shape (n_gas,).
        """
        ln_n = math.log(max(n_gas_total, 1e-300))
        if ln_P_ratio is None:
            ln_P_ratio = math.log(P / _P_REF)
        if g0_arr is None:
            g0_arr = np.array([sp.reduced_gibbs(T) for sp in gas_species])
        # For zero-moles species, clamp ln(nj/n) to _LOG_CONC_TOL (≈ -18.4).
        # Using ln(1e-300) ≈ -690 would produce delta_ln_nj ≈ 690 and collapse
        # the damping factor to ~0.003 on every iteration, preventing convergence.
        ln_nj_over_n = np.where(
            n_gas_arr > 0.0,
            np.log(np.where(n_gas_arr > 0.0, n_gas_arr, 1.0)) - ln_n,
            _LOG_CONC_TOL,
        )
        return g0_arr + ln_nj_over_n + ln_P_ratio

    @staticmethod
    def _compute_damping(
        ln_nj: np.ndarray,
        ln_n: float,
        delta_ln_nj: np.ndarray,
        delta_ln_n: float,
        delta_ln_T: float,
    ) -> float:
        """Compute the damping factor λ ∈ (0, 1] (CEA strategy).

        Two limits are applied:

        * **λ₁**: cap the maximum log-space step to 2, i.e. no species
          changes by more than e² ≈ 7.4× in one step.  Only considers
          *positive* corrections for species *above* the concentration
          floor (matching CEA ``compute_damped_update_factor``).
        * **λ₂**: prevent trace species (currently near the concentration
          floor) from being driven below the floor by a large positive step.

        Args:
            ln_nj: Current ln(nⱼ) for gas species, shape (n_gas,).
            ln_n: Current ln(n_gas_total).
            delta_ln_nj: Proposed log-space gas corrections, shape (n_gas,).
            delta_ln_n: Proposed ln(n) correction.
            delta_ln_T: Proposed ln(T) correction.

        Returns:
            Damping factor λ.
        """
        _FACTOR = -9.2103404  # log(1e-4)
        _SIZE = -_LOG_CONC_TOL  # 18.420681 = -log(1e-8)

        # λ₁: overall step-size limit (cap to ≤2 in log space).
        # CEA only considers *positive* dln_nj for species *above* the
        # concentration floor.  Negative corrections and floor species are
        # excluded — including them crushes lambda to ~0.04 and converts
        # quadratic Newton to linear convergence.
        l1_denom = 5.0 * max(abs(delta_ln_T), abs(delta_ln_n))
        lam2 = 1.0

        for i in range(len(delta_ln_nj)):
            d_i = delta_ln_nj[i]
            if d_i > 0.0:
                if ln_nj[i] - ln_n + _SIZE <= 0.0:
                    # Floor species with positive correction → lambda2
                    l2_denom = abs(d_i - delta_ln_n)
                    if l2_denom >= (_SIZE + _FACTOR):
                        lam2 = min(lam2, abs(_FACTOR - ln_nj[i] + ln_n) / l2_denom)
                elif d_i > l1_denom:
                    # Non-floor species with positive correction
                    l1_denom = d_i

        lam1 = 2.0 / l1_denom if l1_denom > 2.0 else 1.0

        return min(1.0, lam1, lam2)

    @staticmethod
    def _apply_update(
        mixture: Mixture,
        active_cnd_local: List[int],
        lam: float,
        ln_nj: np.ndarray,
        delta_ln_nj: np.ndarray,
        delta_n_cnd: np.ndarray,
        delta_ln_n: float,
        ln_n: float = None,
    ) -> None:
        """Apply the damped Newton step in-place.

        Gas species are updated in log-space; species below the concentration
        floor are set to zero.  Condensed species are updated linearly.

        Args:
            mixture: Modified in-place.
            active_cnd_local: Local indices (within the condensed sub-array)
                of currently active condensed species.
            lam: Damping factor.
            ln_nj: Current ln(nⱼ) for gas species.
            delta_ln_nj: Log-space gas corrections.
            delta_n_cnd: Linear condensed corrections (active only).
            delta_ln_n: Correction to ln(n_gas_total) — used to track the
                floor relative to total moles.
            ln_n: ln(n_var) for the floor reference.  If None, falls back to
                ln(sum(gas_moles)) (old behaviour).
        """
        n_gas = mixture.n_gas
        n_gas_total_new = 0.0

        # Update gas species in log-space
        new_ln_nj = ln_nj + lam * delta_ln_nj
        _ln_n_base = (
            ln_n
            if ln_n is not None
            else math.log(max(float(mixture.gas_moles().sum()), 1e-300))
        )
        new_ln_n = _ln_n_base + lam * delta_ln_n

        for i in range(n_gas):
            if new_ln_nj[i] - new_ln_n <= _LOG_CONC_TOL:
                mixture.moles[i] = 0.0
            else:
                mixture.moles[i] = math.exp(new_ln_nj[i])
                n_gas_total_new += mixture.moles[i]

        # Update active condensed species (linear space)
        for local_idx, delta in zip(active_cnd_local, delta_n_cnd):
            global_idx = n_gas + local_idx
            mixture.moles[global_idx] = max(
                0.0, mixture.moles[global_idx] + lam * delta
            )

    # ------------------------------------------------------------------
    # Convergence and condensed-phase management
    # ------------------------------------------------------------------

    def _check_convergence(
        self,
        mixture: Mixture,
        active_cnd_local: List[int],
        delta_ln_nj: np.ndarray,
        delta_n_cnd: np.ndarray,
        delta_ln_n: float,
        delta_ln_T: float,
    ) -> bool:
        """Test RP-1311 / cpropep convergence criteria.

        Args:
            mixture: Current composition.
            active_cnd_local: Active condensed local indices.
            delta_ln_nj: Gas log-space corrections.
            delta_n_cnd: Active condensed linear corrections.
            delta_ln_n: Total-moles log correction.
            delta_ln_T: Temperature log correction.

        Returns:
            True if all criteria pass.
        """
        n_gas = mixture.n_gas
        n_gas_arr = mixture.gas_moles()
        n_gas_total = float(n_gas_arr.sum())
        n_total = mixture.total_moles
        tol = self.tolerance

        # Gas species: |nⱼ · Δln(nⱼ)| / n_gas < tol
        for j in range(n_gas):
            if (
                n_gas_arr[j] > 0
                and abs(n_gas_arr[j] * delta_ln_nj[j]) / max(n_gas_total, 1e-300) > tol
            ):
                return False

        # Active condensed: |Δnc| / n_total < tol
        for i, d in zip(active_cnd_local, delta_n_cnd):
            if abs(d) / max(n_total, 1e-300) > tol:
                return False

        # Total moles: n · |Δln(n)| / n_gas < tol
        if math.isnan(delta_ln_n):
            return False
        n_total_est = math.exp(math.log(max(n_gas_total, 1e-300)) + delta_ln_n)
        if n_total_est * abs(delta_ln_n) / max(n_gas_total, 1e-300) > tol:
            return False

        # Temperature (HP/SP): |Δln(T)| < 1e-4
        if math.isnan(delta_ln_T) or abs(delta_ln_T) > 1e-4:
            return False

        return True

    # _manage_condensed_phases and _active_condensed_indices are
    # inherited from EquilibriumSolver (the common base class).
