"""
Rocket performance calculations (Frozen and Shifting equilibrium).
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
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from loguru import logger

from prometheus_equilibrium.core.constants import UNIVERSAL_GAS_CONSTANT as _R
from prometheus_equilibrium.equilibrium.diagnostics import NonConvergenceReason
from prometheus_equilibrium.equilibrium.element_matrix import ElementMatrix
from prometheus_equilibrium.equilibrium.problem import EquilibriumProblem, ProblemType
from prometheus_equilibrium.equilibrium.solution import EquilibriumSolution
from prometheus_equilibrium.equilibrium.solver import (
    EquilibriumSolver,
    GordonMcBrideSolver,
)
from prometheus_equilibrium.equilibrium.species import CalibratedSpecies


def _condensed_transition_temperature(species, T: float):
    """Return the nearest boundary of *species*' polynomial validity range.

    Used to find the calibration temperature T_tr when a condensed species'
    polynomial is out of range at *T*.  Returns the lower boundary if
    ``T < T_low``, the upper boundary if ``T > T_high``, or ``None`` if the
    range cannot be determined.
    """
    if hasattr(species, "T_low") and hasattr(species, "T_high"):
        # NASASevenCoeff
        return species.T_low if T < species.T_low else species.T_high
    if hasattr(species, "temperatures") and len(species.temperatures) >= 2:
        # NASANineCoeff, ShomateCoeff
        t_min = species.temperatures[0]
        t_max = species.temperatures[-1]
        return t_min if T < t_min else t_max
    return None


@dataclass
class RocketPerformanceResult:
    """Complete rocket performance results from chamber to exit."""

    chamber: EquilibriumSolution
    throat: EquilibriumSolution
    exit: EquilibriumSolution

    # Performance metrics
    cstar: float
    isp_actual: float
    isp_vac: float
    isp_sl: float

    # Expansion properties
    area_ratio: float
    pressure_ratio: float

    # Mode
    shifting: bool  # True if shifting, False if frozen

    # Profile for plotting
    # List of states from chamber to exit
    profile: List[EquilibriumSolution] = None


@dataclass
class RocketPerformanceComparison:
    """Paired rocket performance results for shifting and frozen expansion."""

    shifting: RocketPerformanceResult
    frozen: RocketPerformanceResult
    ambient_pressure: float


class PerformanceSolver:
    """Calculates rocket performance by isentropic nozzle expansion.

    Wraps an :class:`~prometheus.equilibrium.solver.EquilibriumSolver` to
    perform the full chamber → throat → exit calculation sequence for both
    shifting and frozen equilibrium assumptions.

    The recommended entry point for scripting is
    :py:meth:`solve_from_mixture`, which accepts a
    :class:`~prometheus.propellants.PropellantMixture` directly and handles
    product-species lookup, H₀ assembly, and problem construction internally.
    For lower-level control, :py:meth:`solve` and :py:meth:`solve_pair`
    accept a pre-built :class:`~prometheus.equilibrium.problem.EquilibriumProblem`.

    Attributes:
        solver: The underlying equilibrium solver instance.  Defaults to
            :class:`~prometheus.equilibrium.solver.GordonMcBrideSolver`.

    Example:
        ::

            perf = PerformanceSolver()
            result = perf.solve_from_mixture(
                mixture, db, p_chamber=70e5, area_ratio=10.0
            )
            print(f"c*  = {result.shifting.cstar:.1f} m/s")
            print(f"Isp = {result.shifting.isp_vac:.1f} m/s  (vacuum, shifting)")
    """

    def __init__(
        self,
        solver: Optional[EquilibriumSolver] = None,
        db=None,
    ):
        self.solver = solver or GordonMcBrideSolver()
        self.db = db  # Optional SpeciesDatabase for condensed phase transitions

    def solve(
        self,
        problem: EquilibriumProblem,
        pe_pa: Optional[float] = None,
        area_ratio: Optional[float] = None,
        shifting: bool = True,
        ambient_pressure: float = 101325.0,
        compute_profile: bool = True,
    ) -> RocketPerformanceResult:
        """Calculate performance from chamber to a fixed exit pressure or area ratio.

        Set ``compute_profile=False`` to skip intermediate expansion states and
        avoid extra SP/frozen solves when profiling or batch-sweeping cases.

        Note:
            ``cstar`` is computed from the converged throat state
            (``chamber.characteristic_velocity(throat)``), so in shifting mode it
            can show small solver-to-solver differences when the throat Mach-root
            solve lands on slightly different SP states. The current ``Isp``
            implementation is primarily chamber-to-exit based (enthalpy drop plus
            exit pressure thrust), so matching ``Isp`` does not guarantee identical
            throat-derived ``cstar``.
        """
        if pe_pa is None and area_ratio is None:
            raise ValueError("Must specify either exit pressure or area ratio.")

        logger.info(
            f"Starting performance calculation ({'Shifting' if shifting else 'Frozen'})"
        )

        # Store the full product species list from the original problem so that
        # the shifting SP solver can include species that are thermo-invalid at
        # the chamber temperature but become valid at lower expansion temps
        # (e.g. solid Al2O3 valid 300-2327K, excluded at chamber T ~3300K).
        self._full_products = list(problem.products)

        # 1. Chamber Solve (HP)
        chamber = self.solver.solve(problem)
        if not chamber.converged:
            raise RuntimeError("Chamber solve did not converge.")

        # 2. Throat Solve
        throat = self._find_throat(chamber, shifting)
        if not throat.converged:
            raise RuntimeError(
                f"Throat solve did not converge ({throat.failure_reason})."
            )

        # 3. Exit Solve
        if pe_pa is not None:
            exit_sol = self._solve_at_p(chamber, pe_pa, shifting)
        else:
            exit_sol = self._find_exit_at_area_ratio(
                chamber, throat, area_ratio, shifting
            )
        if not exit_sol.converged:
            raise RuntimeError(
                f"Exit solve did not converge ({exit_sol.failure_reason})."
            )

        # 4. Metrics
        # c* is throat-derived; Isp is computed from chamber/exit terms in
        # EquilibriumSolution.specific_impulse, so these can show different
        # cross-solver sensitivity in shifting mode.
        cstar = chamber.characteristic_velocity(throat)
        isp_actual = chamber.specific_impulse(
            throat, exit_sol, ambient_pressure=ambient_pressure
        )
        isp_vac = chamber.specific_impulse(throat, exit_sol, ambient_pressure=0.0)
        isp_sl = chamber.specific_impulse(throat, exit_sol, ambient_pressure=101325.0)

        logger.debug(
            f"Metrics: cstar={cstar:.2f}, isp_actual={isp_actual:.2f}, isp_vac={isp_vac:.2f}"
        )
        logger.debug(
            f"Enthalpies: Hc={chamber.total_enthalpy:.2f}, He={exit_sol.total_enthalpy:.2f}"
        )

        # Calculate resulting area ratio if P was fixed.
        # Use total_enthalpy (J per basis mass) for a mass-consistent velocity,
        # matching the formula used in _calculate_profile and specific_impulse.
        v_t = throat.speed_of_sound
        dH_exit = chamber.total_enthalpy - exit_sol.total_enthalpy
        v_e = math.sqrt(max(0.0, 2 * dH_exit))

        if area_ratio is None:
            calc_area_ratio = (
                (throat.density * v_t) / (exit_sol.density * v_e)
                if v_e > 0
                else float("nan")
            )
        else:
            calc_area_ratio = area_ratio

        # 5. Optional profile for plotting (additional SP/frozen solves)
        profile = (
            self._calculate_profile(chamber, exit_sol, shifting)
            if compute_profile
            else []
        )

        return RocketPerformanceResult(
            chamber=chamber,
            throat=throat,
            exit=exit_sol,
            cstar=cstar,
            isp_actual=isp_actual,
            isp_vac=isp_vac,
            isp_sl=isp_sl,
            area_ratio=calc_area_ratio,
            pressure_ratio=chamber.pressure / exit_sol.pressure,
            shifting=shifting,
            profile=profile,
        )

    def solve_pair(
        self,
        problem: EquilibriumProblem,
        pe_pa: Optional[float] = None,
        area_ratio: Optional[float] = None,
        ambient_pressure: float = 101325.0,
        compute_profile: bool = True,
    ) -> RocketPerformanceComparison:
        """Solve both shifting and frozen expansions for the same chamber case.

        Args:
            problem: Chamber equilibrium problem (typically HP).
            pe_pa: Optional exit pressure in Pa.
            area_ratio: Optional nozzle area ratio Ae/At.
            ambient_pressure: Ambient back pressure in Pa for actual Isp.
            compute_profile: Whether to compute intermediate chamber-to-exit
                profile states for plotting.

        Returns:
            Paired results for shifting and frozen expansion states.

        Raises:
            ValueError: If neither ``pe_pa`` nor ``area_ratio`` is provided.
        """
        shifting_result = self.solve(
            problem,
            pe_pa=pe_pa,
            area_ratio=area_ratio,
            shifting=True,
            ambient_pressure=ambient_pressure,
            compute_profile=compute_profile,
        )
        frozen_result = self.solve(
            problem,
            pe_pa=pe_pa,
            area_ratio=area_ratio,
            shifting=False,
            ambient_pressure=ambient_pressure,
            compute_profile=compute_profile,
        )
        return RocketPerformanceComparison(
            shifting=shifting_result,
            frozen=frozen_result,
            ambient_pressure=ambient_pressure,
        )

    def solve_from_mixture(
        self,
        mixture,
        db,
        p_chamber: float,
        pe_pa: Optional[float] = None,
        area_ratio: Optional[float] = None,
        max_atoms: int = 20,
        t_init: float = 3500.0,
        ambient_pressure: float = 101325.0,
        compute_profile: bool = True,
    ) -> RocketPerformanceComparison:
        """Convenience entry point: solve rocket performance from a PropellantMixture.

        Handles all boilerplate — product-species lookup, H₀ assembly, and
        :class:`~prometheus.equilibrium.problem.EquilibriumProblem` construction
        — so the caller only needs to supply the mixture, the species database,
        and the operating conditions.

        Args:
            mixture: A :class:`~prometheus.propellants.PropellantMixture`
                returned by
                :py:meth:`~prometheus.propellants.PropellantDatabase.mix` or
                :py:meth:`~prometheus.propellants.PropellantDatabase.expand`.
                Carries ``reactants`` [mol/kg], ``enthalpy`` [J/kg], and
                ``elements``.
            db: Loaded :class:`~prometheus.equilibrium.SpeciesDatabase`.
                Used to look up candidate product species for the element set.
            p_chamber: Chamber pressure [Pa].
            pe_pa: Exit pressure [Pa].  One of ``pe_pa`` or ``area_ratio``
                must be provided.
            area_ratio: Nozzle exit-to-throat area ratio Ae/At.  One of
                ``pe_pa`` or ``area_ratio`` must be provided.
            max_atoms: Maximum atom count for product-species candidates
                passed to ``db.get_species``.  Lower values exclude large
                molecules and speed up the solve.  Default 20.
            t_init: Initial temperature guess for the chamber Newton
                iteration [K].  Default 3500 K.
            ambient_pressure: Back pressure [Pa] for the actual-Isp
                calculation.  Default 101 325 Pa (1 atm).
            compute_profile: Whether to compute intermediate chamber-to-exit
                profile states for plotting.

        Returns:
            :class:`RocketPerformanceComparison` with paired shifting and
            frozen results.  Key fields on each
            :class:`RocketPerformanceResult`:

            - ``cstar``      — characteristic velocity c* [m/s]
            - ``isp_vac``    — vacuum specific impulse [m/s]
            - ``isp_sl``     — sea-level specific impulse [m/s]
            - ``isp_actual`` — specific impulse at *ambient_pressure* [m/s]
            - ``area_ratio`` — nozzle area ratio Ae/At
            - ``chamber``    — :class:`~prometheus.equilibrium.solution.EquilibriumSolution` at the chamber
            - ``throat``     — solution at the throat
            - ``exit``       — solution at the nozzle exit

        Raises:
            ValueError: If neither ``pe_pa`` nor ``area_ratio`` is given.
            RuntimeError: If the chamber solve does not converge.

        Example:
            ::

                from prometheus_equilibrium.equilibrium import SpeciesDatabase, PerformanceSolver
                from prometheus_equilibrium.propellants import PropellantDatabase

                db = SpeciesDatabase(
                    nasa7_path="prometheus/thermo_data/nasa7.json",
                    nasa9_path="prometheus/thermo_data/nasa9.json",
                )
                db.load()

                prop_db = PropellantDatabase(
                    "prometheus/propellants/propellants.toml"
                )
                prop_db.load()

                mixture = prop_db.mix([
                    ("AMMONIUM_PERCHLORATE",   0.68),
                    ("ALUMINUM_PURE_CRYSTALINE", 0.18),
                    ("HTPB_R_45HT",             0.14),
                ])

                result = PerformanceSolver().solve_from_mixture(
                    mixture, db, p_chamber=70e5, area_ratio=10.0
                )
                print(f"c*  = {result.shifting.cstar:.1f} m/s")
                print(f"Isp = {result.shifting.isp_vac:.1f} m/s  (vacuum, shifting)")
                print(f"Isp = {result.frozen.isp_vac:.1f} m/s  (vacuum, frozen)")
        """
        products = db.get_species(mixture.elements, max_atoms=max_atoms)

        problem = EquilibriumProblem(
            reactants=mixture.reactants,
            products=products,
            problem_type=ProblemType.HP,
            constraint1=mixture.enthalpy,
            constraint2=p_chamber,
            t_init=t_init,
        )

        return self.solve_pair(
            problem,
            pe_pa=pe_pa,
            area_ratio=area_ratio,
            ambient_pressure=ambient_pressure,
            compute_profile=compute_profile,
        )

    def _calculate_profile(self, chamber, exit_sol, shifting):
        """Calculate intermediate points for expansion plots."""
        pts = []
        n_points = 15
        # Log-space pressures from Pc down to Pe
        pressures = np.geomspace(chamber.pressure, exit_sol.pressure, n_points)

        last_guess = chamber
        for p in pressures:
            sol = self._solve_at_p(
                chamber,
                p,
                shifting,
                guess=last_guess,
                log_failure=False,
            )
            if sol.converged:
                # Add Mach number to the solution object dynamically for plotting
                # v = sqrt(2 * (Hc_mass - H_mass))
                dH_mass = chamber.total_enthalpy - sol.total_enthalpy
                v = math.sqrt(max(0.0, 2 * dH_mass))
                setattr(sol, "mach_number", v / sol.speed_of_sound)
                pts.append(sol)
                last_guess = sol
        return pts

    def _find_exit_at_area_ratio(
        self,
        chamber: EquilibriumSolution,
        throat: EquilibriumSolution,
        area_ratio_target: float,
        shifting: bool,
    ) -> EquilibriumSolution:
        """Find exit state corresponding to a given area ratio Ae/At."""
        # 1. Warm start with frozen solve to get a good initial guess
        frozen_exit = self._find_exit_at_area_ratio_frozen(
            chamber, throat, area_ratio_target
        )
        p_guess = frozen_exit.pressure

        if not shifting:
            return frozen_exit

        # 2. Newton-Raphson refinement for shifting equilibrium
        v_t = throat.speed_of_sound
        mdot_per_at = throat.density * v_t

        def get_ar_error(p, guess):
            sol = self._solve_at_p(
                chamber,
                p,
                shifting=True,
                guess=guess,
                log_failure=False,
            )
            if not sol.converged:
                return float("nan"), sol
            # v = sqrt(2 * (Hc_mass - He_mass))
            dH_mass = chamber.total_enthalpy - sol.total_enthalpy
            v = math.sqrt(max(0.0, 2 * dH_mass))
            ar = mdot_per_at / (sol.density * v) if v > 0 else 1e10
            return ar - area_ratio_target, sol

        p_curr = p_guess
        last_sol = frozen_exit

        # Newton iterations
        for _ in range(10):
            err, sol = get_ar_error(p_curr, last_sol)
            if math.isnan(err):
                break
            last_sol = sol

            if abs(err) < 1e-4:
                break

            # Numerical derivative d(Error)/dP
            dp = p_curr * 0.01
            err_p, _ = get_ar_error(p_curr + dp, last_sol)
            dedp = (err_p - err) / dp

            if abs(dedp) < 1e-20:
                break

            step = err / dedp
            # Damp the step
            p_curr -= max(-0.2 * p_curr, min(0.2 * p_curr, step))
            p_curr = max(1.0, p_curr)

        return last_sol

    def _find_exit_at_area_ratio_frozen(
        self,
        chamber: EquilibriumSolution,
        throat: EquilibriumSolution,
        area_ratio_target: float,
    ) -> EquilibriumSolution:
        """Legacy binary search for frozen exit (fast)."""
        p_low = 1.0
        p_high = throat.pressure
        v_t = throat.speed_of_sound
        mdot_per_at = throat.density * v_t

        def get_area_ratio(p):
            sol = self._solve_frozen_at_p(chamber, p)
            # v = sqrt(2 * (Hc_mass - He_mass))
            dH_mass = chamber.total_enthalpy - sol.total_enthalpy
            v = math.sqrt(max(0.0, 2 * dH_mass))
            if v <= 0:
                return 1e10
            return mdot_per_at / (sol.density * v)

        p_e = 0.1 * chamber.pressure
        for _ in range(25):
            ar = get_area_ratio(p_e)
            if ar > area_ratio_target:
                p_low = p_e
            else:
                p_high = p_e
            p_e = 0.5 * (p_low + p_high)
            if abs(p_high - p_low) / p_e < 1e-7:
                break
        return self._solve_frozen_at_p(chamber, p_e)

    def _find_throat(
        self, chamber: EquilibriumSolution, shifting: bool
    ) -> EquilibriumSolution:
        """Find the throat state (where Mach = 1)."""
        # 1. Warm start with frozen solve
        frozen_throat = self._find_throat_frozen(chamber)
        p_guess = frozen_throat.pressure

        if not shifting:
            return frozen_throat

        # 2. Bisection refinement for shifting equilibrium.
        #
        # A pure Newton-Raphson loop requires a smooth dMach/dP estimate, but
        # the shifting equilibrium solver (especially MajorSpeciesSolver) can
        # produce Mach values that vary non-monotonically at the ~1e-3 level
        # across individual Newton steps, making numerical derivatives
        # unreliable and causing oscillation.  Bisection is unconditionally
        # convergent for a monotone Mach(P) function and remains robust even
        # when individual evaluations are slightly noisy.
        def get_mach(p, guess):
            sol = self._solve_at_p(
                chamber,
                p,
                shifting=True,
                guess=guess,
                log_failure=False,
            )
            if not sol.converged:
                return float("nan"), sol
            dH_mass = chamber.total_enthalpy - sol.total_enthalpy
            v = math.sqrt(max(0.0, 2 * dH_mass))
            return v / sol.speed_of_sound, sol

        # Bracket the sonic point.  The frozen throat (subsonic side) gives
        # a good initial bracket; expand the lower bound until we find a
        # supersonic point or fall back to the frozen solution.
        p_low = 0.1 * chamber.pressure
        p_high = p_guess

        mach_high, sol_high = get_mach(p_high, frozen_throat)
        if math.isnan(mach_high):
            return frozen_throat  # can't evaluate at frozen guess

        # Ensure p_high is subsonic (Mach < 1) and p_low is supersonic.
        # Mach increases as P decreases, so we scan downward for a supersonic
        # bracket point.
        if mach_high >= 1.0:
            # Frozen throat is already supersonic — scan upward for p_high
            p_scan = p_high
            sol_scan = sol_high
            for _ in range(6):
                p_scan = min(0.95 * chamber.pressure, p_scan * 1.5)
                mach_scan, sol_scan = get_mach(p_scan, sol_scan)
                if not math.isnan(mach_scan) and mach_scan < 1.0:
                    p_high = p_scan
                    sol_high = sol_scan
                    mach_high = mach_scan
                    break
            else:
                return frozen_throat  # couldn't bracket

        mach_low, sol_low = get_mach(p_low, frozen_throat)
        if math.isnan(mach_low) or mach_low <= 1.0:
            # Try to find a supersonic bracket near the frozen guess
            p_low = p_guess * 0.9
            mach_low, sol_low = get_mach(p_low, frozen_throat)
            if math.isnan(mach_low) or mach_low <= 1.0:
                return frozen_throat

        last_sol = sol_high
        for _ in range(30):
            if (p_high - p_low) / chamber.pressure < 1e-6:
                break
            p_mid = 0.5 * (p_low + p_high)
            mach_mid, sol_mid = get_mach(p_mid, last_sol)
            if math.isnan(mach_mid):
                break
            last_sol = sol_mid
            if mach_mid > 1.0:
                p_low = p_mid
            else:
                p_high = p_mid
                sol_high = sol_mid

        return sol_high

    def _find_throat_frozen(self, chamber: EquilibriumSolution) -> EquilibriumSolution:
        """Binary search for frozen throat (fast warm-start for shifting solver)."""
        p_low = 0.1 * chamber.pressure
        p_high = 0.98 * chamber.pressure

        def get_mach(p):
            sol = self._solve_frozen_at_p(chamber, p)
            dH_mass = chamber.total_enthalpy - sol.total_enthalpy
            v = math.sqrt(max(0.0, 2 * dH_mass))
            return v / sol.speed_of_sound, sol

        # Initial estimate from RP-1311 Eq. 6.15: P_t = P_c / ((γ/2+0.5)^(γ/(γ-1)))
        g = chamber.gamma
        try:
            _exp = g / (g - 1.0)
            p_t_est = chamber.pressure / ((g / 2.0 + 0.5) ** _exp)
        except (ZeroDivisionError, OverflowError):
            p_t_est = 0.55 * chamber.pressure
        p_t = max(p_low, min(p_high, p_t_est))

        last_sol = chamber
        for _ in range(20):
            mach, sol = get_mach(p_t)
            last_sol = sol
            if mach > 1.0:
                p_low = p_t
            else:
                p_high = p_t
            p_t = 0.5 * (p_low + p_high)
            if abs(p_high - p_low) / chamber.pressure < 1e-6:
                break
        return last_sol

    def _solve_at_p(
        self,
        chamber: EquilibriumSolution,
        p_target: float,
        shifting: bool,
        guess: Optional[EquilibriumSolution] = None,
        log_failure: bool = True,
    ) -> EquilibriumSolution:
        """Solve for state at a given pressure with frozen or shifting constraints."""
        if shifting:
            if guess is None and chamber.pressure > 1.5 * p_target:
                continuation_sol = self._solve_shifting_with_continuation(
                    chamber,
                    p_target,
                    log_failure=log_failure,
                )
                if continuation_sol.converged:
                    return continuation_sol
                logger.warning(
                    "Continuation SP solve did not converge at P={:.3e} Pa; retrying direct SP solve.",
                    p_target,
                )
            return self._solve_shifting_sp_mode(
                chamber,
                p_target,
                guess=guess,
                log_failure=log_failure,
            )
        else:
            # Frozen: fix composition at chamber values
            return self._solve_frozen_at_p(chamber, p_target)

    def _solve_shifting_sp_mode(
        self,
        chamber: EquilibriumSolution,
        p_target: float,
        guess: Optional[EquilibriumSolution],
        log_failure: bool,
        t_init_override: Optional[float] = None,
    ) -> EquilibriumSolution:
        """Solve one shifting SP state at fixed pressure."""
        from prometheus_equilibrium.equilibrium.problem import EquilibriumProblem as EP

        chamber_mix = chamber.mixture
        guess_mix = guess.mixture if guess else None

        # Use the full product list from the original HP problem so that
        # species valid only at lower temperatures (e.g. solid Al2O3, valid
        # 300-2327K) can be reintroduced by the solver as T drops during
        # nozzle expansion.  Without this, the chamber solve (at ~3300K)
        # excludes low-T condensed species and the shifting solver cannot
        # condense them at exit conditions.
        full_products = getattr(self, "_full_products", None) or chamber_mix.species

        sp_products = list(full_products)

        reactant_dict = {
            sp: n for sp, n in zip(chamber_mix.species, chamber_mix.moles) if n > 0
        }
        entropy_target = chamber_mix.total_entropy(
            chamber.temperature, chamber.pressure
        )
        sp_prob = EP(
            reactants=reactant_dict,
            products=sp_products,
            problem_type=ProblemType.SP,
            constraint1=entropy_target,
            constraint2=p_target,
            t_init=(
                float(t_init_override)
                if t_init_override is not None
                else (guess.temperature if guess else chamber.temperature * 0.8)
            ),
        )
        solution = self.solver.solve(
            sp_prob,
            guess=guess_mix,
            log_failure=log_failure,
        )
        if solution.converged:
            gs = self._compute_gamma_s(solution)
            if gs is not None:
                solution.gamma_s = gs

        logger.debug(
            "SP @P={:.3e} converged={} T={:.2f}K S_target={:.6e} gamma_s={}",
            p_target,
            solution.converged,
            solution.temperature,
            entropy_target,
            solution.gamma_s,
        )
        return solution

    def _solve_shifting_with_continuation(
        self,
        chamber: EquilibriumSolution,
        p_target: float,
        log_failure: bool,
    ) -> EquilibriumSolution:
        """March SP solution from chamber pressure to target pressure in small steps.

        The continuation path improves branch stability near condensed phase
        transitions for whole-flow entropy constraints.
        """
        n_steps = int(
            np.clip(
                math.ceil(math.log(chamber.pressure / p_target) / math.log(1.35)), 6, 30
            )
        )
        pressures = np.geomspace(chamber.pressure, p_target, n_steps + 1)[1:]

        last = chamber
        for idx, p_step in enumerate(pressures):
            sol = self._solve_shifting_sp_mode(
                chamber,
                float(p_step),
                guess=last,
                log_failure=False if idx < (len(pressures) - 1) else log_failure,
            )
            if not sol.converged:
                return sol

            if self._is_branch_jump(last, sol):
                retry = self._retry_smooth_branch(
                    chamber=chamber,
                    p_target=float(p_step),
                    reference=last,
                    log_failure=False if idx < (len(pressures) - 1) else log_failure,
                )
                if retry is not None and retry.converged:
                    if self._branch_distance(
                        reference=last, candidate=retry
                    ) < self._branch_distance(reference=last, candidate=sol):
                        logger.warning(
                            "SP continuation branch guard selected alternate state at P={:.3e} Pa.",
                            p_step,
                        )
                        sol = retry

            last = sol
        return last

    @staticmethod
    def _is_branch_jump(
        previous: EquilibriumSolution, current: EquilibriumSolution
    ) -> bool:
        """Return whether the current SP point shows a likely non-physical branch jump."""
        if current.gas_mean_molar_mass <= previous.gas_mean_molar_mass:
            return False
        mass_ratio = current.gas_mean_molar_mass / max(
            previous.gas_mean_molar_mass, 1e-12
        )
        temperature_drop = previous.temperature - current.temperature
        return mass_ratio > 1.25 and temperature_drop > 120.0

    @staticmethod
    def _branch_distance(
        reference: EquilibriumSolution, candidate: EquilibriumSolution
    ) -> float:
        """Compute a smoothness score against the previous continuation point."""
        mass_ratio = candidate.gas_mean_molar_mass / max(
            reference.gas_mean_molar_mass, 1e-12
        )
        d_mass = abs(math.log(max(mass_ratio, 1e-12)))
        d_temp = abs(candidate.temperature - reference.temperature) / max(
            reference.temperature, 1.0
        )
        return d_mass + d_temp

    def _retry_smooth_branch(
        self,
        chamber: EquilibriumSolution,
        p_target: float,
        reference: EquilibriumSolution,
        log_failure: bool,
    ) -> Optional[EquilibriumSolution]:
        """Retry one pressure point with nearby temperature seeds and pick smoothest state."""
        seeds = (
            reference.temperature * 0.98,
            reference.temperature * 1.00,
            reference.temperature * 1.02,
        )
        best: Optional[EquilibriumSolution] = None
        best_score = float("inf")

        # Candidate family 1: reuse the previous shifting state with nearby T seeds.
        for seed in seeds:
            candidate = self._solve_shifting_sp_mode(
                chamber,
                p_target,
                guess=reference,
                log_failure=log_failure,
                t_init_override=seed,
            )
            if not candidate.converged:
                continue
            score = self._branch_distance(reference=reference, candidate=candidate)
            if score < best_score:
                best = candidate
                best_score = score

        # Candidate family 2: seed from frozen state to probe alternate whole-flow branch.
        frozen_seed = self._solve_frozen_at_p(chamber, p_target)
        frozen_seed.temperature = reference.temperature
        for seed in seeds:
            candidate = self._solve_shifting_sp_mode(
                chamber,
                p_target,
                guess=frozen_seed,
                log_failure=log_failure,
                t_init_override=seed,
            )
            if not candidate.converged:
                continue
            score = self._branch_distance(reference=reference, candidate=candidate)
            if score < best_score:
                best = candidate
                best_score = score
        return best

    def _apply_phase_transitions(self, mix: "Mixture", T: float) -> "Mixture":
        """Replace condensed species whose thermo data is invalid at *T*.

        For each condensed species in *mix* whose Cp evaluates to nan or zero
        at *T*, looks up a phase partner in ``self.db`` valid at *T* and
        substitutes a :class:`~prometheus_equilibrium.equilibrium.species.CalibratedSpecies`
        that is enthalpy/entropy-continuous at the transition boundary,
        regardless of the database source of each phase.

        The calibration offsets are computed at the nearest boundary of the
        out-of-range species' polynomial:

        .. math::

            \\Delta h = H_{\\text{old}}(T_{\\text{tr}}) - H_{\\text{new}}(T_{\\text{tr}})

            \\Delta s = S_{\\text{old}}(T_{\\text{tr}}) - S_{\\text{new}}(T_{\\text{tr}})

        This ensures that enthalpy and entropy are continuous at the phase
        boundary so that the Newton iteration in ``_solve_frozen_at_p`` remains
        consistent, even when the two phases come from databases with different
        reference states.  ``Cp`` is not offset — the new phase's Cp is used
        directly, which is the physically correct choice.

        Args:
            mix: Current mixture whose condensed species may need replacing.
            T: Target temperature [K].

        Returns:
            The original *mix* if no substitutions were needed, otherwise a
            new :class:`Mixture` with calibrated replacement species and the
            same moles.
        """
        if self.db is None:
            return mix
        new_species = list(mix.species)
        changed = False
        for idx, sp in enumerate(new_species):
            if sp.condensed == 0:
                continue
            cp = sp.specific_heat_capacity(T)
            if not math.isfinite(cp) or cp <= 0.0:
                # Find the base species to determine the transition boundary.
                # CalibratedSpecies delegates Cp to its base, so OOR detection
                # means the base is also OOR — use the base's boundary.
                base_sp = sp._base if isinstance(sp, CalibratedSpecies) else sp
                T_tr = _condensed_transition_temperature(base_sp, T)

                partner = self.db.condensed_phase_partner(sp, T)
                if partner is not None and partner is not sp and T_tr is not None:
                    h_offset = sp.enthalpy(T_tr) - partner.enthalpy(T_tr)
                    s_offset = sp.entropy(T_tr) - partner.entropy(T_tr)
                    new_species[idx] = CalibratedSpecies(partner, h_offset, s_offset)
                    changed = True
        if not changed:
            return mix
        from prometheus_equilibrium.equilibrium.mixture import Mixture

        return Mixture(new_species, mix.moles.copy())

    @staticmethod
    def _compute_gamma_s(solution: EquilibriumSolution) -> Optional[float]:
        """Compute the equilibrium isentropic exponent γₛ (NASA RP-1311 §2.5).

        Solves two auxiliary linear systems (const-P and const-T sensitivities)
        sharing the same (ne+1)×(ne+1) Jacobian built from the converged gas-
        phase species.  Condensed phases are excluded from the compressibility
        calculation per the standard ideal-gas assumption for nozzle flow.

        Returns ``None`` if the system is underdetermined (no gas species,
        singular Jacobian, or non-finite thermo values).  The caller should
        fall back to the frozen γ in that case.

        Args:
            solution: Converged SP equilibrium solution.

        Returns:
            γₛ > 1 (equilibrium isentropic exponent), or ``None``.
        """
        mix = solution.mixture
        T = solution.temperature
        n_gas = mix.n_gas
        if n_gas == 0:
            return None

        n_gas_arr = mix.gas_moles()  # shape (n_gas,)
        n_gas_total = float(n_gas_arr.sum())
        if n_gas_total <= 0.0:
            return None

        # Build the stoichiometric matrix using only the full mixture
        # (Mixture keeps gas before condensed, so gas_rows() is safe).
        em = ElementMatrix.from_mixture(mix)
        A_gas = em.gas_rows()  # (n_gas, ne)
        ne = A_gas.shape[1]

        # Dimensionless enthalpies h_j = H_j(T)/(R·T)
        h_gas = np.array([sp.reduced_enthalpy(T) for sp in mix.species[:n_gas]])
        if not np.all(np.isfinite(h_gas)):
            return None

        # ---- Build (ne+1)×(ne+1) Jacobian (RP-1311 Tables 2.3/2.4) ----
        # J[:ne, :ne] = A_gas.T @ diag(n_gas) @ A_gas   (element-element)
        # J[:ne, ne] = J[ne, :ne] = A_gas.T @ n_gas      (element–Δln(n))
        # J[ne, ne] = 0  (diagonal for Δln(n) row is zero in CEA's formulation)
        J = np.zeros((ne + 1, ne + 1))
        J[:ne, :ne] = A_gas.T @ (n_gas_arr[:, None] * A_gas)
        col = A_gas.T @ n_gas_arr  # shape (ne,)
        J[:ne, ne] = col
        J[ne, :ne] = col
        # J[ne, ne] = 0.0 already (from np.zeros)

        cond = np.linalg.cond(J)
        if not math.isfinite(cond) or cond > 1e12:
            return None

        # ---- Const-P RHS (∂/∂lnT at const P, RP-1311 Table 2.3) ----
        # rhs_p[i] = -Σⱼ nⱼ·Aᵢⱼ·hⱼ   for element row i
        # rhs_p[ne] = -Σⱼ nⱼ·hⱼ         for total-moles row
        rhs_p = np.zeros(ne + 1)
        rhs_p[:ne] = -(A_gas.T @ (n_gas_arr * h_gas))
        rhs_p[ne] = -float(n_gas_arr @ h_gas)

        # ---- Const-T RHS (∂/∂lnP at const T, RP-1311 Table 2.4) ----
        # rhs_t[:ne] = A_gas.T @ n_gas  (element abundances)
        # rhs_t[ne]  = n_gas_total
        rhs_t = np.zeros(ne + 1)
        rhs_t[:ne] = col
        rhs_t[ne] = n_gas_total

        try:
            X = np.linalg.solve(J, np.column_stack([rhs_p, rhs_t]))
        except np.linalg.LinAlgError:
            return None

        # Extract sensitivities (last row = dn/dlnT and dn/dlnP)
        dn_dlnT = float(X[ne, 0])
        dn_dlnP = float(X[ne, 1])
        dpi_dlnT = X[:ne, 0]  # shape (ne,)

        dlnV_dlnT = 1.0 + dn_dlnT
        dlnV_dlnP = -1.0 + dn_dlnP

        # ---- Equilibrium Cp/R (RP-1311 Eq. 2.59, pure-gas terms) ----
        # Term 4 (frozen):  Σⱼ nⱼ·Cpⱼ/R
        # Term 1:           Σᵢ (Σⱼ nⱼ·Aᵢⱼ·hⱼ) · (∂πᵢ/∂lnT)
        # Term 3:           (Σⱼ nⱼ·hⱼ) · dn_dlnT
        # Term 5:           Σⱼ nⱼ·hⱼ²
        cp_fr_over_R = mix.total_gas_cp(T) / _R
        nA_dot_h = A_gas.T @ (n_gas_arr * h_gas)  # (ne,): Σⱼ nⱼ·Aᵢⱼ·hⱼ per element
        cp_eq_over_R = (
            cp_fr_over_R
            + float(nA_dot_h @ dpi_dlnT)
            + float(n_gas_arr @ h_gas) * dn_dlnT
            + float(n_gas_arr @ (h_gas * h_gas))
        )

        if cp_eq_over_R <= 0.0 or not math.isfinite(cp_eq_over_R):
            return None

        # ---- γₛ = −1 / (∂lnV/∂lnP + (∂lnV/∂lnT)² · n / Cp_eq) ----
        denom = dlnV_dlnP + (dlnV_dlnT**2) * n_gas_total / cp_eq_over_R
        if not math.isfinite(denom) or abs(denom) < 1e-12:
            return None

        gamma_s = -1.0 / denom
        if not math.isfinite(gamma_s) or gamma_s <= 1.0:
            return None

        return gamma_s

    def _solve_frozen_at_p(
        self, chamber: EquilibriumSolution, p_target: float
    ) -> EquilibriumSolution:
        """Find T such that S(T, P_target, n_chamber) = S_chamber."""
        s_target = chamber.entropy
        mix = chamber.mixture.copy()

        # Use Newton to find T, with safety clamping
        T = chamber.temperature * 0.8
        i = 0
        converged = False
        failure_reason: Optional[NonConvergenceReason] = None
        last_step_norm = float("nan")
        for i in range(50):
            # Clamp T to valid range for thermo data
            T = max(200.0, min(6000.0, T))
            mix = self._apply_phase_transitions(mix, T)
            s_curr = mix.entropy(T, p_target)
            cp_curr = mix.cp(T)

            if not (
                math.isfinite(s_curr) and math.isfinite(cp_curr) and math.isfinite(T)
            ):
                failure_reason = NonConvergenceReason.INVALID_THERMO_PROPERTIES
                break

            ds_dT = cp_curr / T
            if not math.isfinite(ds_dT) or abs(ds_dT) < 1e-10:
                failure_reason = NonConvergenceReason.INVALID_THERMO_PROPERTIES
                break

            dT = (s_target - s_curr) / ds_dT
            if not math.isfinite(dT):
                failure_reason = NonConvergenceReason.INVALID_THERMO_PROPERTIES
                break

            last_step_norm = abs(dT)
            # Damp the Newton step
            T += max(-200.0, min(200.0, dT))

            if abs(dT) < 1e-3:
                converged = True
                break

        if not converged and failure_reason is None:
            failure_reason = NonConvergenceReason.MAX_ITERATIONS_REACHED

        return EquilibriumSolution(
            mixture=mix,
            temperature=T,
            pressure=p_target,
            converged=converged,
            iterations=i + 1,
            residuals=np.zeros(0),
            lagrange_multipliers=np.zeros(0),
            failure_reason=failure_reason if not converged else None,
            element_balance_error=0.0,
            last_step_norm=last_step_norm,
        )
