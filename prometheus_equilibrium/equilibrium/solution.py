"""
EquilibriumSolution — converged state from a chemical equilibrium calculation.

For rocket propulsion the two primary calculations are:

1. **Combustion chamber** (HP problem): reactants at known enthalpy and
   chamber pressure → equilibrium temperature T_c, species composition,
   mixture properties (Cp, γ, M̄).

2. **Nozzle expansion** (SP problem): chamber state isentropically expanded
   to throat or exit → T_e, P_e, frozen/shifting species, specific impulse.

This class holds the output of either calculation and provides properties
relevant to both.  Rocket-specific derived quantities (c*, Isp, ṁ, …) are
computed from combinations of the basic mixture properties.
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
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from prometheus_equilibrium.core.constants import UNIVERSAL_GAS_CONSTANT as R
from prometheus_equilibrium.equilibrium.diagnostics import NonConvergenceReason
from prometheus_equilibrium.equilibrium.mixture import Mixture


@dataclass
class ConvergenceStep:
    """Snapshot of solver state at a single iteration."""

    temperature: float
    max_residual: float
    mole_fractions: Dict[str, float]


@dataclass
class EquilibriumSolution:
    """Converged thermodynamic state from an equilibrium calculation.

    Attributes:
        mixture: Converged species mixture with mole amounts nⱼ.
        temperature: Equilibrium temperature T [K].
        pressure: Equilibrium pressure P [Pa].
        converged: True if all convergence criteria were satisfied.
        iterations: Number of Newton iterations taken.
        residuals: Final element-balance residuals b₀ − Aᵀ·n, shape (n_elements,).
        lagrange_multipliers: Converged reduced Lagrange multipliers π,
            shape (n_elements,).  The chemical potential of element k at
            equilibrium is λₖ = −R·T·πₖ.
        history: List of states at each iteration, used for convergence plots.
        failure_reason: If ``converged`` is False, the enum value describing
            why the solver stopped.  ``None`` on a successful solve.
        element_balance_error: ``max(|b₀ − Aᵀ·n|)`` over all elements at the
            final iteration.  Always populated (even on convergence) so it can
            be used to verify element conservation independently.
        last_step_norm: Solver-specific convergence criterion on the final
            iteration — the same quantity that is compared against ``tolerance``.
            For Newton solvers this is ``max(|nⱼ·Δln nⱼ| / n_gas, |Δln n|)``;
            for PEP and the outer temperature search it is the energy/element
            residual norm at the last step.  Always populated.
    """

    mixture: Mixture
    temperature: float
    pressure: float
    converged: bool
    iterations: int
    residuals: np.ndarray
    lagrange_multipliers: np.ndarray
    history: Optional[List[ConvergenceStep]] = None
    failure_reason: Optional[NonConvergenceReason] = None
    element_balance_error: Optional[float] = None
    last_step_norm: Optional[float] = None

    # ------------------------------------------------------------------
    # Composition
    # ------------------------------------------------------------------

    @property
    def mole_fractions(self) -> Dict[str, float]:
        """Mole fractions {species_formula: xⱼ}.

        Returns:
            Dict mapping each species' human-readable formula to its
            mole fraction.
        """
        xj = self.mixture.mole_fractions
        return {sp.formula: float(x) for sp, x in zip(self.mixture.species, xj)}

    def major_species(self, threshold: float = 1e-4) -> Dict[str, float]:
        """Mole fractions of species above *threshold* (default 0.01 %).

        Args:
            threshold: Minimum mole fraction to include in the result.

        Returns:
            Dict of ``{element_string: mole_fraction}`` for all species
            with xⱼ ≥ threshold, sorted descending by mole fraction.
        """
        return {
            k: v
            for k, v in sorted(self.mole_fractions.items(), key=lambda kv: -kv[1])
            if v >= threshold
        }

    # ------------------------------------------------------------------
    # Mixture thermodynamic properties
    # ------------------------------------------------------------------

    @property
    def mean_molar_mass(self) -> float:
        """Mean molar mass M̄ [kg/mol].

        For a two-phase mixture this includes both gas and condensed
        species weighted by mole fraction.  Use
        :py:attr:`gas_mean_molar_mass` for rocket performance calculations.
        """
        return self.mixture.mean_molar_mass

    @property
    def gas_mean_molar_mass(self) -> float:
        """Gas-phase mean molar mass M̄_gas [kg/mol].

        Used for speed of sound and density in the presence of condensed
        products (e.g. Al₂O₃ in aluminised propellants).
        """
        return self.mixture.gas_mean_molar_mass

    @property
    def cp(self) -> float:
        """Mixture Cp at equilibrium T [J/(mol·K)].

        Molar Cp over all species (gas + condensed), per mole of mixture.
        This is the *frozen* Cp (composition fixed).
        """
        return self.mixture.cp(self.temperature)

    @property
    def cv(self) -> float:
        """Mixture Cv at equilibrium T [J/(mol·K)].

        Ideal-gas approximation: Cv = Cp − R.  Only valid for the
        gas-phase contribution; for mixtures with significant condensed
        mass the correction is smaller.
        """
        return self.cp - R

    @property
    def gamma(self) -> float:
        """Ratio of specific heats γ = Cp / Cv (frozen-flow isentropic exponent).

        Returns:
            Dimensionless γ > 1.  For an ideal monatomic gas γ = 5/3;
            for a diatomic gas γ ≈ 7/5.
        """
        return self.cp / self.cv

    @property
    def enthalpy(self) -> float:
        """Mixture absolute enthalpy H [J/mol] at equilibrium T."""
        return self.mixture.enthalpy(self.temperature)

    @property
    def entropy(self) -> float:
        """Mixture entropy S [J/(mol·K)] at equilibrium T and P."""
        return self.mixture.entropy(self.temperature, self.pressure)

    @property
    def gibbs(self) -> float:
        """Mixture Gibbs free energy G = H − T·S [J/mol]."""
        return self.enthalpy - self.temperature * self.entropy

    # ------------------------------------------------------------------
    # Rocket-specific derived properties
    # ------------------------------------------------------------------

    @property
    def speed_of_sound(self) -> float:
        """Frozen speed of sound a [m/s] at the equilibrium state.

        Uses the gas-phase mean molar mass (excludes condensed species):

        .. math::

            a = \\sqrt{\\gamma \\cdot R \\cdot T \\,/\\, \\bar{M}_{\\text{gas}}}

        Returns:
            Speed of sound in m/s.

        Raises:
            ValueError: If there are no gas-phase species.
        """
        M_gas = self.gas_mean_molar_mass
        if M_gas <= 0.0:
            raise ValueError("No gas-phase species — cannot compute speed of sound.")
        return math.sqrt(self.gamma * R * self.temperature / M_gas)

    @property
    def density(self) -> float:
        """Gas-phase mixture density ρ [kg/m³] at equilibrium T and P.

        From the ideal-gas law:

        .. math::

            \\rho = \\frac{P \\cdot \\bar{M}_{\\text{gas}}}{R \\cdot T}
        """
        return self.pressure * self.gas_mean_molar_mass / (R * self.temperature)

    def characteristic_velocity(self, throat: "EquilibriumSolution") -> float:
        """Characteristic velocity c* [m/s].

        Computed from the throat conditions (isentropic, frozen flow):

        .. math::

            c^* = \\frac{a_t}{\\Gamma}
            \\quad\\text{where }\\Gamma = \\sqrt{\\gamma_t}
                \\left(\\frac{2}{\\gamma_t+1}\\right)^{(\\gamma_t+1)/[2(\\gamma_t-1)]}

        Args:
            throat: The converged state at the nozzle throat (Mach 1).

        Returns:
            Characteristic velocity c* in m/s.
        """
        g = throat.gamma
        gamma_factor = math.sqrt(g) * (2 / (g + 1)) ** ((g + 1) / (2 * (g - 1)))
        return throat.speed_of_sound / gamma_factor

    def specific_impulse(
        self,
        throat: "EquilibriumSolution",
        exit: "EquilibriumSolution",
        ambient_pressure: float = 0.0,
    ) -> float:
        """Specific impulse Isp [s].

        Vacuum Isp (ambient_pressure=0) is the standard figure of merit.
        Uses the enthalpy-drop formula for exit velocity:

        .. math::

            v_e = \\sqrt{2 (h_c - h_e)}

        where h_c and h_e are the extensive enthalpies [J/kg] at
        chamber and exit conditions.

        Args:
            throat: Converged state at the nozzle throat.
            exit: Converged state at the nozzle exit plane.
            ambient_pressure: Ambient back-pressure [Pa].  Use 0 for vacuum Isp.

        Returns:
            Specific impulse in seconds (referenced to standard g₀ = 9.80665 m/s²).
        """
        g0 = 9.80665  # m/s²

        # Enthalpy drop on a mass basis [J/kg]
        # self.total_enthalpy returns J (extensive)
        # Mixtures in solutions derived from PropellantDatabase always represent 1 kg.
        dh_j_kg = self.total_enthalpy - exit.total_enthalpy

        if dh_j_kg <= 0:
            v_exit = 0.0
        else:
            v_exit = math.sqrt(2 * dh_j_kg)

        # Pressure thrust term: (P_e - P_a) / (rho_e * v_e)
        # thrust / mdot = v_e + (Pe - Pa) * Ae / mdot
        # Ae / mdot = 1 / (rho_e * v_e)
        if exit.density * v_exit > 0:
            F_pressure = (exit.pressure - ambient_pressure) / (exit.density * v_exit)
            v_eff = v_exit + F_pressure
        else:
            v_eff = v_exit

        return v_eff / g0

    @property
    def total_enthalpy(self) -> float:
        """Total mixture enthalpy H_total = Σ nⱼ·Hⱼ°(T) [J]."""
        return self.mixture.total_enthalpy(self.temperature)

    @property
    def total_entropy(self) -> float:
        """Total mixture entropy S_total = Σ nⱼ·Sⱼ_mix(T,P) [J/K]."""
        return self.mixture.total_entropy(self.temperature, self.pressure)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Human-readable summary of the equilibrium state."""
        lines = [
            "EquilibriumSolution",
            f"  T          = {self.temperature:.2f} K",
            f"  P          = {self.pressure:.4g} Pa",
            f"  converged  = {self.converged}  ({self.iterations} iterations)",
            f"  M_gas      = {self.gas_mean_molar_mass * 1000:.4f} g/mol",
        ]
        if self.converged:
            try:
                lines += [
                    f"  Cp         = {self.cp:.4f} J/(mol·K)",
                    f"  γ          = {self.gamma:.4f}",
                    f"  a          = {self.speed_of_sound:.2f} m/s",
                ]
            except (ValueError, ZeroDivisionError):
                pass
        else:
            if self.failure_reason is not None:
                lines.append(f"  failure    = {self.failure_reason.name}")
            if self.element_balance_error is not None:
                lines.append(f"  el_balance = {self.element_balance_error:.3e}")
            if self.last_step_norm is not None:
                lines.append(f"  step_norm  = {self.last_step_norm:.3e}")
        lines.append("  Major species (xⱼ ≥ 1e-4):")
        for name, x in self.major_species().items():
            lines.append(f"    {name:30s}  {x:.6f}")
        return "\n".join(lines)

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"EquilibriumSolution(T={self.temperature:.1f} K, "
            f"P={self.pressure:.3g} Pa, converged={self.converged})"
        )
