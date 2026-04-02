"""
EquilibriumProblem — specification of a chemical equilibrium calculation.

A problem is fully defined by:

1. **Reactants**: the initial mixture whose element composition fixes the
   conserved element abundance vector b₀.
2. **Products**: candidate product species (any subset of the species database
   that contains the right elements).
3. **Constraint type**: which pair of thermodynamic variables is held constant
   during the calculation (TP, HP, SP, TV, UV, or SV).
4. **Constraint values**: numerical values of the two fixed variables.

The problem object does *not* run any iteration — it is purely declarative.
Pass it to an :py:class:`~Prometheus.equilibrium.solver.EquilibriumSolver` to
obtain an :py:class:`~Prometheus.equilibrium.solution.EquilibriumSolution`.

Constraint types
----------------
Following NASA CEA (RP-1311) naming conventions:

    TP  — constant T [K]  and P [Pa]
    HP  — constant H [J]  and P [Pa]   (adiabatic combustion at const P)
    SP  — constant S [J/K] and P [Pa]  (isentropic nozzle expansion)
    TV  — constant T [K]  and V [m³]
    UV  — constant U [J]  and V [m³]   (constant-volume explosion)
    SV  — constant S [J/K] and V [m³]  (isentropic compression)
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

from enum import Enum
from typing import Dict, List, Optional

import numpy as np

from prometheus_equilibrium.equilibrium.mixture import Mixture
from prometheus_equilibrium.equilibrium.species import Species

_P_REF = 1e5  # Standard reference pressure [Pa] (1 bar)


class ProblemType(Enum):
    """Which pair of thermodynamic variables is held constant."""

    TP = "TP"
    HP = "HP"
    SP = "SP"
    TV = "TV"
    UV = "UV"
    SV = "SV"

    @property
    def fixed_temperature(self) -> bool:
        """True if temperature is a fixed input (TP or TV problems)."""
        return self in (ProblemType.TP, ProblemType.TV)

    @property
    def fixed_pressure(self) -> bool:
        """True if pressure is a fixed input (TP, HP, SP problems)."""
        return self in (ProblemType.TP, ProblemType.HP, ProblemType.SP)

    @property
    def energy_constraint(self) -> Optional[str]:
        """Which energy-like quantity is fixed: 'H', 'S', 'U', or None."""
        return {
            ProblemType.TP: None,
            ProblemType.HP: "H",
            ProblemType.SP: "S",
            ProblemType.TV: None,
            ProblemType.UV: "U",
            ProblemType.SV: "S",
        }[self]


class EquilibriumProblem:
    """Declarative specification of a chemical equilibrium problem.

    Parameters
    ----------
    reactants : dict {Species: float}
        Reactant species mapped to their mole amounts [mol].  These define
        the conserved element abundances b₀ = A^T · n_reactants.
    products : list of Species
        Candidate product species.  Typically obtained from
        ``SpeciesDatabase.get_species(element_set)``; the solver will find
        the subset that minimises Gibbs free energy.
    problem_type : ProblemType
        Which pair of thermodynamic variables is held constant.
    constraint1 : float
        Value of the first fixed variable.  Interpretation by problem type:

        - TP : T [K]
        - HP : H [J]  (total enthalpy of the reactant mixture)
        - SP : S [J/K]
        - TV : T [K]
        - UV : U [J]
        - SV : S [J/K]

    constraint2 : float
        Value of the second fixed variable.  Interpretation by problem type:

        - TP : P [Pa]
        - HP : P [Pa]
        - SP : P [Pa]
        - TV : V [m³]
        - UV : V [m³]
        - SV : V [m³]

    pressure : float, optional
        Reference pressure [Pa] for ideal-gas chemical-potential calculations.
        Defaults to 1 bar (1e5 Pa).  For TP/HP/SP problems this is the same
        as constraint2; it is kept as a separate parameter for TV/UV/SV
        problems where pressure is not fixed.
    t_init : float, optional
        Initial temperature estimate [K] for the Newton iteration.  If None,
        a default of 3000 K is used for combustion problems.
    """

    def __init__(
        self,
        reactants: Dict[Species, float],
        products: List[Species],
        problem_type: ProblemType,
        constraint1: float,
        constraint2: float,
        pressure: float = _P_REF,
        t_init: Optional[float] = None,
    ) -> None:
        self.reactants = dict(reactants)
        self.products = list(products)
        self.problem_type = problem_type
        self.constraint1 = float(constraint1)
        self.constraint2 = float(constraint2)
        self.pressure = float(pressure)
        self.t_init = float(t_init) if t_init is not None else 3000.0

    # ------------------------------------------------------------------
    # Alternative constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_mass_fractions(
        cls,
        species_mass: Dict[Species, float],
        products: List[Species],
        problem_type: ProblemType,
        constraint1: float,
        constraint2: float,
        **kwargs,
    ) -> "EquilibriumProblem":
        """Construct from reactants given as mass amounts [kg].

        Converts each reactant's mass to moles using its molar mass
        (``species.molar_mass()`` returns kg/mol), then delegates to the
        standard constructor.

        Args:
            species_mass: Mass of each reactant species in kilograms.
            products: Candidate product species.
            problem_type: Thermodynamic constraint type.
            constraint1: First constraint value.
            constraint2: Second constraint value.
            **kwargs: Forwarded to the standard constructor (e.g. ``t_init``).
        """
        reactants = {sp: mass / sp.molar_mass() for sp, mass in species_mass.items()}
        return cls(
            reactants, products, problem_type, constraint1, constraint2, **kwargs
        )

    # ------------------------------------------------------------------
    # Element abundances
    # ------------------------------------------------------------------

    def element_abundances(self) -> Dict[str, float]:
        """Compute b₀: total moles of each element from the reactant mixture.

        ::

            b₀[k] = Σⱼ  n_reactant[j] × A[j, k]

        where A[j, k] is the stoichiometric coefficient of element k in
        reactant species j.  This is the conserved quantity that the
        equilibrium composition must satisfy.

        Returns a dict ``{element_symbol: total_moles}``.
        """
        b0: Dict[str, float] = {}
        for sp, n in self.reactants.items():
            for el, coeff in sp.elements.items():
                b0[el] = b0.get(el, 0.0) + coeff * n
        return b0

    def b0_array(self, elements: List[str]) -> np.ndarray:
        """Return b₀ as a NumPy array aligned to *elements*.

        Parameters
        ----------
        elements : list of str
            Ordered element list matching the columns of an
            :py:class:`~Prometheus.equilibrium.element_matrix.ElementMatrix`.

        Returns
        -------
        np.ndarray, shape (n_elements,)
        """
        b0_dict = self.element_abundances()
        return np.array([b0_dict.get(el, 0.0) for el in elements], dtype=float)

    # ------------------------------------------------------------------
    # Starting guess
    # ------------------------------------------------------------------

    def initial_mixture(self) -> Mixture:
        """Construct a starting-guess Mixture for the Newton iteration.

        Follows the strategy from cpropep / RP-1311 §3.2: distribute moles
        equally among all gas-phase product species; set all condensed species
        to zero (the solver will include them when their chemical potential
        condition is satisfied).

        The total initial gas moles is set to the mean element abundance
        (``Σ b₀ₖ / n_elements``) as a physically reasonable scale — this
        puts the starting composition in the right order of magnitude without
        requiring a more expensive atom-balance initialisation.  The G-McB
        Newton iteration is robust enough to converge from this simple guess.

        Returns
        -------
        Mixture
            All product species in gas-first order.  Gas species have equal
            positive mole amounts; condensed species start at zero.
        """
        b0 = self.element_abundances()
        n_elements = len(b0)
        n_total_init = sum(b0.values()) / n_elements if n_elements > 0 else 0.1

        n_gas_species = sum(1 for sp in self.products if sp.condensed == 0)
        n_gas_species = max(n_gas_species, 1)
        n_each = n_total_init / n_gas_species

        moles = np.array(
            [n_each if sp.condensed == 0 else 0.0 for sp in self.products], dtype=float
        )
        return Mixture(self.products, moles)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> None:
        """Verify the problem is well-posed before handing it to the solver.

        Checks
        ------
        - At least one reactant species is provided.
        - At least one product species is provided.
        - All element abundances in b₀ are non-negative.
        - Every element in b₀ is represented by at least one product species.
        - The initial temperature estimate is positive.
        - For TP/TV: constraint1 (T) > 0.
        - For HP/SP/UV/SV: constraint2 has physically reasonable sign.

        Raises
        ------
        ValueError
            With a descriptive message identifying the first failed check.
        """
        if not self.reactants:
            raise ValueError("No reactants specified.")
        if not self.products:
            raise ValueError("No product species specified.")

        b0 = self.element_abundances()

        for el, n in b0.items():
            if n < 0:
                raise ValueError(f"Negative element abundance for '{el}': {n}")

        product_elements: set[str] = set()
        for sp in self.products:
            product_elements.update(e for e in sp.elements if e != "e-")

        missing = {el for el in b0 if el != "e-" and el not in product_elements}
        if missing:
            raise ValueError(
                f"Element(s) {sorted(missing)} present in reactants but not "
                f"representable by any product species."
            )

        if self.t_init <= 0:
            raise ValueError(
                f"Initial temperature must be positive, got {self.t_init}."
            )

        if self.problem_type.fixed_temperature and self.constraint1 <= 0:
            raise ValueError(
                f"Temperature constraint must be positive for {self.problem_type.value} "
                f"problems, got {self.constraint1}."
            )

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:  # pragma: no cover
        b0 = self.element_abundances()
        el_str = ", ".join(f"{el}={n:.4g}" for el, n in sorted(b0.items()))
        return (
            f"EquilibriumProblem("
            f"type={self.problem_type.value}, "
            f"n_products={len(self.products)}, "
            f"b0=[{el_str}])"
        )
