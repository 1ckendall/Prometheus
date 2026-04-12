"""
Prometheus.equilibrium — Chemical equilibrium solver infrastructure.

This package provides the data structures and abstract interfaces needed to
solve chemical equilibrium problems by Gibbs free energy minimisation.

The algorithm follows the modified Lagrange multiplier method described in:

    Gordon, S. and McBride, B.J. (1994).
    Computer Program for Calculation of Complex Chemical Equilibrium
    Compositions and Applications.
    NASA Reference Publication 1311.

Six thermodynamic problem types are supported (matching NASA CEA nomenclature):

    TP  — fixed temperature and pressure
    HP  — fixed enthalpy and pressure        (e.g. adiabatic combustion)
    SP  — fixed entropy and pressure         (e.g. isentropic nozzle flow)
    TV  — fixed temperature and volume
    UV  — fixed internal energy and volume   (e.g. constant-volume explosion)
    SV  — fixed entropy and volume

Quick-start example::

    from prometheus_equilibrium.equilibrium import (
        SpeciesDatabase, EquilibriumProblem, GordonMcBrideSolver, ProblemType
    )

    db = SpeciesDatabase(nasa7_path, nasa9_path, janaf_path)
    db.load()

    h2, o2 = db["H2_G"], db["O2_G"]
    products = db.get_species({"H", "O"})
    H0 = h2.enthalpy(298.15) * 2.0 + o2.enthalpy(298.15) * 1.0

    problem = EquilibriumProblem(
        reactants={h2: 2.0, o2: 1.0},
        products=products,
        problem_type=ProblemType.HP,
        constraint1=H0,   # total enthalpy (J)
        constraint2=30e5, # pressure (Pa)
    )

    solution = GordonMcBrideSolver().solve(problem)
    print(solution.summary())
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

from prometheus_equilibrium.equilibrium.diagnostics import NonConvergenceReason
from prometheus_equilibrium.equilibrium.element_matrix import ElementMatrix
from prometheus_equilibrium.equilibrium.mixture import Mixture
from prometheus_equilibrium.equilibrium.performance import (
    PerformanceSolver,
    RocketPerformanceComparison,
    RocketPerformanceResult,
)
from prometheus_equilibrium.equilibrium.problem import EquilibriumProblem, ProblemType
from prometheus_equilibrium.equilibrium.solution import EquilibriumSolution
from prometheus_equilibrium.equilibrium.solver import (
    EquilibriumSolver,
    GordonMcBrideSolver,
    HybridSolver,
    MajorSpeciesSolver,
    PEPSolver,
)
from prometheus_equilibrium.equilibrium.species import SpeciesDatabase

__all__ = [
    "Mixture",
    "ElementMatrix",
    "EquilibriumProblem",
    "ProblemType",
    "EquilibriumSolution",
    "NonConvergenceReason",
    "EquilibriumSolver",
    "MajorSpeciesSolver",
    "PEPSolver",
    "GordonMcBrideSolver",
    "HybridSolver",
    "PerformanceSolver",
    "RocketPerformanceResult",
    "RocketPerformanceComparison",
    "SpeciesDatabase",
]
