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

"""Prometheus — combustion equilibrium solver.

Top-level convenience imports::

    from prometheus import (
        SpeciesDatabase,
        EquilibriumProblem,
        ProblemType,
        GordonMcBrideSolver,
        MajorSpeciesSolver,
        PerformanceSolver,
    )
"""

from prometheus_equilibrium.equilibrium import (
    ElementMatrix,
    EquilibriumProblem,
    EquilibriumSolution,
    EquilibriumSolver,
    GordonMcBrideSolver,
    HybridSolver,
    MajorSpeciesSolver,
    Mixture,
    PerformanceSolver,
    ProblemType,
    RocketPerformanceComparison,
    RocketPerformanceResult,
    SpeciesDatabase,
)
from prometheus_equilibrium.propellants import PropellantDatabase, PropellantMixture

__all__ = [
    "SpeciesDatabase",
    "EquilibriumProblem",
    "ProblemType",
    "EquilibriumSolution",
    "EquilibriumSolver",
    "GordonMcBrideSolver",
    "HybridSolver",
    "MajorSpeciesSolver",
    "PerformanceSolver",
    "RocketPerformanceResult",
    "RocketPerformanceComparison",
    "Mixture",
    "ElementMatrix",
    "PropellantDatabase",
    "PropellantMixture",
]
