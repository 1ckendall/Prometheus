"""
Non-convergence diagnostics for equilibrium solvers.

When an equilibrium solve fails, the returned :class:`EquilibriumSolution`
carries a :class:`NonConvergenceReason` that explains why the solver stopped.
This module defines that enum so it can be imported independently from the
heavier solver module.
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

from enum import IntEnum


class NonConvergenceReason(IntEnum):
    """Reason why an equilibrium solver did not satisfy its convergence criteria.

    The value is stored in :attr:`EquilibriumSolution.failure_reason` and is
    ``None`` when the solve converged normally.

    Attributes:
        MAX_ITERATIONS_REACHED: The iteration counter hit ``max_iterations``
            (or the outer temperature-search limit) without the step-size
            criterion dropping below ``tolerance``.
        SINGULAR_JACOBIAN: ``numpy.linalg.LinAlgError`` was raised inside the
            Newton solve, indicating a singular or near-singular Jacobian.
            Typically caused by a degenerate species set or extreme composition.
        GAS_MOLES_COLLAPSED: Total gas-phase moles fell to zero during
            iteration.  Only raised by ``GordonMcBrideSolver``.
        CONDENSED_PHASE_CYCLING: The condensed-phase set changed more than
            ``3 * n_elements`` times without composition converging.  Only
            raised by ``MajorSpeciesSolver``.
        NO_BASIS_FOUND: ``ElementMatrix.select_basis`` raised ``RuntimeError``
            because no linearly independent Browne basis could be selected from
            the current mole amounts.  Only raised by ``PEPSolver``.
        INVALID_THERMO_PROPERTIES: One or more thermodynamic property evaluations
            returned non-finite values (NaN/inf) at the attempted state.
    """

    MAX_ITERATIONS_REACHED = 1
    SINGULAR_JACOBIAN = 2
    GAS_MOLES_COLLAPSED = 3
    CONDENSED_PHASE_CYCLING = 4
    NO_BASIS_FOUND = 5
    INVALID_THERMO_PROPERTIES = 6
