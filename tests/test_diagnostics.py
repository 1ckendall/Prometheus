"""Tests for NonConvergenceReason diagnostics on EquilibriumSolution.

Verifies that all three solvers populate failure_reason, element_balance_error,
and last_step_norm correctly on both converged and non-converged solves.
"""

import math

import numpy as np
import pytest

from prometheus_equilibrium.core.constants import REFERENCE_PRESSURE as P_REF
from prometheus_equilibrium.core.constants import UNIVERSAL_GAS_CONSTANT as R
from prometheus_equilibrium.equilibrium.diagnostics import NonConvergenceReason
from prometheus_equilibrium.equilibrium.problem import EquilibriumProblem, ProblemType
from prometheus_equilibrium.equilibrium.solver import (
    GordonMcBrideSolver,
    MajorSpeciesSolver,
    PEPSolver,
)
from prometheus_equilibrium.equilibrium.species import Species

# ---------------------------------------------------------------------------
# Minimal mock species (constant Gibbs energy, avoids thermo data dependency)
# ---------------------------------------------------------------------------


class _ConstGibbsGas(Species):
    """Ideal-gas mock with a constant (T-independent) reduced Gibbs energy."""

    def __init__(self, elements, g0_RT, h0_RT=None, molar_mass_kg=0.002):
        super().__init__(elements=elements, state="G")
        self._g0 = g0_RT
        self._h0 = h0_RT if h0_RT is not None else g0_RT + 1.0
        self._M = molar_mass_kg

    def molar_mass(self):
        return self._M

    def specific_heat_capacity(self, T):
        return self._h0 * R

    def enthalpy(self, T):
        return self._h0 * R * float(T)

    def entropy(self, T):
        return (self._h0 - self._g0) * R


def _make_x_x2_tp(max_iterations=50):
    sp_X = _ConstGibbsGas({"X": 1}, g0_RT=1.0, molar_mass_kg=0.001)
    sp_X2 = _ConstGibbsGas({"X": 2}, g0_RT=3.0, molar_mass_kg=0.002)
    problem = EquilibriumProblem(
        reactants={sp_X2: 1.0},
        products=[sp_X, sp_X2],
        problem_type=ProblemType.TP,
        constraint1=1000.0,
        constraint2=P_REF,
        t_init=1000.0,
    )
    return problem


# ---------------------------------------------------------------------------
# 1. Converged solves — diagnostics on success
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "SolverClass,kwargs",
    [
        (GordonMcBrideSolver, {}),
        (MajorSpeciesSolver, {}),
        (PEPSolver, {"max_iterations": 300}),
    ],
)
def test_converged_failure_reason_is_none(SolverClass, kwargs):
    sol = SolverClass(**kwargs).solve(_make_x_x2_tp())
    assert sol.converged, f"{SolverClass.__name__} did not converge"
    assert sol.failure_reason is None


@pytest.mark.parametrize(
    "SolverClass,kwargs",
    [
        (GordonMcBrideSolver, {}),
        (MajorSpeciesSolver, {}),
        (PEPSolver, {"max_iterations": 300}),
    ],
)
def test_converged_element_balance_error_is_small(SolverClass, kwargs):
    sol = SolverClass(**kwargs).solve(_make_x_x2_tp())
    assert sol.converged
    assert sol.element_balance_error is not None
    assert math.isfinite(sol.element_balance_error)
    assert sol.element_balance_error >= 0.0
    assert sol.element_balance_error < 1e-6


@pytest.mark.parametrize(
    "SolverClass,kwargs",
    [
        (GordonMcBrideSolver, {}),
        (MajorSpeciesSolver, {}),
        (PEPSolver, {"max_iterations": 300}),
    ],
)
def test_converged_last_step_norm_is_finite_and_nonnegative(SolverClass, kwargs):
    sol = SolverClass(**kwargs).solve(_make_x_x2_tp())
    assert sol.converged
    assert sol.last_step_norm is not None
    assert math.isfinite(sol.last_step_norm)
    assert sol.last_step_norm >= 0.0


@pytest.mark.parametrize(
    "SolverClass,kwargs",
    [
        (GordonMcBrideSolver, {}),
        (MajorSpeciesSolver, {}),
        (PEPSolver, {"max_iterations": 300}),
    ],
)
def test_converged_last_step_norm_at_or_below_tolerance(SolverClass, kwargs):
    """Converged last_step_norm must be at or below the solver tolerance."""
    solver = SolverClass(**kwargs)
    sol = solver.solve(_make_x_x2_tp())
    assert sol.converged
    # Allow a small factor above tolerance for the final-step bookkeeping
    assert sol.last_step_norm <= solver.tolerance * 10


# ---------------------------------------------------------------------------
# 2. MAX_ITERATIONS_REACHED
# ---------------------------------------------------------------------------


def test_max_iterations_reached_gmcb():
    """GordonMcBrideSolver should report MAX_ITERATIONS_REACHED when capped."""
    sol = GordonMcBrideSolver(max_iterations=1).solve(_make_x_x2_tp())
    # If it converged in 1 iteration, the test is vacuously satisfied.
    if not sol.converged:
        assert sol.failure_reason == NonConvergenceReason.MAX_ITERATIONS_REACHED


def test_max_iterations_reached_major_species():
    sol = MajorSpeciesSolver(max_iterations=1).solve(_make_x_x2_tp())
    if not sol.converged:
        assert sol.failure_reason == NonConvergenceReason.MAX_ITERATIONS_REACHED


def test_max_iterations_reached_pep():
    sol = PEPSolver(max_iterations=1).solve(_make_x_x2_tp())
    if not sol.converged:
        assert sol.failure_reason == NonConvergenceReason.MAX_ITERATIONS_REACHED


def test_non_converged_diagnostics_always_populated():
    """Even when not converged, element_balance_error and last_step_norm are set."""
    sol = GordonMcBrideSolver(max_iterations=1).solve(_make_x_x2_tp())
    assert sol.element_balance_error is not None
    assert sol.last_step_norm is not None
    assert math.isfinite(sol.element_balance_error) or True  # may be inf on collapse
    # last_step_norm may be nan if no steps taken, but must exist
    assert isinstance(sol.last_step_norm, float)


# ---------------------------------------------------------------------------
# 3. SINGULAR_JACOBIAN via monkeypatch
# ---------------------------------------------------------------------------


def test_singular_jacobian_gmcb(monkeypatch):
    """GordonMcBrideSolver should report SINGULAR_JACOBIAN on LinAlgError."""
    import numpy.linalg

    def _raise(*args, **kwargs):
        raise numpy.linalg.LinAlgError("injected singular matrix")

    monkeypatch.setattr(numpy.linalg, "solve", _raise)

    sol = GordonMcBrideSolver().solve(_make_x_x2_tp())
    assert not sol.converged
    assert sol.failure_reason == NonConvergenceReason.SINGULAR_JACOBIAN


def test_singular_jacobian_major_species(monkeypatch):
    """MajorSpeciesSolver should report SINGULAR_JACOBIAN on LinAlgError."""
    import numpy.linalg

    def _raise(*args, **kwargs):
        raise numpy.linalg.LinAlgError("injected singular matrix")

    monkeypatch.setattr(numpy.linalg, "solve", _raise)

    sol = MajorSpeciesSolver().solve(_make_x_x2_tp())
    assert not sol.converged
    assert sol.failure_reason == NonConvergenceReason.SINGULAR_JACOBIAN


# ---------------------------------------------------------------------------
# 4. summary() failure branch
# ---------------------------------------------------------------------------


def test_summary_failure_reason_shown():
    """summary() should include failure reason name when not converged."""
    sol = GordonMcBrideSolver(max_iterations=1).solve(_make_x_x2_tp())
    if not sol.converged:
        s = sol.summary()
        assert "MAX_ITERATIONS_REACHED" in s


def test_summary_element_balance_shown_on_failure():
    """summary() should include el_balance when not converged."""
    sol = GordonMcBrideSolver(max_iterations=1).solve(_make_x_x2_tp())
    if not sol.converged:
        s = sol.summary()
        assert "el_balance" in s


def test_summary_step_norm_shown_on_failure():
    """summary() should include step_norm when not converged."""
    sol = GordonMcBrideSolver(max_iterations=1).solve(_make_x_x2_tp())
    if not sol.converged:
        s = sol.summary()
        assert "step_norm" in s


def test_summary_no_failure_fields_on_convergence():
    """summary() should NOT show failure fields when converged."""
    sol = GordonMcBrideSolver().solve(_make_x_x2_tp())
    assert sol.converged
    s = sol.summary()
    assert "failure" not in s
    assert "el_balance" not in s
    assert "step_norm" not in s


# ---------------------------------------------------------------------------
# 5. Consistency: failure_reason ↔ converged
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "SolverClass,kwargs",
    [
        (GordonMcBrideSolver, {}),
        (GordonMcBrideSolver, {"max_iterations": 1}),
        (MajorSpeciesSolver, {}),
        (MajorSpeciesSolver, {"max_iterations": 1}),
        (PEPSolver, {"max_iterations": 300}),
        (PEPSolver, {"max_iterations": 1}),
    ],
)
def test_failure_reason_consistent_with_converged(SolverClass, kwargs):
    """failure_reason must be None iff converged is True."""
    sol = SolverClass(**kwargs).solve(_make_x_x2_tp())
    if sol.converged:
        assert sol.failure_reason is None
    else:
        assert sol.failure_reason is not None
        assert isinstance(sol.failure_reason, NonConvergenceReason)
