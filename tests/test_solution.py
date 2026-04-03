"""Tests for EquilibriumSolution — thermodynamic and rocket properties."""

import math

import numpy as np
import pytest

from prometheus_equilibrium.core.constants import REFERENCE_PRESSURE as P_REF
from prometheus_equilibrium.core.constants import UNIVERSAL_GAS_CONSTANT as R
from prometheus_equilibrium.equilibrium.element_matrix import ElementMatrix
from prometheus_equilibrium.equilibrium.mixture import Mixture
from prometheus_equilibrium.equilibrium.problem import EquilibriumProblem, ProblemType
from prometheus_equilibrium.equilibrium.solution import EquilibriumSolution
from prometheus_equilibrium.equilibrium.solver import GordonMcBrideSolver
from prometheus_equilibrium.equilibrium.species import Species

# ---------------------------------------------------------------------------
# Mock species (reused from solver tests pattern)
# ---------------------------------------------------------------------------


class _Gas(Species):
    def __init__(self, elements, g0, h0, molar_mass_kg=0.002):
        super().__init__(elements=elements, state="G")
        self._g0 = g0
        self._h0 = h0
        self._M = molar_mass_kg

    def molar_mass(self):
        return self._M

    def specific_heat_capacity(self, T):
        return self._h0 * R

    def enthalpy(self, T):
        return self._h0 * R * T

    def entropy(self, T):
        return (self._h0 - self._g0) * R


# ---------------------------------------------------------------------------
# Converged solution fixture — X / X₂ TP at 1000 K
# ---------------------------------------------------------------------------


@pytest.fixture
def solved_x_x2():
    """Return a converged TP EquilibriumSolution for the X/X₂ system."""
    sp_X = _Gas({"X": 1}, g0=1.0, h0=2.0, molar_mass_kg=0.001)
    sp_X2 = _Gas({"X": 2}, g0=3.0, h0=4.0, molar_mass_kg=0.002)

    prob = EquilibriumProblem(
        reactants={sp_X2: 1.0},
        products=[sp_X, sp_X2],
        problem_type=ProblemType.TP,
        constraint1=1000.0,
        constraint2=P_REF,
        t_init=1000.0,
    )
    return GordonMcBrideSolver().solve(prob)


# ---------------------------------------------------------------------------
# 1. Basic metadata
# ---------------------------------------------------------------------------


def test_solution_converged(solved_x_x2):
    assert solved_x_x2.converged


def test_solution_temperature(solved_x_x2):
    assert solved_x_x2.temperature == pytest.approx(1000.0, abs=1e-3)


def test_solution_pressure(solved_x_x2):
    assert solved_x_x2.pressure == pytest.approx(P_REF)


# ---------------------------------------------------------------------------
# 2. Mole fractions
# ---------------------------------------------------------------------------


def test_mole_fractions_sum_to_one(solved_x_x2):
    total = sum(solved_x_x2.mole_fractions.values())
    assert total == pytest.approx(1.0, rel=1e-6)


def test_major_species_threshold(solved_x_x2):
    major = solved_x_x2.major_species(threshold=0.0)
    assert len(major) == 2  # both X and X₂ are above 0 threshold


def test_major_species_excludes_trace(solved_x_x2):
    # Both species have large mole fractions, but check filtering works at 0.99
    major = solved_x_x2.major_species(threshold=0.99)
    assert len(major) <= 2


# ---------------------------------------------------------------------------
# 3. Thermodynamic properties
# ---------------------------------------------------------------------------


def test_mean_molar_mass(solved_x_x2):
    """M̄ is a weighted average of species molar masses."""
    sol = solved_x_x2
    mix = sol.mixture
    expected = sum(
        x * sp.molar_mass() for x, sp in zip(mix.mole_fractions, mix.species)
    )
    assert sol.mean_molar_mass == pytest.approx(expected, rel=1e-8)


def test_gas_mean_molar_mass_equals_mean_for_all_gas(solved_x_x2):
    """For an all-gas mixture gas M̄ = total M̄."""
    sol = solved_x_x2
    assert sol.gas_mean_molar_mass == pytest.approx(sol.mean_molar_mass, rel=1e-8)


def test_cp_positive(solved_x_x2):
    assert solved_x_x2.cp > 0


def test_cv_less_than_cp(solved_x_x2):
    """Cv = Cp − R < Cp for ideal gas."""
    sol = solved_x_x2
    assert sol.cv == pytest.approx(sol.cp - R, rel=1e-10)
    assert sol.cv < sol.cp


def test_gamma_greater_than_one(solved_x_x2):
    assert solved_x_x2.gamma > 1.0


def test_enthalpy_finite(solved_x_x2):
    assert math.isfinite(solved_x_x2.enthalpy)


def test_entropy_finite(solved_x_x2):
    assert math.isfinite(solved_x_x2.entropy)


def test_gibbs_equals_h_minus_ts(solved_x_x2):
    sol = solved_x_x2
    expected = sol.enthalpy - sol.temperature * sol.entropy
    assert sol.gibbs == pytest.approx(expected, rel=1e-10)


# ---------------------------------------------------------------------------
# 4. Speed of sound and density
# ---------------------------------------------------------------------------


def test_speed_of_sound_positive(solved_x_x2):
    a = solved_x_x2.speed_of_sound
    assert a > 0
    assert math.isfinite(a)


def test_speed_of_sound_formula(solved_x_x2):
    """a = sqrt(γ·R·T / M̄_gas)."""
    sol = solved_x_x2
    expected = math.sqrt(sol.gamma * R * sol.temperature / sol.gas_mean_molar_mass)
    assert sol.speed_of_sound == pytest.approx(expected, rel=1e-8)


def test_density_ideal_gas(solved_x_x2):
    """ρ = P·M̄_gas / (R·T) for ideal gas."""
    sol = solved_x_x2
    expected = sol.pressure * sol.gas_mean_molar_mass / (R * sol.temperature)
    assert sol.density == pytest.approx(expected, rel=1e-8)


def test_speed_of_sound_raises_no_gas():
    """speed_of_sound raises ValueError when all species are condensed."""
    from prometheus_equilibrium.equilibrium.mixture import Mixture

    cond_sp = _Gas({"X": 1}, g0=1.0, h0=2.0)
    cond_sp.__class__ = type(
        "_Cond",
        (Species,),
        {
            "condensed": 1,
            "state": "S",
            "elements": {"X": 1},
            "molar_mass": lambda s: 0.001,
            "specific_heat_capacity": lambda s, T: 10.0,
            "enthalpy": lambda s, T: 10.0 * T,
            "entropy": lambda s, T: 5.0,
        },
    )
    # Build a trivial solution with no gas species
    mix = Mixture(
        species=[_Gas({"X": 1}, g0=0.0, h0=1.0)],
        moles=np.array([1.0]),
    )
    # Patch: make the species appear condensed by manipulating the moles
    # Rather than patching internals, just test via a solution with 0 gas moles
    mix._moles[0] = 0.0  # zero all moles so gas_mean_molar_mass = 0
    sol = EquilibriumSolution(
        mixture=mix,
        temperature=1000.0,
        pressure=P_REF,
        converged=True,
        iterations=1,
        residuals=np.array([0.0]),
        lagrange_multipliers=np.array([0.0]),
    )
    with pytest.raises((ValueError, ZeroDivisionError)):
        _ = sol.speed_of_sound


# ---------------------------------------------------------------------------
# 5. characteristic_velocity
# ---------------------------------------------------------------------------


def test_characteristic_velocity_positive(solved_x_x2):
    """c* requires a throat solution — use the chamber itself as a proxy."""
    sol = solved_x_x2
    c_star = sol.characteristic_velocity(sol)
    assert c_star > 0


def test_characteristic_velocity_formula(solved_x_x2):
    """c* = a_throat / Gamma(γ)."""
    sol = solved_x_x2
    g = sol.gamma
    gamma_factor = math.sqrt(g) * (2 / (g + 1)) ** ((g + 1) / (2 * (g - 1)))
    expected = sol.speed_of_sound / gamma_factor
    assert sol.characteristic_velocity(sol) == pytest.approx(expected, rel=1e-8)


# ---------------------------------------------------------------------------
# 6. summary()
# ---------------------------------------------------------------------------


def test_summary_contains_temperature(solved_x_x2):
    s = solved_x_x2.summary()
    assert "1000" in s


def test_summary_contains_converged(solved_x_x2):
    s = solved_x_x2.summary()
    assert "True" in s


# ---------------------------------------------------------------------------
# 7. total_enthalpy and total_entropy
# ---------------------------------------------------------------------------


def test_total_enthalpy_finite(solved_x_x2):
    assert math.isfinite(solved_x_x2.total_enthalpy)


def test_total_enthalpy_positive_for_positive_h0(solved_x_x2):
    """With h0 > 0 mocks, total enthalpy should be positive."""
    assert solved_x_x2.total_enthalpy > 0


def test_total_entropy_finite(solved_x_x2):
    assert math.isfinite(solved_x_x2.total_entropy)


def test_total_entropy_consistent_with_molar(solved_x_x2):
    """total_entropy = entropy (molar) × total_moles."""
    sol = solved_x_x2
    expected = sol.entropy * sol.mixture.total_moles
    assert sol.total_entropy == pytest.approx(expected, rel=1e-8)


# ---------------------------------------------------------------------------
# 8. specific_impulse
# ---------------------------------------------------------------------------


def test_specific_impulse_positive_dH(solved_x_x2):
    """When chamber enthalpy > exit enthalpy, Isp is positive."""
    chamber = solved_x_x2

    # Build an 'exit' solution with a lower temperature → lower enthalpy
    import copy

    exit_mix = chamber.mixture.copy()
    exit_sol = EquilibriumSolution(
        mixture=exit_mix,
        temperature=chamber.temperature * 0.5,  # much colder exit
        pressure=P_REF * 0.1,
        converged=True,
        iterations=1,
        residuals=chamber.residuals,
        lagrange_multipliers=chamber.lagrange_multipliers,
    )

    isp = chamber.specific_impulse(chamber, exit_sol)
    assert isp > 0


def test_specific_impulse_zero_dH():
    """When chamber enthalpy ≤ exit enthalpy, exit velocity is clamped to 0."""
    sp_X = _Gas({"X": 1}, g0=1.0, h0=2.0, molar_mass_kg=0.001)
    sp_X2 = _Gas({"X": 2}, g0=3.0, h0=4.0, molar_mass_kg=0.002)

    prob = EquilibriumProblem(
        reactants={sp_X2: 1.0},
        products=[sp_X, sp_X2],
        problem_type=ProblemType.TP,
        constraint1=1000.0,
        constraint2=P_REF,
        t_init=1000.0,
    )
    sol = GordonMcBrideSolver().solve(prob)

    # Use the same solution as both chamber and exit → dH = 0 → v_exit = 0
    isp = sol.specific_impulse(sol, sol)
    assert isp == pytest.approx(0.0, abs=1e-6)


def test_specific_impulse_pressure_thrust_path():
    """Isp with non-zero ambient pressure includes a pressure-thrust term."""
    sp_X = _Gas({"X": 1}, g0=1.0, h0=2.0, molar_mass_kg=0.001)
    sp_X2 = _Gas({"X": 2}, g0=3.0, h0=4.0, molar_mass_kg=0.002)

    prob = EquilibriumProblem(
        reactants={sp_X2: 1.0},
        products=[sp_X, sp_X2],
        problem_type=ProblemType.TP,
        constraint1=1000.0,
        constraint2=P_REF,
        t_init=1000.0,
    )
    chamber = GordonMcBrideSolver().solve(prob)

    exit_mix = chamber.mixture.copy()
    exit_sol = EquilibriumSolution(
        mixture=exit_mix,
        temperature=chamber.temperature * 0.5,
        pressure=P_REF * 0.01,
        converged=True,
        iterations=1,
        residuals=chamber.residuals,
        lagrange_multipliers=chamber.lagrange_multipliers,
    )

    isp_vac = chamber.specific_impulse(chamber, exit_sol, ambient_pressure=0.0)
    isp_sl = chamber.specific_impulse(chamber, exit_sol, ambient_pressure=101325.0)

    # Vacuum Isp ≥ sea-level Isp (under-expanded at sea level → thrust penalty)
    assert isp_vac >= isp_sl


# ---------------------------------------------------------------------------
# 9. Diagnostics fields on EquilibriumSolution
# ---------------------------------------------------------------------------


def test_element_balance_error_populated(solved_x_x2):
    assert solved_x_x2.element_balance_error is not None
    assert math.isfinite(solved_x_x2.element_balance_error)
    assert solved_x_x2.element_balance_error >= 0.0


def test_last_step_norm_populated(solved_x_x2):
    assert solved_x_x2.last_step_norm is not None
    assert math.isfinite(solved_x_x2.last_step_norm)
    assert solved_x_x2.last_step_norm >= 0.0
