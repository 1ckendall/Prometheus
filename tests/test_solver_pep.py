"""Tests for PEPSolver — convergence, element balance, equilibrium, HP."""

import math

import numpy as np
import pytest

from prometheus_equilibrium.core.constants import REFERENCE_PRESSURE as P_REF
from prometheus_equilibrium.core.constants import UNIVERSAL_GAS_CONSTANT as R
from prometheus_equilibrium.equilibrium.element_matrix import ElementMatrix
from prometheus_equilibrium.equilibrium.mixture import Mixture
from prometheus_equilibrium.equilibrium.problem import EquilibriumProblem, ProblemType
from prometheus_equilibrium.equilibrium.solver import (
    EquilibriumSolver,
    MajorSpeciesSolver,
    PEPSolver,
)
from prometheus_equilibrium.equilibrium.species import Species

# ---------------------------------------------------------------------------
# Mock species (reuse pattern from test_solver_gmcb.py)
# ---------------------------------------------------------------------------


class _ConstGibbsGas(Species):
    """Ideal-gas mock with a constant (T-independent) reduced Gibbs energy."""

    def __init__(
        self,
        elements: dict,
        g0_RT: float,
        h0_RT: float | None = None,
        molar_mass_kg: float = 0.002,
    ):
        super().__init__(elements=elements, state="G")
        self._g0 = g0_RT
        self._h0 = h0_RT if h0_RT is not None else g0_RT + 1.0
        self._M = molar_mass_kg

    def molar_mass(self) -> float:
        return self._M

    def specific_heat_capacity(self, T: float) -> float:
        return self._h0 * R

    def enthalpy(self, T: float) -> float:
        return self._h0 * R * T

    def entropy(self, T: float) -> float:
        return (self._h0 - self._g0) * R


class _ConstGibbsCond(Species):
    """Condensed-phase mock with constant reduced Gibbs energy."""

    def __init__(self, elements: dict, g0_RT: float, molar_mass_kg: float = 0.030):
        super().__init__(elements=elements, state="S")
        self._g0 = g0_RT
        self._M = molar_mass_kg

    def molar_mass(self) -> float:
        return self._M

    def specific_heat_capacity(self, T: float) -> float:
        return 30.0

    def enthalpy(self, T: float) -> float:
        return 30.0 * T

    def entropy(self, T: float) -> float:
        return 10.0


class _WindowedConstGibbsGas(_ConstGibbsGas):
    """Const-gibbs gas that is thermo-valid only inside [t_min, t_max]."""

    def __init__(self, *args, t_min: float, t_max: float, **kwargs):
        super().__init__(*args, **kwargs)
        self._t_min = float(t_min)
        self._t_max = float(t_max)

    def reduced_gibbs(self, T):
        t_val = float(np.asarray(T))
        if self._t_min <= t_val <= self._t_max:
            return float(self._g0)
        return float("nan")


# ---------------------------------------------------------------------------
# Shared test problem: X / X₂ dissociation (same as G-McB tests)
#
# Analytical answer:  e = exp(1),  n_X = sqrt(4e/(4+e))  ≈ 1.2722
# ---------------------------------------------------------------------------

_G0_X = 1.0
_G0_X2 = 3.0
_T_TEST = 1000.0
_P_TEST = P_REF


def _expected_equilibrium():
    e = math.exp(1.0)
    n_X = math.sqrt(4 * e / (4 + e))
    n_X2 = (2.0 - n_X) / 2.0
    return n_X, n_X2


@pytest.fixture
def x_x2_problem():
    sp_X = _ConstGibbsGas({"X": 1}, g0_RT=_G0_X, molar_mass_kg=0.001)
    sp_X2 = _ConstGibbsGas({"X": 2}, g0_RT=_G0_X2, molar_mass_kg=0.002)
    return EquilibriumProblem(
        reactants={sp_X2: 1.0},
        products=[sp_X, sp_X2],
        problem_type=ProblemType.TP,
        constraint1=_T_TEST,
        constraint2=_P_TEST,
        t_init=_T_TEST,
    )


# ---------------------------------------------------------------------------
# 1. Basic TP convergence
# ---------------------------------------------------------------------------


def test_tp_converges(x_x2_problem):
    sol = PEPSolver(max_iterations=200).solve(x_x2_problem)
    assert sol.converged, f"Did not converge in {sol.iterations} iterations"


def test_history_default_full_capture_pep_major_species(x_x2_problem):
    """PEP and major-species solver should capture history by default."""
    sol_pep = PEPSolver(max_iterations=200).solve(x_x2_problem)
    sol_hyb = MajorSpeciesSolver(max_iterations=120).solve(x_x2_problem)
    assert sol_pep.converged
    assert sol_hyb.converged
    assert sol_pep.history is not None and len(sol_pep.history) > 0
    assert sol_hyb.history is not None and len(sol_hyb.history) > 0


def test_history_can_be_disabled_pep_major_species(x_x2_problem):
    """PEP and major-species solver should support disabling history capture."""
    sol_pep = PEPSolver(max_iterations=200, capture_history=False).solve(x_x2_problem)
    sol_hyb = MajorSpeciesSolver(max_iterations=120, capture_history=False).solve(
        x_x2_problem
    )
    assert sol_pep.converged
    assert sol_hyb.converged
    assert sol_pep.history is None
    assert sol_hyb.history is None


def test_tp_element_balance(x_x2_problem):
    sol = PEPSolver(max_iterations=200).solve(x_x2_problem)
    assert sol.converged
    assert np.all(
        np.abs(sol.residuals) < 1e-4
    ), f"Element balance residuals: {sol.residuals}"


def test_tp_composition_matches_analytical(x_x2_problem):
    sol = PEPSolver(max_iterations=200).solve(x_x2_problem)
    assert sol.converged

    n_X_exp, n_X2_exp = _expected_equilibrium()
    mix = sol.mixture
    idx_X = next(i for i, sp in enumerate(mix.species) if sp.elements == {"X": 1})
    idx_X2 = next(i for i, sp in enumerate(mix.species) if sp.elements == {"X": 2})

    assert mix.moles[idx_X] == pytest.approx(n_X_exp, rel=1e-3)
    assert mix.moles[idx_X2] == pytest.approx(n_X2_exp, rel=1e-3)


# ---------------------------------------------------------------------------
# 2. Two-element system — checks basis selection and TWID
# ---------------------------------------------------------------------------


def test_tp_two_element_system():
    sp_A = _ConstGibbsGas({"A": 1}, g0_RT=2.0, molar_mass_kg=0.001)
    sp_B = _ConstGibbsGas({"B": 1}, g0_RT=1.5, molar_mass_kg=0.001)
    sp_AB = _ConstGibbsGas({"A": 1, "B": 1}, g0_RT=2.0, molar_mass_kg=0.002)
    sp_AB2 = _ConstGibbsGas({"A": 1, "B": 2}, g0_RT=4.0, molar_mass_kg=0.003)

    prob = EquilibriumProblem(
        reactants={sp_AB2: 1.0},
        products=[sp_A, sp_B, sp_AB, sp_AB2],
        problem_type=ProblemType.TP,
        constraint1=1000.0,
        constraint2=P_REF,
        t_init=1000.0,
    )
    sol = PEPSolver(max_iterations=300).solve(prob)
    assert sol.converged

    em = ElementMatrix.from_mixture(sol.mixture).reduced()
    b0 = prob.b0_array(em.elements)
    residuals = em.element_residuals(sol.mixture.moles, b0)
    np.testing.assert_allclose(residuals, 0.0, atol=1e-3)


# ---------------------------------------------------------------------------
# 3. PEP and major-species solver agree on TP composition
# ---------------------------------------------------------------------------


def test_pep_matches_major_species_tp(x_x2_problem):
    sol_pep = PEPSolver(max_iterations=300).solve(x_x2_problem)
    sol_hyb = MajorSpeciesSolver(max_iterations=100).solve(x_x2_problem)

    assert sol_pep.converged, "PEP did not converge"
    assert sol_hyb.converged, "Major-species solver did not converge"

    mix_p = sol_pep.mixture
    mix_h = sol_hyb.mixture

    for i, sp in enumerate(mix_p.species):
        n_p = mix_p.moles[i]
        n_h = mix_h.moles[i]
        if max(n_p, n_h) > 1e-6:
            assert (
                abs(n_p - n_h) / max(n_p, n_h) < 1e-2
            ), f"Species {i} mismatch: PEP={n_p:.6g}, MajorSpecies={n_h:.6g}"


# ---------------------------------------------------------------------------
# 4. HP problem — PEP and major-species solver agree on temperature
# ---------------------------------------------------------------------------


def test_hp_pep_matches_major_species():
    """HP equilibrium: PEP temperature should match major-species to within 1 %."""
    sp_X = _ConstGibbsGas({"X": 1}, g0_RT=_G0_X, h0_RT=2.0, molar_mass_kg=0.001)
    sp_X2 = _ConstGibbsGas({"X": 2}, g0_RT=_G0_X2, h0_RT=4.0, molar_mass_kg=0.002)

    # Compute reactant enthalpy at T_ref = 1000 K
    T_ref = 1000.0
    H0 = 1.0 * sp_X2.enthalpy(T_ref)

    prob = EquilibriumProblem(
        reactants={sp_X2: 1.0},
        products=[sp_X, sp_X2],
        problem_type=ProblemType.HP,
        constraint1=H0,
        constraint2=P_REF,
        t_init=T_ref,
    )

    sol_pep = PEPSolver(max_iterations=300).solve(prob)
    sol_hyb = MajorSpeciesSolver(max_iterations=100).solve(prob)

    assert sol_pep.converged, "PEP HP did not converge"
    assert sol_hyb.converged, "Major-species HP did not converge"

    assert (
        abs(sol_pep.temperature - sol_hyb.temperature) / sol_hyb.temperature < 0.01
    ), f"Temperature mismatch: PEP={sol_pep.temperature:.1f}, MajorSpecies={sol_hyb.temperature:.1f}"


# ---------------------------------------------------------------------------
# 5. _equilibrium_constants unit test
# ---------------------------------------------------------------------------


def test_equilibrium_constants_basis_species_are_zero():
    """ln_K for a basis species must be zero (reacts with itself)."""
    sp_X = _ConstGibbsGas({"X": 1}, g0_RT=1.0, molar_mass_kg=0.001)
    sp_X2 = _ConstGibbsGas({"X": 2}, g0_RT=3.0, molar_mass_kg=0.002)
    mixture = Mixture([sp_X, sp_X2], np.array([1.0, 1.0]))
    em = ElementMatrix.from_mixture(mixture).reduced()

    basis_indices = [0, 1] if em.n_elements >= 2 else [0]
    basis_indices = basis_indices[: em.n_elements]
    nu = em.reaction_coefficients(basis_indices)
    g_all = np.array([sp.reduced_gibbs(1000.0) for sp in mixture.species])
    g_basis = g_all[np.array(basis_indices)]

    ln_K = PEPSolver._equilibrium_constants(nu, g_all, g_basis)

    for jj, j in enumerate(basis_indices):
        assert (
            abs(ln_K[j]) < 1e-10
        ), f"ln_K for basis species {j} should be 0, got {ln_K[j]}"


# ---------------------------------------------------------------------------
# 6. SP problem via _temperature_search
# ---------------------------------------------------------------------------


class _LogEntropyGas(Species):
    """Ideal gas with Cp=const and S(T) ~ ln(T), needed for SP tests."""

    def __init__(self, elements, cp_over_r, s_ref_over_r, molar_mass_kg=0.002):
        super().__init__(elements=elements, state="G")
        self._cp_r = float(cp_over_r)
        self._s0_r = float(s_ref_over_r)
        self._M = float(molar_mass_kg)

    def molar_mass(self):
        return self._M

    def specific_heat_capacity(self, T):
        return self._cp_r * R

    def enthalpy(self, T):
        return self._cp_r * R * float(T)

    def entropy(self, T):
        import math

        return (self._cp_r * math.log(float(T)) + self._s0_r) * R


def _make_hp_log_problem():
    """HP problem using _LogEntropyGas species (needed for SP tests later)."""
    sp_X = _LogEntropyGas(
        {"X": 1}, cp_over_r=4.0, s_ref_over_r=2.0, molar_mass_kg=0.001
    )
    sp_X2 = _LogEntropyGas(
        {"X": 2}, cp_over_r=5.0, s_ref_over_r=1.0, molar_mass_kg=0.002
    )
    T_ref = 1000.0
    H0 = 1.0 * sp_X2.enthalpy(T_ref)
    return (
        EquilibriumProblem(
            reactants={sp_X2: 1.0},
            products=[sp_X, sp_X2],
            problem_type=ProblemType.HP,
            constraint1=H0,
            constraint2=30 * P_REF,
            t_init=T_ref,
        ),
        sp_X,
        sp_X2,
    )


def test_sp_major_species_converges():
    """MajorSpeciesSolver should converge on a SP problem."""
    sp_X = _LogEntropyGas(
        {"X": 1}, cp_over_r=4.0, s_ref_over_r=2.0, molar_mass_kg=0.001
    )
    sp_X2 = _LogEntropyGas(
        {"X": 2}, cp_over_r=5.0, s_ref_over_r=1.0, molar_mass_kg=0.002
    )

    # First solve HP to get the chamber state
    T_ref = 1000.0
    H0 = 1.0 * sp_X2.enthalpy(T_ref)
    hp_problem = EquilibriumProblem(
        reactants={sp_X2: 1.0},
        products=[sp_X, sp_X2],
        problem_type=ProblemType.HP,
        constraint1=H0,
        constraint2=30 * P_REF,
        t_init=T_ref,
    )
    chamber = MajorSpeciesSolver().solve(hp_problem)
    assert chamber.converged, f"HP chamber solve failed: {chamber.failure_reason}"

    # Then solve SP to lower pressure (nozzle expansion)
    from prometheus_equilibrium.equilibrium.mixture import Mixture as _Mixture

    S_chamber = chamber.mixture.total_entropy(chamber.temperature, chamber.pressure)
    sp_problem = EquilibriumProblem(
        reactants={
            sp: n
            for sp, n in zip(chamber.mixture.species, chamber.mixture.moles)
            if n > 0
        },
        products=[sp_X, sp_X2],
        problem_type=ProblemType.SP,
        constraint1=S_chamber,
        constraint2=P_REF,
        t_init=chamber.temperature * 0.7,
    )
    sol = MajorSpeciesSolver().solve(sp_problem)
    assert sol.converged, f"SP solve failed: {sol.failure_reason}"
    # Isentropic expansion should cool the gas
    assert sol.temperature < chamber.temperature


def test_sp_pep_converges():
    """PEPSolver should converge on a SP problem."""
    sp_X = _LogEntropyGas(
        {"X": 1}, cp_over_r=4.0, s_ref_over_r=2.0, molar_mass_kg=0.001
    )
    sp_X2 = _LogEntropyGas(
        {"X": 2}, cp_over_r=5.0, s_ref_over_r=1.0, molar_mass_kg=0.002
    )

    T_ref = 1000.0
    H0 = 1.0 * sp_X2.enthalpy(T_ref)
    hp_problem = EquilibriumProblem(
        reactants={sp_X2: 1.0},
        products=[sp_X, sp_X2],
        problem_type=ProblemType.HP,
        constraint1=H0,
        constraint2=30 * P_REF,
        t_init=T_ref,
    )
    chamber = PEPSolver(max_iterations=300).solve(hp_problem)
    assert chamber.converged, f"HP chamber solve failed"

    S_chamber = chamber.mixture.total_entropy(chamber.temperature, chamber.pressure)
    sp_problem = EquilibriumProblem(
        reactants={
            sp: n
            for sp, n in zip(chamber.mixture.species, chamber.mixture.moles)
            if n > 0
        },
        products=[sp_X, sp_X2],
        problem_type=ProblemType.SP,
        constraint1=S_chamber,
        constraint2=P_REF,
        t_init=chamber.temperature * 0.7,
    )
    sol = PEPSolver(max_iterations=300).solve(sp_problem)
    assert sol.converged, f"SP solve failed: {sol.failure_reason}"
    assert sol.temperature < chamber.temperature


def test_refresh_thermo_species_set_drops_and_reintroduces_species():
    """Thermo refresh should remove invalid species and re-add newly valid ones."""
    sp_hot = _WindowedConstGibbsGas(
        {"X": 1},
        g0_RT=1.0,
        h0_RT=2.0,
        molar_mass_kg=0.001,
        t_min=2000.0,
        t_max=6000.0,
    )
    sp_cool = _WindowedConstGibbsGas(
        {"X": 1},
        g0_RT=1.5,
        h0_RT=2.5,
        molar_mass_kg=0.001,
        t_min=200.0,
        t_max=1500.0,
    )
    species_pool = [sp_hot, sp_cool]
    active_elements = ["X"]

    mix0 = Mixture([sp_hot], np.array([1.0]))

    mix_hot, _ = EquilibriumSolver._refresh_thermo_species_set(
        mix0, species_pool, active_elements, T=3000.0
    )
    assert mix_hot.species == [sp_hot]
    assert mix_hot.moles[0] == pytest.approx(1.0)

    mix_cool, _ = EquilibriumSolver._refresh_thermo_species_set(
        mix_hot, species_pool, active_elements, T=1000.0
    )
    assert mix_cool.species == [sp_cool]
    # Newly valid species are reintroduced at zero and then seeded to tiny positive gas moles.
    assert mix_cool.moles[0] > 0.0


# ---------------------------------------------------------------------------
# 7. guess parameter
# ---------------------------------------------------------------------------


def test_guess_accepted_major_species(x_x2_problem):
    """MajorSpeciesSolver should accept and use a guess mixture."""
    # Solve once to get a good starting guess
    sol1 = MajorSpeciesSolver().solve(x_x2_problem)
    assert sol1.converged

    # Solve again with the converged mixture as a guess — should converge faster
    sol2 = MajorSpeciesSolver().solve(x_x2_problem, guess=sol1.mixture)
    assert sol2.converged


def test_guess_accepted_pep(x_x2_problem):
    """PEPSolver should accept and use a guess mixture."""
    sol1 = PEPSolver(max_iterations=300).solve(x_x2_problem)
    assert sol1.converged

    sol2 = PEPSolver(max_iterations=300).solve(x_x2_problem, guess=sol1.mixture)
    assert sol2.converged
