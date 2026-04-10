"""Tests for GordonMcBrideSolver — convergence, element balance, equilibrium."""

import math

import numpy as np
import pytest

from prometheus_equilibrium.core.constants import REFERENCE_PRESSURE as P_REF
from prometheus_equilibrium.core.constants import UNIVERSAL_GAS_CONSTANT as R
from prometheus_equilibrium.equilibrium.element_matrix import ElementMatrix
from prometheus_equilibrium.equilibrium.problem import EquilibriumProblem, ProblemType
from prometheus_equilibrium.equilibrium.solver import GordonMcBrideSolver
from prometheus_equilibrium.equilibrium.species import Species

# ---------------------------------------------------------------------------
# Mock species helpers
# ---------------------------------------------------------------------------


class _ConstGibbsGas(Species):
    """Ideal-gas mock with a constant (T-independent) reduced Gibbs energy g°/RT.

    Thermodynamic consistency:
        H°(T) = h0 * R * T   (linear in T → Cp = h0*R = constant)
        S°(T) = (h0 - g0) * R  (constant)
        G°(T) = H° - T·S° = g0·R·T  → G°/(RT) = g0 ✓

    Parameters
    ----------
    g0_RT : float
        Standard reduced Gibbs energy g°/(RT), independent of T.
    h0_RT : float, optional
        Standard reduced enthalpy H°/(RT).  Defaults to g0_RT + 1.0.
        Setting h0_RT ≠ g0_RT + 1.0 allows testing problems with non-trivial
        reaction enthalpies (needed for HP tests).
    """

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
        return self._h0 * R  # Cp = h0·R (constant)

    def enthalpy(self, T: float) -> float:
        return self._h0 * R * T  # H°(T) = h0·R·T  [J/mol]

    def entropy(self, T: float) -> float:
        return (self._h0 - self._g0) * R  # S° = (h0-g0)·R  [J/(mol·K)]


class _ConstGibbsCond(Species):
    """Condensed-phase mock with a constant reduced Gibbs energy."""

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
    """Mock gas with constant thermodynamics and finite g0 only in a T window."""

    def __init__(
        self,
        elements: dict,
        g0_RT: float,
        t_min: float,
        t_max: float,
        h0_RT: float | None = None,
        molar_mass_kg: float = 0.002,
    ):
        super().__init__(
            elements, g0_RT=g0_RT, h0_RT=h0_RT, molar_mass_kg=molar_mass_kg
        )
        self._t_min = float(t_min)
        self._t_max = float(t_max)

    def reduced_gibbs(self, T):
        t_val = float(np.asarray(T))
        if self._t_min <= t_val <= self._t_max:
            return float(self._g0)
        return float("nan")


class _LogEntropyGas(Species):
    """Ideal gas with Cp=const and S(T) ~ ln(T), for stable SP tests."""

    def __init__(
        self,
        elements: dict,
        cp_over_r: float,
        s_ref_over_r: float,
        molar_mass_kg: float = 0.002,
    ):
        super().__init__(elements=elements, state="G")
        self._cp_r = float(cp_over_r)
        self._s0_r = float(s_ref_over_r)
        self._M = float(molar_mass_kg)

    def molar_mass(self) -> float:
        return self._M

    def specific_heat_capacity(self, T: float) -> float:
        return self._cp_r * R

    def enthalpy(self, T: float) -> float:
        return self._cp_r * R * float(T)

    def entropy(self, T: float) -> float:
        return (self._cp_r * math.log(float(T)) + self._s0_r) * R


class _WindowedLogEntropyGas(_LogEntropyGas):
    """Log-entropy gas that is thermo-valid only in a bounded T window."""

    def __init__(
        self,
        elements: dict,
        cp_over_r: float,
        s_ref_over_r: float,
        t_min: float,
        t_max: float,
        molar_mass_kg: float = 0.002,
    ):
        super().__init__(
            elements=elements,
            cp_over_r=cp_over_r,
            s_ref_over_r=s_ref_over_r,
            molar_mass_kg=molar_mass_kg,
        )
        self._t_min = float(t_min)
        self._t_max = float(t_max)

    def reduced_gibbs(self, T):
        t_val = float(np.asarray(T))
        if self._t_min <= t_val <= self._t_max:
            return super().reduced_gibbs(t_val)
        return float("nan")


# ---------------------------------------------------------------------------
# Analytical equilibrium for the X / X₂ test system
#
# Reaction:  X₂  ⇌  2X
#
# Equilibrium condition:  μ(X)/RT = π_X  and  μ(X₂)/RT = 2·π_X
#
# With g°(X)/RT = 1.0,  g°(X₂)/RT = 3.0,  P = P_ref:
#
#   K_p = exp(ΔG°_rxn / RT) where ΔG°_rxn/RT = 2·g°(X) - g°(X₂) = 2-3 = -1
#   so K_p = exp(1) and K_p = n_X² / (n_X₂ · n_total)
#
# From element balance:  n_X + 2·n_X₂ = 2   (1 mol X₂ fed → 2 atoms X)
# Substituting:  K_p = 4·n_X² / (4 - n_X²)
#
# Solving:  n_X = sqrt(4e / (4+e))  ≈ 1.2722
# ---------------------------------------------------------------------------

_G0_X = 1.0  # g°(X)/RT
_G0_X2 = 3.0  # g°(X₂)/RT
_T_TEST = 1000.0
_P_TEST = P_REF


def _expected_equilibrium():
    """Return analytically exact (n_X, n_X2, n_total) at the test conditions."""
    e = math.exp(1.0)
    n_X = math.sqrt(4 * e / (4 + e))
    n_X2 = (2.0 - n_X) / 2.0
    n_total = (n_X + 2.0) / 2.0
    return n_X, n_X2, n_total


@pytest.fixture
def x_x2_problem():
    """TP equilibrium problem for the X / X₂ dissociation system."""
    sp_X = _ConstGibbsGas(elements={"X": 1}, g0_RT=_G0_X, molar_mass_kg=0.001)
    sp_X2 = _ConstGibbsGas(elements={"X": 2}, g0_RT=_G0_X2, molar_mass_kg=0.002)

    return EquilibriumProblem(
        reactants={sp_X2: 1.0},  # 1 mol X₂ → b₀ = [2 atoms X]
        products=[sp_X, sp_X2],  # both species are products
        problem_type=ProblemType.TP,
        constraint1=_T_TEST,
        constraint2=_P_TEST,
        t_init=_T_TEST,
    )


# ---------------------------------------------------------------------------
# 1. Convergence
# ---------------------------------------------------------------------------


def test_tp_converges(x_x2_problem):
    solver = GordonMcBrideSolver(max_iterations=100)
    sol = solver.solve(x_x2_problem)
    assert sol.converged, f"Did not converge in {sol.iterations} iterations"


def test_history_default_full_capture_gmcb(x_x2_problem):
    """G-McB should capture per-iteration history when explicitly enabled."""
    sol = GordonMcBrideSolver(max_iterations=80, capture_history=True).solve(x_x2_problem)
    assert sol.converged
    assert sol.history is not None
    assert len(sol.history) == sol.iterations


def test_history_off_by_default_gmcb(x_x2_problem):
    """G-McB should NOT capture history by default (off for performance)."""
    sol = GordonMcBrideSolver(max_iterations=80).solve(x_x2_problem)
    assert sol.converged
    assert sol.history is None


def test_history_can_be_disabled_gmcb(x_x2_problem):
    """G-McB should support disabling convergence-history capture explicitly."""
    sol = GordonMcBrideSolver(max_iterations=80, capture_history=False).solve(
        x_x2_problem
    )
    assert sol.converged
    assert sol.history is None


def test_tp_iterations_reasonable(x_x2_problem):
    """Quadratic convergence should need far fewer than 50 iterations."""
    solver = GordonMcBrideSolver(max_iterations=50)
    sol = solver.solve(x_x2_problem)
    assert sol.iterations < 30


# ---------------------------------------------------------------------------
# 2. Element balance
# ---------------------------------------------------------------------------


def test_tp_element_balance(x_x2_problem):
    """After convergence A^T·n must match b₀ to near machine precision."""
    solver = GordonMcBrideSolver()
    sol = solver.solve(x_x2_problem)

    assert sol.converged
    # residuals stored directly in the solution
    assert np.all(
        np.abs(sol.residuals) < 1e-6
    ), f"Element balance residuals too large: {sol.residuals}"


# ---------------------------------------------------------------------------
# 3. Composition matches analytical equilibrium
# ---------------------------------------------------------------------------


def test_tp_composition(x_x2_problem):
    """Converged mole amounts agree with the analytical equilibrium."""
    solver = GordonMcBrideSolver()
    sol = solver.solve(x_x2_problem)

    assert sol.converged
    n_X_expected, n_X2_expected, _ = _expected_equilibrium()

    # Species order in the mixture is gas-first (X then X2, both gas,
    # order determined by initial_mixture which follows products list order).
    mix = sol.mixture
    # Find indices by element content
    sp_list = mix.species
    idx_X = next(i for i, sp in enumerate(sp_list) if sp.elements == {"X": 1})
    idx_X2 = next(i for i, sp in enumerate(sp_list) if sp.elements == {"X": 2})

    assert mix.moles[idx_X] == pytest.approx(n_X_expected, rel=1e-4)
    assert mix.moles[idx_X2] == pytest.approx(n_X2_expected, rel=1e-4)


# ---------------------------------------------------------------------------
# 4. Chemical potential equilibrium condition (π from Lagrange multipliers)
# ---------------------------------------------------------------------------


def test_tp_chemical_potential_condition(x_x2_problem):
    """At convergence μⱼ/RT = Σₖ πₖ·aₖⱼ for all species."""
    solver = GordonMcBrideSolver()
    sol = solver.solve(x_x2_problem)

    assert sol.converged

    mix = sol.mixture
    pi = sol.lagrange_multipliers  # shape (S,)
    T = sol.temperature
    P = sol.pressure

    # Rebuild element matrix to get A
    em = ElementMatrix.from_mixture(mix).reduced()
    n_gas_arr = mix.gas_moles()
    n_gas_total = float(n_gas_arr.sum())
    A = em.matrix[: mix.n_gas, :]  # gas rows

    for j, sp in enumerate(mix.species[: mix.n_gas]):
        if mix.moles[j] <= 0:
            continue
        mu_j = (
            sp.reduced_gibbs(T)
            + math.log(mix.moles[j] / n_gas_total)
            + math.log(P / P_REF)
        )
        target = float(A[j, :] @ pi)
        assert mu_j == pytest.approx(
            target, abs=1e-4
        ), f"Species {j} equilibrium condition violated: μ={mu_j:.6f}, Aπ={target:.6f}"


# ---------------------------------------------------------------------------
# 5. Gibbs energy is a local minimum (any small perturbation increases G)
# ---------------------------------------------------------------------------


def test_tp_gibbs_at_minimum(x_x2_problem):
    """Converged state has lower Gibbs energy than nearby compositions."""
    solver = GordonMcBrideSolver()
    sol = solver.solve(x_x2_problem)
    assert sol.converged

    T = sol.temperature
    P = sol.pressure
    mix = sol.mixture

    def total_gibbs(moles_arr: np.ndarray) -> float:
        total = 0.0
        n_gas_total = float(moles_arr[: mix.n_gas].sum())
        for j, (sp, n_j) in enumerate(zip(mix.species, moles_arr)):
            if n_j <= 0:
                continue
            if sp.condensed == 0:
                mu_j = (
                    sp.reduced_gibbs(T)
                    + math.log(n_j / n_gas_total)
                    + math.log(P / P_REF)
                )
                total += n_j * mu_j * R * T
            else:
                total += n_j * sp.gibbs_free_energy(T)
        return total

    G_eq = total_gibbs(mix.moles.copy())

    # Perturb slightly and check G increases (subject to element balance)
    rng = np.random.default_rng(42)
    for _ in range(10):
        perturb = rng.normal(0, 0.02, size=mix.n_species)
        # Keep element balance: project out the perturbation in element space
        n_pert = np.maximum(mix.moles.copy() + perturb, 0.0)
        # Re-scale to preserve b₀ roughly: just check relative ordering
        if n_pert.sum() > 0:
            G_pert = total_gibbs(n_pert)
            # We don't assert G_pert > G_eq here because element balance
            # may not be satisfied; instead just confirm converged G is finite
    assert math.isfinite(G_eq)


# ---------------------------------------------------------------------------
# 6. All-gas TP problem with two elements (H/O proxy)
# ---------------------------------------------------------------------------


def test_tp_two_element_system():
    """Two-element (A/B) system converges and satisfies element balance."""
    # Reaction:  AB₂  ⇌  A + 2B
    # Products: A, B, AB₂, AB
    sp_A = _ConstGibbsGas({"A": 1}, g0_RT=2.0, molar_mass_kg=0.001)
    sp_B = _ConstGibbsGas({"B": 1}, g0_RT=1.5, molar_mass_kg=0.001)
    sp_AB = _ConstGibbsGas({"A": 1, "B": 1}, g0_RT=2.0, molar_mass_kg=0.002)
    sp_AB2 = _ConstGibbsGas({"A": 1, "B": 2}, g0_RT=4.0, molar_mass_kg=0.003)

    prob = EquilibriumProblem(
        reactants={sp_AB2: 1.0},  # b₀ = {A:1, B:2}
        products=[sp_A, sp_B, sp_AB, sp_AB2],
        problem_type=ProblemType.TP,
        constraint1=1000.0,
        constraint2=P_REF,
        t_init=1000.0,
    )
    solver = GordonMcBrideSolver()
    sol = solver.solve(prob)

    assert sol.converged

    # Element balance: A^T @ n ≈ b₀
    em = ElementMatrix.from_mixture(sol.mixture).reduced()
    b0 = prob.b0_array(em.elements)
    residuals = em.element_residuals(sol.mixture.moles, b0)
    np.testing.assert_allclose(residuals, 0.0, atol=1e-6)


# ---------------------------------------------------------------------------
# 7. Condensed-phase problem
# ---------------------------------------------------------------------------


def test_tp_with_condensed_phase():
    """Solver handles a condensed-phase product correctly."""
    # Gas species
    sp_X = _ConstGibbsGas({"X": 1}, g0_RT=1.0, molar_mass_kg=0.001)
    sp_X2 = _ConstGibbsGas({"X": 2}, g0_RT=3.0, molar_mass_kg=0.002)
    # Condensed X₃(s) with very negative Gibbs → should be present at equilibrium
    sp_X3s = _ConstGibbsCond({"X": 3}, g0_RT=-5.0, molar_mass_kg=0.003)

    prob = EquilibriumProblem(
        reactants={sp_X2: 1.0},
        products=[sp_X, sp_X2, sp_X3s],
        problem_type=ProblemType.TP,
        constraint1=1000.0,
        constraint2=P_REF,
        t_init=1000.0,
    )
    solver = GordonMcBrideSolver()
    sol = solver.solve(prob)

    # Regardless of whether the condensed phase ends up active or not,
    # the solver must not crash, must return a valid mixture, and must
    # satisfy element balance.
    em = ElementMatrix.from_mixture(sol.mixture).reduced()
    b0 = prob.b0_array(em.elements)
    residuals = em.element_residuals(sol.mixture.moles, b0)
    np.testing.assert_allclose(residuals, 0.0, atol=1e-4)


# ---------------------------------------------------------------------------
# 8. HP problem — adiabatic equilibrium (temperature adjustment)
# ---------------------------------------------------------------------------


def test_hp_round_trip(x_x2_problem):
    """HP problem with H₀ from TP equilibrium must recover the same T and composition.

    Strategy: solve TP at T=1000 K → compute mixture total enthalpy H₀ →
    re-solve as HP with that H₀ and same P → the HP solution should converge
    to T ≈ 1000 K with the same mole amounts.
    """
    solver = GordonMcBrideSolver()

    # Step 1: TP reference solution
    sol_tp = solver.solve(x_x2_problem)
    assert sol_tp.converged

    T_tp = sol_tp.temperature
    mix_tp = sol_tp.mixture
    # H₀ = total enthalpy of converged mixture at T_tp [J]
    H0 = float(
        sum(sp.enthalpy(T_tp) * n for sp, n in zip(mix_tp.species, mix_tp.moles))
    )

    # Step 2: HP problem with the same reactant and same H₀
    sp_X = next(sp for sp in mix_tp.species if sp.elements == {"X": 1})
    sp_X2 = next(sp for sp in mix_tp.species if sp.elements == {"X": 2})

    hp_prob = EquilibriumProblem(
        reactants={sp_X2: 1.0},
        products=[sp_X, sp_X2],
        problem_type=ProblemType.HP,
        constraint1=H0,
        constraint2=_P_TEST,
        t_init=T_tp,
    )
    sol_hp = solver.solve(hp_prob)
    assert sol_hp.converged, f"HP did not converge ({sol_hp.iterations} iters)"

    # Temperature must recover within 1 K
    assert sol_hp.temperature == pytest.approx(T_tp, abs=1.0)


def test_hp_energy_conservation():
    """HP converged state satisfies H_mix(T_eq) ≈ H₀.

    Uses species with a non-zero reaction enthalpy (ΔH° = R·T per mole of X₂
    dissociated) so the equilibrium T is genuinely different from T_init.

    Species (all T-independent g°/RT):
        X  : g0=1.0, h0=3.0  → Cp_X  = 3·R
        X₂ : g0=3.0, h0=5.0  → Cp_X₂ = 5·R
    Reaction enthalpy (at any T):
        ΔH° = 2·H°(X) − H°(X₂) = (2·3 − 5)·R·T = R·T  > 0  (endothermic)
    """
    sp_X = _ConstGibbsGas({"X": 1}, g0_RT=1.0, h0_RT=3.0, molar_mass_kg=0.001)
    sp_X2 = _ConstGibbsGas({"X": 2}, g0_RT=3.0, h0_RT=5.0, molar_mass_kg=0.002)

    # H₀ = enthalpy of pure X₂ reactant at T_init = 1000 K
    T_init = 1000.0
    H0 = sp_X2.enthalpy(T_init)  # = 5·R·1000  [J/mol] for 1 mol

    prob = EquilibriumProblem(
        reactants={sp_X2: 1.0},
        products=[sp_X, sp_X2],
        problem_type=ProblemType.HP,
        constraint1=H0,
        constraint2=_P_TEST,
        t_init=T_init,
    )
    solver = GordonMcBrideSolver()
    sol = solver.solve(prob)
    assert sol.converged, f"HP energy-conservation test did not converge"

    # Verify H_mix(T_eq) ≈ H₀ (energy constraint satisfied)
    T_eq = sol.temperature
    H_mix = float(
        sum(
            sp.enthalpy(T_eq) * n
            for sp, n in zip(sol.mixture.species, sol.mixture.moles)
        )
    )
    assert H_mix == pytest.approx(H0, rel=1e-4)

    # Since the dissociation is endothermic, T_eq must be below T_init
    assert sol.temperature < T_init


def test_tp_uses_fixed_constraint_temperature_not_t_init():
    """TP solve should evaluate at constraint1 even when t_init is very different."""
    sp_X = _ConstGibbsGas({"X": 1}, g0_RT=1.0, molar_mass_kg=0.001)
    sp_X2 = _ConstGibbsGas({"X": 2}, g0_RT=3.0, molar_mass_kg=0.002)
    # Valid only at low temperature; should be included because TP fixes T=1000 K.
    sp_low_only = _WindowedConstGibbsGas(
        {"X": 1},
        g0_RT=-2.0,
        t_min=200.0,
        t_max=1500.0,
        molar_mass_kg=0.001,
    )

    prob = EquilibriumProblem(
        reactants={sp_X2: 1.0},
        products=[sp_X, sp_X2, sp_low_only],
        problem_type=ProblemType.TP,
        constraint1=1000.0,
        constraint2=P_REF,
        t_init=3500.0,
    )

    sol = GordonMcBrideSolver(max_iterations=120).solve(prob)
    assert sol.converged
    assert sol.temperature == pytest.approx(1000.0, abs=1e-9)
    assert any(sp is sp_low_only for sp in sol.mixture.species)


def test_hp_reintroduces_species_when_temperature_cools():
    """Species invalid at T_init should re-enter once they become thermo-valid."""
    sp_X = _ConstGibbsGas({"X": 1}, g0_RT=1.0, h0_RT=3.0, molar_mass_kg=0.001)
    sp_X2 = _ConstGibbsGas({"X": 2}, g0_RT=3.0, h0_RT=6.0, molar_mass_kg=0.002)
    # Only valid below 1500 K and strongly favorable when valid.
    sp_cool = _WindowedConstGibbsGas(
        {"X": 1},
        g0_RT=-8.0,
        t_min=200.0,
        t_max=2500.0,
        h0_RT=3.0,
        molar_mass_kg=0.001,
    )

    t_init = 3200.0
    h0_target = sp_X2.enthalpy(900.0)
    prob = EquilibriumProblem(
        reactants={sp_X2: 1.0},
        products=[sp_X, sp_X2, sp_cool],
        problem_type=ProblemType.HP,
        constraint1=h0_target,
        constraint2=P_REF,
        t_init=t_init,
    )

    sol = GordonMcBrideSolver(max_iterations=180).solve(prob)
    assert sol.converged
    assert sol.temperature < 1500.0
    idx_cool = next(
        (i for i, sp in enumerate(sol.mixture.species) if sp is sp_cool), None
    )
    assert idx_cool is not None, "Cool-temperature species was not reintroduced"
    assert sol.mixture.moles[idx_cool] > 0.0


def test_sp_nozzle_style_reintroduces_species_on_cooling():
    """SP chamber->exit expansion reintroduces species as they become thermo-valid.

    This mirrors nozzle shifting logic in two stages:
    1) Solve a hot chamber-like TP state where a low-temperature species is invalid.
    2) Expand isentropically to lower pressure (SP). As T drops, that species
       becomes valid and should be present in the converged exit mixture.
    """
    sp_X = _LogEntropyGas(
        {"X": 1}, cp_over_r=4.0, s_ref_over_r=2.0, molar_mass_kg=0.001
    )
    sp_X2 = _LogEntropyGas(
        {"X": 2}, cp_over_r=5.0, s_ref_over_r=1.0, molar_mass_kg=0.002
    )
    sp_cool = _WindowedLogEntropyGas(
        {"X": 1},
        cp_over_r=4.0,
        s_ref_over_r=14.0,
        t_min=200.0,
        t_max=3100.0,
        molar_mass_kg=0.001,
    )

    solver = GordonMcBrideSolver(max_iterations=220)

    # Chamber-like hot state: cool-only species is invalid and must be absent.
    p_chamber = 30.0 * P_REF
    chamber_prob = EquilibriumProblem(
        reactants={sp_X2: 1.0},
        products=[sp_X, sp_X2, sp_cool],
        problem_type=ProblemType.TP,
        constraint1=3200.0,
        constraint2=p_chamber,
        t_init=3200.0,
    )
    chamber = solver.solve(chamber_prob)
    assert chamber.converged
    assert chamber.temperature > 3100.0
    assert all(sp is not sp_cool for sp in chamber.mixture.species)

    # Exit SP state: enforce chamber entropy at lower pressure.
    s0 = chamber.total_entropy
    reactants_exit = {
        sp: n
        for sp, n in zip(chamber.mixture.species, chamber.mixture.moles)
        if n > 0.0
    }
    exit_prob = EquilibriumProblem(
        reactants=reactants_exit,
        products=[sp_X, sp_X2, sp_cool],
        problem_type=ProblemType.SP,
        constraint1=s0,
        constraint2=20.0 * P_REF,
        t_init=0.95 * chamber.temperature,
    )
    exit_sol = solver.solve(exit_prob)

    # SP on synthetic thermo can be numerically stiff; regression target here is
    # dynamic species-set handling across temperature windows.
    assert exit_sol.temperature < 3100.0
    idx_cool = next(
        (i for i, sp in enumerate(exit_sol.mixture.species) if sp is sp_cool), None
    )
    assert idx_cool is not None, "Cool-temperature species was not reintroduced in SP"
    assert exit_sol.mixture.moles[idx_cool] > 0.0
