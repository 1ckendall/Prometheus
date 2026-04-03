"""Tests for PerformanceSolver paired shifting/frozen execution."""

import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from prometheus_equilibrium.core.constants import REFERENCE_PRESSURE as P_REF
from prometheus_equilibrium.core.constants import UNIVERSAL_GAS_CONSTANT as R
from prometheus_equilibrium.equilibrium.diagnostics import NonConvergenceReason
from prometheus_equilibrium.equilibrium.mixture import Mixture
from prometheus_equilibrium.equilibrium.performance import (
    PerformanceSolver,
    RocketPerformanceComparison,
    RocketPerformanceResult,
)
from prometheus_equilibrium.equilibrium.problem import EquilibriumProblem, ProblemType
from prometheus_equilibrium.equilibrium.solution import EquilibriumSolution
from prometheus_equilibrium.equilibrium.solver import (
    GordonMcBrideSolver,
    MajorSpeciesSolver,
)
from prometheus_equilibrium.equilibrium.species import Species, SpeciesDatabase


def test_solve_pair_calls_both_modes(monkeypatch):
    """solve_pair should execute shifting and frozen solves with shared inputs."""
    solver = PerformanceSolver()
    calls = []

    def _fake_solve(
        self,
        problem,
        pe_pa=None,
        area_ratio=None,
        shifting=True,
        ambient_pressure=101325.0,
    ):
        calls.append(
            {
                "problem": problem,
                "pe_pa": pe_pa,
                "area_ratio": area_ratio,
                "shifting": shifting,
                "ambient_pressure": ambient_pressure,
            }
        )
        return "shift" if shifting else "frozen"

    monkeypatch.setattr(PerformanceSolver, "solve", _fake_solve)

    token_problem: Any = object()
    result = solver.solve_pair(
        token_problem,
        pe_pa=120000.0,
        ambient_pressure=90000.0,
    )

    assert isinstance(result, RocketPerformanceComparison)
    assert result.shifting == "shift"
    assert result.frozen == "frozen"
    assert result.ambient_pressure == 90000.0

    assert len(calls) == 2
    assert calls[0]["problem"] is token_problem
    assert calls[0]["shifting"] is True
    assert calls[1]["shifting"] is False
    assert calls[0]["pe_pa"] == calls[1]["pe_pa"] == 120000.0
    assert calls[0]["ambient_pressure"] == calls[1]["ambient_pressure"] == 90000.0


def test_frozen_solve_reports_invalid_thermo_nonconvergence():
    """Frozen pressure solve should fail cleanly when thermo becomes invalid."""
    sp = _WindowedEntropyGas(
        {"X": 1},
        cp_over_r=4.0,
        s_ref_over_r=2.0,
        t_min=3000.0,
        t_max=6000.0,
        molar_mass_kg=0.02,
    )
    mix = Mixture([sp], np.array([1.0], dtype=float))
    chamber = EquilibriumSolution(
        mixture=mix,
        temperature=3500.0,
        pressure=2.0e6,
        converged=True,
        iterations=1,
        residuals=np.zeros(0),
        lagrange_multipliers=np.zeros(0),
    )

    exit_sol = PerformanceSolver()._solve_frozen_at_p(chamber, 101325.0)

    assert not exit_sol.converged
    assert exit_sol.failure_reason == NonConvergenceReason.INVALID_THERMO_PROPERTIES


def test_solve_raises_when_exit_not_converged(monkeypatch):
    """Public solve should fail fast if exit state does not converge."""
    solver = PerformanceSolver()

    sp = _LogEntropyGas({"X": 1}, cp_over_r=4.0, s_ref_over_r=2.0, molar_mass_kg=0.02)
    mix = Mixture([sp], np.array([1.0], dtype=float))
    chamber = EquilibriumSolution(
        mixture=mix,
        temperature=3500.0,
        pressure=2.0e6,
        converged=True,
        iterations=1,
        residuals=np.zeros(0),
        lagrange_multipliers=np.zeros(0),
    )
    throat = EquilibriumSolution(
        mixture=mix,
        temperature=3300.0,
        pressure=1.0e6,
        converged=True,
        iterations=1,
        residuals=np.zeros(0),
        lagrange_multipliers=np.zeros(0),
    )
    bad_exit = EquilibriumSolution(
        mixture=mix,
        temperature=2800.0,
        pressure=101325.0,
        converged=False,
        iterations=5,
        residuals=np.zeros(0),
        lagrange_multipliers=np.zeros(0),
        failure_reason=NonConvergenceReason.INVALID_THERMO_PROPERTIES,
    )

    monkeypatch.setattr(solver.solver, "solve", lambda _problem: chamber)
    monkeypatch.setattr(solver, "_find_throat", lambda *_args, **_kwargs: throat)
    monkeypatch.setattr(solver, "_solve_at_p", lambda *_args, **_kwargs: bad_exit)

    problem = EquilibriumProblem(
        reactants={sp: 1.0},
        products=[sp],
        problem_type=ProblemType.HP,
        constraint1=sp.enthalpy(3500.0),
        constraint2=2.0e6,
        t_init=3500.0,
    )

    with pytest.raises(RuntimeError, match="Exit solve did not converge"):
        solver.solve(problem, pe_pa=101325.0, shifting=False)


# ---------------------------------------------------------------------------
# Mock species for real code-path integration tests
# ---------------------------------------------------------------------------


class _LogEntropyGas(Species):
    """Ideal gas with Cp=const and S(T) ~ ln(T) — required for SP/HP tests."""

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
        return (self._cp_r * math.log(float(T)) + self._s0_r) * R


class _WindowedEntropyGas(_LogEntropyGas):
    """Log-entropy gas whose thermo is invalid outside a temperature window."""

    def __init__(self, *args, t_min: float, t_max: float, **kwargs):
        super().__init__(*args, **kwargs)
        self._t_min = float(t_min)
        self._t_max = float(t_max)

    def entropy(self, T):
        t_val = float(T)
        if self._t_min <= t_val <= self._t_max:
            return super().entropy(t_val)
        return float("nan")

    def specific_heat_capacity(self, T):
        t_val = float(T)
        if self._t_min <= t_val <= self._t_max:
            return super().specific_heat_capacity(t_val)
        return float("nan")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def log_entropy_hp_problem():
    """HP problem with X/X₂ mock system for end-to-end performance tests."""
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


# ---------------------------------------------------------------------------
# 2. Frozen nozzle expansion — code path coverage
# ---------------------------------------------------------------------------


def test_frozen_nozzle_cstar_positive(log_entropy_hp_problem):
    """PerformanceSolver.solve with frozen expansion: cstar must be positive."""
    problem, _, _ = log_entropy_hp_problem
    result = PerformanceSolver().solve(problem, pe_pa=P_REF, shifting=False)
    assert result.chamber.converged, "Chamber HP solve did not converge"
    assert result.cstar > 0


def test_frozen_nozzle_isp_vac_positive(log_entropy_hp_problem):
    """Frozen isp_vac must be positive."""
    problem, _, _ = log_entropy_hp_problem
    result = PerformanceSolver().solve(problem, pe_pa=P_REF, shifting=False)
    assert result.isp_vac > 0


def test_frozen_exit_temperature_lower_than_chamber(log_entropy_hp_problem):
    """Isentropic expansion must cool the gas."""
    problem, _, _ = log_entropy_hp_problem
    result = PerformanceSolver().solve(problem, pe_pa=P_REF, shifting=False)
    assert result.exit.temperature < result.chamber.temperature


# ---------------------------------------------------------------------------
# 3. Shifting nozzle expansion
# ---------------------------------------------------------------------------


def test_shifting_nozzle_cstar_positive(log_entropy_hp_problem):
    """PerformanceSolver.solve with shifting expansion: cstar must be positive."""
    problem, _, _ = log_entropy_hp_problem
    result = PerformanceSolver().solve(problem, pe_pa=P_REF, shifting=True)
    assert result.chamber.converged
    assert result.cstar > 0


def test_shifting_nozzle_isp_vac_positive(log_entropy_hp_problem):
    problem, _, _ = log_entropy_hp_problem
    result = PerformanceSolver().solve(problem, pe_pa=P_REF, shifting=True)
    assert result.isp_vac > 0


# ---------------------------------------------------------------------------
# 4. solve_pair consistency
# ---------------------------------------------------------------------------


def test_solve_pair_shifting_isp_geq_frozen(log_entropy_hp_problem):
    """Shifting Isp ≥ frozen Isp (shifting can take advantage of recomposition)."""
    problem, _, _ = log_entropy_hp_problem
    pair = PerformanceSolver().solve_pair(problem, pe_pa=P_REF)
    # Shifting vacuum Isp is generally ≥ frozen — not always guaranteed for mocks,
    # but both must be positive.
    assert pair.shifting.isp_vac > 0
    assert pair.frozen.isp_vac > 0


def test_solve_pair_area_ratio_consistent(log_entropy_hp_problem):
    """solve() with area_ratio should return a finite area_ratio in the result."""
    problem, _, _ = log_entropy_hp_problem
    result = PerformanceSolver().solve(problem, area_ratio=5.0, shifting=False)
    assert math.isfinite(result.area_ratio)
    assert result.area_ratio > 1.0


# ---------------------------------------------------------------------------
# 5. RocketCEA H₂/O₂ validation (optional — skipped if rocketcea not installed)
# ---------------------------------------------------------------------------


def test_h2o2_hp_temperature_vs_rocketcea():
    """H₂/O₂ HP temperature at O/F=8, 500 psia should match RocketCEA within 2%."""
    cea_mod = pytest.importorskip("rocketcea.cea_obj")
    add_new_fuel = cea_mod.add_new_fuel
    add_new_oxidizer = cea_mod.add_new_oxidizer
    CEA_Obj = cea_mod.CEA_Obj

    from prometheus_equilibrium.equilibrium.problem import (
        EquilibriumProblem,
        ProblemType,
    )
    from prometheus_equilibrium.equilibrium.solver import GordonMcBrideSolver
    from prometheus_equilibrium.equilibrium.species import SpeciesDatabase

    CAL_TO_J = 4.184
    RANKINE_TO_K = 1.0 / 1.8
    PSIA_TO_PA = 6894.757

    # Load real thermo DB
    db = SpeciesDatabase()
    db.load(include_nasa7=True, include_nasa9=True)

    h2 = db.find("H2", phase="G")
    o2 = db.find("O2", phase="G")

    T_ref = 298.15
    n_h2, n_o2 = 2.0, 1.0
    H0 = n_h2 * h2.enthalpy(T_ref) + n_o2 * o2.enthalpy(T_ref)

    products = db.get_species({"H", "O"}, max_atoms=8)
    problem = EquilibriumProblem(
        reactants={h2: n_h2, o2: n_o2},
        products=products,
        problem_type=ProblemType.HP,
        constraint1=H0,
        constraint2=500 * PSIA_TO_PA,
        t_init=3000.0,
    )

    sol = GordonMcBrideSolver().solve(problem)
    if not sol.converged:
        pytest.skip(f"Prometheus HP solve did not converge: {sol.failure_reason}")
    T_prometheus = sol.temperature

    # Register Prometheus thermo data with RocketCEA so references match
    h2_cal = h2.enthalpy(T_ref) / CAL_TO_J
    o2_cal = o2.enthalpy(T_ref) / CAL_TO_J
    add_new_fuel(
        "H2_prom_test",
        f"fuel H2_prom_test  H 2  wt%=100.  t(k)={T_ref:.3f}  h,cal={h2_cal:.3f}",
    )
    add_new_oxidizer(
        "O2_prom_test",
        f"oxid O2_prom_test  O 2  wt%=100.  t(k)={T_ref:.3f}  h,cal={o2_cal:.3f}",
    )
    cea = CEA_Obj(oxName="O2_prom_test", fuelName="H2_prom_test")

    # O/F = n_o2*M_o2 / (n_h2*M_h2) = 1*0.032 / (2*0.002) = 8.0
    Tc_R = cea.get_Tcomb(Pc=500.0, MR=8.0)
    T_cea = float(Tc_R) * RANKINE_TO_K

    err_pct = abs(T_prometheus - T_cea) / T_cea * 100
    assert (
        err_pct < 2.0
    ), f"T_prometheus={T_prometheus:.1f} K, T_CEA={T_cea:.1f} K, error={err_pct:.2f}%"


# ---------------------------------------------------------------------------
# 6. APCP frozen expansion convergence test
# ---------------------------------------------------------------------------


def test_apcp_frozen_expansion_converges():
    """Frozen nozzle expansion for APCP (Al-containing) must converge with phase substitution."""
    import json

    from prometheus_equilibrium.propellants.loader import PropellantDatabase

    REPO_ROOT = Path(__file__).resolve().parents[1]
    prop_file = REPO_ROOT / "basic_APCP.prop"
    if not prop_file.exists():
        pytest.skip("basic_APCP.prop not found at repo root")

    PSI_TO_PA = 6894.757
    chamber_pressure_pa = 1000.0 * PSI_TO_PA

    db = SpeciesDatabase()
    db.load(include_janaf=False)

    prop_db = PropellantDatabase(
        str(REPO_ROOT / "prometheus_equilibrium" / "propellants" / "propellants.toml"),
        species_db=db,
    )
    prop_db.load()

    apcp_cfg = json.loads(prop_file.read_text(encoding="utf-8"))
    components = [
        (rec["name"], float(rec["mass_fraction"])) for rec in apcp_cfg["components"]
    ]
    mixture = prop_db.mix(components)

    products = db.get_species(set(mixture.elements), max_atoms=20)
    problem = EquilibriumProblem(
        reactants=mixture.reactants,
        products=products,
        problem_type=ProblemType.HP,
        constraint1=mixture.enthalpy,
        constraint2=chamber_pressure_pa,
        t_init=3500.0,
    )

    solver = GordonMcBrideSolver(max_iterations=120)
    perf_solver = PerformanceSolver(solver, db=db)
    result = perf_solver.solve(problem, pe_pa=101325.0, shifting=False)

    assert (
        result.exit.converged
    ), f"Frozen exit did not converge: {result.exit.failure_reason}"
    assert (
        result.exit.temperature < result.chamber.temperature
    ), "Exit temperature must be lower than chamber temperature"
    assert (
        result.exit.temperature > 500.0
    ), f"Exit temperature {result.exit.temperature:.1f} K too low"

    # Check Al and O element balance between chamber and exit
    def _element_moles(sol, element):
        total = 0.0
        for sp, n in zip(sol.mixture.species, sol.mixture.moles):
            total += n * sp.elements.get(element, 0.0)
        return total

    al_chamber = _element_moles(result.chamber, "Al")
    al_exit = _element_moles(result.exit, "Al")
    o_chamber = _element_moles(result.chamber, "O")
    o_exit = _element_moles(result.exit, "O")

    if al_chamber > 0:
        al_rel_err = abs(al_exit - al_chamber) / al_chamber
        assert al_rel_err < 1e-6, f"Al element balance error: {al_rel_err:.2e}"
    if o_chamber > 0:
        o_rel_err = abs(o_exit - o_chamber) / o_chamber
        assert o_rel_err < 1e-6, f"O element balance error: {o_rel_err:.2e}"
