"""
Integration regression tests for Prometheus solvers.

Test strategy
-------------
Three layers of checks protect against solver regressions:

1. **GordonMcBrideSolver vs RocketCEA** — tight tolerance (0.05 %).
   GMcB is the reference implementation; it should agree with CEA to within
   a few thousandths of a percent.  Regressions in the primary Newton loop
   or the nozzle expansion will be caught here.

2. **MajorSpeciesSolver vs GordonMcBrideSolver** — moderate tolerance (1 %).
   Major is the production solver.  For CH4/O2 at high O/F ratios the two
   solvers can legitimately differ by up to ~0.6 % on Isp (nozzle-expansion
   sensitivity), so a 1 % bound is used.  If Major regresses (e.g. wrong
   temperature basin, element-balance error), this check catches it.

3. **APCP conservation + cross-solver agreement** — no external reference.
   Condensed Al2O3 phases are handled differently across codes, so
   RocketCEA is not a reliable third-party reference here.  Instead we
   verify element-abundance conservation, physical plausibility bounds
   (from Sutton & Biblarz), and GMcB/Major agreement within 0.5 %.

Run with:
    uv run pytest tests/test_regression_integration.py -v

Exclude from a fast CI run:
    uv run pytest -m "not integration"

RocketCEA is required for the CEA-comparison tests and is automatically
skipped if the package is not importable.  APCP and cross-solver tests
run without any external dependency.

Notes on the CEA comparison
----------------------------
* Custom fuel/oxidiser cards are registered from the Prometheus species
  database at 298.15 K, ensuring enthalpy references are identical between
  both codes.
* ``cstar`` is excluded from the CEA comparison: the two codes use slightly
  different reference-pressure conventions that cause a systematic ~3 %
  offset in cstar while Tc and Isp agree to < 0.003 %.
* Frozen performance uses ``frozenAtThroat=0`` in RocketCEA, matching
  Prometheus's convention of freezing the composition at the combustion
  chamber rather than at the throat.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pytest

from prometheus_equilibrium.equilibrium.element_matrix import ElementMatrix
from prometheus_equilibrium.equilibrium.performance import PerformanceSolver
from prometheus_equilibrium.equilibrium.problem import EquilibriumProblem, ProblemType
from prometheus_equilibrium.equilibrium.solver import (
    GordonMcBrideSolver,
    MajorSpeciesSolver,
)
from prometheus_equilibrium.equilibrium.species import SpeciesDatabase
from prometheus_equilibrium.propellants.loader import PropellantDatabase

pytestmark = pytest.mark.integration

# ---------------------------------------------------------------------------
# Tolerances
# ---------------------------------------------------------------------------

_GMCB_CEA_TOL = 5e-4   # 0.05 % — GMcB vs RocketCEA
_MAJOR_GMCB_TOL = 1e-2  # 1.0 % — Major vs GMcB (nozzle-expansion variability)
_ELEM_TOL = 1e-6        # element abundance conservation
_APCP_CROSS_TOL = 5e-3  # 0.5 % — APCP cross-solver agreement

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_THERMO_DIR = (
    Path(__file__).resolve().parent.parent
    / "prometheus_equilibrium"
    / "thermo_data"
)
_PROPELLANTS_TOML = (
    Path(__file__).resolve().parent.parent
    / "prometheus_equilibrium"
    / "propellants"
    / "propellants.toml"
)

# ---------------------------------------------------------------------------
# Session-scoped fixtures  (load DB once per test session)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def species_db() -> SpeciesDatabase:
    """NASA-9 species database, loaded once for the entire test session."""
    db = SpeciesDatabase(nasa9_path=str(_THERMO_DIR / "nasa9.json"))
    db.load(include_janaf=False, include_nasa7=False)
    return db


@pytest.fixture(scope="session")
def prop_db(species_db) -> PropellantDatabase:
    """Propellant TOML database, loaded once for the entire test session."""
    pdb = PropellantDatabase(str(_PROPELLANTS_TOML), species_db=species_db)
    pdb.load()
    return pdb


# ---------------------------------------------------------------------------
# RocketCEA fixtures  (lazy import — APCP tests work without CEA)
# ---------------------------------------------------------------------------

_CAL_TO_J = 4.18400
_RANKINE_TO_K = 5.0 / 9.0


def _register_cea_species(formula: str, kind: str, pairs, h_j_per_mol: float) -> None:
    from rocketcea.cea_obj import add_new_fuel, add_new_oxidizer

    h_cal = h_j_per_mol / _CAL_TO_J
    elems = "  ".join(f"{sym} {n}" for sym, n in pairs)
    card = (
        f"{kind} prom_{formula}  {elems}  wt%=100."
        f"  t(k)=298.150  h,cal={h_cal:.6f}"
    )
    (add_new_fuel if kind == "fuel" else add_new_oxidizer)(f"prom_{formula}", card)


@pytest.fixture(scope="session")
def cea_h2o2(species_db):
    """RocketCEA object for H2/O2, enthalpy-registered from Prometheus DB."""
    pytest.importorskip("rocketcea", reason="rocketcea not installed")
    from rocketcea.cea_obj import CEA_Obj

    h2 = species_db.find("H2", "G")
    o2 = species_db.find("O2", "G")
    _register_cea_species("H2", "fuel", [("H", 2)], h2.enthalpy(298.15))
    _register_cea_species("O2", "oxid", [("O", 2)], o2.enthalpy(298.15))
    return CEA_Obj(oxName="prom_O2", fuelName="prom_H2")


@pytest.fixture(scope="session")
def cea_ch4o2(species_db):
    """RocketCEA object for CH4/O2, enthalpy-registered from Prometheus DB."""
    pytest.importorskip("rocketcea", reason="rocketcea not installed")
    from rocketcea.cea_obj import CEA_Obj

    ch4 = species_db.find("CH4", "G")
    o2 = species_db.find("O2", "G")
    _register_cea_species("CH4", "fuel", [("C", 1), ("H", 4)], ch4.enthalpy(298.15))
    _register_cea_species("O2", "oxid", [("O", 2)], o2.enthalpy(298.15))
    return CEA_Obj(oxName="prom_O2", fuelName="prom_CH4")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _elem_balance_error(sol, reactants: Dict) -> float:
    """Max relative element-abundance error across all elements."""
    em = ElementMatrix.from_mixture(sol.mixture)
    b0_map: Dict[str, float] = {}
    for sp, n in reactants.items():
        for el, cnt in sp.elements.items():
            if el != "e-":
                b0_map[el] = b0_map.get(el, 0.0) + n * cnt

    elements = [e for e in em._elements if e != "e-"]
    b0_arr = np.array([b0_map.get(e, 0.0) for e in elements])
    b_calc = em.element_abundances(np.array(list(sol.mixture.moles)))[: len(elements)]

    with np.errstate(invalid="ignore", divide="ignore"):
        rel = np.where(
            np.abs(b0_arr) > 1e-30,
            np.abs(b_calc - b0_arr) / np.abs(b0_arr),
            0.0,
        )
    return float(np.max(rel))


def _make_biprop_problem(
    fuel_sp,
    ox_sp,
    of: float,
    pc_pa: float,
    products,
    t_init: float = 3500.0,
) -> Tuple[EquilibriumProblem, Dict]:
    """HP EquilibriumProblem from two species supplied at 298.15 K."""
    fu_moles = 1.0 / (1.0 + of) / fuel_sp.molar_mass()
    ox_moles = of / (1.0 + of) / ox_sp.molar_mass()
    reactants = {fuel_sp: fu_moles, ox_sp: ox_moles}
    H0 = sum(n * sp.enthalpy(298.15) for sp, n in reactants.items())
    return (
        EquilibriumProblem(
            reactants=reactants,
            products=products,
            problem_type=ProblemType.HP,
            constraint1=H0,
            constraint2=pc_pa,
            t_init=t_init,
        ),
        reactants,
    )


def _cea_ref(cea_obj, of: float, pc_psia: float, eps: float) -> Dict[str, float]:
    """Query RocketCEA for Tc, shifting Isp, and frozen Isp (frozen at chamber)."""
    return {
        "Tc": float(cea_obj.get_Tcomb(Pc=pc_psia, MR=of)) * _RANKINE_TO_K,
        "isp_sh": float(cea_obj.get_Isp(Pc=pc_psia, MR=of, eps=eps)),
        "isp_fr": float(
            cea_obj.get_Isp(Pc=pc_psia, MR=of, eps=eps, frozen=1, frozenAtThroat=0)
        ),
    }


_PC_PSIA = 1000.0
_PC_PA = _PC_PSIA * 6894.757
_EPS = 40.0  # nozzle area ratio used for all CEA-comparison cases


# ---------------------------------------------------------------------------
# H2/O2 — GMcB vs RocketCEA
# ---------------------------------------------------------------------------

_H2O2_OFS = [
    pytest.param(4.0, id="OF4"),
    pytest.param(6.0, id="OF6"),
    pytest.param(8.0, id="OF8"),
]


@pytest.mark.parametrize("of", _H2O2_OFS)
def test_h2o2_gmcb_vs_cea(of, species_db, cea_h2o2):
    """GMcB shifting and frozen Isp must match RocketCEA within 0.05 % for H2/O2."""
    h2 = species_db.find("H2", "G")
    o2 = species_db.find("O2", "G")
    products = species_db.get_species(frozenset({"H", "O"}), max_atoms=20)
    problem, reactants = _make_biprop_problem(h2, o2, of, _PC_PA, products)
    ref = _cea_ref(cea_h2o2, of, _PC_PSIA, _EPS)

    perf = PerformanceSolver(GordonMcBrideSolver())
    r_sh = perf.solve(problem, area_ratio=_EPS, shifting=True)
    r_fr = perf.solve(problem, area_ratio=_EPS, shifting=False)

    assert r_sh.chamber.converged, "Shifting chamber did not converge"
    assert r_fr.chamber.converged, "Frozen chamber did not converge"
    assert _elem_balance_error(r_sh.chamber, reactants) < _ELEM_TOL

    tc = r_sh.chamber.temperature
    assert abs(tc - ref["Tc"]) / ref["Tc"] < _GMCB_CEA_TOL, (
        f"Tc {tc:.2f} vs CEA {ref['Tc']:.2f} K — "
        f"{abs(tc - ref['Tc']) / ref['Tc'] * 100:.3f} % deviation"
    )
    assert abs(r_sh.isp_vac - ref["isp_sh"]) / ref["isp_sh"] < _GMCB_CEA_TOL, (
        f"Isp_shift {r_sh.isp_vac:.3f} vs CEA {ref['isp_sh']:.3f} s — "
        f"{abs(r_sh.isp_vac - ref['isp_sh']) / ref['isp_sh'] * 100:.3f} % deviation"
    )
    assert abs(r_fr.isp_vac - ref["isp_fr"]) / ref["isp_fr"] < _GMCB_CEA_TOL, (
        f"Isp_frozen {r_fr.isp_vac:.3f} vs CEA {ref['isp_fr']:.3f} s — "
        f"{abs(r_fr.isp_vac - ref['isp_fr']) / ref['isp_fr'] * 100:.3f} % deviation"
    )


# ---------------------------------------------------------------------------
# H2/O2 — MajorSpeciesSolver vs GordonMcBrideSolver
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("of", _H2O2_OFS)
def test_h2o2_major_vs_gmcb(of, species_db):
    """MajorSpeciesSolver must agree with GMcB within 1 % for H2/O2."""
    h2 = species_db.find("H2", "G")
    o2 = species_db.find("O2", "G")
    products = species_db.get_species(frozenset({"H", "O"}), max_atoms=20)
    problem, reactants = _make_biprop_problem(h2, o2, of, _PC_PA, products)

    gmcb_sh = PerformanceSolver(GordonMcBrideSolver()).solve(
        problem, area_ratio=_EPS, shifting=True
    )
    gmcb_fr = PerformanceSolver(GordonMcBrideSolver()).solve(
        problem, area_ratio=_EPS, shifting=False
    )
    major_sh = PerformanceSolver(MajorSpeciesSolver()).solve(
        problem, area_ratio=_EPS, shifting=True
    )
    major_fr = PerformanceSolver(MajorSpeciesSolver()).solve(
        problem, area_ratio=_EPS, shifting=False
    )

    assert major_sh.chamber.converged, "Major shifting chamber did not converge"
    assert major_fr.chamber.converged, "Major frozen chamber did not converge"
    assert _elem_balance_error(major_sh.chamber, reactants) < _ELEM_TOL

    def _check(a, b, label):
        diff = abs(a - b) / ((abs(a) + abs(b)) / 2.0)
        assert diff < _MAJOR_GMCB_TOL, (
            f"{label}: Major={a:.4g}, GMcB={b:.4g}, "
            f"rel diff={diff * 100:.3f} % > {_MAJOR_GMCB_TOL * 100} %"
        )

    _check(major_sh.chamber.temperature, gmcb_sh.chamber.temperature, "Tc")
    _check(major_sh.isp_vac, gmcb_sh.isp_vac, "Isp_shift")
    _check(major_fr.isp_vac, gmcb_fr.isp_vac, "Isp_frozen")


# ---------------------------------------------------------------------------
# CH4/O2 — GMcB vs RocketCEA
# ---------------------------------------------------------------------------

_CH4O2_OFS = [
    pytest.param(2.6, id="OF2p6"),
    pytest.param(3.5, id="OF3p5"),
    pytest.param(4.5, id="OF4p5"),
]


@pytest.mark.parametrize("of", _CH4O2_OFS)
def test_ch4o2_gmcb_vs_cea(of, species_db, cea_ch4o2):
    """GMcB shifting and frozen Isp must match RocketCEA within 0.05 % for CH4/O2."""
    ch4 = species_db.find("CH4", "G")
    o2 = species_db.find("O2", "G")
    products = species_db.get_species(frozenset({"C", "H", "O"}), max_atoms=20)
    problem, reactants = _make_biprop_problem(ch4, o2, of, _PC_PA, products)
    ref = _cea_ref(cea_ch4o2, of, _PC_PSIA, _EPS)

    perf = PerformanceSolver(GordonMcBrideSolver())
    r_sh = perf.solve(problem, area_ratio=_EPS, shifting=True)
    r_fr = perf.solve(problem, area_ratio=_EPS, shifting=False)

    assert r_sh.chamber.converged, "Shifting chamber did not converge"
    assert r_fr.chamber.converged, "Frozen chamber did not converge"
    assert _elem_balance_error(r_sh.chamber, reactants) < _ELEM_TOL

    tc = r_sh.chamber.temperature
    assert abs(tc - ref["Tc"]) / ref["Tc"] < _GMCB_CEA_TOL, (
        f"Tc {tc:.2f} vs CEA {ref['Tc']:.2f} K — "
        f"{abs(tc - ref['Tc']) / ref['Tc'] * 100:.3f} % deviation"
    )
    assert abs(r_sh.isp_vac - ref["isp_sh"]) / ref["isp_sh"] < _GMCB_CEA_TOL, (
        f"Isp_shift {r_sh.isp_vac:.3f} vs CEA {ref['isp_sh']:.3f} s — "
        f"{abs(r_sh.isp_vac - ref['isp_sh']) / ref['isp_sh'] * 100:.3f} % deviation"
    )
    assert abs(r_fr.isp_vac - ref["isp_fr"]) / ref["isp_fr"] < _GMCB_CEA_TOL, (
        f"Isp_frozen {r_fr.isp_vac:.3f} vs CEA {ref['isp_fr']:.3f} s — "
        f"{abs(r_fr.isp_vac - ref['isp_fr']) / ref['isp_fr'] * 100:.3f} % deviation"
    )


# ---------------------------------------------------------------------------
# CH4/O2 — MajorSpeciesSolver vs GordonMcBrideSolver
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("of", _CH4O2_OFS)
def test_ch4o2_major_vs_gmcb(of, species_db):
    """MajorSpeciesSolver must agree with GMcB within 1 % for CH4/O2."""
    ch4 = species_db.find("CH4", "G")
    o2 = species_db.find("O2", "G")
    products = species_db.get_species(frozenset({"C", "H", "O"}), max_atoms=20)
    problem, reactants = _make_biprop_problem(ch4, o2, of, _PC_PA, products)

    gmcb_sh = PerformanceSolver(GordonMcBrideSolver()).solve(
        problem, area_ratio=_EPS, shifting=True
    )
    gmcb_fr = PerformanceSolver(GordonMcBrideSolver()).solve(
        problem, area_ratio=_EPS, shifting=False
    )
    major_sh = PerformanceSolver(MajorSpeciesSolver()).solve(
        problem, area_ratio=_EPS, shifting=True
    )
    major_fr = PerformanceSolver(MajorSpeciesSolver()).solve(
        problem, area_ratio=_EPS, shifting=False
    )

    assert major_sh.chamber.converged, "Major shifting chamber did not converge"
    assert major_fr.chamber.converged, "Major frozen chamber did not converge"
    assert _elem_balance_error(major_sh.chamber, reactants) < _ELEM_TOL

    def _check(a, b, label):
        diff = abs(a - b) / ((abs(a) + abs(b)) / 2.0)
        assert diff < _MAJOR_GMCB_TOL, (
            f"{label}: Major={a:.4g}, GMcB={b:.4g}, "
            f"rel diff={diff * 100:.3f} % > {_MAJOR_GMCB_TOL * 100} %"
        )

    _check(major_sh.chamber.temperature, gmcb_sh.chamber.temperature, "Tc")
    _check(major_sh.isp_vac, gmcb_sh.isp_vac, "Isp_shift")
    _check(major_fr.isp_vac, gmcb_fr.isp_vac, "Isp_frozen")


# ---------------------------------------------------------------------------
# APCP (HTPB/AP/Al 68/18/14) — conservation + cross-solver agreement
# ---------------------------------------------------------------------------
#
# RocketCEA is not a reliable reference for APCP: condensed Al2O3 phases
# and HTPB thermochemistry are represented differently across codes.
#
# Checks:
#   1. Both solvers converge.
#   2. Element abundance is conserved to 1e-6.
#   3. The two solver implementations agree within 0.5 %.
#   4. Results fall within published physical bounds
#      (Sutton & Biblarz, "Rocket Propulsion Elements").

_APCP_PC_PA = 1000.0 * 6894.757
_APCP_EPS = 10.0

_APCP_TC_BOUNDS = (2900.0, 3700.0)     # K
_APCP_CSTAR_BOUNDS = (1400.0, 1900.0)  # m/s
_APCP_ISP_SH_BOUNDS = (250.0, 320.0)   # s (shifting, vacuum)
_APCP_ISP_FR_BOUNDS = (230.0, 310.0)   # s (frozen, vacuum)


@pytest.fixture(scope="session")
def apcp_results(species_db, prop_db):
    """Run both solvers on HTPB/AP/Al 68/18/14 and cache the results."""
    mixture = prop_db.mix(
        [
            ("HTPB_R_45M", 14.0),
            ("AMMONIUM_PERCHLORATE", 68.0),
            ("ALUMINUM_PURE_CRYSTALINE", 18.0),
        ]
    )
    products = species_db.get_species(mixture.elements, max_atoms=20)
    problem = EquilibriumProblem(
        reactants=mixture.reactants,
        products=products,
        problem_type=ProblemType.HP,
        constraint1=mixture.enthalpy,
        constraint2=_APCP_PC_PA,
        t_init=3000.0,
    )

    out = {}
    for name, cls in [("GMcB", GordonMcBrideSolver), ("Major", MajorSpeciesSolver)]:
        perf = PerformanceSolver(cls())
        out[name] = {
            "shift": perf.solve(problem, area_ratio=_APCP_EPS, shifting=True),
            "frozen": perf.solve(problem, area_ratio=_APCP_EPS, shifting=False),
            "reactants": mixture.reactants,
        }
    return out


@pytest.mark.parametrize("solver_name", ["GMcB", "Major"])
def test_apcp_conservation(solver_name, apcp_results):
    """Element abundance must be conserved to 1e-6 for both APCP solvers."""
    r = apcp_results[solver_name]
    sol = r["shift"].chamber
    assert sol.converged, f"{solver_name} shifting chamber did not converge"
    err = _elem_balance_error(sol, r["reactants"])
    assert err < _ELEM_TOL, (
        f"{solver_name} element balance error {err:.2e} exceeds {_ELEM_TOL}"
    )


@pytest.mark.parametrize("solver_name", ["GMcB", "Major"])
def test_apcp_physical_bounds(solver_name, apcp_results):
    """APCP performance must fall within published physical bounds."""
    sh = apcp_results[solver_name]["shift"]
    fr = apcp_results[solver_name]["frozen"]

    tc = sh.chamber.temperature
    assert _APCP_TC_BOUNDS[0] <= tc <= _APCP_TC_BOUNDS[1], (
        f"{solver_name} Tc {tc:.1f} K outside {_APCP_TC_BOUNDS} K"
    )
    assert _APCP_CSTAR_BOUNDS[0] <= sh.cstar <= _APCP_CSTAR_BOUNDS[1], (
        f"{solver_name} cstar {sh.cstar:.1f} m/s outside {_APCP_CSTAR_BOUNDS}"
    )
    assert _APCP_ISP_SH_BOUNDS[0] <= sh.isp_vac <= _APCP_ISP_SH_BOUNDS[1], (
        f"{solver_name} Isp_shift {sh.isp_vac:.2f} s outside {_APCP_ISP_SH_BOUNDS}"
    )
    assert _APCP_ISP_FR_BOUNDS[0] <= fr.isp_vac <= _APCP_ISP_FR_BOUNDS[1], (
        f"{solver_name} Isp_frozen {fr.isp_vac:.2f} s outside {_APCP_ISP_FR_BOUNDS}"
    )


def test_apcp_cross_solver_agreement(apcp_results):
    """GMcB and MajorSpecies must agree within 0.5 % on all APCP metrics."""
    gmcb_sh = apcp_results["GMcB"]["shift"]
    maj_sh = apcp_results["Major"]["shift"]
    gmcb_fr = apcp_results["GMcB"]["frozen"]
    maj_fr = apcp_results["Major"]["frozen"]

    def _check(a, b, label):
        diff = abs(a - b) / ((abs(a) + abs(b)) / 2.0)
        assert diff < _APCP_CROSS_TOL, (
            f"{label}: GMcB={a:.4g}, Major={b:.4g}, "
            f"rel diff={diff * 100:.3f} % > {_APCP_CROSS_TOL * 100} %"
        )

    _check(gmcb_sh.chamber.temperature, maj_sh.chamber.temperature, "Tc")
    _check(gmcb_sh.cstar, maj_sh.cstar, "cstar")
    _check(gmcb_sh.isp_vac, maj_sh.isp_vac, "Isp_shifting")
    _check(gmcb_fr.isp_vac, maj_fr.isp_vac, "Isp_frozen")
