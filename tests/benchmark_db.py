"""
benchmark_db.py — Database configuration comparison vs RocketCEA.

Runs the same four propellant systems as benchmark.py with five different
database configurations, using GordonMcBrideSolver throughout:

  NASA      -- NASA-7 + NASA-9 only (current production default)
  AFCESIC   -- afcesic.json only (RF/CH corrected to NASA reference)
  TERRA     -- terra.json only
  JANAF     -- janaf.csv only (with REF-phase elements included as gas)
  MIXED     -- NASA + AFCESIC + TERRA + JANAF (NASA priority where available)

Compares all five against RocketCEA and reports:
  - T_c error (primary metric)
  - Convergence rate per database
  - Which database each product species comes from

Known JANAF limitation: the JANAF tables list standard-state reference elements
(H2, O2, N2) under phase "REF" rather than "G".  SpeciesDatabase._load_janaf
now treats REF as gas so these species are available.  Even after this fix,
JANAF is missing some exotic combustion product species (radicals, ions) that
NASA-9 covers, so convergence and accuracy are somewhat lower for JANAF-only.

Run with (from repo root):
    uv run python tests/benchmark_db.py
"""

from __future__ import annotations

import math
import sys
import time
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from loguru import logger
from rocketcea.cea_obj import CEA_Obj, add_new_fuel, add_new_oxidizer

from prometheus.equilibrium.problem import EquilibriumProblem, ProblemType
from prometheus.equilibrium.solver import GordonMcBrideSolver
from prometheus.equilibrium.species import (AFCESICCoeff, NASANineCoeff,
                                         NASASevenCoeff, Species,
                                         SpeciesDatabase)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

R_UNIV = 8.314462
CAL_TO_J = 4.184
PSIA_TO_PA = 6894.757
FPS_TO_MPS = 0.3048
RANKINE_TO_K = 1.0 / 1.8

_SOURCE_LABEL = {
    "AFCESICCoeff": "AFCESIC",
    "NASANineCoeff": "NASA-9",
    "NASASevenCoeff": "NASA-7",
    "JANAF": "JANAF",
    "ShomateCoeff": "Shomate",
}

_REPO_ROOT = Path(__file__).resolve().parent.parent
_THERMO_DIR = _REPO_ROOT / "prometheus" / "thermo_data"

# ---------------------------------------------------------------------------
# Database configurations
# ---------------------------------------------------------------------------


def _build_databases() -> Dict[str, SpeciesDatabase]:
    """Return one SpeciesDatabase per configuration."""
    nasa_only = SpeciesDatabase(
        nasa7_path=str(_THERMO_DIR / "nasa7.json"),
        nasa9_path=str(_THERMO_DIR / "nasa9.json"),
        janaf_path=str(_THERMO_DIR / "janaf.csv"),
    )
    nasa_only.load(
        include_nasa7=True,
        include_nasa9=True,
        include_afcesic=False,
        include_terra=False,
    )

    afcesic_only = SpeciesDatabase(
        nasa7_path=str(_THERMO_DIR / "nasa7.json"),
        nasa9_path=str(_THERMO_DIR / "nasa9.json"),
        janaf_path=str(_THERMO_DIR / "janaf.csv"),
        afcesic_path=str(_THERMO_DIR / "afcesic.json"),
    )
    afcesic_only.load(
        include_nasa7=False,
        include_nasa9=False,
        include_afcesic=True,
        include_terra=False,
    )

    terra_only = SpeciesDatabase(
        nasa7_path=str(_THERMO_DIR / "nasa7.json"),
        nasa9_path=str(_THERMO_DIR / "nasa9.json"),
        janaf_path=str(_THERMO_DIR / "janaf.csv"),
        terra_path=str(_THERMO_DIR / "terra.json"),
    )
    terra_only.load(
        include_nasa7=False,
        include_nasa9=False,
        include_afcesic=False,
        include_terra=True,
    )

    janaf_only = SpeciesDatabase(
        nasa7_path=str(_THERMO_DIR / "nasa7.json"),
        nasa9_path=str(_THERMO_DIR / "nasa9.json"),
        janaf_path=str(_THERMO_DIR / "janaf.csv"),
    )
    janaf_only.load(
        include_nasa7=False,
        include_nasa9=False,
        include_afcesic=False,
        include_terra=False,
        include_janaf=True,
    )

    mixed = SpeciesDatabase(
        nasa7_path=str(_THERMO_DIR / "nasa7.json"),
        nasa9_path=str(_THERMO_DIR / "nasa9.json"),
        janaf_path=str(_THERMO_DIR / "janaf.csv"),
        afcesic_path=str(_THERMO_DIR / "afcesic.json"),
        terra_path=str(_THERMO_DIR / "terra.json"),
    )
    mixed.load(
        include_nasa7=True,
        include_nasa9=True,
        include_afcesic=True,
        include_terra=True,
        include_janaf=True,
    )

    return {
        "NASA": nasa_only,
        "AFCESIC": afcesic_only,
        "TERRA": terra_only,
        "JANAF": janaf_only,
        "MIXED": mixed,
    }


# ---------------------------------------------------------------------------
# Propellant systems
# ---------------------------------------------------------------------------


@dataclass
class PropSystem:
    name: str
    fuel_formula: str
    ox_formula: str
    product_elements: set
    of_ratios: List[float]
    pressures_psia: List[float]
    fuel_sp: Optional[Species] = field(default=None, repr=False)
    ox_sp: Optional[Species] = field(default=None, repr=False)
    fuel_T_ref: float = 298.15
    ox_T_ref: float = 298.15
    cea: Optional[CEA_Obj] = field(default=None, repr=False)


def _t_ref(sp: Species) -> float:
    if isinstance(sp, NASANineCoeff):
        return max(sp.temperatures[0], 298.15)
    if isinstance(sp, AFCESICCoeff):
        return max(sp.T_low, 298.15)
    return 298.15


def _register_cea(
    tag: str, kind: str, pairs: List[Tuple[str, int]], t_ref: float, h_j: float
) -> None:
    elems = "  ".join(f"{s} {n}" for s, n in pairs)
    card = (
        f"{kind} {tag}  {elems}  wt%=100.  t(k)={t_ref:.3f}  h,cal={h_j/CAL_TO_J:.3f}"
    )
    (add_new_fuel if kind == "fuel" else add_new_oxidizer)(tag, card)


def _build_systems(db: SpeciesDatabase, db_name: str = "") -> List[PropSystem]:
    systems: List[PropSystem] = []
    _prefix = db_name.replace(" ", "_") + "_" if db_name else ""

    def _setup(formula, kind, pairs):
        sp = db.find(formula)
        t = _t_ref(sp)
        tag = f"_dbtest_{_prefix}{formula.replace('-','m').replace('+','p')}"
        h = sp.enthalpy(t)
        if not math.isfinite(h):
            raise ValueError(f"enthalpy({t} K) = {h} for {formula!r}")
        _register_cea(tag, kind, pairs, t, h)
        return sp, t, tag

    h2_sp, h2_t, h2_tag = _setup("H2", "fuel", [("H", 2)])
    o2_sp, o2_t, o2_tag = _setup("O2", "oxid", [("O", 2)])
    systems.append(
        PropSystem(
            name="H2/O2",
            fuel_formula="H2",
            ox_formula="O2",
            product_elements={"H", "O"},
            of_ratios=[2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0, 16.0, 20.0],
            pressures_psia=[50.0, 200.0, 500.0, 1000.0, 3000.0],
            fuel_sp=h2_sp,
            ox_sp=o2_sp,
            fuel_T_ref=h2_t,
            ox_T_ref=o2_t,
            cea=CEA_Obj(oxName=o2_tag, fuelName=h2_tag),
        )
    )

    try:
        ch4_sp, ch4_t, ch4_tag = _setup("CH4", "fuel", [("C", 1), ("H", 4)])
        systems.append(
            PropSystem(
                name="CH4/O2",
                fuel_formula="CH4",
                ox_formula="O2",
                product_elements={"C", "H", "O"},
                of_ratios=[1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0],
                pressures_psia=[50.0, 200.0, 500.0, 1000.0, 3000.0],
                fuel_sp=ch4_sp,
                ox_sp=o2_sp,
                fuel_T_ref=ch4_t,
                ox_T_ref=o2_t,
                cea=CEA_Obj(oxName=o2_tag, fuelName=ch4_tag),
            )
        )
    except Exception as e:
        print(f"[SKIP] CH4/O2: {e}")

    try:
        n2h4_sp, n2h4_t, n2h4_tag = _setup("N2H4", "fuel", [("N", 2), ("H", 4)])
        n2o4_sp, n2o4_t, n2o4_tag = _setup("N2O4", "oxid", [("N", 2), ("O", 4)])
        systems.append(
            PropSystem(
                name="N2H4/N2O4",
                fuel_formula="N2H4",
                ox_formula="N2O4",
                product_elements={"N", "H", "O"},
                of_ratios=[0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0],
                pressures_psia=[50.0, 200.0, 500.0, 1000.0, 3000.0],
                fuel_sp=n2h4_sp,
                ox_sp=n2o4_sp,
                fuel_T_ref=n2h4_t,
                ox_T_ref=n2o4_t,
                cea=CEA_Obj(oxName=n2o4_tag, fuelName=n2h4_tag),
            )
        )
    except Exception as e:
        print(f"[SKIP] N2H4/N2O4: {e}")

    try:
        nh3_sp, nh3_t, nh3_tag = _setup("NH3", "fuel", [("N", 1), ("H", 3)])
        systems.append(
            PropSystem(
                name="NH3/O2",
                fuel_formula="NH3",
                ox_formula="O2",
                product_elements={"N", "H", "O"},
                of_ratios=[1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0],
                pressures_psia=[50.0, 200.0, 500.0, 1000.0, 3000.0],
                fuel_sp=nh3_sp,
                ox_sp=o2_sp,
                fuel_T_ref=nh3_t,
                ox_T_ref=o2_t,
                cea=CEA_Obj(oxName=o2_tag, fuelName=nh3_tag),
            )
        )
    except Exception as e:
        print(f"[SKIP] NH3/O2: {e}")

    return systems


# ---------------------------------------------------------------------------
# CEA reference query
# ---------------------------------------------------------------------------


def _query_cea(sys: PropSystem, of: float, p_psia: float) -> dict:
    cea = sys.cea
    T_K = float(cea.get_Tcomb(Pc=p_psia, MR=of)) * RANKINE_TO_K
    mw_f, _ = cea.get_Chamber_MolWt_gamma(Pc=p_psia, MR=of)
    mw = float(mw_f)
    cp_fr = float(cea.get_Chamber_Cp(Pc=p_psia, MR=of, frozen=1)) * CAL_TO_J * mw
    gamma_fr = cp_fr / (cp_fr - R_UNIV)
    a_fr = math.sqrt(gamma_fr * R_UNIV * T_K / (mw / 1000))
    return {"T": T_K, "M": mw, "gamma_fr": gamma_fr, "cp_fr": cp_fr, "a_fr": a_fr}


# ---------------------------------------------------------------------------
# Prometheus solve
# ---------------------------------------------------------------------------

_PRODUCTS_CACHE: Dict[Tuple[str, frozenset], list] = {}
_MAX_ATOMS = 20


def _get_products(db_name: str, db: SpeciesDatabase, elements: set) -> list:
    key = (db_name, frozenset(elements))
    if key not in _PRODUCTS_CACHE:
        _PRODUCTS_CACHE[key] = db.get_species(elements, max_atoms=_MAX_ATOMS)
    return _PRODUCTS_CACHE[key]


def _solve(
    sys: PropSystem,
    of: float,
    p_pa: float,
    db: SpeciesDatabase,
    db_name: str,
    solver: GordonMcBrideSolver,
) -> dict:
    M_f, M_o = sys.fuel_sp.molar_mass(), sys.ox_sp.molar_mass()
    n_f = 1.0
    n_o = of * n_f * M_f / M_o
    H0 = (
        sys.fuel_sp.enthalpy(sys.fuel_T_ref) * n_f
        + sys.ox_sp.enthalpy(sys.ox_T_ref) * n_o
    )
    products = _get_products(db_name, db, sys.product_elements)
    prob = EquilibriumProblem(
        reactants={sys.fuel_sp: n_f, sys.ox_sp: n_o},
        products=products,
        problem_type=ProblemType.HP,
        constraint1=H0,
        constraint2=p_pa,
        t_init=3500.0,
    )
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        t0 = time.perf_counter()
        sol = solver.solve(prob)
        dt = time.perf_counter() - t0
    return {
        "T": sol.temperature,
        "M": sol.gas_mean_molar_mass * 1000,
        "gamma": sol.gamma,
        "cp": sol.cp,
        "a": sol.speed_of_sound,
        "converged": sol.converged,
        "iters": sol.iterations,
        "time": dt,
    }


def _pct(val: float, ref: float) -> float:
    if not math.isfinite(val) or not math.isfinite(ref) or abs(ref) < 1e-12:
        return float("nan")
    return (val - ref) / ref * 100.0


# ---------------------------------------------------------------------------
# Species-source breakdown
# ---------------------------------------------------------------------------


def _print_source_breakdown(databases: Dict[str, SpeciesDatabase]) -> None:
    """For each DB config, show how many product species (H/O/C sets) come from each source."""
    element_sets = [
        ("H/O (H2+O2 products)", {"H", "O"}),
        ("C/H/O (CH4+O2 products)", {"C", "H", "O"}),
        ("N/H/O (N2H4/NH3 products)", {"N", "H", "O"}),
    ]
    print(f"\n{'='*70}")
    print("  PRODUCT SPECIES SOURCE BREAKDOWN (max_atoms=20)")
    print(f"{'='*70}")
    for db_name, db in databases.items():
        print(f"\n  [{db_name}]")
        for set_label, elems in element_sets:
            products = db.get_species(elems, max_atoms=_MAX_ATOMS)
            by_src: Dict[str, int] = defaultdict(int)
            for sp in products:
                # Use attribution if present, otherwise class name mapping
                attr = getattr(sp, "source_attribution", None)
                if attr:
                    by_src[attr] += 1
                else:
                    by_src[_SOURCE_LABEL.get(type(sp).__name__, type(sp).__name__)] += 1
            total = len(products)
            src_str = ", ".join(f"{s}: {n}" for s, n in sorted(by_src.items()))
            print(f"    {set_label:<35} {total:>4} species  [{src_str}]")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run() -> None:
    output_txt = "benchmark_db_output.txt"

    class _Tee:
        def __init__(self, *s):
            self._s = s

        def write(self, d):
            for s in self._s:
                s.write(d)
            return len(d)

        def flush(self):
            for s in self._s:
                s.flush()

        @property
        def encoding(self):
            return getattr(self._s[0], "encoding", "utf-8")

        @property
        def errors(self):
            return getattr(self._s[0], "errors", "replace")

    _f = open(output_txt, "w", encoding="utf-8")
    _orig = sys.stdout
    sys.stdout = _Tee(_orig, _f)
    logger.remove()

    try:
        _run_inner()
    finally:
        sys.stdout = _orig
        _f.close()
        print(f"\nOutput written to: {output_txt}")


def _print_janaf_coverage(databases: Dict[str, SpeciesDatabase]) -> None:
    """Identify key combustion product species and their source/enthalpy per DB.

    Also reports the JANAF-vs-NASA-9 enthalpy offset for OH and H2O at 298.15 K
    and 3000 K — the primary sources of JANAF temperature inaccuracy:

    * OH  -- JANAF 3rd-edition ΔHf°(298.15 K) = 38.99 kJ/mol vs NASA-9 37.28 kJ/mol
             (1.71 kJ/mol offset; drives ~2% temperature over-prediction in fuel-lean flames)
    * H2O -- JANAF Cp slightly lower than NASA-9 above 1500 K, causing a growing
             enthalpy offset (~1.7 kJ/mol at 3500 K) that raises equilibrium temperature
    * Before the REF-phase fix, H2 / O2 / N2 were stored as H2_REF / O2_REF / N2_REF
      in the JANAF CSV and silently excluded by _load_janaf, making JANAF-only solves
      fail entirely for any H/O or N-containing system.
    """
    key_species = [
        "H2", "O2", "H2O", "OH", "H", "O",
        "HO2", "H2O2",
        "CO2", "CO", "CH4", "C",
        "N2", "NO", "N2O", "NO2", "NH3", "N",
    ]
    print(f"\n{'='*80}")
    print("  KEY SPECIES COVERAGE CHECK")
    print(f"{'='*80}")
    print(f"  {'Species':<10}" + "".join(f"  {n:>8}" for n in databases))
    print(f"  {'-'*70}")
    for formula in key_species:
        row = f"  {formula:<10}"
        for db_name, db in databases.items():
            try:
                sp = db.find(formula)
                src = getattr(sp, "source_attribution", type(sp).__name__)
                row += f"  {src:>8}"
            except (KeyError, RuntimeError):
                row += f"  {'MISSING':>8}"
        print(row)

    # Quantify the OH and H2O enthalpy offsets between JANAF and NASA-9
    nasa_db = databases.get("NASA")
    janaf_db = databases.get("JANAF")
    if nasa_db is None or janaf_db is None:
        return
    print(f"\n  JANAF vs NASA-9 enthalpy offsets for key species [J/mol]:")
    print(f"  {'Species':<10}  {'T [K]':>8}  {'NASA-9':>12}  {'JANAF':>12}  {'diff':>10}")
    print(f"  {'-'*60}")
    for formula, temps in [("OH", [298.15, 1000.0, 3000.0]), ("H2O", [298.15, 1000.0, 3000.0])]:
        try:
            sp_n = nasa_db.find(formula)
            sp_j = janaf_db.find(formula)
        except Exception:
            continue
        for T in temps:
            h_n = sp_n.enthalpy(T)
            h_j = sp_j.enthalpy(T)
            print(f"  {formula:<10}  {T:>8.2f}  {h_n:>12.1f}  {h_j:>12.1f}  {h_j-h_n:>10.1f}")


def _run_inner() -> None:
    print("Building databases...")
    databases = _build_databases()
    for name, db in databases.items():
        print(f"  {name}: {len(db.species)} species")

    _print_source_breakdown(databases)
    _print_janaf_coverage(databases)

    # Each DB configuration needs its own solver and propellant system
    # (systems depend on which DB is active for reactant lookup)
    DB_NAMES = list(databases.keys())
    solver = GordonMcBrideSolver()

    # Build systems once per DB (reactants looked up from that DB)
    print("\nBuilding propellant systems per database...")
    db_systems: Dict[str, List[PropSystem]] = {}
    for db_name, db in databases.items():
        try:
            db_systems[db_name] = _build_systems(db, db_name)
            print(f"  {db_name}: {len(db_systems[db_name])} propellant systems")
        except Exception as e:
            print(f"  {db_name}: FAILED — {e}")
            db_systems[db_name] = []

    # Use NASA systems as the template for propellant names/conditions
    # (all DBs should produce the same set of systems)
    template_systems = db_systems.get("NASA", [])
    if not template_systems:
        print("ERROR: Could not build NASA propellant systems.")
        return

    # Accumulators
    errs: Dict[str, Dict[str, List[float]]] = {n: defaultdict(list) for n in DB_NAMES}
    n_conv: Dict[str, int] = {n: 0 for n in DB_NAMES}
    n_fail: Dict[str, int] = {n: 0 for n in DB_NAMES}
    timing: Dict[str, List[float]] = {n: [] for n in DB_NAMES}
    total_cases = 0

    for sys_idx, template_sys in enumerate(template_systems):
        n_cases = len(template_sys.of_ratios) * len(template_sys.pressures_psia)
        total_cases += n_cases

        print(f"\n{'='*88}")
        print(f"  {template_sys.name}  —  {n_cases} cases")
        print(f"{'='*88}")
        hdr_dbs = "".join(f"  {n:>10} {'err%':>7}" for n in DB_NAMES)
        print(f"  {'O/F':>5}  {'P':>6}  {'CEA T [K]':>10}{hdr_dbs}")
        print(f"  {'-'*80}")

        for p_psia in template_sys.pressures_psia:
            p_pa = p_psia * PSIA_TO_PA
            for of in template_sys.of_ratios:
                label = f"O/F={of:.1f} P={p_psia:.0f}psia {template_sys.name}"

                try:
                    cea = _query_cea(template_sys, of, p_psia)
                except Exception as e:
                    print(f"  {of:>5.1f}  {p_psia:>6.0f}  CEA ERROR: {e}")
                    continue

                results: Dict[str, dict] = {}
                for db_name, db in databases.items():
                    sys_list = db_systems.get(db_name, [])
                    if sys_idx >= len(sys_list):
                        results[db_name] = {
                            "converged": False,
                            "T": float("nan"),
                            "M": float("nan"),
                            "gamma": float("nan"),
                            "cp": float("nan"),
                            "a": float("nan"),
                            "time": 0.0,
                        }
                        continue
                    this_sys = sys_list[sys_idx]
                    try:
                        results[db_name] = _solve(
                            this_sys, of, p_pa, db, db_name, solver
                        )
                    except Exception as e:
                        results[db_name] = {
                            "converged": False,
                            "T": float("nan"),
                            "M": float("nan"),
                            "gamma": float("nan"),
                            "cp": float("nan"),
                            "a": float("nan"),
                            "time": 0.0,
                            "error": str(e),
                        }

                # Tally and print temperature row
                cells = ""
                for db_name in DB_NAMES:
                    r = results[db_name]
                    ok = r.get("converged") and math.isfinite(r["T"])
                    if ok:
                        n_conv[db_name] += 1
                        timing[db_name].append(r["time"])
                        e_pct = _pct(r["T"], cea["T"])
                        errs[db_name]["T"].append(e_pct)
                        cells += f"  {r['T']:>10.1f} {e_pct:>6.2f}%"
                    else:
                        n_fail[db_name] += 1
                        cells += f"  {'FAIL':>10} {'---':>7}"

                print(f"  {of:>5.1f}  {p_psia:>6.0f}  {cea['T']:>10.1f}{cells}")

    # Summary
    print(f"\n\n{'='*88}")
    print(
        f"  DATABASE COMPARISON SUMMARY  —  {total_cases} cases  x  {len(DB_NAMES)} configs"
    )
    print(f"{'='*88}")
    print(
        f"\n  {'Config':<10}  {'OK':>6}  {'FAIL':>6}  {'OK%':>7}  {'Mean T err%':>12}  "
        f"{'90th T err%':>12}  {'Max T err%':>12}  {'Mean t [s]':>11}"
    )
    print(f"  {'-'*86}")
    for db_name in DB_NAMES:
        nc = n_conv[db_name]
        nf = n_fail[db_name]
        nt = nc + nf
        ok_pct = 100 * nc / max(nt, 1)
        t_errs = [abs(e) for e in errs[db_name]["T"] if math.isfinite(e)]
        mean_err = float(np.mean(t_errs)) if t_errs else float("nan")
        p90_err = float(np.percentile(t_errs, 90)) if t_errs else float("nan")
        max_err = float(np.max(t_errs)) if t_errs else float("nan")
        mean_t = float(np.mean(timing[db_name])) if timing[db_name] else float("nan")
        print(
            f"  {db_name:<10}  {nc:>6}  {nf:>6}  {ok_pct:>6.1f}%  "
            f"{mean_err:>11.4f}%  {p90_err:>11.4f}%  {max_err:>11.4f}%  {mean_t:>10.4f}s"
        )

    # Per-propellant T error breakdown
    print(f"\n  T error distribution per propellant system (|err%|, converged cases):")
    bins = [0, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, float("inf")]
    bin_labels = [
        "<0.05%",
        "0.05-0.1%",
        "0.1-0.25%",
        "0.25-0.5%",
        "0.5-1%",
        "1-2%",
        "2-5%",
        ">5%",
    ]
    for db_name in DB_NAMES:
        t_errs = [abs(e) for e in errs[db_name]["T"] if math.isfinite(e)]
        counts = [0] * len(bin_labels)
        for e in t_errs:
            for k, (lo, hi) in enumerate(zip(bins, bins[1:])):
                if lo <= e < hi:
                    counts[k] += 1
                    break
        print(f"\n    {db_name} (n={len(t_errs)}):")
        for lbl, cnt in zip(bin_labels, counts):
            bar = "#" * int(cnt * 40 / max(len(t_errs), 1))
            print(f"      {lbl:>12}  {cnt:>4}  {bar}")

    print(f"\n{'='*88}\n")


if __name__ == "__main__":
    run()
