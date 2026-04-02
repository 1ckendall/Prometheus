"""
benchmark.py — Comprehensive Prometheus vs RocketCEA solver comparison.

Propellant systems tested:
  1. H2  / O2    — gaseous hydrogen/oxygen (baseline, wide O/F sweep)
  2. CH4 / O2    — methane/oxygen (Raptor-type)
  3. N2H4 / N2O4 — hydrazine / nitrogen tetroxide (hypergolic)
  4. NH3 / O2    — ammonia/oxygen

Conditions swept per system:
  - O/F mass ratio: system-appropriate range including lean, stoich, rich
  - Chamber pressure: 50, 200, 500, 1000, 3000 psia

Comparison metrics (all vs RocketCEA reference):
  - Adiabatic flame temperature T_c  [K]
  - Gas mean molar mass M̄           [g/mol]
  - Frozen gamma gamma_fr
  - Frozen Cp                        [J/(mol·K)]
  - Frozen speed of sound a_fr       [m/s]

GordonMcBrideSolver, MajorSpeciesSolver, and PEPSolver are run for each case.
Non-convergent cases are flagged and excluded from error statistics.

Run with (from repo root):
    uv run python tests/benchmark.py
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

# ---------------------------------------------------------------------------
# Tee: mirror sys.stdout to both console and a text file
# ---------------------------------------------------------------------------


class _Tee:
    """Write to multiple streams simultaneously.  Used to capture print()
    output to a file while still showing it on the terminal."""

    def __init__(self, *streams):
        self._streams = streams

    def write(self, data: str) -> int:
        for s in self._streams:
            s.write(data)
        return len(data)

    def flush(self) -> None:
        for s in self._streams:
            s.flush()

    # Proxy attributes that some libraries check (e.g. click, loguru)
    @property
    def encoding(self):
        return getattr(self._streams[0], "encoding", "utf-8")

    @property
    def errors(self):
        return getattr(self._streams[0], "errors", "replace")


from prometheus.equilibrium.problem import EquilibriumProblem, ProblemType
from prometheus.equilibrium.solver import (
    GordonMcBrideSolver,
    MajorSpeciesSolver,
    PEPSolver,
)
from prometheus.equilibrium.species import NASANineCoeff, Species, SpeciesDatabase

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

R_UNIV = 8.314462  # J/(mol·K)
CAL_TO_J = 4.184
PSIA_TO_PA = 6894.757
FPS_TO_MPS = 0.3048
RANKINE_TO_K = 1.0 / 1.8

# ---------------------------------------------------------------------------
# Database + solvers (module-level singletons)
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent.parent
_THERMO_DIR = _REPO_ROOT / "prometheus" / "thermo_data"

_DB = SpeciesDatabase(
    nasa7_path=str(_THERMO_DIR / "nasa7.json"),
    nasa9_path=str(_THERMO_DIR / "nasa9.json"),
    afcesic_path=str(_THERMO_DIR / "afcesic.json"),
    terra_path=str(_THERMO_DIR / "terra.json"),
    janaf_path=str(_THERMO_DIR / "janaf.csv"),
)
_DB.load(include_nasa7=True,
         include_nasa9=False,
         include_janaf=False,
         include_afcesic=True,
         include_terra = True)

_GMCB = GordonMcBrideSolver()
_MAJOR_SPECIES = MajorSpeciesSolver()
_PEP = PEPSolver()

SOLVER_NAMES = ["G-McB", "Major-Species", "PEP"]
SOLVERS = [_GMCB, _MAJOR_SPECIES, _PEP]


# ---------------------------------------------------------------------------
# Propellant system definition
# ---------------------------------------------------------------------------


@dataclass
class PropSystem:
    """One propellant combination with its test conditions."""

    name: str
    fuel_formula: str
    ox_formula: str
    product_elements: set
    of_ratios: List[float]
    pressures_psia: List[float]
    # Filled by _build_systems():
    fuel_sp: Optional[Species] = field(default=None, repr=False)
    ox_sp: Optional[Species] = field(default=None, repr=False)
    fuel_T_ref: float = 298.15  # K — may be > 298.15 if T_low > 298.15
    ox_T_ref: float = 298.15
    cea: Optional[CEA_Obj] = field(default=None, repr=False)


def _t_ref(sp: Species) -> float:
    """Lowest valid temperature for the species polynomial [K]."""
    if isinstance(sp, NASANineCoeff):
        t_lo = sp.temperatures[0]
        return max(t_lo, 298.15)
    return 298.15


def _register_cea_species(
    tag: str,
    kind: str,
    formula_pairs: List[Tuple[str, int]],
    t_ref: float,
    h_j_per_mol: float,
) -> None:
    """Register a custom fuel or oxidiser with RocketCEA."""
    elems = "  ".join(f"{sym} {n}" for sym, n in formula_pairs)
    h_cal = h_j_per_mol / CAL_TO_J  # J/mol → cal/mol
    card = f"{kind} {tag}  {elems}  wt%=100.  t(k)={t_ref:.3f}  h,cal={h_cal:.3f}"
    if kind == "fuel":
        add_new_fuel(tag, card)
    else:
        add_new_oxidizer(tag, card)


def _build_systems() -> List[PropSystem]:
    """Construct and return all propellant systems for the test."""
    systems: List[PropSystem] = []

    def _setup(
        formula: str, kind: str, formula_pairs: List[Tuple[str, int]]
    ) -> Tuple[Species, float, str]:
        sp = _DB.find(formula)
        t = _t_ref(sp)
        tag = f"_pep_{formula.replace('-','m').replace('+','p')}"
        h = sp.enthalpy(t)
        if not math.isfinite(h):
            raise ValueError(
                f"enthalpy({t} K) = {h} for {formula!r}; cannot set up CEA reactant."
            )
        _register_cea_species(tag, kind, formula_pairs, t, h)
        return sp, t, tag

    # H2 / O2
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

    # CH4 / O2
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

    # N2H4 / N2O4
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
    except (KeyError, ValueError) as e:
        print(f"[SKIP] N2H4/N2O4: {e}")

    # NH3 / O2
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
    except (KeyError, ValueError) as e:
        print(f"[SKIP] NH3/O2: {e}")

    return systems


# ---------------------------------------------------------------------------
# CEA query
# ---------------------------------------------------------------------------


def _query_cea(sys: PropSystem, of: float, p_psia: float) -> dict:
    """Return chamber properties from RocketCEA for one (O/F, P) case."""
    cea = sys.cea
    Tc_R = cea.get_Tcomb(Pc=p_psia, MR=of)
    T_K = float(Tc_R) * RANKINE_TO_K

    mw_f, gamma_eq = cea.get_Chamber_MolWt_gamma(Pc=p_psia, MR=of)
    cp_eq_cal = cea.get_Chamber_Cp(Pc=p_psia, MR=of)
    cp_fr_cal = cea.get_Chamber_Cp(Pc=p_psia, MR=of, frozen=1)
    a_eq_fps = cea.get_Chamber_SonicVel(Pc=p_psia, MR=of)

    mw = float(mw_f)
    cp_eq = float(cp_eq_cal) * CAL_TO_J * mw  # J/(mol·K)
    cp_fr = float(cp_fr_cal) * CAL_TO_J * mw

    gamma_fr = cp_fr / (cp_fr - R_UNIV)
    a_fr = math.sqrt(gamma_fr * R_UNIV * T_K / (mw / 1000))

    return {
        "T": T_K,
        "M": mw,
        "gamma_eq": float(gamma_eq),
        "gamma_fr": gamma_fr,
        "cp_eq": cp_eq,
        "cp_fr": cp_fr,
        "a_eq": float(a_eq_fps) * FPS_TO_MPS,
        "a_fr": a_fr,
    }


# ---------------------------------------------------------------------------
# Prometheus solve
# ---------------------------------------------------------------------------

_PRODUCTS_CACHE: Dict[frozenset, list] = {}
_MAX_PRODUCT_ATOMS = 20


def _get_products(element_set: set) -> list:
    key = frozenset(element_set)
    if key not in _PRODUCTS_CACHE:
        _PRODUCTS_CACHE[key] = _DB.get_species(
            element_set, max_atoms=_MAX_PRODUCT_ATOMS
        )
    return _PRODUCTS_CACHE[key]


def _solve_pep(sys: PropSystem, of: float, p_pa: float, solver) -> dict:
    """Run one HP equilibrium solve with Prometheus.  Returns result dict."""
    fuel = sys.fuel_sp
    ox = sys.ox_sp
    M_f = fuel.molar_mass()
    M_o = ox.molar_mass()
    n_f = 1.0
    n_o = of * n_f * M_f / M_o

    H0 = fuel.enthalpy(sys.fuel_T_ref) * n_f + ox.enthalpy(sys.ox_T_ref) * n_o

    products = _get_products(sys.product_elements)
    prob = EquilibriumProblem(
        reactants={fuel: n_f, ox: n_o},
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pct(val: float, ref: float) -> float:
    if not math.isfinite(val) or not math.isfinite(ref) or abs(ref) < 1e-12:
        return float("nan")
    return (val - ref) / ref * 100.0


# ---------------------------------------------------------------------------
# Main test loop
# ---------------------------------------------------------------------------


def run() -> None:
    # ------------------------------------------------------------------
    # Output setup: tee print() to file + configure solver debug logs
    # ------------------------------------------------------------------
    output_txt = "benchmark_output.txt"
    debug_log = "benchmark_debug.log"

    _out_file = open(output_txt, "w", encoding="utf-8")
    _orig_stdout = sys.stdout
    sys.stdout = _Tee(_orig_stdout, _out_file)

    # Remove loguru's default stderr sink so solver logs don't pollute
    # the terminal.  Write all DEBUG+ messages from prometheus solvers to
    # a dedicated file; format includes timestamp, level, solver name.
    logger.remove()
    _log_id = logger.add(
        debug_log,
        level="DEBUG",
        mode="w",
        format="{time:HH:mm:ss.SSS} | {level:<8} | {extra[solver]:<14} | {message}",
        filter=lambda r: "solver" in r["extra"],
        encoding="utf-8",
    )

    try:
        _run_inner()
    finally:
        sys.stdout = _orig_stdout
        _out_file.close()
        logger.remove(_log_id)
        print(f"\nOutput written to: {output_txt}")
        print(f"Solver debug log:  {debug_log}")


def _run_inner() -> None:
    systems = _build_systems()

    errs: Dict[str, Dict[str, List[float]]] = {
        s: defaultdict(list) for s in SOLVER_NAMES
    }
    n_conv: Dict[str, int] = {s: 0 for s in SOLVER_NAMES}
    n_maxiter: Dict[str, int] = {s: 0 for s in SOLVER_NAMES}
    n_fail: Dict[str, int] = {s: 0 for s in SOLVER_NAMES}
    worst: Dict[str, Dict[str, Tuple[float, str]]] = {s: {} for s in SOLVER_NAMES}
    total_cases = 0
    timing: Dict[str, List[float]] = {s: [] for s in SOLVER_NAMES}

    QUANTITIES = [
        ("T", "T_c [K]", lambda r: r["T"], lambda c: c["T"]),
        ("M", "M [g/mol]", lambda r: r["M"], lambda c: c["M"]),
        ("gamma_fr", "gamma_fr", lambda r: r["gamma"], lambda c: c["gamma_fr"]),
        ("cp_fr", "Cp_fr [J/(mol·K)]", lambda r: r["cp"], lambda c: c["cp_fr"]),
        ("a_fr", "a_fr [m/s]", lambda r: r["a"], lambda c: c["a_fr"]),
    ]

    for sys in systems:
        n_cases = len(sys.of_ratios) * len(sys.pressures_psia)
        total_cases += n_cases

        print(f"\n{'='*88}")
        print(
            f"  {sys.name}  —  {n_cases} cases  "
            f"(fuel T_ref={sys.fuel_T_ref:.2f} K, ox T_ref={sys.ox_T_ref:.2f} K)"
        )
        print(f"  O/F: {sys.of_ratios}")
        print(f"  P (psia): {sys.pressures_psia}")
        print(f"{'='*88}")

        solver_cols = "".join(f"  {s:>11} {'err%':>7}" for s in SOLVER_NAMES)
        hdr = f"  {'O/F':>5}  {'P':>6}  {'Qty':<16}  {'CEA':>11}{solver_cols}"
        print(hdr)
        print(f"  {'-'*len(hdr.rstrip())}")

        for p_psia in sys.pressures_psia:
            p_pa = p_psia * PSIA_TO_PA
            for of in sys.of_ratios:
                label = f"O/F={of:.1f} P={p_psia:.0f}psia {sys.name}"

                try:
                    cea = _query_cea(sys, of, p_psia)
                except Exception as e:
                    print(f"  {of:>5.1f}  {p_psia:>6.0f}  CEA ERROR: {e}")
                    continue

                pep: Dict[str, dict] = {}
                for sname, solver in zip(SOLVER_NAMES, SOLVERS):
                    try:
                        pep[sname] = _solve_pep(sys, of, p_pa, solver)
                    except Exception as e:
                        pep[sname] = {
                            "converged": False,
                            "error": str(e),
                            "T": float("nan"),
                            "M": float("nan"),
                            "gamma": float("nan"),
                            "cp": float("nan"),
                            "a": float("nan"),
                            "iters": 0,
                            "time": 0.0,
                        }

                def _status(sname: str) -> str:
                    r = pep[sname]
                    if r.get("error"):
                        return "FAIL"
                    if r.get("converged"):
                        return "OK"
                    if math.isfinite(r["T"]):
                        return "MAX"
                    return "FAIL"

                for sname in SOLVER_NAMES:
                    st = _status(sname)
                    r = pep[sname]
                    if st == "OK":
                        n_conv[sname] += 1
                        timing[sname].append(r["time"])
                    elif st == "MAX":
                        n_maxiter[sname] += 1
                        timing[sname].append(r["time"])
                    else:
                        n_fail[sname] += 1

                first = True
                for qty_key, qty_label, get_pep, get_cea in QUANTITIES:
                    cea_val = get_cea(cea)

                    solver_cells = ""
                    for sname in SOLVER_NAMES:
                        st = _status(sname)
                        val = get_pep(pep[sname])
                        ok = st in ("OK", "MAX") and math.isfinite(val)
                        ann = "*" if st == "MAX" else " "
                        v_str = f"{val:>11.4f}{ann}" if ok else f"{'FAIL':>11} "
                        e = _pct(val, cea_val)
                        e_str = (
                            f"{e:>7.3f}%" if ok and math.isfinite(e) else f"{'---':>8}"
                        )
                        solver_cells += f"  {v_str} {e_str}"

                    of_str = f"{of:>5.1f}" if first else f"{'':>5}"
                    p_str = f"{p_psia:>6.0f}" if first else f"{'':>6}"
                    first = False

                    print(
                        f"  {of_str}  {p_str}  {qty_label:<16}  {cea_val:>11.4f}{solver_cells}"
                    )

                for sname in SOLVER_NAMES:
                    if _status(sname) not in ("OK", "MAX"):
                        continue
                    r = pep[sname]
                    for qty_key, _, gpep, gcea in QUANTITIES:
                        pv = gpep(r)
                        cv = gcea(cea)
                        e = _pct(pv, cv)
                        if math.isfinite(e):
                            errs[sname][qty_key].append(e)
                            cur_worst = worst[sname].get(qty_key, (0.0, ""))
                            if abs(e) > abs(cur_worst[0]):
                                worst[sname][qty_key] = (e, label)
                print()

    # ---------------------------------------------------------------------------
    # Summary tables
    # ---------------------------------------------------------------------------
    print(f"\n\n{'='*88}")
    print(
        f"  TORTURE TEST SUMMARY  —  {total_cases} cases  ×  {len(SOLVER_NAMES)} solvers"
    )
    print(f"{'='*88}")

    print(
        f"\n  Convergence  (* = hit max_iterations but T finite, included in error stats):"
    )
    print(
        f"  {'Solver':<12}  {'OK':>6}  {'MAX*':>6}  {'FAIL':>6}  {'Total':>6}  {'OK rate':>8}  {'Mean time':>10}"
    )
    print(f"  {'-'*66}")
    for sname in SOLVER_NAMES:
        nc, nm, nf = n_conv[sname], n_maxiter[sname], n_fail[sname]
        nt = nc + nm + nf
        mean_t = np.mean(timing[sname]) if timing[sname] else float("nan")
        print(
            f"  {sname:<12}  {nc:>6}  {nm:>6}  {nf:>6}  {nt:>6}  {100*nc/nt:>7.1f}%  {mean_t:>9.4f}s"
        )

    print(f"\n  Error statistics vs RocketCEA (OK + MAX* cases, frozen properties):")
    qty_labels = {
        "T": "T_c [K]",
        "M": "M [g/mol]",
        "gamma_fr": "gamma_fr",
        "cp_fr": "Cp_fr",
        "a_fr": "a_fr [m/s]",
    }
    print(
        f"\n  {'Quantity':<16}  {'Solver':<12}  {'n':>5}  "
        f"{'Mean |err|':>12}  {'Max |err|':>12}  {'Worst case':<}"
    )
    print(f"  {'-'*90}")
    for qty_key in ["T", "M", "gamma_fr", "cp_fr", "a_fr"]:
        for sname in SOLVER_NAMES:
            ev = errs[sname][qty_key]
            if not ev:
                continue
            ae = np.abs(ev)
            print(
                f"  {qty_labels[qty_key]:<16}  {sname:<12}  {len(ev):>5}  "
                f"{float(np.mean(ae)):>11.4f}%  {float(np.max(ae)):>11.4f}%  {worst[sname].get(qty_key, ('',''))[1]}"
            )

    # ---------------------------------------------------------------------------
    # Breakdown of Temperature Error Distribution by Solver
    # ---------------------------------------------------------------------------
    print(f"\n  Temperature error distribution breakdown (|err| %, OK + MAX* cases):")
    bins = [0, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, float("inf")]
    labels = [
        "<0.05%",
        "0.05-0.1%",
        "0.1-0.25%",
        "0.25-0.5%",
        "0.5-1%",
        "1-2%",
        "2-5%",
        ">5%",
    ]

    for sname in SOLVER_NAMES:
        s_errs = [abs(e) for e in errs[sname]["T"]]
        counts = [0] * len(labels)
        for e in s_errs:
            for k, (lo, hi) in enumerate(zip(bins, bins[1:])):
                if lo <= e < hi:
                    counts[k] += 1
                    break

        print(f"\n    {sname} solver (n={len(s_errs)}):")
        for lbl, cnt in zip(labels, counts):
            bar = "#" * int(cnt * 40 / max(len(s_errs), 1)) if len(s_errs) > 0 else ""
            print(f"      {lbl:>12}  {cnt:>4}  {bar}")

    print(f"\n{'='*88}\n")


if __name__ == "__main__":
    run()
