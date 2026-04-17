"""Diagnostic runner for Bismuth APCP optimisation config.

Runs with n_workers=1 (sequential) so exceptions surface cleanly.
Prints per-start traces, final compositions, SLSQP status, and failure reasons.
"""

from __future__ import annotations

import json
import math
import sys
import traceback
from pathlib import Path

# Force UTF-8 output on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from loguru import logger as _loguru_logger

_loguru_logger.disable("prometheus_equilibrium")

_PKG = Path(__file__).resolve().parent / "prometheus_equilibrium"
_THERMO = _PKG / "thermo_data"
_PROP_DB = _PKG / "propellants" / "propellants.toml"
_CONFIG = Path(__file__).resolve().parent / "Bismuth_APCP_Optimisation.prop-opt.json"


# ---------------------------------------------------------------------------
# Monkey-patch _run_single_start to capture SLSQP result status
# ---------------------------------------------------------------------------
import prometheus_equilibrium.optimization.gradient_engine as _ge

_orig_run = _ge._run_single_start


def _patched_run(
    x0,
    problem,
    objective,
    operating_point,
    enabled_databases,
    max_atoms,
    max_iter,
    fd_step,
    prop_db,
    spec_db,
    perf_solver,
    start_idx=None,
    per_iter_callback=None,
):
    from scipy.optimize import Bounds, minimize

    from prometheus_equilibrium.optimization._eval import evaluate_composition
    from prometheus_equilibrium.optimization.gradient_engine import (
        _PENALTY,
        _TRACE_FEASIBILITY_TOL,
        _build_scipy_bounds,
        _build_scipy_constraints,
        _constraint_diagnostics,
    )

    var_ids = [v.ingredient_id for v in problem.variables]
    bounds = _build_scipy_bounds(problem)
    constraints = _build_scipy_constraints(problem)

    def _obj(x):
        composition = {vid: float(x[i]) for i, vid in enumerate(var_ids)}
        try:
            log_fom, _, _, _ = evaluate_composition(
                composition,
                prop_db=prop_db,
                spec_db=spec_db,
                perf_solver=perf_solver,
                enabled_databases=enabled_databases,
                max_atoms=max_atoms,
                operating_point=operating_point,
                objective=objective,
            )
            return -log_fom
        except Exception as exc:
            return _PENALTY

    trace = []

    def _callback(xk):
        composition = {vid: float(xk[i]) for i, vid in enumerate(var_ids)}
        try:
            log_fom, _, _, _ = evaluate_composition(
                composition,
                prop_db=prop_db,
                spec_db=spec_db,
                perf_solver=perf_solver,
                enabled_databases=enabled_databases,
                max_atoms=max_atoms,
                operating_point=operating_point,
                objective=objective,
            )
            trace.append((len(trace), float(log_fom), dict(composition)))
            if per_iter_callback and start_idx is not None:
                per_iter_callback(start_idx, [(t, v) for t, v, _ in trace])
        except Exception:
            trace.append((len(trace), None, None))

    # Evaluate x0
    x0_comp = {vid: float(x0[i]) for i, vid in enumerate(var_ids)}
    try:
        lf0, _, isp0, rho0 = evaluate_composition(
            x0_comp,
            prop_db=prop_db,
            spec_db=spec_db,
            perf_solver=perf_solver,
            enabled_databases=enabled_databases,
            max_atoms=max_atoms,
            operating_point=operating_point,
            objective=objective,
        )
        x0_status = f"log_fom={lf0:.6f}, Isp={isp0:.2f}, rho={rho0:.1f}"
    except Exception as exc:
        x0_status = f"FAILED: {exc}"

    print(f"\n  Start {start_idx}: x0 eval -> {x0_status}", flush=True)
    comp_str = "  ".join(f"{k}={v*100:.2f}%" for k, v in sorted(x0_comp.items()))
    print(f"    x0 composition: {comp_str}", flush=True)

    res = minimize(
        _obj,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        callback=_callback,
        options={"maxiter": max_iter, "ftol": 1e-9, "eps": fd_step, "disp": False},
    )

    print(
        f"  Start {start_idx}: SLSQP status={res.status} ({res.message}), "
        f"nit={res.nit}, fun={res.fun:.6f}",
        flush=True,
    )

    if trace:
        vals = [v for _, v, _ in trace if v is not None]
        if vals:
            print(
                f"    trace: iter 0→{len(vals)-1}, "
                f"log_fom {vals[0]:.6f} → {vals[-1]:.6f}, "
                f"delta={vals[-1]-vals[0]:.2e}",
                flush=True,
            )
            # Show last few iters to spot stalling
            last = trace[-min(5, len(trace)) :]
            for it, v, comp in last:
                if v is not None and comp is not None:
                    cs = "  ".join(
                        f"{k}={cv*100:.2f}%" for k, cv in sorted(comp.items())
                    )
                    print(f"    iter {it}: log_fom={v:.8f}  {cs}", flush=True)

    x_opt = res.x
    comp_opt = {vid: float(x_opt[i]) for i, vid in enumerate(var_ids)}
    try:
        log_fom, fom, isp, density = evaluate_composition(
            comp_opt,
            prop_db=prop_db,
            spec_db=spec_db,
            perf_solver=perf_solver,
            enabled_databases=enabled_databases,
            max_atoms=max_atoms,
            operating_point=operating_point,
            objective=objective,
        )
        print(
            f"  Start {start_idx}: CONVERGED → log_fom={log_fom:.6f}, "
            f"Isp={isp:.2f}, rho={density:.1f}",
            flush=True,
        )
        opt_str = "  ".join(f"{k}={v*100:.2f}%" for k, v in sorted(comp_opt.items()))
        print(f"    opt composition: {opt_str}", flush=True)
        # Re-build trace/meta from scratch for return value
        result = _orig_run(
            x0,
            problem,
            objective,
            operating_point,
            enabled_databases,
            max_atoms,
            max_iter,
            fd_step,
            prop_db,
            spec_db,
            perf_solver,
            start_idx=start_idx,
            per_iter_callback=per_iter_callback,
        )
        return result
    except Exception as exc:
        print(f"  Start {start_idx}: FINAL EVAL FAILED → {exc}", flush=True)
        traceback.print_exc(file=sys.stdout)
        opt_str = "  ".join(f"{k}={v*100:.2f}%" for k, v in sorted(comp_opt.items()))
        print(f"    failed composition: {opt_str}", flush=True)
        return None


_ge._run_single_start = _patched_run


# ---------------------------------------------------------------------------
# Main diagnostic run
# ---------------------------------------------------------------------------


def main() -> None:
    from prometheus_equilibrium.equilibrium.solver import HybridSolver
    from prometheus_equilibrium.equilibrium.species import SpeciesDatabase
    from prometheus_equilibrium.optimization.config import (
        load_gradient_config,
        load_objective,
        load_operating_point,
        load_problem,
        load_solver_settings,
    )
    from prometheus_equilibrium.optimization.gradient_engine import (
        MultiStartGradientOptimizer,
    )
    from prometheus_equilibrium.propellants import PropellantDatabase

    with open(_CONFIG, encoding="utf-8") as f:
        config = json.load(f)

    solver_type, enabled_databases, max_atoms = load_solver_settings(config)
    print(f"Databases: {enabled_databases}, max_atoms={max_atoms}")

    spec_db = SpeciesDatabase(
        nasa7_path=str(_THERMO / "nasa7.json"),
        nasa9_path=str(_THERMO / "nasa9.json"),
        janaf_path=str(_THERMO / "janaf.csv"),
        afcesic_path=str(_THERMO / "afcesic.json"),
        terra_path=str(_THERMO / "terra.json"),
    )
    spec_db.load(
        include_nasa7="NASA-7" in enabled_databases,
        include_nasa9="NASA-9" in enabled_databases,
        include_afcesic="AFCESIC" in enabled_databases,
        include_terra="TERRA" in enabled_databases,
        include_janaf="JANAF" in enabled_databases,
    )
    prop_db = PropellantDatabase(str(_PROP_DB), species_db=spec_db)
    prop_db.load()

    problem = load_problem(config)
    objective = load_objective(config)
    operating_point = load_operating_point(config)
    n_starts, max_iter, fd_step, _, seed = load_gradient_config(config)

    solver = HybridSolver(capture_history=False)
    optimizer = MultiStartGradientOptimizer(
        problem=problem,
        objective=objective,
        operating_point=operating_point,
        prop_db=prop_db,
        spec_db=spec_db,
        solver=solver,
        enabled_databases=enabled_databases,
        max_atoms=max_atoms,
    )

    print(
        f"\nRunning {n_starts} starts, max_iter={max_iter}, seed={seed}, n_workers=1\n"
    )

    result = optimizer.optimize(
        n_starts=n_starts,
        max_iter_per_start=max_iter,
        fd_step=fd_step,
        seed=seed,
        n_workers=1,  # sequential for clean output
    )

    print(f"\n{'='*60}")
    print(f"Completed: {result.completed_trials}, Failed: {result.pruned_trials}")
    print(f"Best Isp*rho^n: {result.best_objective:.4f}")
    print(f"Best Isp: {result.best_isp:.2f} s")
    print(f"Best density: {result.best_density:.1f} kg/m3")
    print("Best composition:")
    for k, v in sorted(result.best_composition.items()):
        print(f"  {k}: {v*100:.4f}%")


if __name__ == "__main__":
    main()
