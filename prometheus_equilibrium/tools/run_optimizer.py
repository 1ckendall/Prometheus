"""Headless optimizer runner.

Runs a Prometheus gradient optimizer from a saved config file without
launching the GUI.  Progress is printed to stderr; results are written as
JSON to stdout or to a file.

Usage::

    prometheus-optimize config.prop-opt.json
    prometheus-optimize config.prop-opt.json --output results.json
    prometheus-optimize config.prop-opt.json --thermo-dir /path/to/thermo --prop-db /path/to/propellants.toml
    prometheus-optimize config.prop-opt.json --n-starts 8 --max-iter 20 --seed 0   # CLI overrides
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_PKG = Path(__file__).resolve().parents[1]
_DEFAULT_THERMO_DIR = _PKG / "thermo_data"
_DEFAULT_PROP_DB = _PKG / "propellants" / "propellants.toml"

_SOLVER_LABELS = {
    "gmcb": "Gordon-McBride",
    "mss": "Major-Species",
    "hybrid": "Hybrid",
}


def _load_databases(thermo_dir: Path, prop_db_path: Path, enabled_databases: list[str]):
    """Load species and propellant databases.

    Args:
        thermo_dir: Directory containing thermo JSON files.
        prop_db_path: Path to propellants TOML file.
        enabled_databases: Database labels to activate.

    Returns:
        ``(spec_db, prop_db)`` tuple.
    """
    from prometheus_equilibrium.equilibrium.species import SpeciesDatabase
    from prometheus_equilibrium.propellants import PropellantDatabase

    spec_db = SpeciesDatabase(
        nasa7_path=str(thermo_dir / "nasa7.json"),
        nasa9_path=str(thermo_dir / "nasa9.json"),
        janaf_path=str(thermo_dir / "janaf.csv"),
        afcesic_path=str(thermo_dir / "afcesic.json"),
        terra_path=str(thermo_dir / "terra.json"),
    )
    spec_db.load(
        include_nasa7="NASA-7" in enabled_databases,
        include_nasa9="NASA-9" in enabled_databases,
        include_afcesic="AFCESIC" in enabled_databases,
        include_terra="TERRA" in enabled_databases,
        include_janaf="JANAF" in enabled_databases,
    )

    prop_db = PropellantDatabase(str(prop_db_path), species_db=spec_db)
    prop_db.load()

    return spec_db, prop_db


def _make_solver(solver_type: str):
    """Instantiate a solver by type key.

    Args:
        solver_type: One of ``"gmcb"``, ``"mss"``, ``"hybrid"``.

    Returns:
        Solver instance.
    """
    from prometheus_equilibrium.equilibrium.solver import (
        GordonMcBrideSolver,
        HybridSolver,
        MajorSpeciesSolver,
    )

    if solver_type == "hybrid":
        return HybridSolver(capture_history=False)
    if solver_type == "mss":
        return MajorSpeciesSolver(capture_history=False)
    return GordonMcBrideSolver(capture_history=False)


def run_from_config(
    config: dict,
    thermo_dir: Path,
    prop_db_path: Path,
    n_starts_override: int | None = None,
    max_iter_override: int | None = None,
    n_workers_override: int | None = None,
    seed_override: int | None = None,
) -> dict:
    """Execute an optimization run from a config dict.

    Args:
        config: Parsed config dict (schema version 1).
        thermo_dir: Directory containing thermo JSON files.
        prop_db_path: Path to propellants TOML file.
        n_starts_override: If set, overrides ``run.n_starts`` from config.
        max_iter_override: If set, overrides ``run.max_iter_per_start`` from config.
        n_workers_override: If set, overrides ``run.n_workers`` from config.
        seed_override: If set, overrides ``run.seed`` from config.

    Returns:
        Results dict with keys ``best_objective``, ``best_isp``, ``best_density``,
        ``best_composition``, ``completed_trials``, ``pruned_trials``, ``trial_history``.
    """
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

    problem = load_problem(config)
    objective = load_objective(config)
    operating_point = load_operating_point(config)
    n_starts, max_iter_per_start, fd_step, n_workers, seed = load_gradient_config(
        config
    )
    solver_type, enabled_databases, max_atoms = load_solver_settings(config)

    if n_starts_override is not None:
        n_starts = n_starts_override
    if max_iter_override is not None:
        max_iter_per_start = max_iter_override
    if n_workers_override is not None:
        n_workers = n_workers_override
    if seed_override is not None:
        seed = seed_override

    print(
        f"Loading databases: {', '.join(enabled_databases) or '(none)'}",
        file=sys.stderr,
    )
    spec_db, prop_db = _load_databases(thermo_dir, prop_db_path, enabled_databases)

    print(
        f"Solver: {_SOLVER_LABELS.get(solver_type, solver_type)} | "
        f"max_atoms={max_atoms} | "
        f"n_starts={n_starts} | max_iter={max_iter_per_start} | "
        f"n_workers={n_workers} | seed={seed}",
        file=sys.stderr,
    )

    solver = _make_solver(solver_type)
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

    def _progress(payload: dict) -> None:
        s = int(payload.get("start", 0)) + 1
        n = int(payload.get("n_starts", n_starts))
        best = payload.get("best_value")
        converged = payload.get("converged", False)
        if best is not None:
            print(
                f"  [start {s}/{n}] best log FoM = {best:.6f}",
                file=sys.stderr,
            )
        else:
            status = "converged" if converged else "failed"
            print(f"  [start {s}/{n}] {status}", file=sys.stderr)

    print("Running...", file=sys.stderr)
    result = optimizer.optimize(
        n_starts=n_starts,
        max_iter_per_start=max_iter_per_start,
        fd_step=fd_step,
        seed=seed,
        n_workers=n_workers,
        progress_callback=_progress,
    )

    return {
        "best_objective": result.best_objective,
        "best_isp": result.best_isp,
        "best_density": result.best_density,
        "best_composition": result.best_composition,
        "completed_trials": result.completed_trials,
        "pruned_trials": result.pruned_trials,
        "trial_history": result.trial_history,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="prometheus-optimize",
        description="Run a Prometheus gradient optimizer headlessly from a config file.",
    )
    parser.add_argument(
        "config", help="Path to optimizer config (.prop-opt.json) file."
    )
    parser.add_argument(
        "--output",
        "-o",
        metavar="PATH",
        help="Write results JSON to this file (default: print to stdout).",
    )
    parser.add_argument(
        "--thermo-dir",
        default=str(_DEFAULT_THERMO_DIR),
        dest="thermo_dir",
        metavar="DIR",
        help="Directory containing thermo database JSON files. "
        f"Default: {_DEFAULT_THERMO_DIR}",
    )
    parser.add_argument(
        "--prop-db",
        default=str(_DEFAULT_PROP_DB),
        dest="prop_db",
        metavar="FILE",
        help="Path to propellants TOML file. " f"Default: {_DEFAULT_PROP_DB}",
    )
    parser.add_argument(
        "--n-starts",
        type=int,
        default=None,
        dest="n_starts",
        metavar="N",
        help="Override the number of SLSQP starts from the config.",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=None,
        dest="max_iter",
        metavar="N",
        help="Override max iterations per start from the config.",
    )
    parser.add_argument(
        "--n-workers",
        type=int,
        default=None,
        dest="n_workers",
        metavar="N",
        help="Override the worker count (0 = auto).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        metavar="N",
        help="Override the random seed from the config.",
    )
    return parser


def main() -> None:
    """Entry point for ``prometheus-optimize``."""
    parser = _build_parser()
    args = parser.parse_args()

    if not str(args.config).lower().endswith(".prop-opt.json"):
        parser.error("config file must use the .prop-opt.json extension")

    with open(args.config, encoding="utf-8") as f:
        config = json.load(f)

    results = run_from_config(
        config,
        thermo_dir=Path(args.thermo_dir),
        prop_db_path=Path(args.prop_db),
        n_starts_override=args.n_starts,
        max_iter_override=args.max_iter,
        n_workers_override=args.n_workers,
        seed_override=args.seed,
    )

    output_json = json.dumps(results, indent=2)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(output_json)
            f.write("\n")
        print(f"Results written to {args.output}", file=sys.stderr)
    else:
        print(output_json)

    # Human-readable summary on stderr regardless of --output
    r = results
    print("\n--- Result ---", file=sys.stderr)
    print(
        f"Isp x rho^n: {r['best_objective']:.4f}   "
        f"Isp: {r['best_isp']:.2f} s   "
        f"rho: {r['best_density']:.1f} kg/m3",
        file=sys.stderr,
    )
    print(
        f"Starts: {r['completed_trials']} converged, {r['pruned_trials']} failed",
        file=sys.stderr,
    )
    print("Best composition:", file=sys.stderr)
    for ingredient_id, value in sorted(r["best_composition"].items()):
        print(f"  {ingredient_id}: {value:.6f}", file=sys.stderr)


if __name__ == "__main__":
    main()
