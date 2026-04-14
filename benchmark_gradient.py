"""Benchmark multi-start gradient optimizer configurations on optimisation_test_1.json.

Usage:
    uv run python benchmark_gradient.py

Compares sequential vs parallel execution and different start/iteration budgets.
Metrics reported per configuration x seed:
  - best_objective (Isp * rho^n)
  - best_isp
  - best_density (kg/m3)
  - wall time (s)
  - completed starts
"""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

from loguru import logger

logger.remove()  # suppress solver debug/info output during benchmark

_ROOT = Path(__file__).parent
_CONFIG_PATH = _ROOT / "optimisation_test_1.json"
_PKG = _ROOT / "prometheus_equilibrium"
_THERMO_DIR = _PKG / "thermo_data"
_PROP_DB = _PKG / "propellants" / "propellants.toml"

SEEDS = [42, 123, 456]

# Configurations to benchmark
CONFIGS = [
    {"label": "4-starts-seq", "n_starts": 4, "max_iter": 10, "n_workers": 1},
    {"label": "4-starts-par", "n_starts": 4, "max_iter": 10, "n_workers": 0},
    {"label": "8-starts-par", "n_starts": 8, "max_iter": 10, "n_workers": 0},
]


def _load_everything():
    from prometheus_equilibrium.equilibrium.solver import GordonMcBrideSolver
    from prometheus_equilibrium.equilibrium.species import SpeciesDatabase
    from prometheus_equilibrium.optimization.config import (
        load_objective,
        load_operating_point,
        load_problem,
        load_solver_settings,
    )
    from prometheus_equilibrium.propellants import PropellantDatabase

    config = json.loads(_CONFIG_PATH.read_text())
    _, enabled_databases, max_atoms = load_solver_settings(config)

    print("Loading databases...", flush=True)
    spec_db = SpeciesDatabase(
        nasa7_path=str(_THERMO_DIR / "nasa7.json"),
        nasa9_path=str(_THERMO_DIR / "nasa9.json"),
        janaf_path=str(_THERMO_DIR / "janaf.csv"),
        afcesic_path=str(_THERMO_DIR / "afcesic.json"),
        terra_path=str(_THERMO_DIR / "terra.json"),
    )
    spec_db.load(
        include_nasa7="NASA-7" in enabled_databases,
        include_nasa9="NASA-9" in enabled_databases,
        include_terra="TERRA" in enabled_databases,
    )
    prop_db = PropellantDatabase(str(_PROP_DB), species_db=spec_db)
    prop_db.load()

    problem = load_problem(config)
    objective = load_objective(config)
    operating_point = load_operating_point(config)

    solver = GordonMcBrideSolver(capture_history=False)
    return (
        spec_db,
        prop_db,
        solver,
        problem,
        objective,
        operating_point,
        enabled_databases,
        max_atoms,
    )


def _run_gradient(
    spec_db,
    prop_db,
    solver,
    problem,
    objective,
    operating_point,
    enabled_databases,
    max_atoms,
    seed: int,
    n_starts: int,
    max_iter: int,
    n_workers: int,
):
    from prometheus_equilibrium.optimization.gradient_engine import (
        MultiStartGradientOptimizer,
    )

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
    t0 = time.perf_counter()
    result = optimizer.optimize(
        n_starts=n_starts,
        max_iter_per_start=max_iter,
        fd_step=1e-4,
        seed=seed,
        n_workers=n_workers,
    )
    elapsed = time.perf_counter() - t0
    return result, elapsed


def _summarise(label: str, runs: list[tuple]) -> dict:
    best_objs = [r.best_objective for r, _ in runs]
    best_isps = [r.best_isp for r, _ in runs]
    times = [t for _, t in runs]
    return {
        "label": label,
        "best_obj_mean": statistics.mean(best_objs),
        "best_obj_stdev": statistics.stdev(best_objs) if len(best_objs) > 1 else 0.0,
        "best_isp_mean": statistics.mean(best_isps),
        "time_mean": statistics.mean(times),
        "best_objs": best_objs,
        "runs": runs,
    }


def _print_summary_table(summaries: list[dict]) -> None:
    col_w = 14
    header = (
        f"  {'Config':22}  {'Best FoM (mean)':>{col_w}}  {'stdev':>{col_w}}"
        f"  {'Isp mean (s)':>{col_w}}  {'Time mean (s)':>{col_w}}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    for s in summaries:
        print(
            f"  {s['label']:22}  {s['best_obj_mean']:>{col_w}.4f}"
            f"  {s['best_obj_stdev']:>{col_w}.4f}"
            f"  {s['best_isp_mean']:>{col_w}.3f}"
            f"  {s['time_mean']:>{col_w}.1f}"
        )


def _print_per_seed(summaries: list[dict]) -> None:
    print(f"\n  Per-seed best objectives:")
    print(f"  {'Seed':>6}  " + "  ".join(f"{s['label']:>22}" for s in summaries))
    for i, seed in enumerate(SEEDS):
        row = f"  {seed:>6}  "
        for s in summaries:
            row += f"  {s['best_objs'][i]:>22.4f}"
        print(row)


def main():
    (
        spec_db,
        prop_db,
        solver,
        problem,
        objective,
        operating_point,
        enabled_databases,
        max_atoms,
    ) = _load_everything()

    print(f"\nBenchmark: Multi-Start Gradient Optimizer  |  seeds={SEEDS}")
    print("=" * 70)

    all_summaries: list[dict] = []

    for gcfg in CONFIGS:
        label = gcfg["label"]
        runs = []
        for seed in SEEDS:
            print(f"\n  {label}  seed={seed} ...", end="", flush=True)
            r, t = _run_gradient(
                spec_db,
                prop_db,
                solver,
                problem,
                objective,
                operating_point,
                enabled_databases,
                max_atoms,
                seed,
                n_starts=gcfg["n_starts"],
                max_iter=gcfg["max_iter"],
                n_workers=gcfg["n_workers"],
            )
            runs.append((r, t))
            print(
                f"  best={r.best_objective:.4f}  Isp={r.best_isp:.2f} s"
                f"  rho={r.best_density:.0f} kg/m3  t={t:.1f}s"
                f"  starts={r.completed_trials}/{gcfg['n_starts']}"
            )
        all_summaries.append(_summarise(label, runs))

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    _print_summary_table(all_summaries)
    _print_per_seed(all_summaries)

    print("\n" + "=" * 70)
    print("Verdict")
    print("=" * 70)
    best_method = max(all_summaries, key=lambda s: s["best_obj_mean"])
    fastest = min(all_summaries, key=lambda s: s["time_mean"])
    most_consistent = min(all_summaries, key=lambda s: s["best_obj_stdev"])
    print(
        f"  Highest mean FoM:  {best_method['label']:22} ({best_method['best_obj_mean']:.4f})"
    )
    print(f"  Fastest wall time: {fastest['label']:22} ({fastest['time_mean']:.1f}s)")
    print(
        f"  Most consistent:   {most_consistent['label']:22} (stdev {most_consistent['best_obj_stdev']:.4f})"
    )
    print()


if __name__ == "__main__":
    main()
