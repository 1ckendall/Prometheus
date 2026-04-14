"""Multi-start gradient-descent optimizer for constrained propellant formulations.

Uses SLSQP (Sequential Least Squares Programming) from ``scipy.optimize`` to
perform local gradient descent from multiple random feasible starting
compositions.  Constraints are expressed directly in composition space (box
bounds, ratio locks, sum-group bounds, mass balance) rather than through the
hierarchical sampler parameterisation, so SLSQP can handle them natively.

Gradients are estimated by finite differences.  The default step size
``fd_step=1e-4`` (0.01 % mass fraction) is large enough to sit above
equilibrium-solver noise while still resolving gradient structure.

Parallel execution
------------------
When ``n_workers != 1`` (or ``n_workers=0`` for auto), the independent SLSQP
starts are distributed across a :class:`~concurrent.futures.ProcessPoolExecutor`
worker pool.  Large shared state (propellant database, species database, solver)
is sent to each worker process once via the pool initializer; per-task pickle
overhead is then limited to the small starting-point vector and problem
parameters.  On Windows the ``spawn`` start method is used automatically.
"""

from __future__ import annotations

import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Callable

import numpy as np
from loguru import logger
from scipy.optimize import Bounds, minimize

from prometheus_equilibrium.equilibrium.performance import PerformanceSolver

from ._eval import evaluate_composition
from .constraints import FormulationConstraintCompiler, InfeasibleTrialError
from .engine import OptimizationResult
from .problem import (
    ObjectiveSpec,
    OperatingPoint,
    OptimizationProblem,
    validate_objective,
    validate_operating_point,
)

# Penalty returned when the equilibrium solver fails at a trial point.
_PENALTY = 1e6

# ---------------------------------------------------------------------------
# Per-process worker state (set once by _worker_init; used by _worker_run_start)
# ---------------------------------------------------------------------------

_W_PROP_DB = None
_W_SPEC_DB = None
_W_PERF_SOLVER = None


def _worker_init(prop_db, spec_db, solver) -> None:
    """Initialise shared state in each worker process.

    Called once per worker process by :class:`~concurrent.futures.ProcessPoolExecutor`
    before any tasks are dispatched to that worker.

    Args:
        prop_db: Loaded :class:`~prometheus_equilibrium.propellants.PropellantDatabase`.
        spec_db: Loaded :class:`~prometheus_equilibrium.equilibrium.species.SpeciesDatabase`.
        solver: Equilibrium solver instance.
    """
    global _W_PROP_DB, _W_SPEC_DB, _W_PERF_SOLVER
    # Keep spawned worker processes quiet during GUI optimization runs.
    logger.disable("prometheus_equilibrium.equilibrium")
    _W_PROP_DB = prop_db
    _W_SPEC_DB = spec_db
    _W_PERF_SOLVER = PerformanceSolver(solver, db=spec_db)


def _worker_run_start(
    x0: np.ndarray,
    problem: OptimizationProblem,
    objective: ObjectiveSpec,
    operating_point: OperatingPoint,
    enabled_databases: list[str],
    max_atoms: int,
    max_iter: int,
    fd_step: float,
) -> dict[str, object]:
    """Run one SLSQP start in a worker process.

    Uses the module-level ``_W_*`` globals populated by :func:`_worker_init`.

    Args:
        x0: Starting composition vector (0–1 mass fractions).
        problem: Validated optimization problem.
        objective: Isp variant and density exponent.
        operating_point: Chamber / expansion conditions.
        enabled_databases: Thermo database labels for species selection.
        max_atoms: Product-species atom-count filter.
        max_iter: Maximum SLSQP iterations.
        fd_step: Finite-difference step size.

    Returns:
        Dict with ``worker_id`` and ``result`` keys where ``result`` is
        ``(log_fom, fom, isp, density, composition)`` on success or ``None``.
    """
    result = _run_single_start(
        x0,
        problem,
        objective,
        operating_point,
        enabled_databases,
        max_atoms,
        max_iter,
        fd_step,
        _W_PROP_DB,
        _W_SPEC_DB,
        _W_PERF_SOLVER,
    )
    return {
        "worker_id": f"pid-{os.getpid()}",
        "result": result,
    }


# ---------------------------------------------------------------------------
# Constraint and bounds builders
# ---------------------------------------------------------------------------


class _RandomTrial:
    """Implements ``FloatTrialLike`` using a numpy random generator.

    Args:
        rng: Numpy random generator used for sampling.
    """

    def __init__(self, rng: np.random.Generator) -> None:
        self._rng = rng

    def suggest_float(self, _name: str, low: float, high: float) -> float:
        """Return a uniform random float from ``[low, high]``.

        Args:
            _name: Parameter name (unused).
            low: Lower bound.
            high: Upper bound.

        Returns:
            Sampled float.
        """
        return float(self._rng.uniform(low, high))


def _build_scipy_bounds(problem: OptimizationProblem) -> Bounds:
    """Build a :class:`scipy.optimize.Bounds` object from per-ingredient bounds.

    Args:
        problem: Validated optimization problem.

    Returns:
        Bounds aligned with ``problem.variables`` order.
    """
    lbs = [v.minimum for v in problem.variables]
    ubs = [v.maximum for v in problem.variables]
    return Bounds(lb=lbs, ub=ubs, keep_feasible=True)


def _build_scipy_constraints(problem: OptimizationProblem) -> list[dict]:
    """Build the scipy constraint list for SLSQP from an ``OptimizationProblem``.

    Constraints produced:

    - **Mass balance** (equality): ``Σ xᵢ = total_mass_fraction``
    - **Ratio locks** (equality, N-1 per fixed group): ``xₐ · αᵦ = xᵦ · αₐ``
    - **Sum-group upper bound** (inequality ≥ 0): ``max_total − Σ xᵢ ≥ 0``
    - **Sum-group lower bound** (inequality ≥ 0): ``Σ xᵢ − min_total ≥ 0``

    All Jacobians are provided analytically (all constraints are linear).

    Args:
        problem: Validated optimization problem.

    Returns:
        List of scipy constraint dicts suitable for ``scipy.optimize.minimize``
        with ``method='SLSQP'``.
    """
    var_ids = [v.ingredient_id for v in problem.variables]
    idx = {vid: i for i, vid in enumerate(var_ids)}
    n = len(var_ids)
    constraints: list[dict] = []

    # --- Mass balance equality ---
    total = problem.total_mass_fraction
    ones = np.ones(n)
    constraints.append(
        {
            "type": "eq",
            "fun": lambda x, _t=total: float(np.sum(x)) - _t,
            "jac": lambda x: ones,
        }
    )

    # --- Fixed-proportion group ratio locks ---
    for grp in problem.fixed_proportion_groups:
        ratios_sum = sum(grp.ratios)
        alphas = [r / ratios_sum for r in grp.ratios]
        i0 = idx[grp.members[0]]
        a0 = alphas[0]
        for k in range(1, len(grp.members)):
            ik = idx[grp.members[k]]
            ak = alphas[k]
            # Equality: x[i0] * ak − x[ik] * a0 = 0
            jac_vec = np.zeros(n)
            jac_vec[i0] = ak
            jac_vec[ik] = -a0
            jac_static = jac_vec.copy()
            constraints.append(
                {
                    "type": "eq",
                    "fun": lambda x, _i0=i0, _ak=ak, _ik=ik, _a0=a0: (
                        float(x[_i0]) * _ak - float(x[_ik]) * _a0
                    ),
                    "jac": lambda x, _j=jac_static: _j,
                }
            )

    # --- Sum-group bounds ---
    for grp in problem.sum_to_total_groups:
        member_idxs = [idx[m] for m in grp.members]
        low, high = grp.total_bounds()

        # Upper bound: high − Σ members ≥ 0
        ub_jac = np.zeros(n)
        for i in member_idxs:
            ub_jac[i] = -1.0
        ub_jac_static = ub_jac.copy()
        constraints.append(
            {
                "type": "ineq",
                "fun": lambda x, _idxs=member_idxs, _h=high: (
                    _h - sum(float(x[i]) for i in _idxs)
                ),
                "jac": lambda x, _j=ub_jac_static: _j,
            }
        )

        # Lower bound: Σ members − low ≥ 0 (only add if non-trivial)
        if low > 1e-12:
            lb_jac = np.zeros(n)
            for i in member_idxs:
                lb_jac[i] = 1.0
            lb_jac_static = lb_jac.copy()
            constraints.append(
                {
                    "type": "ineq",
                    "fun": lambda x, _idxs=member_idxs, _l=low: (
                        sum(float(x[i]) for i in _idxs) - _l
                    ),
                    "jac": lambda x, _j=lb_jac_static: _j,
                }
            )

    return constraints


# ---------------------------------------------------------------------------
# Core single-start logic (shared by sequential and parallel paths)
# ---------------------------------------------------------------------------


def _run_single_start(
    x0: np.ndarray,
    problem: OptimizationProblem,
    objective: ObjectiveSpec,
    operating_point: OperatingPoint,
    enabled_databases: list[str],
    max_atoms: int,
    max_iter: int,
    fd_step: float,
    prop_db,
    spec_db,
    perf_solver,
) -> (
    tuple[float, float, float, float, dict[str, float], list[tuple[int, float]]] | None
):
    """Run one SLSQP solve from ``x0`` using the supplied solver state.

    Builds bounds and constraints from ``problem``, runs SLSQP, then
    re-evaluates the optimum with full detail.

    Args:
        x0: Starting composition vector (0–1 mass fractions).
        problem: Validated optimization problem.
        objective: Isp variant and density exponent.
        operating_point: Chamber / expansion conditions.
        enabled_databases: Thermo database labels for species selection.
        max_atoms: Product-species atom-count filter.
        max_iter: Maximum SLSQP iterations.
        fd_step: Finite-difference step size.
        prop_db: Loaded propellant database.
        spec_db: Loaded species database.
        perf_solver: Performance solver instance.

    Returns:
        Tuple ``(log_fom, fom, isp, density, composition, trace)`` if feasible,
        ``None`` if the solver fails or the composition yields an invalid
        objective.
    """
    bounds = _build_scipy_bounds(problem)
    constraints = _build_scipy_constraints(problem)
    var_ids = [v.ingredient_id for v in problem.variables]

    def _obj(x: np.ndarray) -> float:
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
        except (ValueError, RuntimeError):
            return _PENALTY

    def _log_fom_from_x(x: np.ndarray) -> float | None:
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
            return log_fom
        except (ValueError, RuntimeError):
            return None

    trace: list[tuple[int, float]] = []
    initial_log_fom = _log_fom_from_x(x0)
    if initial_log_fom is not None:
        trace.append((0, float(initial_log_fom)))

    def _callback(xk: np.ndarray) -> None:
        log_fom = _log_fom_from_x(xk)
        if log_fom is not None:
            trace.append((len(trace), float(log_fom)))

    res = minimize(
        _obj,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        callback=_callback,
        options={
            "maxiter": max_iter,
            "ftol": 1e-9,
            "eps": fd_step,
            "disp": False,
        },
    )

    x_opt = res.x
    composition = {vid: float(x_opt[i]) for i, vid in enumerate(var_ids)}

    try:
        log_fom, fom, isp, density = evaluate_composition(
            composition,
            prop_db=prop_db,
            spec_db=spec_db,
            perf_solver=perf_solver,
            enabled_databases=enabled_databases,
            max_atoms=max_atoms,
            operating_point=operating_point,
            objective=objective,
        )
    except (ValueError, RuntimeError):
        return None
    if not trace or trace[-1][1] != float(log_fom):
        trace.append((len(trace), float(log_fom)))

    return log_fom, fom, isp, density, composition, trace


# ---------------------------------------------------------------------------
# Public optimizer class
# ---------------------------------------------------------------------------


class MultiStartGradientOptimizer:
    """Multi-start SLSQP optimizer for constrained propellant formulations.

    Launches ``n_starts`` independent SLSQP runs from random feasible starting
    compositions (generated via :class:`~.constraints.FormulationConstraintCompiler`)
    and returns the best result found across all starts.

    When ``n_workers != 1`` the starts are distributed across a
    :class:`~concurrent.futures.ProcessPoolExecutor`.  Each worker process
    receives the database state once (via the pool initializer) and handles
    many FD evaluations without repeated pickle overhead.  Set
    ``n_workers=1`` to force sequential execution (useful for debugging or
    when the pool startup cost outweighs the parallelism gain for very small
    problems).

    Args:
        problem: Constraint definition for composition generation.
        objective: Objective settings.
        operating_point: Chamber / expansion conditions.
        prop_db: Loaded propellant database.
        spec_db: Loaded species database.
        solver: Equilibrium solver instance.
        enabled_databases: Enabled thermo database labels for species selection.
        max_atoms: Product-species max-atoms filter.
    """

    def __init__(
        self,
        problem: OptimizationProblem,
        objective: ObjectiveSpec,
        operating_point: OperatingPoint,
        prop_db,
        spec_db,
        solver,
        enabled_databases: list[str],
        max_atoms: int,
    ) -> None:
        validate_objective(objective)
        validate_operating_point(operating_point)
        problem.validate()

        self.problem = problem
        self.objective = objective
        self.operating_point = operating_point
        self.prop_db = prop_db
        self.spec_db = spec_db
        self.solver = solver
        self.enabled_databases = enabled_databases
        self.max_atoms = max_atoms

        self._compiler = FormulationConstraintCompiler(problem)
        self._perf_solver = PerformanceSolver(solver, db=spec_db)
        self._var_ids = [v.ingredient_id for v in problem.variables]

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def optimize(
        self,
        n_starts: int = 8,
        max_iter_per_start: int = 100,
        fd_step: float = 1e-4,
        seed: int | None = None,
        n_workers: int = 0,
        progress_callback: Callable[[dict], None] | None = None,
        should_stop: Callable[[], bool] | None = None,
    ) -> OptimizationResult:
        """Run multi-start SLSQP and return the best result.

        Args:
            n_starts: Number of independent random starting points.
            max_iter_per_start: Maximum SLSQP iterations per start.
            fd_step: Finite-difference step size for gradient estimation
                (in mass-fraction units, i.e. 0–1 scale).
            seed: Optional random seed for reproducible starts.
            n_workers: Number of parallel worker processes.  ``0`` (default)
                means automatic: ``min(n_starts, os.cpu_count())``.  ``1``
                forces sequential execution.
            progress_callback: Optional callback receiving per-start progress
                dicts with keys ``start``, ``converged``, ``objective_value``,
                ``start_trace``, ``best_value``, ``isp``, ``n_starts``.
            should_stop: Optional callback; if it returns ``True`` the run
                stops after the current start completes (sequential) or
                cancels pending futures (parallel).

        Returns:
            :class:`~.engine.OptimizationResult` summarising the best start.

        Raises:
            RuntimeError: If no start produces a feasible result.
        """
        rng = np.random.default_rng(seed)
        x0s = [self._random_start(rng) for _ in range(n_starts)]

        # Resolve effective worker count.
        if n_workers <= 0:
            effective = min(n_starts, os.cpu_count() or 1)
        else:
            effective = min(n_workers, n_starts)

        best_log_fom = -math.inf
        best_fom = -math.inf
        best_isp = float("nan")
        best_density = float("nan")
        best_composition: dict[str, float] = {}
        trial_history: list[tuple[int, float]] = []
        start_history: dict[int, list[tuple[int, float]]] = {}
        completed = 0
        failed = 0

        if effective <= 1:
            # ---- Sequential path ----
            for start_idx, x0 in enumerate(x0s):
                if should_stop is not None and should_stop():
                    break
                result = _run_single_start(
                    x0,
                    self.problem,
                    self.objective,
                    self.operating_point,
                    self.enabled_databases,
                    self.max_atoms,
                    max_iter_per_start,
                    fd_step,
                    self.prop_db,
                    self.spec_db,
                    self._perf_solver,
                )
                (
                    best_log_fom,
                    best_fom,
                    best_isp,
                    best_density,
                    best_composition,
                    trial_history,
                    start_history,
                    completed,
                    failed,
                ) = _record_start(
                    start_idx,
                    result,
                    best_log_fom,
                    best_fom,
                    best_isp,
                    best_density,
                    best_composition,
                    trial_history,
                    start_history,
                    completed,
                    failed,
                    n_starts,
                    progress_callback,
                )
        else:
            # ---- Parallel path ----
            with ProcessPoolExecutor(
                max_workers=effective,
                initializer=_worker_init,
                initargs=(self.prop_db, self.spec_db, self.solver),
            ) as pool:
                futures = {
                    pool.submit(
                        _worker_run_start,
                        x0,
                        self.problem,
                        self.objective,
                        self.operating_point,
                        self.enabled_databases,
                        self.max_atoms,
                        max_iter_per_start,
                        fd_step,
                    ): i
                    for i, x0 in enumerate(x0s)
                }
                pending = set(futures)
                for fut in as_completed(futures):
                    pending.discard(fut)
                    start_idx = futures[fut]
                    try:
                        payload = fut.result()
                        if isinstance(payload, dict):
                            result = payload.get("result")
                        else:
                            result = None
                    except Exception:
                        result = None
                    (
                        best_log_fom,
                        best_fom,
                        best_isp,
                        best_density,
                        best_composition,
                        trial_history,
                        start_history,
                        completed,
                        failed,
                    ) = _record_start(
                        start_idx,
                        result,
                        best_log_fom,
                        best_fom,
                        best_isp,
                        best_density,
                        best_composition,
                        trial_history,
                        start_history,
                        completed,
                        failed,
                        n_starts,
                        progress_callback,
                    )
                    if should_stop is not None and should_stop():
                        for p in pending:
                            p.cancel()
                        break

        if completed == 0:
            raise RuntimeError(f"No feasible result from any of the {n_starts} starts.")

        return OptimizationResult(
            best_objective=best_fom,
            best_log_objective=best_log_fom,
            best_composition=best_composition,
            best_isp=best_isp,
            best_density=best_density,
            trial_history=trial_history,
            start_history=start_history,
            completed_trials=completed,
            pruned_trials=failed,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _random_start(self, rng: np.random.Generator) -> np.ndarray:
        """Generate a random feasible starting composition.

        Retries up to 20 times if the compiler raises
        :class:`~.constraints.InfeasibleTrialError` (rare for well-formed
        problems).

        Args:
            rng: Numpy random generator.

        Returns:
            Composition vector aligned with ``problem.variables`` order.

        Raises:
            RuntimeError: If no feasible start is found after 20 attempts.
        """
        for _ in range(20):
            trial = _RandomTrial(rng)
            try:
                comp = self._compiler.build_from_trial(trial)
                return np.array([comp[vid] for vid in self._var_ids], dtype=float)
            except InfeasibleTrialError:
                continue
        raise RuntimeError(
            "Could not generate a feasible random start after 20 attempts."
        )


# ---------------------------------------------------------------------------
# Module-level helper (must be top-level for pickle compatibility)
# ---------------------------------------------------------------------------


def _record_start(
    start_idx: int,
    result: tuple | None,
    best_log_fom: float,
    best_fom: float,
    best_isp: float,
    best_density: float,
    best_composition: dict,
    trial_history: list,
    start_history: dict[int, list[tuple[int, float]]],
    completed: int,
    failed: int,
    n_starts: int,
    progress_callback: Callable | None,
) -> tuple:
    """Update running best and emit progress callback.

    Args:
        start_idx: Index of the completed start.
        result: Return value of :func:`_run_single_start`, or ``None``.
        best_log_fom: Best log-FoM seen so far.
        best_fom: Best raw FoM seen so far.
        best_isp: Best Isp seen so far.
        best_density: Best density seen so far.
        best_composition: Best composition dict seen so far.
        trial_history: Accumulated ``(start_idx, best_log_fom)`` list.
        start_history: Accumulated ``start_idx -> [(iter_idx, log_fom)]`` map.
        completed: Count of successful starts so far.
        failed: Count of failed starts so far.
        n_starts: Total starts requested (for callback).
        progress_callback: Optional progress callback.

    Returns:
        Updated tuple
        ``(best_log_fom, best_fom, best_isp, best_density, best_composition,
        trial_history, start_history, completed, failed)``.
    """
    objective_value = None
    start_trace = None
    if result is None:
        failed += 1
        converged = False
        isp_report = None
    else:
        log_fom, fom, isp, density, composition, trace = result
        completed += 1
        converged = True
        isp_report = isp
        objective_value = log_fom
        start_trace = list(trace)
        if log_fom > best_log_fom:
            best_log_fom = log_fom
            best_fom = fom
            best_isp = isp
            best_density = density
            best_composition = composition
        trial_history.append((start_idx, best_log_fom))
        start_history[start_idx] = start_trace

    if progress_callback is not None:
        progress_callback(
            {
                "start": start_idx,
                "converged": converged,
                "objective_value": objective_value,
                "start_trace": start_trace,
                "best_value": best_log_fom if best_log_fom > -math.inf else None,
                "isp": isp_report,
                "n_starts": n_starts,
            }
        )

    return (
        best_log_fom,
        best_fom,
        best_isp,
        best_density,
        best_composition,
        trial_history,
        start_history,
        completed,
        failed,
    )
