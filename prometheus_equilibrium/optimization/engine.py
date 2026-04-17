"""Shared result type for optimization engines."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class OptimizationResult:
    """Summary payload returned by an optimizer run.

    Attributes:
        best_objective: Best raw objective value ``Isp * rho**n``.
        best_log_objective: Best log-objective value used for optimization.
        best_composition: Best mass-fraction composition.
        best_isp: Best-start Isp corresponding to selected Isp variant.
        best_density: Best-start bulk density in kg/m^3.
        trial_history: List of ``(start_index, best_log_value_so_far)`` points.
        start_history: Mapping ``start_index -> [(iteration_index, log_value), ...]``
            containing objective traces for each converged start.
        start_history_meta: Optional mapping ``start_index -> [point_meta, ...]``
            with per-iteration feasibility diagnostics for each converged start.
        completed_trials: Number of starts that produced a valid result.
        pruned_trials: Number of starts that failed (solver error or infeasible).
    """

    best_objective: float
    best_log_objective: float
    best_composition: dict[str, float]
    best_isp: float
    best_density: float
    trial_history: list[tuple[int, float]]
    start_history: dict[int, list[tuple[int, float]]]
    completed_trials: int
    pruned_trials: int
    start_history_meta: dict[int, list[dict[str, object]]] = field(default_factory=dict)
