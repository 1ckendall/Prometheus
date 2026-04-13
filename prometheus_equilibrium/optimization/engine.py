"""Optuna optimizer engine for constrained propellant formulations."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

from prometheus_equilibrium.equilibrium.performance import PerformanceSolver
from prometheus_equilibrium.equilibrium.problem import EquilibriumProblem, ProblemType

from .constraints import FormulationConstraintCompiler, InfeasibleTrialError
from .problem import (
    ObjectiveSpec,
    OperatingPoint,
    OptimizationProblem,
    validate_objective,
    validate_operating_point,
)


@dataclass(frozen=True)
class OptimizationResult:
    """Summary payload returned after a study run.

    Attributes:
        best_objective: Best raw objective value ``Isp * rho**n``.
        best_log_objective: Best log-objective value used for optimization.
        best_composition: Best mass-fraction composition.
        best_isp: Best-trial Isp corresponding to selected Isp variant.
        best_density: Best-trial bulk density in kg/m^3.
        trial_history: List of ``(trial_number, best_log_value_so_far)`` points.
        completed_trials: Number of completed (non-pruned) trials.
        pruned_trials: Number of pruned trials.
    """

    best_objective: float
    best_log_objective: float
    best_composition: dict[str, float]
    best_isp: float
    best_density: float
    trial_history: list[tuple[int, float]]
    completed_trials: int
    pruned_trials: int


class OptunaOptimizer:
    """Run Optuna optimization against Prometheus performance solves.

    Args:
        problem: Constraint definition for composition generation.
        objective: Objective settings.
        operating_point: Chamber/expansion conditions.
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

    def optimize(
        self,
        n_trials: int,
        timeout_seconds: int | None = None,
        seed: int | None = None,
        progress_callback: Callable[[dict], None] | None = None,
        should_stop: Callable[[], bool] | None = None,
    ) -> OptimizationResult:
        """Run a study and return best-trial summary.

        Args:
            n_trials: Maximum number of trials.
            timeout_seconds: Optional wall-clock timeout in seconds.
            seed: Optional random seed for deterministic sampler behavior.
            progress_callback: Optional callback receiving per-trial progress dicts.
            should_stop: Optional callback to request graceful stop between trials.

        Returns:
            OptimizationResult summarizing the best trial and trial history.

        Raises:
            RuntimeError: If Optuna is unavailable or no trial completes.
        """
        try:
            import optuna
        except ImportError as exc:
            raise RuntimeError(
                "Optuna is required for optimization. Install it via your dev/optimizer dependencies."
            ) from exc

        sampler = optuna.samplers.TPESampler(seed=seed)
        study = optuna.create_study(direction="maximize", sampler=sampler)
        trial_history: list[tuple[int, float]] = []

        def _objective(trial):
            try:
                composition = self._compiler.build_from_trial(trial)
            except InfeasibleTrialError as exc:
                trial.set_user_attr("status_kind", "infeasible")
                trial.set_user_attr("status_reason", str(exc))
                raise optuna.TrialPruned(str(exc))

            density = self._mixture_density(composition)
            if density <= 0.0:
                trial.set_user_attr("status_kind", "infeasible")
                trial.set_user_attr("status_reason", "Mixture density is non-positive.")
                raise optuna.TrialPruned("density<=0")

            mixture = self.prop_db.mix(list(composition.items()))
            products = self.spec_db.get_species(
                mixture.elements,
                max_atoms=self.max_atoms,
                enabled_databases=self.enabled_databases,
            )
            problem = EquilibriumProblem(
                reactants=mixture.reactants,
                products=products,
                problem_type=ProblemType.HP,
                constraint1=mixture.enthalpy,
                constraint2=self.operating_point.chamber_pressure_pa,
                t_init=3500.0,
            )
            problem.validate()

            try:
                if self.operating_point.expansion_type == "pressure":
                    pair = self._perf_solver.solve_pair(
                        problem,
                        pe_pa=self.operating_point.expansion_value,
                        ambient_pressure=self.operating_point.ambient_pressure_pa,
                        compute_profile=False,
                    )
                else:
                    pair = self._perf_solver.solve_pair(
                        problem,
                        area_ratio=self.operating_point.expansion_value,
                        ambient_pressure=self.operating_point.ambient_pressure_pa,
                        compute_profile=False,
                    )
            except RuntimeError as exc:
                # Treat solver non-convergence as a pruned trial, not a hard failure.
                trial.set_user_attr("status_kind", "solver_error")
                trial.set_user_attr("status_reason", str(exc))
                raise optuna.TrialPruned(str(exc))

            perf = pair.shifting if self.operating_point.shifting else pair.frozen
            isp = float(getattr(perf, self.objective.isp_variant))
            if isp <= 0.0:
                trial.set_user_attr("status_kind", "infeasible")
                trial.set_user_attr("status_reason", "Isp is non-positive.")
                raise optuna.TrialPruned("isp<=0")

            log_fom = math.log(isp) + self.objective.rho_exponent * math.log(density)
            fom = isp * (density**self.objective.rho_exponent)

            trial.set_user_attr("composition", composition)
            trial.set_user_attr("density", density)
            trial.set_user_attr("isp", isp)
            trial.set_user_attr("fom", fom)
            trial.set_user_attr("status_kind", "complete")
            return log_fom

        def _callback(study_obj, trial_obj):
            if trial_obj.value is not None:
                trial_history.append((trial_obj.number, study_obj.best_value))
            best_value = None
            if trial_history:
                best_value = trial_history[-1][1]
            if progress_callback is not None:
                progress_callback(
                    {
                        "trial": trial_obj.number,
                        "state": str(trial_obj.state),
                        "best_value": best_value,
                        "status_kind": trial_obj.user_attrs.get("status_kind", ""),
                        "status_reason": trial_obj.user_attrs.get("status_reason", ""),
                        "completed_trials": len(
                            [
                                t
                                for t in study_obj.trials
                                if t.state == optuna.trial.TrialState.COMPLETE
                            ]
                        ),
                    }
                )
            if should_stop is not None and should_stop():
                study_obj.stop()

        study.optimize(
            _objective,
            n_trials=n_trials,
            timeout=timeout_seconds,
            callbacks=[_callback],
            show_progress_bar=False,
        )

        complete = [
            t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
        ]
        pruned = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
        if not complete:
            raise RuntimeError("No complete optimization trials were produced.")

        best_trial = study.best_trial
        return OptimizationResult(
            best_objective=float(best_trial.user_attrs["fom"]),
            best_log_objective=float(best_trial.value),
            best_composition=dict(best_trial.user_attrs["composition"]),
            best_isp=float(best_trial.user_attrs["isp"]),
            best_density=float(best_trial.user_attrs["density"]),
            trial_history=trial_history,
            completed_trials=len(complete),
            pruned_trials=len(pruned),
        )

    def _mixture_density(self, composition: dict[str, float]) -> float:
        """Return bulk density from component mass fractions.

        Args:
            composition: Mapping ingredient ID -> mass fraction.

        Returns:
            Mixture density in kg/m^3.

        Raises:
            KeyError: If an ingredient is unknown.
            ValueError: If an ingredient is missing a positive density value.
        """
        denominator = 0.0
        for ingredient_id, mass_fraction in composition.items():
            if mass_fraction <= 0.0:
                continue
            ingredient = self.prop_db.find_ingredient(ingredient_id)
            density = float(ingredient.get("density", 0.0))
            if density <= 0.0:
                raise ValueError(
                    f"Ingredient {ingredient_id!r} is missing a positive density."
                )
            denominator += mass_fraction / density

        if denominator <= 0.0:
            return 0.0
        return 1.0 / denominator
