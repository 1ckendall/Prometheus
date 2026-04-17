"""Shared composition evaluation for optimization engines."""

from __future__ import annotations

import math

from prometheus_equilibrium.equilibrium.problem import EquilibriumProblem, ProblemType

from .problem import ObjectiveSpec, OperatingPoint


def evaluate_composition(
    composition: dict[str, float],
    *,
    prop_db,
    spec_db,
    perf_solver,
    enabled_databases: list[str],
    max_atoms: int,
    operating_point: OperatingPoint,
    objective: ObjectiveSpec,
) -> tuple[float, float, float, float]:
    """Evaluate a composition; return ``(log_fom, fom, isp, density)``.

    Args:
        composition: Mass-fraction mapping of ingredient IDs to fractions (0–1).
        prop_db: Loaded :class:`~prometheus_equilibrium.propellants.PropellantDatabase`.
        spec_db: Loaded :class:`~prometheus_equilibrium.equilibrium.species.SpeciesDatabase`.
        perf_solver: :class:`~prometheus_equilibrium.equilibrium.performance.PerformanceSolver`
            instance.
        enabled_databases: Thermo database labels used for species selection.
        max_atoms: Product-species atom-count filter.
        operating_point: Chamber / expansion conditions.
        objective: Isp variant and density exponent.

    Returns:
        Tuple ``(log_fom, fom, isp, density)`` where ``log_fom = log(Isp) +
        rho_exponent * log(density)`` and ``fom = Isp * density**rho_exponent``.

    Raises:
        ValueError: If the mixture density is non-positive, Isp is non-positive,
            or the equilibrium / performance solver does not converge.
    """
    density = mixture_density(composition, prop_db)
    if density <= 0.0:
        raise ValueError("Mixture density is non-positive.")

    mixture = prop_db.mix(list(composition.items()))
    products = spec_db.get_species(
        mixture.elements,
        max_atoms=max_atoms,
        enabled_databases=enabled_databases,
    )
    eq_problem = EquilibriumProblem(
        reactants=mixture.reactants,
        products=products,
        problem_type=ProblemType.HP,
        constraint1=mixture.enthalpy,
        constraint2=operating_point.chamber_pressure_pa,
        t_init=3500.0,
    )
    eq_problem.validate()

    try:
        if operating_point.expansion_type == "pressure":
            perf = perf_solver.solve(
                eq_problem,
                pe_pa=operating_point.expansion_value,
                shifting=operating_point.shifting,
                ambient_pressure=operating_point.ambient_pressure_pa,
                compute_profile=False,
            )
        else:
            perf = perf_solver.solve(
                eq_problem,
                area_ratio=operating_point.expansion_value,
                shifting=operating_point.shifting,
                ambient_pressure=operating_point.ambient_pressure_pa,
                compute_profile=False,
            )
    except RuntimeError as exc:
        raise ValueError(str(exc)) from exc
    isp = float(getattr(perf, objective.isp_variant))
    if isp <= 0.0:
        raise ValueError("Isp is non-positive.")

    log_fom = math.log(isp) + objective.rho_exponent * math.log(density)
    fom = isp * (density**objective.rho_exponent)
    return log_fom, fom, isp, density


def mixture_density(composition: dict[str, float], prop_db) -> float:
    """Return bulk density from component mass fractions (harmonic mean).

    Args:
        composition: Mass-fraction mapping of ingredient IDs to fractions (0–1).
        prop_db: Loaded propellant database.

    Returns:
        Mixture density in kg/m³, or ``0.0`` if no positive mass fractions.

    Raises:
        ValueError: If any ingredient with positive mass fraction is missing a
            positive density value.
    """
    denominator = 0.0
    for ingredient_id, mass_fraction in composition.items():
        if mass_fraction <= 0.0:
            continue
        ingredient = prop_db.find_ingredient(ingredient_id)
        density = float(ingredient.get("density", 0.0))
        if density <= 0.0:
            raise ValueError(
                f"Ingredient {ingredient_id!r} is missing a positive density."
            )
        denominator += mass_fraction / density
    if denominator <= 0.0:
        return 0.0
    return 1.0 / denominator
