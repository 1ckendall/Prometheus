"""Serialization helpers for optimizer run configurations.

A config is a plain JSON-serializable dict with schema version 1.

**Mass-fraction convention**: all mass-fraction values in the JSON (variable
``minimum``/``maximum``, sum-group ``total``/``minimum_total``/``maximum_total``,
and ``total_mass_fraction``) are expressed as **percentages** (0–100), not
0–1 fractions.  For example, a formulation that sums to 100 % has
``total_mass_fraction: 100.0``, and a solids fraction of 84 % is written as
``maximum_total: 84.0``.  :func:`load_problem` converts these to 0–1
internally; :func:`dump_config` converts the other direction.

Example config::

    {
      "schema_version": 1,
      "problem": {
        "variables": [
          {"ingredient_id": "AP", "minimum": 49.0, "maximum": 91.0, "pinned": false},
          ...
        ],
        "fixed_proportion_groups": [
          {"group_id": "binder", "members": ["HTPB", "IPDI"], "ratios": [0.14, 0.01]}
        ],
        "sum_to_total_groups": [
          {"group_id": "solids", "members": ["AP", "AL"], "minimum_total": 80.0, "maximum_total": 88.0}
        ],
        "total_mass_fraction": 100.0
      },
      "objective": {"isp_variant": "isp_actual", "rho_exponent": 0.25},
      "operating_point": {
        "chamber_pressure_pa": 6895000.0,
        "expansion_type": "area_ratio",
        "expansion_value": 40.0,
        "ambient_pressure_pa": 101325.0,
        "shifting": true
      },
      "run": {"n_starts": 4, "max_iter_per_start": 10, "fd_step": 0.0001, "n_workers": 0, "seed": 42},
      "solver": {"type": "gmcb", "enabled_databases": ["NASA-7", "NASA-9", "TERRA"], "max_atoms": 6}
    }

A pinned variable has ``minimum == maximum``; the ``pinned`` field is stored for
display round-trip convenience but ``minimum`` always carries the fixed value.

There is no ``closure_ingredient_id`` field.  Mass balance is guaranteed by the
hierarchical two-level sampler in
:class:`~prometheus_equilibrium.optimization.constraints.FormulationConstraintCompiler`;
no ingredient needs special designation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .problem import (
    FixedProportionGroup,
    ObjectiveSpec,
    OperatingPoint,
    OptimizationProblem,
    SumToTotalGroup,
    VariableBound,
)

_SCHEMA_VERSION = 1


def dump_config(
    *,
    problem: OptimizationProblem,
    objective: ObjectiveSpec,
    operating_point: OperatingPoint,
    n_starts: int,
    max_iter_per_start: int,
    fd_step: float,
    ftol: float = 1e-4,
    n_workers: int,
    seed: int | None,
    solver_type: str,
    enabled_databases: list[str],
    max_atoms: int,
    staged_enabled: bool = False,
    n_refine: int = 4,
    max_iter_stage2: int = 20,
) -> dict[str, Any]:
    """Serialise a complete optimizer run configuration to a plain dict.

    All mass-fraction values in the output are expressed as **percentages**
    (0–100).  The internal :class:`OptimizationProblem` stores 0–1 fractions,
    which this function multiplies by 100 before writing.

    Args:
        problem: Formulation constraint definition.
        objective: Isp variant and density exponent.
        operating_point: Chamber / expansion conditions.
        n_starts: Number of independent SLSQP starting points.
        max_iter_per_start: Maximum SLSQP iterations per start.
        fd_step: Finite-difference step size (0–1 mass-fraction scale).
        n_workers: Parallel worker count (0 = automatic).
        seed: Random seed, or ``None`` for non-deterministic.
        solver_type: Solver key (``"gmcb"``, ``"mss"``, ``"hybrid"``).
        enabled_databases: Thermo database labels to use.
        max_atoms: Product-species atom-count filter.
        staged_enabled: If ``True``, enable two-stage frozen→shifting mode.
        n_refine: Number of stage-1 optima to carry into stage-2 refinement.
        max_iter_stage2: SLSQP iterations per start in stage 2.

    Returns:
        JSON-serializable dict with mass fractions expressed as percentages.
    """
    variables: list[dict[str, Any]] = []
    for var in problem.variables:
        pinned = abs(var.minimum - var.maximum) <= 1e-12
        variables.append(
            {
                "ingredient_id": var.ingredient_id,
                "minimum": var.minimum * 100.0,
                "maximum": var.maximum * 100.0,
                "pinned": pinned,
            }
        )

    fixed_groups = [
        {
            "group_id": g.group_id,
            "members": list(g.members),
            "ratios": list(g.ratios),
        }
        for g in problem.fixed_proportion_groups
    ]

    sum_groups: list[dict[str, Any]] = []
    for g in problem.sum_to_total_groups:
        entry: dict[str, Any] = {"group_id": g.group_id, "members": list(g.members)}
        if g.total is not None:
            entry["total"] = g.total * 100.0
        if g.minimum_total is not None:
            entry["minimum_total"] = g.minimum_total * 100.0
        if g.maximum_total is not None:
            entry["maximum_total"] = g.maximum_total * 100.0
        sum_groups.append(entry)

    return {
        "schema_version": _SCHEMA_VERSION,
        "problem": {
            "variables": variables,
            "fixed_proportion_groups": fixed_groups,
            "sum_to_total_groups": sum_groups,
            "total_mass_fraction": problem.total_mass_fraction * 100.0,
        },
        "objective": {
            "isp_variant": objective.isp_variant,
            "rho_exponent": objective.rho_exponent,
        },
        "operating_point": {
            "chamber_pressure_pa": operating_point.chamber_pressure_pa,
            "expansion_type": operating_point.expansion_type,
            "expansion_value": operating_point.expansion_value,
            "ambient_pressure_pa": operating_point.ambient_pressure_pa,
            "shifting": operating_point.shifting,
        },
        "run": {
            "n_starts": n_starts,
            "max_iter_per_start": max_iter_per_start,
            "fd_step": fd_step,
            "ftol": ftol,
            "n_workers": n_workers,
            "seed": seed,
        },
        "staged": {
            "enabled": staged_enabled,
            "n_refine": n_refine,
            "max_iter_stage2": max_iter_stage2,
        },
        "solver": {
            "type": solver_type,
            "enabled_databases": list(enabled_databases),
            "max_atoms": max_atoms,
        },
    }


def load_problem(d: dict[str, Any]) -> OptimizationProblem:
    """Deserialise an :class:`OptimizationProblem` from a config dict.

    Mass-fraction values in the config are expressed as **percentages** (0–100)
    and are divided by 100 to produce the 0–1 fractions expected by
    :class:`OptimizationProblem`.

    Args:
        d: Full config dict (schema version 1).

    Returns:
        Validated :class:`OptimizationProblem`.
    """
    p = d["problem"]
    variables: list[VariableBound] = []
    for v in p.get("variables", []):
        pinned = v.get("pinned", False)
        min_val = float(v["minimum"]) / 100.0
        max_val = min_val if pinned else float(v["maximum"]) / 100.0
        variables.append(
            VariableBound(
                ingredient_id=v["ingredient_id"], minimum=min_val, maximum=max_val
            )
        )

    fixed_groups = [
        FixedProportionGroup(
            group_id=g["group_id"],
            members=list(g["members"]),
            ratios=list(g["ratios"]),
        )
        for g in p.get("fixed_proportion_groups", [])
    ]

    def _pct_or_none(val: Any) -> float | None:
        return float(val) / 100.0 if val is not None else None

    sum_groups = [
        SumToTotalGroup(
            group_id=g["group_id"],
            members=list(g["members"]),
            total=_pct_or_none(g.get("total")),
            minimum_total=_pct_or_none(g.get("minimum_total")),
            maximum_total=_pct_or_none(g.get("maximum_total")),
        )
        for g in p.get("sum_to_total_groups", [])
    ]

    return OptimizationProblem(
        variables=variables,
        fixed_proportion_groups=fixed_groups,
        sum_to_total_groups=sum_groups,
        total_mass_fraction=float(p.get("total_mass_fraction", 100.0)) / 100.0,
    )


def load_objective(d: dict[str, Any]) -> ObjectiveSpec:
    """Deserialise an :class:`ObjectiveSpec` from a config dict.

    Args:
        d: Full config dict.

    Returns:
        :class:`ObjectiveSpec`.
    """
    o = d.get("objective", {})
    return ObjectiveSpec(
        isp_variant=o.get("isp_variant", "isp_actual"),
        rho_exponent=float(o.get("rho_exponent", 0.0)),
    )


def load_operating_point(d: dict[str, Any]) -> OperatingPoint:
    """Deserialise an :class:`OperatingPoint` from a config dict.

    Args:
        d: Full config dict.

    Returns:
        :class:`OperatingPoint`.
    """
    op = d["operating_point"]
    return OperatingPoint(
        chamber_pressure_pa=float(op["chamber_pressure_pa"]),
        expansion_type=op["expansion_type"],
        expansion_value=float(op["expansion_value"]),
        ambient_pressure_pa=float(op["ambient_pressure_pa"]),
        shifting=bool(op.get("shifting", True)),
    )


def load_gradient_config(
    d: dict[str, Any],
) -> tuple[int, int, float, int, int | None]:
    """Return gradient-optimizer run settings from a config dict.

    Reads the ``run`` section for gradient-specific keys, with sensible
    defaults when absent.

    Args:
        d: Full config dict.

    Returns:
        Tuple ``(n_starts, max_iter_per_start, fd_step, n_workers, seed)``
        where ``seed`` may be ``None``.
    """
    r = d.get("run", {})
    return (
        int(r.get("n_starts", 4)),
        int(r.get("max_iter_per_start", 10)),
        float(r.get("fd_step", 1e-4)),
        float(r.get("ftol", 1e-4)),
        int(r.get("n_workers", 0)),
        r.get("seed"),
    )


def load_staged_config(d: dict[str, Any]) -> tuple[bool, int, int]:
    """Return staged-optimizer settings from a config dict.

    Args:
        d: Full config dict.

    Returns:
        Tuple ``(staged_enabled, n_refine, max_iter_stage2)``.
    """
    s = d.get("staged", {})
    return (
        bool(s.get("enabled", False)),
        int(s.get("n_refine", 4)),
        int(s.get("max_iter_stage2", 20)),
    )


def load_solver_settings(
    d: dict[str, Any],
) -> tuple[str, list[str], int]:
    """Return ``(solver_type, enabled_databases, max_atoms)`` from a config dict.

    Args:
        d: Full config dict.

    Returns:
        Tuple of solver type string, list of database labels, and max atoms.
    """
    s = d.get("solver", {})
    return (
        s.get("type", "gmcb"),
        list(s.get("enabled_databases", ["NASA-7", "NASA-9", "TERRA"])),
        int(s.get("max_atoms", 6)),
    )


def save_json(path: Path | str, config: dict[str, Any]) -> None:
    """Write a config dict to a JSON file.

    Args:
        path: Destination file path.
        config: Serializable config dict from :func:`dump_config`.
    """
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
        f.write("\n")


def load_json(path: Path | str) -> dict[str, Any]:
    """Load a config dict from a JSON file.

    Args:
        path: Source file path.

    Returns:
        Config dict.
    """
    with open(path, encoding="utf-8") as f:
        return json.load(f)
