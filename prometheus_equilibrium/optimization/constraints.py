"""Constraint compiler for formulation relations.

Sampling strategy
-----------------
Composition is built in two levels:

**Level 1 — unit allocation.**  The total mass fraction is distributed among
*compositional units*:

- Each :class:`~.problem.FixedProportionGroup` is one unit (its scale ``g``).
- Each :class:`~.problem.SumToTotalGroup` is one unit (its allocated total ``T``).
- Each ungrouped ingredient is one unit (its mass fraction directly).

Units are sampled budget-aware in declaration order (fixed groups, then sum
groups, then standalones).  The effective range for each unit is tightened so
that all remaining units can still be satisfied.  The last unit's allocation is
fully determined by the remaining budget — no explicit closure ingredient is
needed.

**Level 2 — intra-unit distribution.**  After allocations are determined:

- Fixed group: ``member_i = alpha_i * g`` (fully determined by the ratio).
- Sum group: members are sampled budget-aware within the group allocation ``T``;
  the last member absorbs the group residual.
- Standalone: the allocation is the ingredient's mass fraction directly.

Each ingredient must belong to **at most one** group.  Cross-membership between
a fixed group and a sum group is rejected.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .problem import FixedProportionGroup, OptimizationProblem, SumToTotalGroup


class FloatTrialLike(Protocol):
    """Protocol for Optuna-like trials used by the constraint compiler."""

    def suggest_float(self, name: str, low: float, high: float) -> float:
        """Return a sampled float from ``[low, high]``."""


class InfeasibleTrialError(RuntimeError):
    """Raised when constraints cannot be satisfied for a sampled trial."""


@dataclass(frozen=True)
class Bound:
    """Closed interval bound for an ingredient or unit mass fraction."""

    low: float
    high: float


# Internal representation of a Level-1 sampling unit.
# kind: "fixed" | "sum" | "standalone"
# data: FixedProportionGroup | SumToTotalGroup | VariableBound
# low/high: effective unit-level allocation bounds
_Unit = tuple[str, object, float, float]


class FormulationConstraintCompiler:
    """Compile and sample constrained formulation variables.

    Supports two constraint types:

    - **Fixed-proportion groups**: all members scaled by a shared scalar ``g``
      derived from their declared ratios.
    - **Sum-to-total groups**: members sum to a declared range; internal
      distribution is sampled budget-aware.

    Standalone (ungrouped) ingredients participate directly as Level-1 units.

    Mass balance is guaranteed by construction via budget-aware sequential
    allocation at Level 1.  No closure ingredient designation is required.

    Args:
        problem: Validated optimization problem.

    Raises:
        ValueError: If any ingredient appears in more than one group, or if a
            fixed group's member bounds are mutually inconsistent.
        InfeasibleTrialError: From :meth:`build_from_trial` when constraints
            cannot be satisfied for a sampled point.
    """

    def __init__(self, problem: OptimizationProblem) -> None:
        self.problem = problem
        problem.validate()
        self._bounds: dict[str, Bound] = {
            var.ingredient_id: Bound(var.minimum, var.maximum)
            for var in problem.variables
        }
        self._validate_group_overlap()
        self._units: list[_Unit] = self._build_units()

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    def _validate_group_overlap(self) -> None:
        """Reject any ingredient that appears in more than one group.

        An ingredient may belong to at most one fixed-proportion group and at
        most one sum-to-total group, but **not both**.  This strict partition
        is required by the two-level hierarchical sampling strategy.

        Raises:
            ValueError: If an ingredient is in multiple fixed groups, multiple
                sum groups, or in both a fixed and a sum group simultaneously.
        """
        fixed_member_groups: dict[str, str] = {}
        for grp in self.problem.fixed_proportion_groups:
            for member in grp.members:
                if member in fixed_member_groups:
                    raise ValueError(
                        f"Ingredient {member!r} appears in multiple fixed-proportion groups "
                        f"({fixed_member_groups[member]!r} and {grp.group_id!r})."
                    )
                fixed_member_groups[member] = grp.group_id

        sum_member_groups: dict[str, str] = {}
        for grp in self.problem.sum_to_total_groups:
            for member in grp.members:
                if member in sum_member_groups:
                    raise ValueError(
                        f"Ingredient {member!r} appears in multiple sum-to-total groups "
                        f"({sum_member_groups[member]!r} and {grp.group_id!r})."
                    )
                sum_member_groups[member] = grp.group_id

        cross = set(fixed_member_groups) & set(sum_member_groups)
        if cross:
            names = ", ".join(sorted(repr(x) for x in cross))
            raise ValueError(
                f"Ingredient(s) {names} appear in both a fixed-proportion group and a "
                "sum-to-total group.  Each ingredient must belong to at most one group."
            )

    def _fixed_group_scale_bounds(
        self, group: FixedProportionGroup
    ) -> tuple[float, float]:
        """Compute the valid scale range ``[g_low, g_high]`` from member bounds.

        Args:
            group: Fixed-proportion group.

        Returns:
            Tuple ``(g_low, g_high)``.

        Raises:
            ValueError: If member bounds are mutually inconsistent.
        """
        ratios_sum = sum(group.ratios)
        alphas = [r / ratios_sum for r in group.ratios]
        g_low = max(self._bounds[m].low / a for m, a in zip(group.members, alphas))
        g_high = min(self._bounds[m].high / a for m, a in zip(group.members, alphas))
        if g_low > g_high + 1e-9:
            raise ValueError(
                f"Fixed group {group.group_id!r} has no feasible scale range "
                f"(member bounds give g_low={g_low:.6g} > g_high={g_high:.6g})."
            )
        return g_low, g_high

    def _sum_group_unit_bounds(self, group: SumToTotalGroup) -> tuple[float, float]:
        """Compute effective Level-1 allocation bounds for a sum group.

        The declared total bounds are intersected with the achievable range
        implied by individual member bounds.

        Args:
            group: Sum-to-total group.

        Returns:
            Tuple ``(low, high)`` for the group's total allocation.
        """
        low_cfg, high_cfg = group.total_bounds()
        member_sum_low = sum(self._bounds[m].low for m in group.members)
        member_sum_high = sum(self._bounds[m].high for m in group.members)
        return max(low_cfg, member_sum_low), min(high_cfg, member_sum_high)

    def _build_units(self) -> list[_Unit]:
        """Build the ordered list of Level-1 sampling units.

        Order: fixed-proportion groups → sum-to-total groups → standalones.
        Within each category the declaration order is preserved.

        Returns:
            List of ``(kind, data, low, high)`` tuples.
        """
        all_grouped = {
            m for grp in self.problem.fixed_proportion_groups for m in grp.members
        } | {m for grp in self.problem.sum_to_total_groups for m in grp.members}
        standalones = [
            v for v in self.problem.variables if v.ingredient_id not in all_grouped
        ]

        units: list[_Unit] = []
        for grp in self.problem.fixed_proportion_groups:
            g_low, g_high = self._fixed_group_scale_bounds(grp)
            units.append(("fixed", grp, g_low, g_high))
        for grp in self.problem.sum_to_total_groups:
            t_low, t_high = self._sum_group_unit_bounds(grp)
            units.append(("sum", grp, t_low, t_high))
        for var in standalones:
            b = self._bounds[var.ingredient_id]
            units.append(("standalone", var, b.low, b.high))
        return units

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def build_from_trial(self, trial: FloatTrialLike) -> dict[str, float]:
        """Build a feasible composition from one trial.

        Args:
            trial: Optuna-like trial object implementing ``suggest_float``.

        Returns:
            Mapping of ingredient ID to mass fraction summing to
            ``problem.total_mass_fraction``.

        Raises:
            InfeasibleTrialError: If the constraint set cannot be satisfied for
                the sampled point.
        """
        target = self.problem.total_mass_fraction
        units = self._units

        # --- Level 1: allocate total among units (budget-aware) ---
        residual = target
        allocations: list[float] = []

        for i, (kind, data, low, high) in enumerate(units):
            remaining = units[i + 1 :]
            rem_low = sum(u[2] for u in remaining)
            rem_high = sum(u[3] for u in remaining)

            eff_low = max(low, residual - rem_high)
            eff_high = min(high, residual - rem_low)

            if eff_low > eff_high + 1e-9:
                unit_name = (
                    data.group_id if hasattr(data, "group_id") else data.ingredient_id
                )
                raise InfeasibleTrialError(
                    f"Unit {unit_name!r} has no feasible allocation "
                    f"[{eff_low:.6g}, {eff_high:.6g}]."
                )

            if abs(eff_high - eff_low) <= 1e-12:
                allocation = (eff_low + eff_high) / 2.0
            elif kind == "fixed":
                allocation = trial.suggest_float(
                    f"g:{data.group_id}", eff_low, eff_high
                )
            elif kind == "sum":
                allocation = trial.suggest_float(
                    f"t:{data.group_id}", eff_low, eff_high
                )
            else:
                allocation = trial.suggest_float(
                    f"x:{data.ingredient_id}", eff_low, eff_high
                )

            allocations.append(allocation)
            residual -= allocation

        # --- Level 2: distribute each unit's allocation ---
        composition: dict[str, float] = {}
        for (kind, data, _low, _high), allocation in zip(units, allocations):
            if kind == "fixed":
                self._distribute_fixed_group(data, allocation, composition)
            elif kind == "sum":
                self._distribute_sum_group(trial, data, allocation, composition)
            else:
                composition[data.ingredient_id] = allocation

        return composition

    # ------------------------------------------------------------------
    # Level-2 distribution helpers
    # ------------------------------------------------------------------

    def _distribute_fixed_group(
        self,
        group: FixedProportionGroup,
        g: float,
        composition: dict[str, float],
    ) -> None:
        """Assign fixed-proportion group members from scale ``g``.

        Args:
            group: Fixed-proportion group.
            g: Allocated group total (scale factor).
            composition: Composition dict to update in place.
        """
        ratios_sum = sum(group.ratios)
        alphas = [r / ratios_sum for r in group.ratios]
        for member, alpha in zip(group.members, alphas):
            composition[member] = alpha * g

    def _distribute_sum_group(
        self,
        trial: FloatTrialLike,
        group: SumToTotalGroup,
        total: float,
        composition: dict[str, float],
    ) -> None:
        """Distribute ``total`` among sum-group members budget-aware.

        The last member absorbs the group residual exactly.

        Args:
            trial: Optuna-like trial object.
            group: Sum-to-total group.
            total: Allocated group total from Level 1.
            composition: Composition dict to update in place.

        Raises:
            InfeasibleTrialError: If any member's individual bounds cannot be
                satisfied within the group's allocated total.
        """
        members = group.members
        residual = total

        for idx, member in enumerate(members):
            bound = self._bounds[member]
            is_last = idx == len(members) - 1

            if is_last:
                val = residual
                if not (bound.low - 1e-9 <= val <= bound.high + 1e-9):
                    raise InfeasibleTrialError(
                        f"Sum group {group.group_id!r}: last member {member!r} value "
                        f"{val:.6g} is outside bounds [{bound.low:.6g}, {bound.high:.6g}]."
                    )
                composition[member] = min(bound.high, max(bound.low, val))
                return

            remaining = members[idx + 1 :]
            rem_low = sum(self._bounds[m].low for m in remaining)
            rem_high = sum(self._bounds[m].high for m in remaining)

            low = max(bound.low, residual - rem_high)
            high = min(bound.high, residual - rem_low)

            if low > high + 1e-9:
                raise InfeasibleTrialError(
                    f"Sum group {group.group_id!r}: no feasible interval for "
                    f"{member!r} (tightened to [{low:.6g}, {high:.6g}])."
                )

            if abs(high - low) <= 1e-12:
                val = (low + high) / 2.0
            else:
                val = trial.suggest_float(f"s:{group.group_id}:{member}", low, high)

            composition[member] = val
            residual -= val
