"""Constraint compiler for formulation relations."""

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
    """Closed interval bound for an ingredient mass fraction."""

    low: float
    high: float


class FormulationConstraintCompiler:
    """Compile and sample constrained formulation variables.

    This compiler supports two relation types:

    - fixed-proportion groups: ``x_i = alpha_i * g``
    - sum-to-total groups: ``sum(x_i) = T``

    plus optional global closure via ``closure_ingredient_id``.

    An ingredient may appear in at most one fixed-proportion group **and** at
    most one sum-to-total group simultaneously.  This lets a ratio-locked
    sub-blend (e.g. HTPB + IPDI at a fixed ratio) also participate in a
    higher-level sum constraint (e.g. binder + AP must total 0.84).  Members
    already assigned by a fixed-proportion group are treated as pre-counted
    when the sum-to-total group is processed; only the remaining unassigned
    members are sampled.

    The closure ingredient is always handled last by :meth:`_apply_global_closure`,
    which overrides whatever value the ingredient may have received from a group.
    This allows the closure to absorb mass-balance slack even when it participates
    in a fixed-proportion group.

    Ungrouped, non-closure variables are drawn with a budget-aware sequential
    strategy: each variable's range is tightened so that the remaining budget
    (including the closure's contribution) can always be distributed among the
    variables that follow.

    Args:
        problem: Validated optimization problem.
    """

    def __init__(self, problem: OptimizationProblem) -> None:
        self.problem = problem
        self.problem.validate()
        self._bounds = {
            var.ingredient_id: Bound(var.minimum, var.maximum)
            for var in self.problem.variables
        }
        self._validate_group_overlap()

    def _validate_group_overlap(self) -> None:
        """Reject same-type overlaps; cross-type (fixed + sum) is permitted.

        Raises:
            ValueError: If an ingredient appears in more than one
                fixed-proportion group, or in more than one sum-to-total group.
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

    def build_from_trial(self, trial: FloatTrialLike) -> dict[str, float]:
        """Build a feasible composition from one trial.

        Args:
            trial: Optuna-like trial object implementing ``suggest_float``.

        Returns:
            Mapping of ingredient ID to mass fraction.

        Raises:
            InfeasibleTrialError: If the constraint set cannot be satisfied.
        """
        composition: dict[str, float] = {}

        for group in self.problem.fixed_proportion_groups:
            self._assign_fixed_group(trial, group, composition)

        for group in self.problem.sum_to_total_groups:
            self._assign_sum_group(trial, group, composition)

        self._assign_free_variables(trial, composition)
        self._apply_global_closure(composition)
        return composition

    def _assign_fixed_group(
        self,
        trial: FloatTrialLike,
        group: FixedProportionGroup,
        composition: dict[str, float],
    ) -> None:
        ratios_sum = sum(group.ratios)
        alphas = [r / ratios_sum for r in group.ratios]

        g_low = max(self._bounds[m].low / a for m, a in zip(group.members, alphas))
        g_high = min(self._bounds[m].high / a for m, a in zip(group.members, alphas))

        if g_low > g_high:
            raise InfeasibleTrialError(
                f"Fixed group {group.group_id!r} has no feasible scale range."
            )

        if abs(g_high - g_low) <= 1e-12:
            g_val = (g_low + g_high) / 2.0
        else:
            g_val = trial.suggest_float(f"g:{group.group_id}", g_low, g_high)
        for member, alpha in zip(group.members, alphas):
            composition[member] = alpha * g_val

    def _assign_sum_group(
        self,
        trial: FloatTrialLike,
        group: SumToTotalGroup,
        composition: dict[str, float],
    ) -> None:
        members = group.members
        low_cfg, high_cfg = group.total_bounds()

        # Members already in composition (e.g. from a fixed-proportion group)
        # count toward the group target but are not re-sampled.
        preassigned_total = sum(composition[m] for m in members if m in composition)
        unassigned = [m for m in members if m not in composition]

        if not unassigned:
            if not (low_cfg - 1e-9 <= preassigned_total <= high_cfg + 1e-9):
                raise InfeasibleTrialError(
                    f"Sum group {group.group_id!r} total {preassigned_total:.6g} violates "
                    f"[{low_cfg:.6g}, {high_cfg:.6g}]."
                )
            return

        # Effective target range for the unassigned members alone.
        eff_low = low_cfg - preassigned_total
        eff_high = high_cfg - preassigned_total

        low_feasible = max(eff_low, sum(self._bounds[m].low for m in unassigned))
        high_feasible = min(eff_high, sum(self._bounds[m].high for m in unassigned))

        if low_feasible > high_feasible:
            raise InfeasibleTrialError(
                f"Sum group {group.group_id!r} has no feasible total range."
            )

        if abs(high_feasible - low_feasible) <= 1e-12:
            residual = low_feasible
        else:
            residual = trial.suggest_float(
                f"t:{group.group_id}", low_feasible, high_feasible
            )

        for idx, member in enumerate(unassigned):
            bound = self._bounds[member]
            is_last = idx == len(unassigned) - 1
            if is_last:
                if not (bound.low - 1e-9 <= residual <= bound.high + 1e-9):
                    raise InfeasibleTrialError(
                        f"Sum group {group.group_id!r} closure for {member!r} "
                        f"is outside bounds: {residual:.6g}."
                    )
                composition[member] = min(bound.high, max(bound.low, residual))
                return

            remaining = unassigned[idx + 1 :]
            rem_low = sum(self._bounds[m].low for m in remaining)
            rem_high = sum(self._bounds[m].high for m in remaining)

            low = max(bound.low, residual - rem_high)
            high = min(bound.high, residual - rem_low)

            if low > high:
                raise InfeasibleTrialError(
                    f"Sum group {group.group_id!r} has no feasible interval for {member!r}."
                )

            if abs(high - low) <= 1e-12:
                val = (low + high) / 2.0
            else:
                val = trial.suggest_float(f"s:{group.group_id}:{member}", low, high)
            composition[member] = val
            residual -= val

    def _assign_free_variables(
        self, trial: FloatTrialLike, composition: dict[str, float]
    ) -> None:
        """Assign ungrouped non-closure variables with budget-aware sequential sampling.

        The closure ingredient is excluded here; it is handled by
        :meth:`_apply_global_closure`.  Each variable's range is tightened to
        ensure that both the remaining free variables *and* the closure ingredient
        can stay within their bounds.

        Args:
            trial: Optuna-like trial object.
            composition: Partially filled composition from group assignments.

        Raises:
            InfeasibleTrialError: If no feasible interval exists for a variable.
        """
        target = self.problem.total_mass_fraction
        closure_id = self.problem.closure_ingredient_id

        # Exclude the closure from sequential sampling.
        free_ids = [
            v.ingredient_id
            for v in self.problem.variables
            if v.ingredient_id not in composition and v.ingredient_id != closure_id
        ]

        if not free_ids:
            return

        # Reserve headroom for the closure ingredient when tightening bounds.
        if closure_id is not None:
            cb = self._bounds[closure_id]
            closure_low, closure_high = cb.low, cb.high
        else:
            closure_low = closure_high = 0.0

        residual = target - sum(composition.values())

        for idx, ingredient_id in enumerate(free_ids):
            bound = self._bounds[ingredient_id]
            remaining = free_ids[idx + 1 :]
            rem_low = sum(self._bounds[m].low for m in remaining)
            rem_high = sum(self._bounds[m].high for m in remaining)

            low = max(bound.low, residual - rem_high - closure_high)
            high = min(bound.high, residual - rem_low - closure_low)

            if low > high:
                raise InfeasibleTrialError(
                    f"Free variable {ingredient_id!r} has no feasible interval "
                    f"(tightened to [{low:.6g}, {high:.6g}])."
                )

            if abs(high - low) <= 1e-12:
                val = (low + high) / 2.0  # pinned / degenerate interval
            else:
                val = trial.suggest_float(f"x:{ingredient_id}", low, high)
            composition[ingredient_id] = val
            residual -= val

    def _apply_global_closure(self, composition: dict[str, float]) -> None:
        """Close the global mass balance via the closure ingredient.

        This step always runs last.  If the closure ingredient was assigned by
        a group (e.g. a fixed-proportion group), its value is overridden here so
        that the total mass fraction equals the target.

        Args:
            composition: Composition dict to update in place.

        Raises:
            InfeasibleTrialError: If the closure value is outside its bounds,
                or if no closure is configured and the total does not match.
        """
        total = sum(composition.values())
        target = self.problem.total_mass_fraction
        tolerance = self.problem.closure_tolerance

        if abs(total - target) <= tolerance:
            return

        closure_id = self.problem.closure_ingredient_id
        if closure_id is None:
            raise InfeasibleTrialError(
                "Global mass-fraction sum does not match target and no closure ingredient "
                f"is configured (got {total:.8f}, target {target:.8f})."
            )

        other_total = total - composition.get(closure_id, 0.0)
        closure_value = target - other_total
        bound = self._bounds[closure_id]
        if not (bound.low <= closure_value <= bound.high):
            raise InfeasibleTrialError(
                f"Closure ingredient {closure_id!r} outside bounds with value "
                f"{closure_value:.8f}."
            )
        composition[closure_id] = closure_value
