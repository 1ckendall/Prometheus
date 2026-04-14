"""Problem definitions for Optuna-based propellant optimization."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class VariableBound:
    """Mass-fraction bounds for a single ingredient.

    Args:
        ingredient_id: Ingredient ID from :class:`PropellantDatabase`.
        minimum: Lower bound (inclusive), as a 0–1 fraction.
        maximum: Upper bound (inclusive), as a 0–1 fraction.
    """

    ingredient_id: str
    minimum: float
    maximum: float


@dataclass(frozen=True)
class FixedProportionGroup:
    """Ingredients constrained to fixed internal ratios.

    Args:
        group_id: Unique group label for diagnostics and trial parameter names.
        members: Ingredient IDs that participate in the ratio lock.
        ratios: Positive ratio coefficients aligned with ``members``.
    """

    group_id: str
    members: list[str]
    ratios: list[float]


@dataclass(frozen=True)
class SumToTotalGroup:
    """Ingredients constrained to sum to a total mass-fraction range.

    Args:
        group_id: Unique group label for diagnostics and trial parameter names.
        members: Ingredient IDs constrained by this group.
        total: Optional exact required sum for the group's members (0–1 fraction).
        minimum_total: Optional lower bound for the group total (0–1 fraction).
        maximum_total: Optional upper bound for the group total (0–1 fraction).
    """

    group_id: str
    members: list[str]
    total: float | None = None
    minimum_total: float | None = None
    maximum_total: float | None = None

    def total_bounds(self) -> tuple[float, float]:
        """Return effective lower/upper bounds for this group's total.

        Returns:
            Tuple ``(low, high)`` of group total bounds as 0–1 fractions.
        """
        if self.total is not None:
            return self.total, self.total

        low = 0.0 if self.minimum_total is None else self.minimum_total
        high = 1.0 if self.maximum_total is None else self.maximum_total
        return low, high


@dataclass(frozen=True)
class ObjectiveSpec:
    """Objective settings for scalar optimization.

    Args:
        isp_variant: One of ``"isp_actual"``, ``"isp_vac"``, or ``"isp_sl"``.
        rho_exponent: Exponent ``n`` in ``Isp * rho**n`` constrained to ``[0, 1]``.
    """

    isp_variant: str = "isp_actual"
    rho_exponent: float = 0.0


@dataclass(frozen=True)
class OperatingPoint:
    """Performance evaluation operating point.

    Args:
        chamber_pressure_pa: Chamber pressure in Pa.
        expansion_type: Either ``"pressure"`` or ``"area_ratio"``.
        expansion_value: ``Pe`` in Pa for pressure mode, or ``Ae/At`` for area-ratio mode.
        ambient_pressure_pa: Ambient pressure in Pa.
        shifting: Whether to score shifting (``True``) or frozen (``False``) expansion.
    """

    chamber_pressure_pa: float
    expansion_type: str
    expansion_value: float
    ambient_pressure_pa: float
    shifting: bool = True


@dataclass(frozen=True)
class OptimizationProblem:
    """Constraint and normalization definition for a formulation search.

    The compiler uses a two-level hierarchical sampling strategy:

    - **Level 1** – allocate the total mass fraction among compositional units
      (fixed-proportion groups, sum-to-total groups, and standalone ingredients)
      using budget-aware sequential sampling.  Mass balance is satisfied by
      construction; no closure ingredient is required.
    - **Level 2** – distribute each unit's allocated total among its members.
      Fixed groups use the ratio lock; sum groups use sequential sampling with
      the last member absorbing the group residual.

    Each ingredient must belong to **at most one** group (fixed or sum).

    Args:
        variables: Per-ingredient bounds (0–1 fractions).
        fixed_proportion_groups: Ratio-locked ingredient groups.
        sum_to_total_groups: Groups constrained to a bounded total mass fraction.
        total_mass_fraction: Required total mass-fraction sum (default 1.0).
    """

    variables: list[VariableBound]
    fixed_proportion_groups: list[FixedProportionGroup] = field(default_factory=list)
    sum_to_total_groups: list[SumToTotalGroup] = field(default_factory=list)
    total_mass_fraction: float = 1.0

    def validate(self) -> None:
        """Validate problem consistency.

        Raises:
            ValueError: If bounds or group definitions are inconsistent.
        """
        if not self.variables:
            raise ValueError("At least one variable bound is required.")

        seen: set[str] = set()
        for var in self.variables:
            if not var.ingredient_id:
                raise ValueError("Variable ingredient_id cannot be blank.")
            if var.ingredient_id in seen:
                raise ValueError(f"Duplicate variable bound for {var.ingredient_id!r}.")
            if var.minimum < 0.0:
                raise ValueError(
                    f"Variable {var.ingredient_id!r} has negative minimum."
                )
            if var.maximum < var.minimum:
                raise ValueError(
                    f"Variable {var.ingredient_id!r} has maximum smaller than minimum."
                )
            seen.add(var.ingredient_id)

        for grp in self.fixed_proportion_groups:
            if len(grp.members) < 2:
                raise ValueError(
                    f"Fixed group {grp.group_id!r} must have >= 2 members."
                )
            if len(grp.members) != len(grp.ratios):
                raise ValueError(
                    f"Fixed group {grp.group_id!r} members/ratios length mismatch."
                )
            if any(r <= 0.0 for r in grp.ratios):
                raise ValueError(
                    f"Fixed group {grp.group_id!r} ratios must be strictly positive."
                )

        for grp in self.sum_to_total_groups:
            if len(grp.members) < 2:
                raise ValueError(f"Sum group {grp.group_id!r} must have >= 2 members.")
            if (
                grp.total is None
                and grp.minimum_total is None
                and grp.maximum_total is None
            ):
                raise ValueError(
                    f"Sum group {grp.group_id!r} must define total, minimum_total, or maximum_total."
                )
            if grp.total is not None and grp.total < 0.0:
                raise ValueError(
                    f"Sum group {grp.group_id!r} total cannot be negative."
                )
            if grp.minimum_total is not None and grp.minimum_total < 0.0:
                raise ValueError(
                    f"Sum group {grp.group_id!r} minimum_total cannot be negative."
                )
            if grp.maximum_total is not None and grp.maximum_total < 0.0:
                raise ValueError(
                    f"Sum group {grp.group_id!r} maximum_total cannot be negative."
                )
            low, high = grp.total_bounds()
            if high < low:
                raise ValueError(
                    f"Sum group {grp.group_id!r} has maximum_total < minimum_total."
                )
            if grp.total is not None and not (low <= grp.total <= high):
                raise ValueError(
                    f"Sum group {grp.group_id!r} exact total is outside configured bounds."
                )
            if high > self.total_mass_fraction + 1e-9:
                raise ValueError(
                    f"Sum group {grp.group_id!r} maximum_total ({high:.6g}) exceeds "
                    f"total_mass_fraction ({self.total_mass_fraction:.6g}). "
                    "Check that values are fractions (0–1), not percentages."
                )

        if self.total_mass_fraction <= 0.0:
            raise ValueError("total_mass_fraction must be > 0.")

        ingredient_ids = {v.ingredient_id for v in self.variables}
        for grp in self.fixed_proportion_groups:
            for member in grp.members:
                if member not in ingredient_ids:
                    raise ValueError(
                        f"Fixed group {grp.group_id!r} references unknown ingredient {member!r}."
                    )
        for grp in self.sum_to_total_groups:
            for member in grp.members:
                if member not in ingredient_ids:
                    raise ValueError(
                        f"Sum group {grp.group_id!r} references unknown ingredient {member!r}."
                    )


def validate_objective(spec: ObjectiveSpec) -> None:
    """Validate objective settings.

    Args:
        spec: Objective settings to validate.

    Raises:
        ValueError: If an unsupported objective option is provided.
    """
    if spec.isp_variant not in {"isp_actual", "isp_vac", "isp_sl"}:
        raise ValueError("isp_variant must be one of: isp_actual, isp_vac, isp_sl")
    if not (0.0 <= spec.rho_exponent <= 1.0):
        raise ValueError("rho_exponent must be in [0, 1].")


def validate_operating_point(spec: OperatingPoint) -> None:
    """Validate operating-point settings.

    Args:
        spec: Operating-point settings to validate.

    Raises:
        ValueError: If operating-point inputs are not physically meaningful.
    """
    if spec.chamber_pressure_pa <= 0.0:
        raise ValueError("chamber_pressure_pa must be > 0.")
    if spec.ambient_pressure_pa < 0.0:
        raise ValueError("ambient_pressure_pa must be >= 0.")
    if spec.expansion_type not in {"pressure", "area_ratio"}:
        raise ValueError("expansion_type must be 'pressure' or 'area_ratio'.")
    if spec.expansion_value <= 0.0:
        raise ValueError("expansion_value must be > 0.")
