"""Unit tests for formulation relation compilation."""

import pytest

from prometheus_equilibrium.optimization.constraints import (
    FormulationConstraintCompiler,
    InfeasibleTrialError,
)
from prometheus_equilibrium.optimization.problem import (
    FixedProportionGroup,
    OptimizationProblem,
    SumToTotalGroup,
    VariableBound,
)


class _MidpointTrial:
    def suggest_float(self, _name, low, high):
        return 0.5 * (low + high)


# ---------------------------------------------------------------------------
# Fixed-proportion groups
# ---------------------------------------------------------------------------


def test_fixed_ratio_group_enforces_internal_proportions():
    """HTPB:IPDI ratio is preserved; AP (standalone, last) absorbs the residual."""
    problem = OptimizationProblem(
        variables=[
            VariableBound("HTPB", 0.08, 0.16),
            VariableBound("IPDI", 0.0064, 0.0128),
            VariableBound("AP", 0.70, 0.90),
        ],
        fixed_proportion_groups=[
            FixedProportionGroup(
                group_id="binder",
                members=["HTPB", "IPDI"],
                ratios=[100.0, 8.0],
            )
        ],
    )

    composition = FormulationConstraintCompiler(problem).build_from_trial(
        _MidpointTrial()
    )
    assert abs(composition["IPDI"] / composition["HTPB"] - 0.08) < 1e-12
    assert abs(sum(composition.values()) - 1.0) < 1e-8


def test_fixed_group_only_partition():
    """Two fixed-proportion groups partition the total exactly."""
    problem = OptimizationProblem(
        variables=[
            VariableBound("A", 0.30, 0.50),
            VariableBound("B", 0.10, 0.20),  # ratio 3:1 with A → g in [0.40, 0.60]
            VariableBound("C", 0.20, 0.40),
            VariableBound("D", 0.20, 0.40),  # ratio 1:1 with C → g in [0.40, 0.80]
        ],
        fixed_proportion_groups=[
            FixedProportionGroup(group_id="g1", members=["A", "B"], ratios=[3.0, 1.0]),
            FixedProportionGroup(group_id="g2", members=["C", "D"], ratios=[1.0, 1.0]),
        ],
    )

    composition = FormulationConstraintCompiler(problem).build_from_trial(
        _MidpointTrial()
    )
    assert abs(composition["B"] / composition["A"] - 1.0 / 3.0) < 1e-12
    assert abs(composition["D"] / composition["C"] - 1.0) < 1e-12
    assert abs(sum(composition.values()) - 1.0) < 1e-8


# ---------------------------------------------------------------------------
# Sum-to-total groups
# ---------------------------------------------------------------------------


def test_sum_to_total_group_respects_exact_total():
    """Sum group with exact total; standalone ingredient absorbs the remainder."""
    problem = OptimizationProblem(
        variables=[
            VariableBound("AP", 0.60, 0.80),
            VariableBound("AL", 0.05, 0.25),
            VariableBound("BINDER", 0.0, 0.35),
        ],
        sum_to_total_groups=[
            SumToTotalGroup(group_id="solids", members=["AP", "AL"], total=0.84)
        ],
    )

    composition = FormulationConstraintCompiler(problem).build_from_trial(
        _MidpointTrial()
    )
    assert abs((composition["AP"] + composition["AL"]) - 0.84) < 1e-8
    assert abs(sum(composition.values()) - 1.0) < 1e-8


def test_sum_to_total_group_respects_min_max_bounds():
    """Sum group with inequality bounds; standalone absorbs remainder."""
    problem = OptimizationProblem(
        variables=[
            VariableBound("AP", 0.60, 0.80),
            VariableBound("AL", 0.05, 0.25),
            VariableBound("BINDER", 0.0, 0.35),
        ],
        sum_to_total_groups=[
            SumToTotalGroup(
                group_id="solids",
                members=["AP", "AL"],
                minimum_total=0.80,
                maximum_total=0.88,
            )
        ],
    )

    composition = FormulationConstraintCompiler(problem).build_from_trial(
        _MidpointTrial()
    )
    solids = composition["AP"] + composition["AL"]
    assert 0.80 <= solids <= 0.88
    assert abs(sum(composition.values()) - 1.0) < 1e-8


# ---------------------------------------------------------------------------
# Hierarchical (group + group) partition — the primary new capability
# ---------------------------------------------------------------------------


def test_fixed_group_and_sum_group_partition_total():
    """Fixed-proportion group + sum-to-total group partition 100 % without closure."""
    # Binder (R45+MDI at 3:1) + Solid (AP+AL+Bi2O3, max 84 %) = 100 %
    problem = OptimizationProblem(
        variables=[
            VariableBound("R45", 0.084, 0.156),
            VariableBound("MDI", 0.028, 0.052),
            VariableBound("AP", 0.35, 0.65),
            VariableBound("AL", 0.098, 0.098),  # pinned
            VariableBound("Bi2O3", 0.14, 0.26),
        ],
        fixed_proportion_groups=[
            FixedProportionGroup(
                group_id="Binder",
                members=["R45", "MDI"],
                ratios=[0.75, 0.25],
            )
        ],
        sum_to_total_groups=[
            SumToTotalGroup(
                group_id="Solid",
                members=["AP", "AL", "Bi2O3"],
                maximum_total=0.84,
            )
        ],
    )

    composition = FormulationConstraintCompiler(problem).build_from_trial(
        _MidpointTrial()
    )

    # Mass balance
    assert abs(sum(composition.values()) - 1.0) < 1e-8
    # Ratio lock
    assert abs(composition["MDI"] / composition["R45"] - 0.25 / 0.75) < 1e-10
    # Solid constraint
    solid_total = composition["AP"] + composition["AL"] + composition["Bi2O3"]
    assert solid_total <= 0.84 + 1e-9
    # Pinned ingredient
    assert abs(composition["AL"] - 0.098) < 1e-10


def test_sum_group_max_enforced_via_binder_budget():
    """Solid ≤ 84 % is satisfied by construction; binder gets the residual."""
    problem = OptimizationProblem(
        variables=[
            VariableBound("R45", 0.084, 0.156),
            VariableBound("MDI", 0.028, 0.052),
            VariableBound("AP", 0.35, 0.65),
            VariableBound("AL", 0.098, 0.098),
            VariableBound("Bi2O3", 0.14, 0.26),
        ],
        fixed_proportion_groups=[
            FixedProportionGroup("Binder", ["R45", "MDI"], [3.0, 1.0])
        ],
        sum_to_total_groups=[
            SumToTotalGroup("Solid", ["AP", "AL", "Bi2O3"], maximum_total=0.84)
        ],
    )
    compiler = FormulationConstraintCompiler(problem)

    # Run multiple trials using different allocation points to check the bound.
    class _HighTrial:
        def suggest_float(self, _name, low, high):
            return high  # maximise every sample

    class _LowTrial:
        def suggest_float(self, _name, low, high):
            return low  # minimise every sample

    for trial in [_MidpointTrial(), _HighTrial(), _LowTrial()]:
        comp = compiler.build_from_trial(trial)
        solid = comp["AP"] + comp["AL"] + comp["Bi2O3"]
        assert solid <= 0.84 + 1e-9, f"Solid constraint violated: {solid:.6f}"
        assert abs(sum(comp.values()) - 1.0) < 1e-8


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_ingredient_in_multiple_fixed_proportion_groups_is_rejected():
    problem = OptimizationProblem(
        variables=[
            VariableBound("A", 0.1, 0.5),
            VariableBound("B", 0.1, 0.5),
            VariableBound("C", 0.0, 0.5),
        ],
        fixed_proportion_groups=[
            FixedProportionGroup(group_id="g1", members=["A", "B"], ratios=[1.0, 1.0]),
            FixedProportionGroup(group_id="g2", members=["B", "C"], ratios=[1.0, 1.0]),
        ],
    )

    with pytest.raises(ValueError, match="multiple fixed-proportion groups"):
        FormulationConstraintCompiler(problem)


def test_ingredient_in_multiple_sum_groups_is_rejected():
    problem = OptimizationProblem(
        variables=[
            VariableBound("A", 0.1, 0.4),
            VariableBound("B", 0.1, 0.4),
            VariableBound("C", 0.1, 0.4),
        ],
        sum_to_total_groups=[
            SumToTotalGroup(
                group_id="g1",
                members=["A", "B"],
                minimum_total=0.3,
                maximum_total=0.7,
            ),
            SumToTotalGroup(
                group_id="g2",
                members=["B", "C"],
                minimum_total=0.2,
                maximum_total=0.6,
            ),
        ],
    )

    with pytest.raises(ValueError, match="multiple sum-to-total groups"):
        FormulationConstraintCompiler(problem)


def test_ingredient_in_fixed_and_sum_group_is_rejected():
    """Cross-type membership is now rejected under the hierarchical model."""
    problem = OptimizationProblem(
        variables=[
            VariableBound("A", 0.1, 0.5),
            VariableBound("B", 0.1, 0.5),
            VariableBound("C", 0.1, 0.5),
        ],
        fixed_proportion_groups=[
            FixedProportionGroup(group_id="fp", members=["A", "B"], ratios=[1.0, 1.0]),
        ],
        sum_to_total_groups=[
            SumToTotalGroup(
                group_id="st",
                members=["B", "C"],
                minimum_total=0.2,
                maximum_total=0.8,
            ),
        ],
    )

    with pytest.raises(
        ValueError, match="both a fixed-proportion group and a sum-to-total group"
    ):
        FormulationConstraintCompiler(problem)


def test_infeasible_when_bounds_cannot_sum_to_total():
    """Trial is pruned when standalone bounds cannot sum to total_mass_fraction."""
    problem = OptimizationProblem(
        variables=[
            VariableBound("A", 0.1, 0.2),
            VariableBound("B", 0.1, 0.2),
        ]
    )

    compiler = FormulationConstraintCompiler(problem)
    with pytest.raises(InfeasibleTrialError):
        compiler.build_from_trial(_MidpointTrial())
