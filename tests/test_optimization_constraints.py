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


def test_fixed_ratio_group_enforces_internal_proportions():
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
        closure_ingredient_id="AP",
    )

    composition = FormulationConstraintCompiler(problem).build_from_trial(
        _MidpointTrial()
    )
    ratio = composition["IPDI"] / composition["HTPB"]

    assert abs(ratio - 0.08) < 1e-12
    assert abs(sum(composition.values()) - 1.0) < 1e-8


def test_sum_to_total_group_respects_total():
    problem = OptimizationProblem(
        variables=[
            VariableBound("AP", 0.60, 0.80),
            VariableBound("AL", 0.05, 0.25),
            VariableBound("BINDER", 0.0, 0.35),
        ],
        sum_to_total_groups=[
            SumToTotalGroup(group_id="solids", members=["AP", "AL"], total=0.84)
        ],
        closure_ingredient_id="BINDER",
    )

    composition = FormulationConstraintCompiler(problem).build_from_trial(
        _MidpointTrial()
    )
    assert abs((composition["AP"] + composition["AL"]) - 0.84) < 1e-8
    assert abs(sum(composition.values()) - 1.0) < 1e-8


def test_sum_to_total_group_respects_min_max_inequality():
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
        closure_ingredient_id="BINDER",
    )

    composition = FormulationConstraintCompiler(problem).build_from_trial(
        _MidpointTrial()
    )
    solids = composition["AP"] + composition["AL"]
    assert 0.80 <= solids <= 0.88
    assert abs(sum(composition.values()) - 1.0) < 1e-8


def test_fixed_proportion_members_can_participate_in_sum_group():
    """An ingredient in a fixed-proportion group may also appear in a sum-to-total group."""
    problem = OptimizationProblem(
        variables=[
            VariableBound("A", 0.1, 0.5),
            VariableBound("B", 0.1, 0.5),
            VariableBound("C", 0.4, 0.9),
        ],
        fixed_proportion_groups=[
            FixedProportionGroup(
                group_id="ab_blend",
                members=["A", "B"],
                ratios=[1.0, 1.0],  # A == B always
            )
        ],
        sum_to_total_groups=[
            SumToTotalGroup(
                group_id="total",
                members=["A", "B", "C"],
                total=1.0,
            )
        ],
    )

    composition = FormulationConstraintCompiler(problem).build_from_trial(
        _MidpointTrial()
    )

    # Fixed proportion respected: A and B are equal
    assert abs(composition["A"] - composition["B"]) < 1e-10
    # Sum constraint satisfied
    assert abs(sum(composition.values()) - 1.0) < 1e-8


def test_fixed_proportion_group_total_counted_against_sum_group():
    """Pre-assigned members reduce the target the sum group must fill."""
    # HTPB+IPDI locked at 100:8; their combined total comes out of the 0.84 solids budget.
    problem = OptimizationProblem(
        variables=[
            VariableBound("HTPB", 0.08, 0.16),
            VariableBound("IPDI", 0.0064, 0.0128),
            VariableBound("AP", 0.60, 0.92),
        ],
        fixed_proportion_groups=[
            FixedProportionGroup(
                group_id="binder",
                members=["HTPB", "IPDI"],
                ratios=[100.0, 8.0],
            )
        ],
        sum_to_total_groups=[
            SumToTotalGroup(
                group_id="propellant",
                members=["HTPB", "IPDI", "AP"],
                total=1.0,
            )
        ],
    )

    composition = FormulationConstraintCompiler(problem).build_from_trial(
        _MidpointTrial()
    )

    # Ratio still locked
    assert abs(composition["IPDI"] / composition["HTPB"] - 8.0 / 100.0) < 1e-10
    # All three sum to 1.0
    assert abs(sum(composition.values()) - 1.0) < 1e-8


def test_ingredient_in_multiple_fixed_proportion_groups_is_rejected():
    """An ingredient cannot appear in two fixed-proportion groups."""
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
    """An ingredient cannot appear in two sum-to-total groups."""
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


def test_missing_closure_ingredient_causes_infeasible_trial():
    """When no closure is configured and free variables cannot sum to target, trial is pruned."""
    problem = OptimizationProblem(
        variables=[
            VariableBound("A", 0.1, 0.2),
            VariableBound("B", 0.1, 0.2),
        ]
    )

    compiler = FormulationConstraintCompiler(problem)
    with pytest.raises(InfeasibleTrialError):
        compiler.build_from_trial(_MidpointTrial())
