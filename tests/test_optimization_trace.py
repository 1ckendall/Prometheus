"""Tests for feasibility-aware optimizer trace logging."""

from __future__ import annotations

from prometheus_equilibrium.optimization.gradient_engine import (
    _constraint_diagnostics,
    _record_start,
)
from prometheus_equilibrium.optimization.problem import (
    FixedProportionGroup,
    OptimizationProblem,
    SumToTotalGroup,
    VariableBound,
)


def _sample_problem() -> OptimizationProblem:
    return OptimizationProblem(
        variables=[
            VariableBound("R45", 0.084, 0.156),
            VariableBound("MDI", 0.028, 0.052),
            VariableBound("AP", 0.35, 0.65),
            VariableBound("AL", 0.098, 0.182),
            VariableBound("BIO", 0.14, 0.26),
        ],
        fixed_proportion_groups=[
            FixedProportionGroup(
                group_id="Binder",
                members=["R45", "MDI"],
                ratios=[0.12, 0.04],
            )
        ],
        sum_to_total_groups=[
            SumToTotalGroup(
                group_id="Solid",
                members=["AP", "AL", "BIO"],
                maximum_total=0.86,
            )
        ],
        total_mass_fraction=1.0,
    )


def test_constraint_diagnostics_marks_infeasible_mass_balance() -> None:
    problem = _sample_problem()
    # Matches the reported transient iterate: sum = 0.82, while ratios remain satisfied.
    x = [0.084, 0.028, 0.35, 0.098, 0.26]
    diag = _constraint_diagnostics(x, problem, ["R45", "MDI", "AP", "AL", "BIO"], 1e-8)

    assert abs(diag["sum_x"] - 0.82) < 1e-12
    assert diag["is_feasible"] is False
    assert abs(diag["eq_residuals"]["mass_balance"] + 0.18) < 1e-12


def test_record_start_reports_infeasible_trace_points() -> None:
    payloads: list[dict] = []
    start_history_meta: dict[int, list[dict[str, object]]] = {}
    result = (
        13.081096,
        480000.0,
        245.0,
        1950.0,
        {"AP": 0.63, "AL": 0.16, "BIO": 0.21},
        [(0, 13.05), (1, 13.28), (2, 13.08)],
        [
            {"iter": 0, "log_fom": 13.05, "is_feasible": True},
            {"iter": 1, "log_fom": 13.28, "is_feasible": False},
            {"iter": 2, "log_fom": 13.08, "is_feasible": True},
        ],
    )

    output = _record_start(
        start_idx=9,
        result=result,
        best_log_fom=float("-inf"),
        best_fom=float("-inf"),
        best_isp=float("nan"),
        best_density=float("nan"),
        best_composition={},
        trial_history=[],
        start_history={},
        start_history_meta=start_history_meta,
        start_compositions={},
        completed=0,
        failed=0,
        n_starts=12,
        progress_callback=payloads.append,
    )

    assert output[6][9] == [(0, 13.05), (1, 13.28), (2, 13.08)]
    assert output[7][9][1]["is_feasible"] is False
    assert output[8][9] == {"AP": 0.63, "AL": 0.16, "BIO": 0.21}
    assert payloads[-1]["infeasible_trace_points"] == 1
    assert isinstance(payloads[-1]["start_trace_meta"], list)
