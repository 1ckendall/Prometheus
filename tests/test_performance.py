"""Tests for PerformanceSolver paired shifting/frozen execution."""

from typing import Any

from prometheus.equilibrium.performance import (
    PerformanceSolver,
    RocketPerformanceComparison,
)


def test_solve_pair_calls_both_modes(monkeypatch):
    """solve_pair should execute shifting and frozen solves with shared inputs."""
    solver = PerformanceSolver()
    calls = []

    def _fake_solve(
        self,
        problem,
        pe_pa=None,
        area_ratio=None,
        shifting=True,
        ambient_pressure=101325.0,
    ):
        calls.append(
            {
                "problem": problem,
                "pe_pa": pe_pa,
                "area_ratio": area_ratio,
                "shifting": shifting,
                "ambient_pressure": ambient_pressure,
            }
        )
        return "shift" if shifting else "frozen"

    monkeypatch.setattr(PerformanceSolver, "solve", _fake_solve)

    token_problem: Any = object()
    result = solver.solve_pair(
        token_problem,
        pe_pa=120000.0,
        ambient_pressure=90000.0,
    )

    assert isinstance(result, RocketPerformanceComparison)
    assert result.shifting == "shift"
    assert result.frozen == "frozen"
    assert result.ambient_pressure == 90000.0

    assert len(calls) == 2
    assert calls[0]["problem"] is token_problem
    assert calls[0]["shifting"] is True
    assert calls[1]["shifting"] is False
    assert calls[0]["pe_pa"] == calls[1]["pe_pa"] == 120000.0
    assert calls[0]["ambient_pressure"] == calls[1]["ambient_pressure"] == 90000.0
