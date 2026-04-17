"""Tests for optimizer composition evaluation mode selection."""

from __future__ import annotations

from prometheus_equilibrium.optimization import _eval
from prometheus_equilibrium.optimization.problem import ObjectiveSpec, OperatingPoint


class _FakeEquilibriumProblem:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def validate(self) -> None:
        return


class _DummyMixture:
    def __init__(self):
        self.elements = {"H": 2.0, "O": 1.0}
        self.reactants = [object()]
        self.enthalpy = -1.0e5


class _DummyPropDB:
    def mix(self, _components):
        return _DummyMixture()

    def find_ingredient(self, _ingredient_id):
        return {"density": 1800.0}


class _DummySpecDB:
    def get_species(self, _elements, max_atoms, enabled_databases):
        assert max_atoms > 0
        assert enabled_databases
        return [object()]


class _DummyPerfResult:
    isp_actual = 230.0
    isp_vac = 240.0
    isp_sl = 220.0


class _DummyPerfSolver:
    def __init__(self):
        self.calls = []

    def solve(self, _problem, **kwargs):
        self.calls.append(kwargs)
        return _DummyPerfResult()

    def solve_pair(self, *_args, **_kwargs):
        raise AssertionError("Optimizer path must not call solve_pair.")


def _run_eval(monkeypatch, *, shifting: bool, expansion_type: str) -> _DummyPerfSolver:
    monkeypatch.setattr(_eval, "EquilibriumProblem", _FakeEquilibriumProblem)

    perf_solver = _DummyPerfSolver()
    operating_point = OperatingPoint(
        chamber_pressure_pa=6.0e6,
        expansion_type=expansion_type,
        expansion_value=0.101325e6 if expansion_type == "pressure" else 10.0,
        ambient_pressure_pa=101325.0,
        shifting=shifting,
    )
    objective = ObjectiveSpec(isp_variant="isp_actual", rho_exponent=0.25)

    _eval.evaluate_composition(
        {"A": 0.5, "B": 0.5},
        prop_db=_DummyPropDB(),
        spec_db=_DummySpecDB(),
        perf_solver=perf_solver,
        enabled_databases=["NASA-9"],
        max_atoms=10,
        operating_point=operating_point,
        objective=objective,
    )
    return perf_solver


def test_evaluate_composition_uses_selected_frozen_mode(monkeypatch) -> None:
    perf_solver = _run_eval(monkeypatch, shifting=False, expansion_type="pressure")

    assert len(perf_solver.calls) == 1
    assert perf_solver.calls[0]["shifting"] is False
    assert "pe_pa" in perf_solver.calls[0]
    assert "area_ratio" not in perf_solver.calls[0]


def test_evaluate_composition_uses_selected_shifting_mode(monkeypatch) -> None:
    perf_solver = _run_eval(monkeypatch, shifting=True, expansion_type="area_ratio")

    assert len(perf_solver.calls) == 1
    assert perf_solver.calls[0]["shifting"] is True
    assert "area_ratio" in perf_solver.calls[0]
    assert "pe_pa" not in perf_solver.calls[0]
