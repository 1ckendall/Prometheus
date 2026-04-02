"""Tests for EquilibriumProblem — declarative problem specification."""

import math

import numpy as np
import pytest

from prometheus_equilibrium.core.constants import UNIVERSAL_GAS_CONSTANT as R
from prometheus_equilibrium.equilibrium.problem import EquilibriumProblem, ProblemType
from prometheus_equilibrium.equilibrium.species import Species

# ---------------------------------------------------------------------------
# Mock species (concrete implementation of abstract Species)
# ---------------------------------------------------------------------------


class _MockSpecies(Species):
    """Minimal concrete Species for problem tests."""

    def __init__(self, elements, state="G", molar_mass_kg=0.002):
        super().__init__(elements=elements, state=state)
        self._M = molar_mass_kg

    def molar_mass(self) -> float:
        return self._M

    def specific_heat_capacity(self, T):
        return 30.0

    def enthalpy(self, T):
        return 30.0 * T

    def entropy(self, T):
        return 200.0


def _h2():
    return _MockSpecies({"H": 2}, molar_mass_kg=0.002016)


def _o2():
    return _MockSpecies({"O": 2}, molar_mass_kg=0.031998)


def _h2o():
    return _MockSpecies({"H": 2, "O": 1}, molar_mass_kg=0.018015)


def _h():
    return _MockSpecies({"H": 1}, molar_mass_kg=0.001008)


def _o():
    return _MockSpecies({"O": 1}, molar_mass_kg=0.015999)


# ---------------------------------------------------------------------------
# 1. Element abundances
# ---------------------------------------------------------------------------


def test_element_abundances_h2_o2():
    """1 mol H₂ + 1 mol O₂ → {H: 2, O: 2}."""
    h2, o2 = _h2(), _o2()
    prob = EquilibriumProblem(
        reactants={h2: 1.0, o2: 1.0},
        products=[_h2o(), _h2(), _o2(), _h(), _o()],
        problem_type=ProblemType.TP,
        constraint1=3000.0,
        constraint2=1e5,
    )
    b0 = prob.element_abundances()
    assert b0 == pytest.approx({"H": 2.0, "O": 2.0})


def test_element_abundances_stoich_h2_o2():
    """2 mol H₂ + 1 mol O₂ → {H: 4, O: 2} (stoichiometric combustion)."""
    h2, o2 = _h2(), _o2()
    prob = EquilibriumProblem(
        reactants={h2: 2.0, o2: 1.0},
        products=[_h2o(), _h2(), _o2(), _h(), _o()],
        problem_type=ProblemType.TP,
        constraint1=3000.0,
        constraint2=1e5,
    )
    b0 = prob.element_abundances()
    assert b0["H"] == pytest.approx(4.0)
    assert b0["O"] == pytest.approx(2.0)


def test_b0_array_alignment():
    """b0_array aligns correctly to a specified element order."""
    h2, o2 = _h2(), _o2()
    prob = EquilibriumProblem(
        reactants={h2: 1.0, o2: 1.0},
        products=[_h2o()],
        problem_type=ProblemType.TP,
        constraint1=3000.0,
        constraint2=1e5,
    )
    b0_HO = prob.b0_array(["H", "O"])
    assert b0_HO[0] == pytest.approx(2.0)  # H
    assert b0_HO[1] == pytest.approx(2.0)  # O

    b0_OH = prob.b0_array(["O", "H"])
    assert b0_OH[0] == pytest.approx(2.0)  # O
    assert b0_OH[1] == pytest.approx(2.0)  # H

    # Missing element → 0
    b0_HC = prob.b0_array(["H", "C"])
    assert b0_HC[0] == pytest.approx(2.0)
    assert b0_HC[1] == pytest.approx(0.0)  # no carbon


# ---------------------------------------------------------------------------
# 2. Validate
# ---------------------------------------------------------------------------


def test_validate_ok():
    """Valid problem raises no error."""
    prob = EquilibriumProblem(
        reactants={_h2(): 1.0, _o2(): 1.0},
        products=[_h2o(), _h2(), _o2(), _h(), _o()],
        problem_type=ProblemType.TP,
        constraint1=3000.0,
        constraint2=1e5,
    )
    prob.validate()  # must not raise


def test_validate_no_reactants():
    prob = EquilibriumProblem(
        reactants={},
        products=[_h2o()],
        problem_type=ProblemType.TP,
        constraint1=3000.0,
        constraint2=1e5,
    )
    with pytest.raises(ValueError, match="[Nn]o reactant"):
        prob.validate()


def test_validate_no_products():
    prob = EquilibriumProblem(
        reactants={_h2(): 1.0},
        products=[],
        problem_type=ProblemType.TP,
        constraint1=3000.0,
        constraint2=1e5,
    )
    with pytest.raises(ValueError, match="[Nn]o product"):
        prob.validate()


def test_validate_missing_element_in_products():
    """If reactant has element not in any product, validate raises."""
    h2 = _h2()  # has H
    co2 = _MockSpecies({"C": 1, "O": 2})  # has C, O but not H
    prob = EquilibriumProblem(
        reactants={h2: 1.0},
        products=[co2],
        problem_type=ProblemType.TP,
        constraint1=3000.0,
        constraint2=1e5,
    )
    with pytest.raises(ValueError, match="'H'"):
        prob.validate()


def test_validate_negative_abundance():
    h2 = _h2()
    prob = EquilibriumProblem(
        reactants={h2: -1.0},
        products=[h2],
        problem_type=ProblemType.TP,
        constraint1=3000.0,
        constraint2=1e5,
    )
    with pytest.raises(ValueError, match="[Nn]egative"):
        prob.validate()


def test_validate_tp_zero_temperature():
    prob = EquilibriumProblem(
        reactants={_h2(): 1.0},
        products=[_h2(), _h()],
        problem_type=ProblemType.TP,
        constraint1=0.0,  # T must be > 0
        constraint2=1e5,
    )
    with pytest.raises(ValueError):
        prob.validate()


# ---------------------------------------------------------------------------
# 3. initial_mixture
# ---------------------------------------------------------------------------


def test_initial_mixture_all_gas_equal():
    """Gas products get equal moles; condensed start at zero."""
    h2o_s = _MockSpecies({"H": 2, "O": 1}, state="S")  # condensed
    products = [_h2(), _o2(), _h2o(), h2o_s]

    prob = EquilibriumProblem(
        reactants={_h2(): 1.0, _o2(): 1.0},
        products=products,
        problem_type=ProblemType.TP,
        constraint1=3000.0,
        constraint2=1e5,
    )
    mix = prob.initial_mixture()

    gas_moles = mix.gas_moles()
    cnd_moles = mix.condensed_moles()

    # All gas moles equal
    assert np.all(gas_moles > 0)
    assert np.allclose(gas_moles, gas_moles[0])

    # All condensed moles zero
    assert np.all(cnd_moles == 0.0)


def test_initial_mixture_species_count():
    """Mixture contains all product species."""
    products = [_h2(), _o2(), _h2o(), _h(), _o()]
    prob = EquilibriumProblem(
        reactants={_h2(): 1.0, _o2(): 1.0},
        products=products,
        problem_type=ProblemType.TP,
        constraint1=3000.0,
        constraint2=1e5,
    )
    mix = prob.initial_mixture()
    assert mix.n_species == len(products)


# ---------------------------------------------------------------------------
# 4. from_mass_fractions factory
# ---------------------------------------------------------------------------


def test_from_mass_fractions_converts_correctly():
    """Moles = mass / molar_mass."""
    h2 = _h2()  # M = 0.002016 kg/mol
    o2 = _o2()  # M = 0.031998 kg/mol

    prob = EquilibriumProblem.from_mass_fractions(
        species_mass={h2: 0.002016, o2: 0.031998},  # 1 mol each
        products=[_h2o(), _h2(), _o2()],
        problem_type=ProblemType.TP,
        constraint1=3000.0,
        constraint2=1e5,
    )
    b0 = prob.element_abundances()
    assert b0["H"] == pytest.approx(2.0, rel=1e-5)
    assert b0["O"] == pytest.approx(2.0, rel=1e-5)


# ---------------------------------------------------------------------------
# 5. ProblemType properties
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "ptype,fixed_T,fixed_P",
    [
        (ProblemType.TP, True, True),
        (ProblemType.HP, False, True),
        (ProblemType.SP, False, True),
        (ProblemType.TV, True, False),
        (ProblemType.UV, False, False),
        (ProblemType.SV, False, False),
    ],
)
def test_problem_type_flags(ptype, fixed_T, fixed_P):
    assert ptype.fixed_temperature == fixed_T
    assert ptype.fixed_pressure == fixed_P


@pytest.mark.parametrize(
    "ptype,energy",
    [
        (ProblemType.TP, None),
        (ProblemType.HP, "H"),
        (ProblemType.SP, "S"),
        (ProblemType.UV, "U"),
        (ProblemType.SV, "S"),
    ],
)
def test_energy_constraint_field(ptype, energy):
    assert ptype.energy_constraint == energy
