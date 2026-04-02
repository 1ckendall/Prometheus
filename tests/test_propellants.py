"""
Unit tests for Prometheus.propellants.

Tests are self-contained: they do NOT load actual database files.
The SpeciesDatabase is provided as a minimal stub where needed.
"""

import math
import tomllib
from io import BytesIO
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from prometheus.core.constants import UNIVERSAL_GAS_CONSTANT as R
from prometheus.propellants import (PropellantDatabase, PropellantMixture,
                                 SyntheticSpecies)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

MINIMAL_TOML = b"""
schema_version = "1"

[[ingredient]]
id        = "FUEL"
name      = "Test Fuel"
phase     = "L"
roles     = ["fuel"]
elements   = {C = 1, H = 4}
molar_mass = 16.043
dHf298     = -74873.0
cp         = 35.7
density    = 425.0
t_supply   = 298.15

[[ingredient]]
id        = "OX"
name      = "Test Oxidizer"
phase     = "G"
roles     = ["oxidizer"]
elements   = {O = 2}
molar_mass = 31.999
dHf298     = 0.0
cp         = 29.4
density    = 1.429
t_supply   = 298.15

[[formulation]]
id   = "FUEL_OX_2to1"
name = "Fuel/Oxidizer 2:1 by mass"
type = "bipropellant_liquid"
components = [
    {ingredient = "FUEL", mass_fraction = 0.6667},
    {ingredient = "OX",   mass_fraction = 0.3333},
]
"""


@pytest.fixture
def prop_db(tmp_path):
    """PropellantDatabase loaded from minimal TOML (no real SpeciesDatabase needed)."""
    toml_file = tmp_path / "propellants.toml"
    toml_file.write_bytes(MINIMAL_TOML)
    db = PropellantDatabase(str(toml_file))
    db.load()
    return db


# ---------------------------------------------------------------------------
# SyntheticSpecies tests
# ---------------------------------------------------------------------------


@pytest.fixture
def methane_synth():
    """SyntheticSpecies representing CH4 (constant-Cp approximation)."""
    return SyntheticSpecies(
        elements={"C": 1, "H": 4},
        state="G",
        dHf298=-74873.0,
        cp=35.7,
        molar_mass_g_mol=16.043,
        alias="CH4 (synthetic)",
    )


def test_synthetic_species_molar_mass(methane_synth):
    """Stored molar mass [kg/mol] is used, not recomputed from elements."""
    assert methane_synth.molar_mass() == pytest.approx(0.016043)


def test_synthetic_species_condensed_flag(methane_synth):
    assert methane_synth.condensed == 0  # gaseous


def test_synthetic_species_enthalpy_at_ref(methane_synth):
    """H°(298.15 K) must equal dHf298 exactly."""
    assert methane_synth.enthalpy(298.15) == pytest.approx(-74873.0)


def test_synthetic_species_enthalpy_above_ref(methane_synth):
    """H°(T) = dHf298 + cp*(T - 298.15)."""
    T = 500.0
    expected = -74873.0 + 35.7 * (500.0 - 298.15)
    assert methane_synth.enthalpy(T) == pytest.approx(expected)


def test_synthetic_species_cp_scalar(methane_synth):
    assert methane_synth.specific_heat_capacity(1000.0) == pytest.approx(35.7)


def test_synthetic_species_cp_array(methane_synth):
    T = np.array([300.0, 500.0, 1000.0])
    cp = methane_synth.specific_heat_capacity(T)
    np.testing.assert_allclose(cp, 35.7)


def test_synthetic_species_enthalpy_array(methane_synth):
    T = np.array([298.15, 500.0])
    H = methane_synth.enthalpy(T)
    assert H[0] == pytest.approx(-74873.0)
    assert H[1] == pytest.approx(-74873.0 + 35.7 * (500.0 - 298.15))


def test_synthetic_species_entropy_at_ref(methane_synth):
    """S°(298.15 K) = 0 by the S298=0 approximation."""
    assert methane_synth.entropy(298.15) == pytest.approx(0.0, abs=1e-10)


def test_synthetic_species_entropy_above_ref(methane_synth):
    T = 500.0
    expected = 35.7 * math.log(500.0 / 298.15)
    assert methane_synth.entropy(T) == pytest.approx(expected)


def test_synthetic_species_reduced_gibbs_scalar(methane_synth):
    T = 1000.0
    H = methane_synth.enthalpy(T)
    S = methane_synth.entropy(T)
    expected = (H - S * T) / (R * T)
    assert methane_synth.reduced_gibbs(T) == pytest.approx(expected)


def test_synthetic_species_reduced_enthalpy_scalar(methane_synth):
    T = 800.0
    expected = methane_synth.enthalpy(T) / (R * T)
    assert methane_synth.reduced_enthalpy(T) == pytest.approx(expected)


def test_synthetic_species_gibbs_free_energy(methane_synth):
    """gibbs_free_energy (inherited) = H - S*T."""
    T = 600.0
    expected = methane_synth.enthalpy(T) - methane_synth.entropy(T) * T
    assert methane_synth.gibbs_free_energy(T) == pytest.approx(expected)


def test_synthetic_species_zero_cp():
    """With cp=0, enthalpy is constant = dHf298 at all temperatures."""
    sp = SyntheticSpecies(
        elements={"Al": 1}, state="S", dHf298=-1000.0, cp=0.0, molar_mass_g_mol=26.98
    )
    assert sp.enthalpy(500.0) == pytest.approx(-1000.0)
    assert sp.enthalpy(1000.0) == pytest.approx(-1000.0)


def test_synthetic_species_condensed():
    sp = SyntheticSpecies(
        elements={"Al": 2, "O": 3},
        state="S",
        dHf298=-1676000.0,
        cp=79.0,
        molar_mass_g_mol=101.96,
    )
    assert sp.condensed == 1


# ---------------------------------------------------------------------------
# PropellantDatabase — loading
# ---------------------------------------------------------------------------


def test_load_ingredients(prop_db):
    assert "FUEL" in prop_db.ingredient_ids
    assert "OX" in prop_db.ingredient_ids


def test_load_formulations(prop_db):
    assert "FUEL_OX_2to1" in prop_db.formulation_ids


def test_find_ingredient(prop_db):
    rec = prop_db.find_ingredient("FUEL")
    assert rec["name"] == "Test Fuel"
    assert "_species" in rec
    assert isinstance(rec["_species"], SyntheticSpecies)


def test_find_ingredient_missing(prop_db):
    with pytest.raises(KeyError, match="NOEXIST"):
        prop_db.find_ingredient("NOEXIST")


def test_find_formulation(prop_db):
    rec = prop_db.find_formulation("FUEL_OX_2to1")
    assert rec["name"] == "Fuel/Oxidizer 2:1 by mass"


def test_find_formulation_missing(prop_db):
    with pytest.raises(KeyError, match="NOFORM"):
        prop_db.find_formulation("NOFORM")


def test_ingredient_species_type(prop_db):
    rec = prop_db.find_ingredient("FUEL")
    assert isinstance(rec["_species"], SyntheticSpecies)
    assert rec["_species"].dHf298 == pytest.approx(-74873.0)


# ---------------------------------------------------------------------------
# PropellantDatabase — mix()
# ---------------------------------------------------------------------------


def test_mix_returns_mixture(prop_db):
    m = prop_db.mix([("FUEL", 1.0), ("OX", 2.0)])
    assert isinstance(m, PropellantMixture)


def test_mix_elements(prop_db):
    m = prop_db.mix([("FUEL", 1.0), ("OX", 2.0)])
    assert "C" in m.elements
    assert "H" in m.elements
    assert "O" in m.elements


def test_mix_moles_per_kg(prop_db):
    """Verify moles are correct: mass_frac / (molar_mass [kg/mol])."""
    # FUEL: 1/(1+2) = 0.3333 kg fraction, MW = 0.016043 kg/mol → 20.78 mol/kg
    # OX:   2/(1+2) = 0.6667 kg fraction, MW = 0.031999 kg/mol → 20.83 mol/kg
    m = prop_db.mix([("FUEL", 1.0), ("OX", 2.0)])
    total_moles = sum(m.reactants.values())
    # Total per kg should be ~41.6 mol
    assert total_moles == pytest.approx(1 / 3 / 0.016043 + 2 / 3 / 0.031999, rel=1e-4)


def test_mix_enthalpy(prop_db):
    """Enthalpy = sum(n_i * H_i(t_supply_i)) per kg mixture."""
    m = prop_db.mix([("FUEL", 1.0), ("OX", 2.0)])
    # Both t_supply = 298.15, so H_i = dHf298_i (constant Cp; no sensible correction)
    n_fuel = (1.0 / 3.0) / 0.016043
    n_ox = (2.0 / 3.0) / 0.031999
    expected = n_fuel * (-74873.0) + n_ox * 0.0
    assert m.enthalpy == pytest.approx(expected, rel=1e-4)


def test_mix_single_ingredient(prop_db):
    """Single-ingredient mix normalises to 1 kg."""
    m = prop_db.mix([("FUEL", 5.0)])
    assert len(m.reactants) == 1
    sp = list(m.reactants)[0]
    assert m.reactants[sp] == pytest.approx(1.0 / sp.molar_mass())


def test_mix_missing_ingredient(prop_db):
    with pytest.raises(KeyError, match="NOEXIST"):
        prop_db.mix([("FUEL", 1.0), ("NOEXIST", 1.0)])


def test_mix_zero_total_mass(prop_db):
    with pytest.raises(ValueError, match="positive"):
        prop_db.mix([("FUEL", 0.0), ("OX", 0.0)])


# ---------------------------------------------------------------------------
# PropellantDatabase — expand()
# ---------------------------------------------------------------------------


def test_expand_returns_mixture(prop_db):
    m = prop_db.expand("FUEL_OX_2to1")
    assert isinstance(m, PropellantMixture)


def test_expand_elements(prop_db):
    m = prop_db.expand("FUEL_OX_2to1")
    assert "C" in m.elements
    assert "H" in m.elements
    assert "O" in m.elements


def test_expand_missing(prop_db):
    with pytest.raises(KeyError, match="NOFORM"):
        prop_db.expand("NOFORM")


def test_expand_vs_mix_equivalent(prop_db):
    """expand('FUEL_OX_2to1') must match mix([('FUEL', 0.6667), ('OX', 0.3333)])."""
    m_expand = prop_db.expand("FUEL_OX_2to1")
    m_mix = prop_db.mix([("FUEL", 0.6667), ("OX", 0.3333)])
    assert m_expand.enthalpy == pytest.approx(m_mix.enthalpy, rel=1e-4)
    assert m_expand.elements == m_mix.elements


# ---------------------------------------------------------------------------
# PropellantDatabase — thermo_id resolution
# ---------------------------------------------------------------------------


def test_thermo_id_resolution(tmp_path):
    """Ingredient with thermo_id is resolved from SpeciesDatabase."""
    toml_bytes = b"""
schema_version = "1"
[[ingredient]]
id        = "LOX"
phase     = "L"
roles     = ["oxidizer"]
thermo_id = "O2_L"
t_supply  = 90.2
"""
    toml_file = tmp_path / "p.toml"
    toml_file.write_bytes(toml_bytes)

    # Create a minimal mock SpeciesDatabase
    mock_sp = MagicMock()
    mock_sp.elements = {"O": 2}
    mock_sp.condensed = 0
    mock_sp.molar_mass.return_value = 0.031999
    mock_sp.enthalpy.return_value = -12400.0  # J/mol at 90.2 K

    mock_sdb = MagicMock(spec=["species", "find"])
    mock_sdb.species = {"O2_L": mock_sp}
    mock_sdb.find.return_value = mock_sp

    db = PropellantDatabase(str(toml_file), species_db=mock_sdb)
    db.load()

    rec = db.find_ingredient("LOX")
    assert rec["_species"] is mock_sp


def test_thermo_id_missing_key(tmp_path):
    """Missing thermo_id key in SpeciesDatabase raises KeyError."""
    toml_bytes = b"""
schema_version = "1"
[[ingredient]]
id        = "X"
phase     = "G"
thermo_id = "NOTEXIST_G"
"""
    toml_file = tmp_path / "p.toml"
    toml_file.write_bytes(toml_bytes)

    mock_sdb = MagicMock(spec=["species", "find"])
    mock_sdb.species = {}
    mock_sdb.find.side_effect = KeyError("NOTEXIST_G")

    db = PropellantDatabase(str(toml_file), species_db=mock_sdb)
    with pytest.raises(KeyError, match="NOTEXIST_G"):
        db.load()


def test_thermo_id_no_species_db(tmp_path):
    """thermo_id ingredient without SpeciesDatabase raises RuntimeError."""
    toml_bytes = b"""
schema_version = "1"
[[ingredient]]
id        = "X"
phase     = "G"
thermo_id = "H2_G"
"""
    toml_file = tmp_path / "p.toml"
    toml_file.write_bytes(toml_bytes)

    db = PropellantDatabase(str(toml_file))  # no species_db
    with pytest.raises(RuntimeError, match="SpeciesDatabase"):
        db.load()


# ---------------------------------------------------------------------------
# PropellantDatabase — inline ingredient validation
# ---------------------------------------------------------------------------


def test_inline_missing_dHf298(tmp_path):
    """Inline ingredient without dHf298 raises ValueError."""
    toml_bytes = b"""
schema_version = "1"
[[ingredient]]
id         = "BAD"
phase      = "S"
elements   = {Al = 1}
molar_mass = 26.98
"""
    toml_file = tmp_path / "p.toml"
    toml_file.write_bytes(toml_bytes)
    db = PropellantDatabase(str(toml_file))
    with pytest.raises(ValueError, match="dHf298"):
        db.load()


# ---------------------------------------------------------------------------
# PropellantMixture — integration with EquilibriumProblem (smoke test)
# ---------------------------------------------------------------------------


def test_mixture_fields(prop_db):
    m = prop_db.mix([("FUEL", 1.0), ("OX", 4.0)])
    assert isinstance(m.reactants, dict)
    assert isinstance(m.enthalpy, float)
    assert isinstance(m.elements, frozenset)
    assert len(m.reactants) == 2
    assert all(v > 0 for v in m.reactants.values())
