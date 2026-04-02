import json
from unittest.mock import patch

import numpy as np
import pytest

from prometheus_equilibrium.core.constants import UNIVERSAL_GAS_CONSTANT as R

# Adjust these imports to match your project's structure
from prometheus_equilibrium.equilibrium.species import (
    JANAF,
    Chemical,
    NASANineCoeff,
    NASASevenCoeff,
    SpeciesDatabase,
)

# ------------------------------------------------------------------
# 1. Base Class Tests
# ------------------------------------------------------------------


@patch(
    "prometheus_equilibrium.equilibrium.species.ELEMENTS_MOLAR_MASSES",
    {"C": 12.011, "O": 15.999},
)
def test_chemical_molar_mass():
    """Test that the base Chemical class correctly computes molar mass in kg/mol."""
    chem = Chemical(elements={"C": 1, "O": 2}, state="G")
    expected_mass = (12.011 + 2 * 15.999) / 1000.0  # kg/mol
    assert chem.molar_mass() == pytest.approx(expected_mass)


# ------------------------------------------------------------------
# 2. JANAF Interpolation Tests
# ------------------------------------------------------------------


@pytest.fixture
def janaf_species():
    """Provide a simple JANAF species with linear dummy data for testing."""
    return JANAF(
        elements={"C": 1},
        state="S",  # Condensed phase
        temperature=(200.0, 300.0, 400.0),
        specific_heat_capacity=(10.0, 20.0, 30.0),
        enthalpy=(100.0, 200.0, 300.0),
        entropy=(1.0, 2.0, 3.0),
        h_formation=5000.0,
    )


def test_janaf_condensed_flag(janaf_species):
    assert janaf_species.condensed == 1


def test_janaf_interpolation_scalars(janaf_species):
    """Test standard scalar interpolation and formation enthalpy offset."""
    T = 300.0
    assert janaf_species.specific_heat_capacity(T) == pytest.approx(20.0)
    assert janaf_species.entropy(T) == pytest.approx(2.0)
    # Enthalpy = sensible + formation
    assert janaf_species.enthalpy(T) == pytest.approx(200.0 + 5000.0)


def test_janaf_interpolation_arrays(janaf_species):
    """Ensure array inputs correctly return array outputs."""
    T_arr = np.array([200.0, 400.0])
    cp_arr = janaf_species.specific_heat_capacity(T_arr)
    np.testing.assert_allclose(cp_arr, [10.0, 30.0])


def test_derived_thermo_properties(janaf_species):
    """Test the base Species methods that compute reduced properties."""
    T = 300.0
    cp = 20.0
    h = 5200.0
    s = 2.0

    assert janaf_species.gibbs_free_energy(T) == pytest.approx(h - T * s)
    assert janaf_species.reduced_gibbs(T) == pytest.approx((h - T * s) / (R * T))
    assert janaf_species.reduced_enthalpy(T) == pytest.approx(h / (R * T))
    assert janaf_species.reduced_entropy(T) == pytest.approx(s / R)
    assert janaf_species.ratio_of_specific_heat_capacities(T) == pytest.approx(
        cp / (cp - R)
    )


def test_janaf_temperature_nudging():
    """Ensure non-strictly-increasing temperatures are corrected."""
    j = JANAF(
        {"C": 1},
        "G",
        temperature=(200.0, 300.0, 300.0),  # Duplicate T
        specific_heat_capacity=(1.0, 2.0, 3.0),
        enthalpy=(1.0, 2.0, 3.0),
        entropy=(1.0, 2.0, 3.0),
    )
    # The internal temperatures should be slightly offset to prevent Pchip errors
    temps = j._specific_heat_capacity_interpolator.x
    assert temps[2] > temps[1]
    assert np.all(np.diff(temps) > 0)


# ------------------------------------------------------------------
# 3. NASA-7 Polynomial Tests
# ------------------------------------------------------------------


@pytest.fixture
def nasa7_co2():
    """NASA-7 CO2 example taken from the module's __main__ block."""
    return NASASevenCoeff(
        elements={"C": 1, "O": 2},
        state="G",
        temperature=(200.0, 1000.0, 6000.0),
        coefficients=(
            # Low T
            (
                2.356813,
                0.0089841299,
                -7.1220632e-06,
                2.4573008e-09,
                -1.4288548e-13,
                -48371.971,
                9.9009035,
            ),
            # High T
            (
                4.6365111,
                0.0027414569,
                -9.9589759e-07,
                1.6038666e-10,
                -9.1619857e-15,
                -49024.904,
                -1.9348955,
            ),
        ),
    )


def test_nasa7_gas_flag(nasa7_co2):
    assert nasa7_co2.condensed == 0


def test_nasa7_piecewise_evaluation(nasa7_co2):
    """Verify evaluating a temperature array bridging the common T (1000 K)."""
    T_arr = np.array([300.0, 2000.0])

    # We aren't testing the exact thermo value accuracy here (that belongs to the database),
    # but rather that the piecewise logic fires and returns valid numeric shapes.
    cp = nasa7_co2.specific_heat_capacity(T_arr)
    h = nasa7_co2.enthalpy(T_arr)
    s = nasa7_co2.entropy(T_arr)

    assert cp.shape == (2,)
    assert h.shape == (2,)
    assert s.shape == (2,)
    assert not np.any(np.isnan(cp))


# ------------------------------------------------------------------
# 4. NASA-9 Polynomial Tests
# ------------------------------------------------------------------


@pytest.fixture
def nasa9_dummy():
    """A minimal NASA-9 species to verify interval selection and integration."""
    return NASANineCoeff(
        elements={"Ar": 1},
        state="G",
        temperatures=(200.0, 1000.0, 6000.0),
        exponents=(
            (-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0),
            (-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0),
        ),
        coefficients=(
            # a1..a7, b1, b2
            (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0),
            (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0),
        ),
    )


def test_nasa9_dispatch(nasa9_dummy):
    """Test evaluation across intervals and out-of-bounds handling."""
    # 500 is in range 0, 3000 is in range 1, 10000 is out of bounds
    T_arr = np.array([500.0, 3000.0, 10000.0])
    cp = nasa9_dummy.specific_heat_capacity(T_arr)

    assert not np.isnan(cp[0])
    assert not np.isnan(cp[1])
    assert np.isnan(cp[2])  # Should return NaN outside defined intervals


def test_nasa9_scalar_vs_array(nasa9_dummy):
    """Test that passing a scalar returns a scalar float, not a 0-d array."""
    T = 500.0
    cp_scalar = nasa9_dummy.specific_heat_capacity(T)
    assert isinstance(cp_scalar, float)


# ------------------------------------------------------------------
# 5. SpeciesDatabase Tests
# ------------------------------------------------------------------


def test_database_element_filtering():
    """Verify that get_species correctly identifies subset elements, ignoring charge."""
    db = SpeciesDatabase("dummy", "dummy", "dummy")

    # Populate the database manually to isolate the filtering logic from the file I/O
    # Added 6000 to the temperature tuple to satisfy (T_low, T_common, T_high)
    sp_h2 = NASASevenCoeff({"H": 2}, "G", (200, 1000, 6000), ((0,) * 7, (0,) * 7))
    sp_h2o = NASASevenCoeff(
        {"H": 2, "O": 1}, "G", (200, 1000, 6000), ((0,) * 7, (0,) * 7)
    )
    sp_co2 = NASASevenCoeff(
        {"C": 1, "O": 2}, "G", (200, 1000, 6000), ((0,) * 7, (0,) * 7)
    )
    sp_ion = NASASevenCoeff(
        {"H": 1, "e-": 1}, "G", (200, 1000, 6000), ((0,) * 7, (0,) * 7)
    )

    db._all_species = [sp_h2, sp_h2o, sp_co2, sp_ion]
    db.species = {"H2": sp_h2, "H2O": sp_h2o, "CO2": sp_co2, "H+": sp_ion}

    result = db.get_species({"H", "O"})

    assert sp_h2 in result
    assert sp_h2o in result
    assert sp_ion in result
    assert sp_co2 not in result


def _build_overlap_species():
    """Create overlapping species records from three database source types."""
    sp_nasa9 = NASANineCoeff(
        elements={"H": 2, "O": 1},
        state="G",
        temperatures=(200.0, 1000.0, 6000.0),
        exponents=((0.0, 1.0, 2.0), (0.0, 1.0, 2.0)),
        coefficients=((1.0, 1.0, 1.0, 0.0, 0.0), (1.0, 1.0, 1.0, 0.0, 0.0)),
    )
    sp_nasa7 = NASASevenCoeff(
        elements={"H": 2, "O": 1},
        state="G",
        temperature=(200.0, 1000.0, 6000.0),
        coefficients=((1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),) * 2,
    )
    sp_janaf = JANAF(
        elements={"H": 2, "O": 1},
        state="G",
        temperature=(298.15, 1000.0, 2000.0),
        specific_heat_capacity=(30.0, 35.0, 40.0),
        enthalpy=(0.0, 10.0, 20.0),
        entropy=(100.0, 110.0, 120.0),
    )
    return sp_nasa9, sp_nasa7, sp_janaf


def test_database_default_priority_prefers_nasa9():
    """Default source priority should prefer NASA-9 over NASA-7 and JANAF."""
    db = SpeciesDatabase("dummy", "dummy", "dummy")
    sp_nasa9, sp_nasa7, sp_janaf = _build_overlap_species()

    deduped = db._deduplicate([sp_janaf, sp_nasa7, sp_nasa9])
    assert deduped["H2O_G"] is sp_nasa9


def test_database_priority_override_prefers_janaf():
    """User-defined source priority should override default deduplication order."""
    db = SpeciesDatabase(
        "dummy",
        "dummy",
        "dummy",
        source_priority=["JANAF", "NASA-9", "NASA-7", "TERRA", "AFCESIC"],
    )
    sp_nasa9, sp_nasa7, sp_janaf = _build_overlap_species()

    deduped = db._deduplicate([sp_nasa7, sp_nasa9, sp_janaf])
    assert deduped["H2O_G"] is sp_janaf


def test_database_default_paths_are_populated():
    """SpeciesDatabase should populate built-in thermo file paths by default."""
    db = SpeciesDatabase()

    assert db.nasa7_path.endswith("nasa7.json")
    assert db.nasa9_path.endswith("nasa9.json")
    assert db.terra_path.endswith("terra.json")


def test_database_load_defaults_and_overrides(monkeypatch):
    """Default load enables NASA-7/NASA-9/TERRA while allowing explicit overrides."""
    db = SpeciesDatabase()
    calls = []

    monkeypatch.setattr(db, "_load_nasa7", lambda: calls.append("nasa7"))
    monkeypatch.setattr(db, "_load_nasa9", lambda: calls.append("nasa9"))
    monkeypatch.setattr(db, "_load_terra", lambda: calls.append("terra"))
    monkeypatch.setattr(db, "_load_afcesic", lambda: calls.append("afcesic"))
    monkeypatch.setattr(db, "_load_janaf", lambda: calls.append("janaf"))
    monkeypatch.setattr(db, "_load_shomate", lambda: calls.append("shomate"))

    db.load()
    assert calls == ["terra", "nasa7", "nasa9"]

    calls.clear()
    db.load(include_terra=False, include_janaf=True, include_shomate=True)
    assert calls == ["nasa7", "nasa9", "janaf", "shomate"]
