import numpy as np
import pytest

from prometheus_equilibrium.core.constants import REFERENCE_PRESSURE as P_REF
from prometheus_equilibrium.core.constants import UNIVERSAL_GAS_CONSTANT as R
from prometheus_equilibrium.equilibrium.mixture import Mixture
from prometheus_equilibrium.equilibrium.species import Species


class MockSpecies(Species):
    """A minimal Species implementation with hardcoded thermodynamic values for testing."""

    def __init__(
        self, name: str, state: str, molar_mass: float, cp: float, h: float, s: float
    ):
        super().__init__(elements={"X": 1}, state=state)
        self.name = name
        self._mock_molar_mass = molar_mass
        self._mock_cp = cp
        self._mock_h = h
        self._mock_s = s

    def molar_mass(self) -> float:
        return self._mock_molar_mass

    def specific_heat_capacity(self, T: float) -> float:
        return self._mock_cp

    def enthalpy(self, T: float) -> float:
        return self._mock_h

    def entropy(self, T: float) -> float:
        return self._mock_s

    def __repr__(self):
        return f"MockSpecies({self.name}, {self.state})"


@pytest.fixture
def mixture_env():
    """Fixture providing a configured Mixture, Temperature, and standard species."""
    gas1 = MockSpecies("GasA", "G", molar_mass=0.020, cp=20.0, h=1000.0, s=100.0)
    gas2 = MockSpecies("GasB", "G", molar_mass=0.040, cp=25.0, h=1500.0, s=150.0)
    cond1 = MockSpecies("SolidC", "S", molar_mass=0.050, cp=30.0, h=500.0, s=50.0)

    # Intentionally pass them out of order to test sorting.
    mix = Mixture(species=[cond1, gas1, gas2], moles=np.array([1.0, 2.0, 3.0]))
    T = 298.15

    return {"mix": mix, "T": T, "gas1": gas1, "gas2": gas2, "cond1": cond1}


def test_ordering(mixture_env):
    """Test that gas species are ordered before condensed species."""
    mix = mixture_env["mix"]
    assert mix.species == [
        mixture_env["gas1"],
        mixture_env["gas2"],
        mixture_env["cond1"],
    ]
    # Verify moles are reordered accordingly (GasA: 2.0, GasB: 3.0, SolidC: 1.0)
    np.testing.assert_array_equal(mix.moles, np.array([2.0, 3.0, 1.0]))


def test_mole_counts(mixture_env):
    """Test total moles and gas moles calculations."""
    mix = mixture_env["mix"]
    assert mix.total_moles == 6.0
    assert mix.total_gas_moles == 5.0


def test_fractions(mixture_env):
    """Test mole and mass fractions."""
    mix = mixture_env["mix"]

    # Mole fractions: 2/6, 3/6, 1/6
    expected_mole_fracs = np.array([2 / 6, 3 / 6, 1 / 6])
    np.testing.assert_allclose(mix.mole_fractions, expected_mole_fracs)

    # Mass calculations
    # GasA: 2.0 * 0.020 = 0.040 kg
    # GasB: 3.0 * 0.040 = 0.120 kg
    # SolidC: 1.0 * 0.050 = 0.050 kg
    # Total mass = 0.210 kg
    expected_mass_fracs = np.array([0.040 / 0.210, 0.120 / 0.210, 0.050 / 0.210])
    np.testing.assert_allclose(mix.mass_fractions, expected_mass_fracs)


def test_mean_molar_mass(mixture_env):
    """Test mixture molar mass."""
    mix = mixture_env["mix"]
    # M_mean = Total Mass / Total Moles = 0.210 / 6.0 = 0.035 kg/mol
    assert mix.mean_molar_mass == pytest.approx(0.035)


def test_total_extensive_properties(mixture_env):
    """Test total mixture Cp and Enthalpy."""
    mix, T = mixture_env["mix"], mixture_env["T"]

    # Total Cp = (2.0 * 20.0) + (3.0 * 25.0) + (1.0 * 30.0) = 40 + 75 + 30 = 145.0
    assert mix.total_cp(T) == pytest.approx(145.0)

    # Total Enthalpy = (2.0 * 1000.0) + (3.0 * 1500.0) + (1.0 * 500.0) = 2000 + 4500 + 500 = 7000.0
    assert mix.total_enthalpy(T) == pytest.approx(7000.0)


def test_entropy_mixing(mixture_env):
    """Test entropy calculations, specifically the ideal gas mixing terms."""
    mix, T = mixture_env["mix"], mixture_env["T"]

    # GasA (2 moles): S_std = 100.0, x_i (gas) = 2/5
    s_gas1 = 2.0 * (100.0 - R * np.log(2 / 5) - R * np.log(1.0))  # P/P_REF = 1.0

    # GasB (3 moles): S_std = 150.0, x_i (gas) = 3/5
    s_gas2 = 3.0 * (150.0 - R * np.log(3 / 5) - R * np.log(1.0))

    # SolidC (1 mole): S_std = 50.0 (No mixing term)
    s_cond1 = 1.0 * 50.0

    expected_total_s = s_gas1 + s_gas2 + s_cond1
    assert mix.total_entropy(T, P_REF) == pytest.approx(expected_total_s)

    # Mixture molar entropy
    expected_molar_s = expected_total_s / 6.0
    assert mix.entropy(T, P_REF) == pytest.approx(expected_molar_s)


def test_molar_properties(mixture_env):
    """Test the intensive/molar properties of the mixture."""
    mix, T = mixture_env["mix"], mixture_env["T"]

    assert mix.cp(T) == pytest.approx(145.0 / 6.0)
    assert mix.enthalpy(T) == pytest.approx(7000.0 / 6.0)

    expected_g = (7000.0 / 6.0) - T * mix.entropy(T, P_REF)
    assert mix.gibbs(T, P_REF) == pytest.approx(expected_g)


def test_log_moles(mixture_env):
    """Test logarithmic mole conversion and trace species handling."""
    mix = mixture_env["mix"]

    log_m = mix.log_moles()
    np.testing.assert_allclose(log_m, np.log([2.0, 3.0, 1.0]))

    # Test trace/inactive species
    mix.moles = np.array([0.0, 3.0, -1.0])
    log_m_trace = mix.log_moles()
    assert np.isneginf(log_m_trace[0])
    assert log_m_trace[1] == pytest.approx(np.log(3.0))
    assert np.isneginf(log_m_trace[2])


def test_set_from_log_moles(mixture_env):
    """Test updating moles from log values."""
    mix = mixture_env["mix"]
    ln_n = np.array([np.log(4.0), np.log(5.0), np.log(2.0)])

    mix.set_from_log_moles(ln_n)
    np.testing.assert_allclose(mix.moles, [4.0, 5.0, 2.0])
