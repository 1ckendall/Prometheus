# Prometheus: An open-source combustion equilibrium solver in Python.
# Copyright (C) 2026 Charles Kendall
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>

import csv
import json
import logging
import math
from abc import abstractmethod
from typing import Dict, List, Literal, Optional, Sequence, Union

import numpy as np
from scipy.interpolate import PchipInterpolator

from prometheus.core.constants import ELEMENTS_MOLAR_MASSES
from prometheus.core.constants import UNIVERSAL_GAS_CONSTANT as R

try:
    from prometheus.equilibrium._thermo_kernels import (_cp_afcesic, _cp_n7,
                                                     _cp_n9, _cp_shomate,
                                                     _enthalpy_afcesic,
                                                     _enthalpy_n7,
                                                     _enthalpy_n9,
                                                     _enthalpy_shomate,
                                                     _gibbs_afcesic, _gibbs_n7,
                                                     _gibbs_n9, _gibbs_shomate)

    _NUMBA_AVAILABLE = True
except ImportError:
    _NUMBA_AVAILABLE = False

# Fast scalar check — avoids the overhead of np.ndim() in the hot path
_SCALAR_TYPES = (int, float, np.floating, np.integer)


class Chemical:
    def __init__(
        self,
        elements: dict[str:float],
        state: Literal["S", "L", "G"],
        phase: Optional[str] = None,
    ):
        self.elements = elements
        self.state = state
        self.phase = phase

    def molar_mass(self) -> float:
        """Return the molar mass of the substance in kg"""
        return (
            np.sum(
                np.fromiter(
                    [
                        ELEMENTS_MOLAR_MASSES[element] * quantity
                        for element, quantity in self.elements.items()
                    ],
                    dtype=float,
                )
            )
            / 1000
        )


class Species(Chemical):
    def __init__(
        self, elements: dict, state: Literal["S", "L", "G"], phase: Optional[str] = None
    ):
        super().__init__(elements, state, phase)
        self.condensed = 0 if self.state == "G" else 1

    @property
    def source(self) -> str:
        """Returns a clean string representation of the database source."""
        # Use explicit source attribution if set (standardized in loaders)
        attribution = getattr(self, "source_attribution", None)
        if attribution:
            return attribution

        # Fallback to class-based mapping for legacy or direct instantiation
        class_to_source = {
            "JANAF": "JANAF",
            "NASASevenCoeff": "NASA-7",
            "NASANineCoeff": "NASA-9",
            "ShomateCoeff": "NASA-9",
            "AFCESICCoeff": "AFCESIC",
            "TERRACoeff": "TERRA",
        }
        return class_to_source.get(self.__class__.__name__, self.__class__.__name__)

    @property
    def formula(self) -> str:
        """Helper to print formula in Hill order (C first, H second, then alphabetical)."""

        # Hill sort logic
        def hill_key(item):
            el = item[0]
            if el == "C":
                return (0, el)
            if el == "H":
                return (1, el)
            return (2, el)

        sorted_elements = sorted(
            [(el, n) for el, n in self.elements.items() if el != "e-"], key=hill_key
        )

        parts = []
        for el, n in sorted_elements:
            if n == 1.0:
                parts.append(el)
            else:
                num = int(n) if float(n).is_integer() else n
                parts.append(f"{el}{num}")

        base_formula = "".join(parts)

        # Handle charge
        charge = 0
        if "e-" in self.elements:
            charge = -int(round(self.elements["e-"]))

        if charge == 0:
            return base_formula
        elif charge == 1:
            return f"{base_formula}+"
        elif charge == -1:
            return f"{base_formula}-"
        elif charge > 1:
            return f"{base_formula}{charge}+"
        else:
            return f"{base_formula}{abs(charge)}-"

    def __str__(self) -> str:
        """User-friendly string representation (used by print)."""
        return f"{self.formula} ({self.state}) - Type: {self.source}"

    def __repr__(self) -> str:
        """Developer-friendly string representation (used in lists and interactive prompts)."""
        return f"<{self.__class__.__name__}: {self.formula}_{self.state}>"

    @abstractmethod
    def specific_heat_capacity(
        self, T: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        """Returns the specific heat capacity in J/mol K"""
        ...

    @abstractmethod
    def enthalpy(self, T: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Computes the sensible enthalpy in J/mol"""
        ...

    @abstractmethod
    def entropy(self, T: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Computes entropy in J/mol K"""
        ...

    def gibbs_free_energy(
        self, T: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        return self.enthalpy(T) - self.entropy(T) * T

    def reduced_gibbs(self, T: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Dimensionless standard Gibbs free energy G°/(RT) = H°/(RT) − S°/R.

        This is the reduced chemical potential μⱼ°/(RT) used in the
        equilibrium condition (Gordon-McBride, RP-1311 §2):

            μⱼ/RT = μⱼ°/RT + ln(nⱼ/n) + ln(P/P°)   [gas phase]

        At equilibrium μⱼ/RT = Σₖ πₖ·A[j,k], where πₖ are the modified
        Lagrange multipliers and A[j,k] is the stoichiometric coefficient of
        element k in species j.
        """
        T_arr = np.asanyarray(T, dtype=float)
        return self.gibbs_free_energy(T_arr) / (R * T_arr)

    def reduced_enthalpy(self, T: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Dimensionless standard enthalpy H°/(RT).

        Appears in the energy-constraint row of the Newton Jacobian (HP and
        UV problems) and in the definition of reduced_gibbs.
        """
        T_arr = np.asanyarray(T, dtype=float)
        return self.enthalpy(T_arr) / (R * T_arr)

    def reduced_entropy(self, T: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Dimensionless standard entropy S°/R.

        Appears in the entropy-constraint row of the Newton Jacobian (SP and
        SV problems) and in the definition of reduced_gibbs.
        """
        return self.entropy(T) / R

    def ratio_of_specific_heat_capacities(
        self, T: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        """Returns the ratio of molar specific heats cp / cv = gamma"""
        return self.specific_heat_capacity(T) / (self.specific_heat_capacity(T) - R)


class JANAF(Species):
    #     VALIDPHASETYPES = [
    #         "cr",
    #         "l",
    #         "cr,l",
    #         "g",
    #         "ref",
    #         "cd",
    #         "fl",
    #         "am",
    #         "vit",
    #         "mon",
    #         "pol",
    #         "sln",
    #         "aq",
    #         "sat",
    #         "l,g",
    #     ]

    def __init__(
        self,
        elements: dict,
        state: Literal["S", "L", "G"],
        temperature: tuple,
        specific_heat_capacity: tuple,
        enthalpy: tuple,
        entropy: tuple,
        phase: Optional[str] = None,
        h_formation: float = 0.0,
    ):
        super().__init__(elements, state, phase)
        self._h_formation = float(h_formation)  # ΔHf°(298.15 K) in J/mol
        self.__temperature = self._ensure_temperature(temperature)
        self.__specific_heat_capacity = np.asarray(specific_heat_capacity)
        self.__entropy = np.asarray(entropy)
        self._specific_heat_capacity_interpolator = PchipInterpolator(
            self.__temperature, self.__specific_heat_capacity, extrapolate=False
        )
        self._enthalpy_interpolator = PchipInterpolator(
            self.__temperature, np.asarray(enthalpy), extrapolate=False
        )
        self._entropy_interpolator = PchipInterpolator(
            self.__temperature, self.__entropy, extrapolate=False
        )

    def _ensure_temperature(self, temperature: tuple) -> np.ndarray:
        """Make sure that the temperatures are strictly increasing by very slightly nudging values so that
        interpolation works properly."""
        temperature = np.asarray(temperature, dtype=float)
        differences = np.diff(temperature)
        if np.any(differences < 0):
            raise Exception("Temperatures must be increasing")
        nudge_count = 0
        for idx, difference in enumerate(differences):
            # print(idx, difference, nudge_count)
            if difference != 0:
                nudge_count = 0
            else:
                nudge_count += 1
                temperature[idx + 1] += 1e-6 * nudge_count
        return temperature

    def specific_heat_capacity(
        self, T: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        return self._specific_heat_capacity_interpolator(T)

    def enthalpy(self, T: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Absolute standard enthalpy H°(T) = ΔHf°(298.15) + [H(T) − H(298.15)] [J/mol]."""
        return self._h_formation + self._enthalpy_interpolator(T)

    def entropy(self, T: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        return self._entropy_interpolator(T)


class NASASevenCoeff(Species):
    r"""
    NASA Seven-Coefficient Polynomial species. Defined by the following equations:

    .. math::
        C_p(T) = R \cdot \left(a_1 + a_2 T + a_3 T^2 + a_4 T^3 + a_5 T^4\right)

        H(T) = R \cdot T \cdot \left(a_1 + \frac{a_2 T}{2} + \frac{a_3 T^2}{3} + \frac{a_4 T^3}{4} + \frac{a_5 T^4}{5} + \frac{a_6}{T}\right)

        S(T) = R \cdot \left(a_1 \ln(T) + a_2 T + \frac{a_3 T^2}{2} + \frac{a_4 T^3}{3} + \frac{a_5 T^4}{4} + a_7\right)

     where :math:`R` is the universal gas constant, and :math:`a_i` are the coefficients for the respective temperature range.
     The coefficients are typically provided for two temperature ranges: low (T_low to T_common) and high (T_common to T_high).
     The class handles both ranges.
    """

    def __init__(
        self,
        elements: dict,
        state: Literal["S", "L", "G"],
        temperature: tuple,
        coefficients: tuple[tuple, tuple],
        phase: Optional[str] = None,
    ):
        super().__init__(elements, state, phase)
        self.T_low, self.T_common, self.T_high = temperature
        self.low_range_coefficients, self.high_range_coefficients = coefficients
        # Pre-packed arrays for numba kernels
        self._nb_T_common = float(self.T_common)
        lo = list(self.low_range_coefficients)
        hi = list(self.high_range_coefficients)
        self._nb_lo = np.array(lo[:7] + [0.0] * max(0, 7 - len(lo)), dtype=np.float64)
        self._nb_hi = np.array(hi[:7] + [0.0] * max(0, 7 - len(hi)), dtype=np.float64)

    def specific_heat_capacity(
        self, T: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        """
        Calculates the specific heat capacity using a piecewise polynomial.
        """
        if isinstance(T, _SCALAR_TYPES):
            T_f = float(T)
            if T_f < self.T_low or T_f > self.T_high:
                return float("nan")
            if _NUMBA_AVAILABLE:
                return _cp_n7(T_f, self._nb_T_common, self._nb_lo, self._nb_hi) * R
            c = (
                self.low_range_coefficients
                if T_f <= self.T_common
                else self.high_range_coefficients
            )
            return (
                c[0] + c[1] * T_f + c[2] * T_f**2 + c[3] * T_f**3 + c[4] * T_f**4
            ) * R
        T = np.asanyarray(T, dtype=float)
        conditions = [
            (self.T_low <= T) & (T <= self.T_common),
            (self.T_common < T) & (T <= self.T_high),
        ]
        functions = [
            lambda t: (
                self.low_range_coefficients[0]
                + self.low_range_coefficients[1] * t
                + self.low_range_coefficients[2] * t**2
                + self.low_range_coefficients[3] * t**3
                + self.low_range_coefficients[4] * t**4
            )
            * R,
            lambda t: (
                self.high_range_coefficients[0]
                + self.high_range_coefficients[1] * t
                + self.high_range_coefficients[2] * t**2
                + self.high_range_coefficients[3] * t**3
                + self.high_range_coefficients[4] * t**4
            )
            * R,
            lambda t: np.full_like(t, np.nan),
        ]
        return np.piecewise(T, conditions, functions)

    def enthalpy(self, T: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        if isinstance(T, _SCALAR_TYPES):
            T_f = float(T)
            if T_f < self.T_low or T_f > self.T_high:
                return float("nan")
            if _NUMBA_AVAILABLE:
                return (
                    _enthalpy_n7(T_f, self._nb_T_common, self._nb_lo, self._nb_hi)
                    * R
                    * T_f
                )
            c = (
                self.low_range_coefficients
                if T_f <= self.T_common
                else self.high_range_coefficients
            )
            return (
                (
                    c[0]
                    + c[1] * T_f / 2
                    + c[2] * T_f**2 / 3
                    + c[3] * T_f**3 / 4
                    + c[4] * T_f**4 / 5
                    + c[5] / T_f
                )
                * R
                * T_f
            )
        T_arr = np.asanyarray(T, dtype=float)
        conditions = [
            (self.T_low <= T_arr) & (T_arr <= self.T_common),
            (self.T_common < T_arr) & (T_arr <= self.T_high),
        ]
        functions = [
            lambda t: (
                self.low_range_coefficients[0]
                + self.low_range_coefficients[1] * t / 2
                + self.low_range_coefficients[2] * t**2 / 3
                + self.low_range_coefficients[3] * t**3 / 4
                + self.low_range_coefficients[4] * t**4 / 5
                + self.low_range_coefficients[5] / t
            )
            * R
            * t,
            lambda t: (
                self.high_range_coefficients[0]
                + self.high_range_coefficients[1] * t / 2
                + self.high_range_coefficients[2] * t**2 / 3
                + self.high_range_coefficients[3] * t**3 / 4
                + self.high_range_coefficients[4] * t**4 / 5
                + self.high_range_coefficients[5] / t
            )
            * R
            * t,
            lambda t: np.full_like(t, np.nan),
        ]
        return np.piecewise(T_arr, conditions, functions)

    def entropy(self, T: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        T_arr = np.asanyarray(T, dtype=float)
        conditions = [
            (self.T_low <= T_arr) & (T_arr <= self.T_common),
            (self.T_common < T_arr) & (T_arr <= self.T_high),
        ]
        functions = [
            lambda t: (
                self.low_range_coefficients[0] * np.log(t)
                + self.low_range_coefficients[1] * t
                + self.low_range_coefficients[2] * t**2 / 2
                + self.low_range_coefficients[3] * t**3 / 3
                + self.low_range_coefficients[4] * t**4 / 4
                + self.low_range_coefficients[6]
            )
            * R,
            lambda t: (
                self.high_range_coefficients[0] * np.log(t)
                + self.high_range_coefficients[1] * t
                + self.high_range_coefficients[2] * t**2 / 2
                + self.high_range_coefficients[3] * t**3 / 3
                + self.high_range_coefficients[4] * t**4 / 4
                + self.high_range_coefficients[6]
            )
            * R,
            lambda t: np.full_like(t, np.nan),
        ]
        return np.piecewise(T_arr, conditions, functions)

    def reduced_gibbs(self, T: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        if isinstance(T, _SCALAR_TYPES):
            T_f = float(T)
            if T_f < self.T_low or T_f > self.T_high:
                return float("nan")
            if _NUMBA_AVAILABLE:
                return _gibbs_n7(T_f, self._nb_T_common, self._nb_lo, self._nb_hi)
            c = (
                self.low_range_coefficients
                if T_f <= self.T_common
                else self.high_range_coefficients
            )
            log_T = math.log(T_f)
            T2 = T_f * T_f
            T3 = T2 * T_f
            T4 = T3 * T_f
            h_rt = (
                c[0]
                + c[1] * T_f / 2
                + c[2] * T2 / 3
                + c[3] * T3 / 4
                + c[4] * T4 / 5
                + c[5] / T_f
            )
            s_r = (
                c[0] * log_T
                + c[1] * T_f
                + c[2] * T2 / 2
                + c[3] * T3 / 3
                + c[4] * T4 / 4
                + c[6]
            )
            return h_rt - s_r
        T_arr = np.asanyarray(T, dtype=float)
        return self.gibbs_free_energy(T_arr) / (R * T_arr)

    def reduced_enthalpy(self, T: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        if isinstance(T, _SCALAR_TYPES):
            T_f = float(T)
            if T_f < self.T_low or T_f > self.T_high:
                return float("nan")
            if _NUMBA_AVAILABLE:
                return _enthalpy_n7(T_f, self._nb_T_common, self._nb_lo, self._nb_hi)
            c = (
                self.low_range_coefficients
                if T_f <= self.T_common
                else self.high_range_coefficients
            )
            T2 = T_f * T_f
            T3 = T2 * T_f
            T4 = T3 * T_f
            return (
                c[0]
                + c[1] * T_f / 2
                + c[2] * T2 / 3
                + c[3] * T3 / 4
                + c[4] * T4 / 5
                + c[5] / T_f
            )
        T_arr = np.asanyarray(T, dtype=float)
        return self.enthalpy(T_arr) / (R * T_arr)


class NASANineCoeff(Species):
    r"""
    NASA Nine-Coefficient Polynomial species.
    """

    def __init__(
        self,
        elements: dict,
        state: Literal["S", "L", "G"],
        temperatures: tuple[float],
        exponents: tuple[tuple[float]],
        coefficients: tuple[tuple[float]],
        phase: Optional[str] = None,
        alias: Optional[str] = None,
        source_attribution: Optional[str] = None,
    ):
        super().__init__(elements, state, phase)
        self.alias = alias
        self.source_attribution = source_attribution
        self.temperatures = temperatures
        self.exponents = exponents
        self.coefficients = coefficients
        # Pre-packed numpy arrays for numba kernels
        n_segs = len(exponents)
        max_n_exps = max(len(e) for e in exponents) if n_segs > 0 else 0
        self._nb_n_segs = n_segs
        self._nb_t_bounds = np.array(temperatures, dtype=np.float64)
        self._nb_n_exps = np.array([len(e) for e in exponents], dtype=np.int64)
        self._nb_exps = np.zeros((n_segs, max_n_exps), dtype=np.float64)
        self._nb_coeffs = np.zeros((n_segs, max_n_exps + 2), dtype=np.float64)
        for k, (exps_k, coeffs_k) in enumerate(zip(exponents, coefficients)):
            ne = len(exps_k)
            self._nb_exps[k, :ne] = exps_k
            self._nb_coeffs[k, :ne] = coeffs_k[:ne]  # a1..an
            self._nb_coeffs[k, ne] = coeffs_k[ne]  # b1
            self._nb_coeffs[k, ne + 1] = coeffs_k[ne + 1]  # b2

    # ------------------------------------------------------------------
    # Per-segment evaluation helpers (static, operate on 1-D arrays)
    # ------------------------------------------------------------------

    @staticmethod
    def _cp_over_R(t: np.ndarray, coeffs: tuple, exps: tuple) -> np.ndarray:
        """Cp°/R = Σ aᵢ · T^eᵢ"""
        result = np.zeros_like(t)
        for i, e in enumerate(exps):
            result += coeffs[i] * t**e
        return result

    @staticmethod
    def _h_over_RT(t: np.ndarray, coeffs: tuple, exps: tuple) -> np.ndarray:
        """H°/(RT) per NASA-9 eq (2n).

        Integration of Cp°/R with respect to T, divided by T, plus b1/T.
        For exponent e:
          e ≠ -1 :  aᵢ · T^e / (e+1)
          e = -1 :  aᵢ · ln(T) / T
        Integration constant b1 stored at coeffs[len(exps)].
        """
        result = np.zeros_like(t)
        for i, e in enumerate(exps):
            if abs(e + 1) < 1e-9:  # e == -1
                result += coeffs[i] * np.log(t) / t
            else:
                result += coeffs[i] * t**e / (e + 1)
        result += coeffs[len(exps)] / t  # b1/T
        return result

    @staticmethod
    def _s_over_R(t: np.ndarray, coeffs: tuple, exps: tuple) -> np.ndarray:
        """S°/R per NASA-9 eq (3n).

        Integration of Cp°/R with respect to ln(T), plus b2.
        For exponent e:
          e ≠ 0 :  aᵢ · T^e / e
          e = 0 :  aᵢ · ln(T)
        Integration constant b2 stored at coeffs[len(exps)+1].
        """
        result = np.zeros_like(t)
        for i, e in enumerate(exps):
            if abs(e) < 1e-9:  # e == 0
                result += coeffs[i] * np.log(t)
            else:
                result += coeffs[i] * t**e / e
        result += coeffs[len(exps) + 1]  # b2
        return result

    # ------------------------------------------------------------------
    # Scalar fast-path helpers (avoid numpy overhead for single floats)
    # ------------------------------------------------------------------

    @staticmethod
    def _h_over_RT_scalar(t: float, coeffs, exps) -> float:
        """H°/(RT) for a single scalar temperature (pure Python, no numpy)."""
        result = 0.0
        n = len(exps)
        for i in range(n):
            e = exps[i]
            if abs(e + 1) < 1e-9:  # e == -1
                result += coeffs[i] * math.log(t) / t
            else:
                result += coeffs[i] * t**e / (e + 1)
        result += coeffs[n] / t  # b1/T
        return result

    @staticmethod
    def _s_over_R_scalar(t: float, coeffs, exps) -> float:
        """S°/R for a single scalar temperature (pure Python, no numpy)."""
        result = 0.0
        n = len(exps)
        for i in range(n):
            e = exps[i]
            if abs(e) < 1e-9:  # e == 0
                result += coeffs[i] * math.log(t)
            else:
                result += coeffs[i] * t**e / e
        result += coeffs[n + 1]  # b2
        return result

    @staticmethod
    def _cp_over_R_scalar(t: float, coeffs, exps) -> float:
        """Cp°/R for a single scalar temperature (pure Python, no numpy)."""
        result = 0.0
        for i in range(len(exps)):
            result += coeffs[i] * t ** exps[i]
        return result

    def _find_segment(self, T: float):
        """Return (coeffs, exps) for the interval containing scalar T, or None."""
        n = len(self.temperatures)
        for k in range(n - 1):
            t_low = self.temperatures[k]
            t_high = self.temperatures[k + 1]
            if k == n - 2:
                if t_low <= T <= t_high:
                    return self.coefficients[k], self.exponents[k]
            else:
                if t_low <= T < t_high:
                    return self.coefficients[k], self.exponents[k]
        return None

    # ------------------------------------------------------------------
    # Interval dispatcher
    # ------------------------------------------------------------------

    def _eval(self, T, segment_fn) -> Union[float, np.ndarray]:
        """Evaluate segment_fn over all temperature intervals.

        segment_fn(t_1d, coeffs, exps) → 1-D array of the same length.
        Returns NaN for T outside all defined intervals.
        """
        scalar = np.ndim(T) == 0
        T_arr = np.atleast_1d(np.asanyarray(T, dtype=float))
        result = np.full_like(T_arr, np.nan)
        n = len(self.temperatures)
        for k in range(n - 1):
            t_low = self.temperatures[k]
            t_high = self.temperatures[k + 1]
            # Last interval is closed on the right; others are half-open [low, high)
            if k == n - 2:
                mask = (T_arr >= t_low) & (T_arr <= t_high)
            else:
                mask = (T_arr >= t_low) & (T_arr < t_high)
            if np.any(mask):
                result[mask] = segment_fn(
                    T_arr[mask], self.coefficients[k], self.exponents[k]
                )
        return result.item() if scalar else result

    # ------------------------------------------------------------------
    # Public thermodynamic methods
    # ------------------------------------------------------------------

    def specific_heat_capacity(
        self, T: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        if isinstance(T, _SCALAR_TYPES):
            if _NUMBA_AVAILABLE:
                return (
                    _cp_n9(
                        float(T),
                        self._nb_t_bounds,
                        self._nb_exps,
                        self._nb_coeffs,
                        self._nb_n_segs,
                        self._nb_n_exps,
                    )
                    * R
                )
            seg = self._find_segment(float(T))
            if seg is None:
                return float("nan")
            return self._cp_over_R_scalar(float(T), *seg) * R
        return self._eval(T, lambda t, c, e: self._cp_over_R(t, c, e) * R)

    def enthalpy(self, T: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        if isinstance(T, _SCALAR_TYPES):
            T_f = float(T)
            if _NUMBA_AVAILABLE:
                return (
                    _enthalpy_n9(
                        T_f,
                        self._nb_t_bounds,
                        self._nb_exps,
                        self._nb_coeffs,
                        self._nb_n_segs,
                        self._nb_n_exps,
                    )
                    * R
                    * T_f
                )
            seg = self._find_segment(T_f)
            if seg is None:
                return float("nan")
            return self._h_over_RT_scalar(T_f, *seg) * R * T_f
        # H° = (H°/RT) · R · T
        return self._eval(T, lambda t, c, e: self._h_over_RT(t, c, e) * R * t)

    def entropy(self, T: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        if isinstance(T, _SCALAR_TYPES):
            T_f = float(T)
            seg = self._find_segment(T_f)
            if seg is None:
                return float("nan")
            return self._s_over_R_scalar(T_f, *seg) * R
        return self._eval(T, lambda t, c, e: self._s_over_R(t, c, e) * R)

    def reduced_gibbs(self, T: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """G°/(RT) = H°/(RT) − S°/R, computed in a single segment lookup for scalars."""
        if isinstance(T, _SCALAR_TYPES):
            T_f = float(T)
            if _NUMBA_AVAILABLE:
                return _gibbs_n9(
                    T_f,
                    self._nb_t_bounds,
                    self._nb_exps,
                    self._nb_coeffs,
                    self._nb_n_segs,
                    self._nb_n_exps,
                )
            seg = self._find_segment(T_f)
            if seg is None:
                return float("nan")
            return self._h_over_RT_scalar(T_f, *seg) - self._s_over_R_scalar(T_f, *seg)
        T_arr = np.asanyarray(T, dtype=float)
        return self.gibbs_free_energy(T_arr) / (R * T_arr)

    def reduced_enthalpy(self, T: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """H°/(RT), computed directly for scalars without the R·T multiply/divide round-trip."""
        if isinstance(T, _SCALAR_TYPES):
            T_f = float(T)
            if _NUMBA_AVAILABLE:
                return _enthalpy_n9(
                    T_f,
                    self._nb_t_bounds,
                    self._nb_exps,
                    self._nb_coeffs,
                    self._nb_n_segs,
                    self._nb_n_exps,
                )
            seg = self._find_segment(T_f)
            if seg is None:
                return float("nan")
            return self._h_over_RT_scalar(T_f, *seg)
        T_arr = np.asanyarray(T, dtype=float)
        return self.enthalpy(T_arr) / (R * T_arr)


class ShomateCoeff(Species):
    """NIST Shomate equation species.

    The Shomate form is used by the NIST WebBook and covers many condensed-phase
    species not present in NASA-7/9 databases. This class also serves as the
    container for species imported from the TERRA database, which uses a
    mathematically compatible reduced Gibbs energy (Phi-star) polynomial.

    Equation Form (with t = T/1000):
        Cp°(T) = A + Bt + Ct² + Dt³ + E/t²                      [J/mol/K]
        H°(T)  = (At + Bt²/2 + Ct³/3 + Dt⁴/4 − E/t + F) × 1000  [J/mol]
        S°(T)  = A·ln(t) + Bt + Ct²/2 + Dt³/3 − E/(2t²) + G     [J/mol/K]

    TERRA Integration:
        TERRA coefficients (f1...f7) are mapped to Shomate (A...G) using the
        following transformations (accounting for TERRA's x = T/10000 scaling)::

            A = f2
            B = 0.2 * f5
            C = 0.06 * f6
            D = 0.012 * f7
            E = 200.0 * f3
            F = (H_meta - 10000*f4 - shift) / 1000
            G = f1 + f2 - f2 * ln(10)

    The F coefficient encodes the standard enthalpy of formation at 298.15 K
    per NIST convention: H°(T) [kJ/mol] = At + Bt²/2 + … + F, where F is
    chosen so that the formula reproduces ΔfH°(298.15 K) at t = 0.29815.
    The H coefficient (= ΔfH°(298.15 K) in kJ/mol) is stored for reference
    only and is not used by the evaluation routines.

    JSON schema (shomate.json)::

        {
          "Bi2O3_S": {
            "elements": {"Bi": 2, "O": 3},
            "phase": "S",
            "alias": "Bi2O3",
            "segments": [
              {
                "t_low": 298.0,
                "t_high": 1097.0,
                "coefficients": [A, B, C, D, E, F, G, H]
              }
            ]
          }
        }

    Attributes:
        temperatures: Segment boundary temperatures [K], length n_segs+1.
        coefficients: Per-segment Shomate coefficients [A,B,C,D,E,F,G,H],
            one tuple of 8 floats per segment.
        alias: Optional original source name.
    """

    def __init__(
        self,
        elements: dict,
        state: Literal["S", "L", "G"],
        temperatures: tuple,
        coefficients: tuple,
        phase: Optional[str] = None,
        alias: Optional[str] = None,
    ):
        super().__init__(elements, state, phase)
        self.temperatures = tuple(float(t) for t in temperatures)
        self.coefficients = tuple(tuple(float(v) for v in seg) for seg in coefficients)
        self.alias = alias

        # Pre-packed arrays for numba kernels
        self._nb_n_segs = len(self.temperatures) - 1
        self._nb_t_bounds = np.array(self.temperatures, dtype=np.float64)
        self._nb_coeffs = np.array(self.coefficients, dtype=np.float64)

    @classmethod
    def from_terra(
        cls,
        elements: dict,
        state: Literal["S", "L", "G"],
        temperatures: tuple,
        terra_coefficients: tuple,
        phase: Optional[str] = None,
        alias: Optional[str] = None,
    ) -> "ShomateCoeff":
        """
        Create a ShomateCoeff species from TERRA f1-f7 coefficients.

        Mapping (x = T/1000):
        TERRA Cp = f2 + 2f3/x^2 + 2f5*x + 6f6*x^2 + 12f7*x^3
        Shomate Cp = A + B*t + C*t^2 + D*t^3 + E/t^2

        A = f[1]
        B = 2*f[4]
        C = 6*f[5]
        D = 12*f[6]
        E = 2*f[2]
        F = -f[3]
        G = f[0] + f[1]
        """
        shomate_segments = []
        for f in terra_coefficients:
            A = f[1]
            B = 2.0 * f[4]
            C = 6.0 * f[5]
            D = 12.0 * f[6]
            E = 2.0 * f[2]
            F = -f[3]
            G = f[0] + f[1]
            # H: NIST convention ΔfH°(298.15 K) in kJ/mol.
            # Shomate H(t) [J/mol] = (At + Bt²/2 + … + F) × 1000.
            # At t=0.29815, this should be ΔfH°(298.15 K) in J/mol.
            t_ref = 0.29815
            h_ref_j_mol = (
                A * t_ref
                + B * t_ref**2 / 2.0
                + C * t_ref**3 / 3.0
                + D * t_ref**4 / 4.0
                - E / t_ref
                + F
            ) * 1000.0
            shomate_segments.append((A, B, C, D, E, F, G, h_ref_j_mol / 1000.0))

        return cls(
            elements=elements,
            state=state,
            temperatures=temperatures,
            coefficients=tuple(shomate_segments),
            phase=phase,
            alias=alias,
        )

    def _find_segment(self, T: float) -> Optional[tuple]:
        """Return the coefficient tuple for the interval containing scalar T, or None."""
        n = len(self.temperatures)
        for k in range(n - 1):
            t_low = self.temperatures[k]
            t_high = self.temperatures[k + 1]
            if k == n - 2:
                if t_low <= T <= t_high:
                    return self.coefficients[k]
            else:
                if t_low <= T < t_high:
                    return self.coefficients[k]
        return None

    def _segment_mask(self, k: int, T_arr: np.ndarray, n: int) -> np.ndarray:
        """Boolean mask for temperatures in segment k."""
        if k == n - 2:
            return (T_arr >= self.temperatures[k]) & (T_arr <= self.temperatures[k + 1])
        return (T_arr >= self.temperatures[k]) & (T_arr < self.temperatures[k + 1])

    def specific_heat_capacity(
        self, T: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        """Cp°(T) = A + Bt + Ct² + Dt³ + E/t²  [J/mol/K]."""
        if isinstance(T, _SCALAR_TYPES):
            T_f = float(T)
            if _NUMBA_AVAILABLE:
                return (
                    _cp_shomate(
                        T_f, self._nb_t_bounds, self._nb_coeffs, self._nb_n_segs, R
                    )
                    * R
                )
            c = self._find_segment(T_f)
            if c is None:
                return float("nan")
            t = T_f * 1e-3
            return c[0] + c[1] * t + c[2] * t * t + c[3] * t * t * t + c[4] / (t * t)
        scalar = np.ndim(T) == 0
        T_arr = np.atleast_1d(np.asanyarray(T, dtype=float))
        result = np.full_like(T_arr, np.nan)
        n = len(self.temperatures)
        for k in range(n - 1):
            mask = self._segment_mask(k, T_arr, n)
            if np.any(mask):
                t = T_arr[mask] * 1e-3
                c = self.coefficients[k]
                result[mask] = c[0] + c[1] * t + c[2] * t**2 + c[3] * t**3 + c[4] / t**2
        return result.item() if scalar else result

    def enthalpy(self, T: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """H°(T) = (At + Bt²/2 + Ct³/3 + Dt⁴/4 − E/t + F) × 1000  [J/mol]."""
        if isinstance(T, _SCALAR_TYPES):
            T_f = float(T)
            if _NUMBA_AVAILABLE:
                return (
                    _enthalpy_shomate(
                        T_f, self._nb_t_bounds, self._nb_coeffs, self._nb_n_segs, R
                    )
                    * R
                    * T_f
                )
            c = self._find_segment(T_f)
            if c is None:
                return float("nan")
            t = T_f * 1e-3
            return (
                c[0] * t
                + c[1] * t * t / 2
                + c[2] * t * t * t / 3
                + c[3] * t * t * t * t / 4
                - c[4] / t
                + c[5]
            ) * 1000.0
        scalar = np.ndim(T) == 0
        T_arr = np.atleast_1d(np.asanyarray(T, dtype=float))
        result = np.full_like(T_arr, np.nan)
        n = len(self.temperatures)
        for k in range(n - 1):
            mask = self._segment_mask(k, T_arr, n)
            if np.any(mask):
                t = T_arr[mask] * 1e-3
                c = self.coefficients[k]
                result[mask] = (
                    c[0] * t
                    + c[1] * t**2 / 2
                    + c[2] * t**3 / 3
                    + c[3] * t**4 / 4
                    - c[4] / t
                    + c[5]
                ) * 1000.0
        return result.item() if scalar else result

    def entropy(self, T: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """S°(T) = A·ln(t) + Bt + Ct²/2 + Dt³/3 − E/(2t²) + G  [J/mol/K]."""
        if isinstance(T, _SCALAR_TYPES):
            T_f = float(T)
            c = self._find_segment(T_f)
            if c is None:
                return float("nan")
            t = T_f * 1e-3
            return (
                c[0] * math.log(t)
                + c[1] * t
                + c[2] * t * t / 2
                + c[3] * t * t * t / 3
                - c[4] / (2 * t * t)
                + c[6]
            )
        scalar = np.ndim(T) == 0
        T_arr = np.atleast_1d(np.asanyarray(T, dtype=float))
        result = np.full_like(T_arr, np.nan)
        n = len(self.temperatures)
        for k in range(n - 1):
            mask = self._segment_mask(k, T_arr, n)
            if np.any(mask):
                t = T_arr[mask] * 1e-3
                c = self.coefficients[k]
                result[mask] = (
                    c[0] * np.log(t)
                    + c[1] * t
                    + c[2] * t**2 / 2
                    + c[3] * t**3 / 3
                    - c[4] / (2 * t**2)
                    + c[6]
                )
        return result.item() if scalar else result

    def reduced_gibbs(self, T: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """G°/(RT) computed without intermediate allocation for scalars."""
        if isinstance(T, _SCALAR_TYPES):
            T_f = float(T)
            if _NUMBA_AVAILABLE:
                return _gibbs_shomate(
                    T_f, self._nb_t_bounds, self._nb_coeffs, self._nb_n_segs, R
                )
            c = self._find_segment(T_f)
            if c is None:
                return float("nan")
            t = T_f * 1e-3
            H = (
                c[0] * t
                + c[1] * t * t / 2
                + c[2] * t * t * t / 3
                + c[3] * t * t * t * t / 4
                - c[4] / t
                + c[5]
            ) * 1000.0
            S = (
                c[0] * math.log(t)
                + c[1] * t
                + c[2] * t * t / 2
                + c[3] * t * t * t / 3
                - c[4] / (2 * t * t)
                + c[6]
            )
            return (H - S * T_f) / (R * T_f)
        T_arr = np.asanyarray(T, dtype=float)
        return self.gibbs_free_energy(T_arr) / (R * T_arr)

    def reduced_enthalpy(self, T: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """H°/(RT)."""
        if isinstance(T, _SCALAR_TYPES):
            T_f = float(T)
            if _NUMBA_AVAILABLE:
                return _enthalpy_shomate(
                    T_f, self._nb_t_bounds, self._nb_coeffs, self._nb_n_segs, R
                )
            h = self.enthalpy(T_f)
            if math.isnan(h):
                return float("nan")
            return h / (R * T_f)
        T_arr = np.asanyarray(T, dtype=float)
        return self.enthalpy(T_arr) / (R * T_arr)


_CAL_TO_J = 4.184
_R_CAL = R / _CAL_TO_J  # 1.98726 cal/(mol·K)


class AFCESICCoeff(Species):
    """AFCESIC thermodynamic polynomial species.

    The AFCESIC database (Air Force Chemical Equilibrium Specific Impulse Code)
    stores heat-capacity data as two sets of five float32 coefficients in
    θ = T/1000.  The low-temperature set uses forward powers of θ while the
    high-temperature set uses *inverse* powers, giving natural convergence at
    combustion temperatures.

    Low-T range (T_low ≤ T ≤ T_mid, typically 300–1200 K)::

        Cp  = b₁ + b₂θ + b₃θ² + b₄θ³ + b₅θ⁴                [cal/(mol·K)]
        H   = 1000·(b₁θ + b₂θ²/2 + b₃θ³/3 + b₄θ⁴/4
              + b₅θ⁵/5) + RF                                  [cal/mol]
        S   = b₁·ln θ + b₂θ + b₃θ²/2 + b₄θ³/3
              + b₅θ⁴/4 + CH                                   [cal/(mol·K)]

    High-T range (T_mid < T ≤ T_high, typically 1200–6000 K)::

        Cp  = a₁ + a₂/θ + a₃/θ² + a₄/θ³ + a₅/θ⁴            [cal/(mol·K)]
        H   = 1000·(a₁θ + a₂·ln θ − a₃/θ − a₄/(2θ²)
              − a₅/(3θ³)) + H₀_hi                             [cal/mol]
        S   = a₁·ln θ − a₂/θ − a₃/(2θ²) − a₄/(3θ³)
              − a₅/(4θ⁴) + S₀_hi                              [cal/(mol·K)]

    The high-T integration constants ``H₀_hi`` and ``S₀_hi`` are derived from
    continuity of H and S at T_mid (θ_mid = 1.2, T = 1200 K).  Note: the
    original AFCESIC format specifies T_mid = 1000 K, but inspection of the
    polynomial coefficients shows the smoothest crossover for this dataset
    occurs at 1200 K; 1000 K produces a visible discontinuity in H and S.

    Args:
        elements: Element composition, e.g. ``{"H": 2, "O": 1}``.
        state: ``"G"``, ``"L"``, or ``"S"``.
        temperature: ``(T_low, T_mid, T_high)`` in Kelvin.
        low_coefficients: ``(b₁, b₂, b₃, b₄, b₅)`` — forward-power Cp
            coefficients for the low-T range [cal/(mol·K)].
        high_coefficients: ``(a₁, a₂, a₃, a₄, a₅)`` — inverse-power Cp
            coefficients for the high-T range [cal/(mol·K)].
        rf: Enthalpy integration constant RF (low-T range) [kcal/mol ÷ 1000].
        ch: Entropy integration constant CH (low-T range) [cal/(mol·K)].
        phase: Optional phase label for display.
    """

    def __init__(
        self,
        elements: dict,
        state: Literal["S", "L", "G"],
        temperature: tuple,
        low_coefficients: tuple,
        high_coefficients: tuple,
        rf: float,
        ch: float,
        phase: Optional[str] = None,
    ):
        super().__init__(elements, state, phase)
        self.T_low, self.T_mid, self.T_high = (float(t) for t in temperature)
        self._lo = tuple(float(c) for c in low_coefficients)  # b₁…b₅
        self._hi = tuple(float(c) for c in high_coefficients)  # a₁…a₅
        self._rf = float(rf)
        self._ch = float(ch)

        # Pre-compute high-T integration constants from continuity at T_mid (θ_mid = T_mid/1000).
        b = self._lo
        a = self._hi
        th_mid = self.T_mid * 1e-3
        inv_mid = 1.0 / th_mid
        # H_lo(θ_mid) = 1000·(b₁θ + b₂θ²/2 + b₃θ³/3 + b₄θ⁴/4 + b₅θ⁵/5) + RF
        h_lo_mid = (
            1000.0
            * (
                b[0] * th_mid
                + b[1] * th_mid**2 / 2
                + b[2] * th_mid**3 / 3
                + b[3] * th_mid**4 / 4
                + b[4] * th_mid**5 / 5
            )
            + rf
        )
        # H_hi(θ_mid) without H₀_hi = 1000·(a₁θ + a₂·ln θ − a₃/θ − a₄/(2θ²) − a₅/(3θ³))
        h_hi_terms_mid = 1000.0 * (
            a[0] * th_mid
            + a[1] * math.log(th_mid)
            - a[2] * inv_mid
            - a[3] * inv_mid**2 / 2
            - a[4] * inv_mid**3 / 3
        )
        self._h0_hi = h_lo_mid - h_hi_terms_mid

        # S_lo(θ_mid) = b₁·ln θ + b₂θ + b₃θ²/2 + b₄θ³/3 + b₅θ⁴/4 + CH
        s_lo_mid = (
            b[0] * math.log(th_mid)
            + b[1] * th_mid
            + b[2] * th_mid**2 / 2
            + b[3] * th_mid**3 / 3
            + b[4] * th_mid**4 / 4
            + ch
        )
        # S_hi(θ_mid) without S₀_hi = a₁·ln θ − a₂/θ − a₃/(2θ²) − a₄/(3θ³) − a₅/(4θ⁴)
        s_hi_terms_mid = (
            a[0] * math.log(th_mid)
            - a[1] * inv_mid
            - a[2] * inv_mid**2 / 2
            - a[3] * inv_mid**3 / 3
            - a[4] * inv_mid**4 / 4
        )
        self._s0_hi = s_lo_mid - s_hi_terms_mid

        # Pre-packed numpy arrays for numba kernels
        self._nb_lo = np.array(self._lo, dtype=np.float64)
        self._nb_hi = np.array(self._hi, dtype=np.float64)

    # ------------------------------------------------------------------
    # Internal helpers (all return values in cal units)
    # ------------------------------------------------------------------

    def _cp_cal(self, T: float) -> float:
        """Cp in cal/(mol·K) for scalar T."""
        th = T * 1e-3
        if T <= self.T_mid:
            b = self._lo
            return b[0] + b[1] * th + b[2] * th**2 + b[3] * th**3 + b[4] * th**4
        a = self._hi
        inv = 1.0 / th
        return a[0] + a[1] * inv + a[2] * inv**2 + a[3] * inv**3 + a[4] * inv**4

    def _h_cal(self, T: float) -> float:
        """H in cal/mol for scalar T."""
        th = T * 1e-3
        if T <= self.T_mid:
            b = self._lo
            return (
                1000.0
                * (
                    b[0] * th
                    + b[1] * th**2 / 2
                    + b[2] * th**3 / 3
                    + b[3] * th**4 / 4
                    + b[4] * th**5 / 5
                )
                + self._rf
            )
        a = self._hi
        return (
            1000.0
            * (
                a[0] * th
                + a[1] * math.log(th)
                - a[2] / th
                - a[3] / (2 * th**2)
                - a[4] / (3 * th**3)
            )
            + self._h0_hi
        )

    def _s_cal(self, T: float) -> float:
        """S in cal/(mol·K) for scalar T."""
        th = T * 1e-3
        if T <= self.T_mid:
            b = self._lo
            return (
                b[0] * math.log(th)
                + b[1] * th
                + b[2] * th**2 / 2
                + b[3] * th**3 / 3
                + b[4] * th**4 / 4
                + self._ch
            )
        a = self._hi
        inv = 1.0 / th
        return (
            a[0] * math.log(th)
            - a[1] * inv
            - a[2] * inv**2 / 2
            - a[3] * inv**3 / 3
            - a[4] * inv**4 / 4
            + self._s0_hi
        )

    # ------------------------------------------------------------------
    # Public thermodynamic methods (SI outputs: J/mol, J/(mol·K))
    # ------------------------------------------------------------------

    def specific_heat_capacity(
        self, T: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        if isinstance(T, _SCALAR_TYPES):
            T_f = float(T)
            if _NUMBA_AVAILABLE:
                return (
                    _cp_afcesic(
                        T_f,
                        self.T_low,
                        self.T_mid,
                        self.T_high,
                        self._nb_lo,
                        self._nb_hi,
                        _R_CAL,
                    )
                    * R
                )
            if T_f < self.T_low or T_f > self.T_high:
                return float("nan")
            return self._cp_cal(T_f) * _CAL_TO_J
        T_arr = np.asanyarray(T, dtype=float)
        th = T_arr * 1e-3
        lo_mask = (self.T_low <= T_arr) & (T_arr <= self.T_mid)
        hi_mask = (self.T_mid < T_arr) & (T_arr <= self.T_high)
        result = np.full_like(T_arr, np.nan)
        b = self._lo
        if np.any(lo_mask):
            t = th[lo_mask]
            result[lo_mask] = (
                b[0] + b[1] * t + b[2] * t**2 + b[3] * t**3 + b[4] * t**4
            ) * _CAL_TO_J
        a = self._hi
        if np.any(hi_mask):
            inv = 1.0 / th[hi_mask]
            result[hi_mask] = (
                a[0] + a[1] * inv + a[2] * inv**2 + a[3] * inv**3 + a[4] * inv**4
            ) * _CAL_TO_J
        return result

    def enthalpy(self, T: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        if isinstance(T, _SCALAR_TYPES):
            T_f = float(T)
            if _NUMBA_AVAILABLE:
                return (
                    _enthalpy_afcesic(
                        T_f,
                        self.T_low,
                        self.T_mid,
                        self.T_high,
                        self._nb_lo,
                        self._nb_hi,
                        self._rf,
                        self._h0_hi,
                        _R_CAL,
                    )
                    * R
                    * T_f
                )
            if T_f < self.T_low or T_f > self.T_high:
                return float("nan")
            return self._h_cal(T_f) * _CAL_TO_J
        T_arr = np.asanyarray(T, dtype=float)
        th = T_arr * 1e-3
        lo_mask = (self.T_low <= T_arr) & (T_arr <= self.T_mid)
        hi_mask = (self.T_mid < T_arr) & (T_arr <= self.T_high)
        result = np.full_like(T_arr, np.nan)
        b = self._lo
        if np.any(lo_mask):
            t = th[lo_mask]
            result[lo_mask] = (
                1000.0
                * (
                    b[0] * t
                    + b[1] * t**2 / 2
                    + b[2] * t**3 / 3
                    + b[3] * t**4 / 4
                    + b[4] * t**5 / 5
                )
                + self._rf
            ) * _CAL_TO_J
        a = self._hi
        if np.any(hi_mask):
            t = th[hi_mask]
            result[hi_mask] = (
                1000.0
                * (
                    a[0] * t
                    + a[1] * np.log(t)
                    - a[2] / t
                    - a[3] / (2 * t**2)
                    - a[4] / (3 * t**3)
                )
                + self._h0_hi
            ) * _CAL_TO_J
        return result

    def entropy(self, T: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        if isinstance(T, _SCALAR_TYPES):
            T_f = float(T)
            if T_f < self.T_low or T_f > self.T_high:
                return float("nan")
            return self._s_cal(T_f) * _CAL_TO_J
        T_arr = np.asanyarray(T, dtype=float)
        th = T_arr * 1e-3
        lo_mask = (self.T_low <= T_arr) & (T_arr <= self.T_mid)
        hi_mask = (self.T_mid < T_arr) & (T_arr <= self.T_high)
        result = np.full_like(T_arr, np.nan)
        b = self._lo
        if np.any(lo_mask):
            t = th[lo_mask]
            result[lo_mask] = (
                b[0] * np.log(t)
                + b[1] * t
                + b[2] * t**2 / 2
                + b[3] * t**3 / 3
                + b[4] * t**4 / 4
                + self._ch
            ) * _CAL_TO_J
        a = self._hi
        if np.any(hi_mask):
            t = th[hi_mask]
            inv = 1.0 / t
            result[hi_mask] = (
                a[0] * np.log(t)
                - a[1] * inv
                - a[2] * inv**2 / 2
                - a[3] * inv**3 / 3
                - a[4] * inv**4 / 4
                + self._s0_hi
            ) * _CAL_TO_J
        return result

    def reduced_gibbs(self, T: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        if isinstance(T, _SCALAR_TYPES):
            T_f = float(T)
            if _NUMBA_AVAILABLE:
                return _gibbs_afcesic(
                    T_f,
                    self.T_low,
                    self.T_mid,
                    self.T_high,
                    self._nb_lo,
                    self._nb_hi,
                    self._rf,
                    self._ch,
                    self._h0_hi,
                    self._s0_hi,
                    _R_CAL,
                )
            if T_f < self.T_low or T_f > self.T_high:
                return float("nan")
            return self._h_cal(T_f) / (_R_CAL * T_f) - self._s_cal(T_f) / _R_CAL
        T_arr = np.asanyarray(T, dtype=float)
        return self.gibbs_free_energy(T_arr) / (R * T_arr)

    def reduced_enthalpy(self, T: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        if isinstance(T, _SCALAR_TYPES):
            T_f = float(T)
            if _NUMBA_AVAILABLE:
                return _enthalpy_afcesic(
                    T_f,
                    self.T_low,
                    self.T_mid,
                    self.T_high,
                    self._nb_lo,
                    self._nb_hi,
                    self._rf,
                    self._h0_hi,
                    _R_CAL,
                )
            if T_f < self.T_low or T_f > self.T_high:
                return float("nan")
            return self._h_cal(T_f) / (_R_CAL * T_f)
        T_arr = np.asanyarray(T, dtype=float)
        return self.enthalpy(T_arr) / (R * T_arr)


class TERRACoeff(Species):
    r"""
    Coefficients from the TERRA computer program,
    developed by Boris Georgievich Trusov at Bauman Moscow State Technical University.

    In TERRA, the thermodynamic properties are derived from the reduced Gibbs energy G*
    (in J/mol K), defined using 7 coefficients:

    .. math::

        G^*(x) = f_1 + f_2 \ln x + f_3 x^{-2} + f_4 x^{-1} + f_5 x + f_6 x^2 + f_7 x^3

    where the scaled temperature x is defined as:

    .. math::

        x = T \cdot 10^{-3}

    Using the thermodynamic relations G* = S - H/T and dG*/dT = H/T^2, we derive:

    .. math::

        H(x) = 1000 \cdot (f_2 x - 2f_3 x^{-1} - f_4 + f_5 x^2 + 2f_6 x^3 + 3f_7 x^4)

        S(x) = f_1 + f_2(1 + \ln x) - f_3 x^{-2} + 2f_5 x + 3f_6 x^2 + 4f_7 x^3

        C_p(x) = f_2 + 2f_3 x^{-2} + 2f_5 x + 6f_6 x^2 + 12f_7 x^3
    """

    def __init__(
        self,
        elements: dict,
        state: Literal["S", "L", "G"],
        temperatures: tuple,
        coefficients: tuple,
        phase: Optional[str] = None,
    ):
        super().__init__(elements, state, phase)
        self.temperatures = tuple(float(t) for t in temperatures)
        self.coefficients = tuple(tuple(float(v) for v in seg) for seg in coefficients)

        # Pre-packed arrays for potential numba kernels
        self._nb_n_segs = len(self.temperatures) - 1
        self._nb_t_bounds = np.array(self.temperatures, dtype=np.float64)
        self._nb_coeffs = np.array(self.coefficients, dtype=np.float64)

    def _find_segment(self, T: float) -> Optional[tuple]:
        """Return the coefficient tuple for the interval containing scalar T, or None."""
        n = len(self.temperatures)
        for k in range(n - 1):
            t_low = self.temperatures[k]
            t_high = self.temperatures[k + 1]
            if k == n - 2:
                if t_low <= T <= t_high:
                    return self.coefficients[k]
            else:
                if t_low <= T < t_high:
                    return self.coefficients[k]
        return None

    def _segment_mask(self, k: int, T_arr: np.ndarray, n: int) -> np.ndarray:
        """Boolean mask for temperatures in segment k."""
        if k == n - 2:
            return (T_arr >= self.temperatures[k]) & (T_arr <= self.temperatures[k + 1])
        return (T_arr >= self.temperatures[k]) & (T_arr < self.temperatures[k + 1])

    def specific_heat_capacity(
        self, T: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        if isinstance(T, _SCALAR_TYPES):
            T_f = float(T)
            f = self._find_segment(T_f)
            if f is None:
                return float("nan")
            x = T_f * 1e-3
            return (
                f[1]
                + 2 * f[2] / (x * x)
                + 2 * f[4] * x
                + 6 * f[5] * x**2
                + 12 * f[6] * x**3
            )

        scalar = np.ndim(T) == 0
        T_arr = np.atleast_1d(np.asanyarray(T, dtype=float))
        result = np.full_like(T_arr, np.nan)
        n = len(self.temperatures)
        for k in range(n - 1):
            mask = self._segment_mask(k, T_arr, n)
            if np.any(mask):
                x = T_arr[mask] * 1e-3
                f = self.coefficients[k]
                result[mask] = (
                    f[1]
                    + 2 * f[2] / x**2
                    + 2 * f[4] * x
                    + 6 * f[5] * x**2
                    + 12 * f[6] * x**3
                )
        return result.item() if scalar else result

    def enthalpy(self, T: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        if isinstance(T, _SCALAR_TYPES):
            T_f = float(T)
            f = self._find_segment(T_f)
            if f is None:
                return float("nan")
            x = T_f * 1e-3
            return 1000.0 * (
                f[1] * x
                - 2 * f[2] / x
                - f[3]
                + f[4] * x**2
                + 2 * f[5] * x**3
                + 3 * f[6] * x**4
            )

        scalar = np.ndim(T) == 0
        T_arr = np.atleast_1d(np.asanyarray(T, dtype=float))
        result = np.full_like(T_arr, np.nan)
        n = len(self.temperatures)
        for k in range(n - 1):
            mask = self._segment_mask(k, T_arr, n)
            if np.any(mask):
                x = T_arr[mask] * 1e-3
                f = self.coefficients[k]
                result[mask] = 1000.0 * (
                    f[1] * x
                    - 2 * f[2] / x
                    - f[3]
                    + f[4] * x**2
                    + 2 * f[5] * x**3
                    + 3 * f[6] * x**4
                )
        return result.item() if scalar else result

    def entropy(self, T: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        if isinstance(T, _SCALAR_TYPES):
            T_f = float(T)
            f = self._find_segment(T_f)
            if f is None:
                return float("nan")
            x = T_f * 1e-3
            return (
                f[0]
                + f[1] * (1 + math.log(x))
                - f[2] / (x * x)
                + 2 * f[4] * x
                + 3 * f[5] * x**2
                + 4 * f[6] * x**3
            )

        scalar = np.ndim(T) == 0
        T_arr = np.atleast_1d(np.asanyarray(T, dtype=float))
        result = np.full_like(T_arr, np.nan)
        n = len(self.temperatures)
        for k in range(n - 1):
            mask = self._segment_mask(k, T_arr, n)
            if np.any(mask):
                x = T_arr[mask] * 1e-3
                f = self.coefficients[k]
                result[mask] = (
                    f[0]
                    + f[1] * (1 + np.log(x))
                    - f[2] / x**2
                    + 2 * f[4] * x
                    + 3 * f[5] * x**2
                    + 4 * f[6] * x**3
                )
        return result.item() if scalar else result

    def reduced_gibbs(self, T: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Dimensionless standard Gibbs free energy G°/(RT)."""
        if isinstance(T, _SCALAR_TYPES):
            T_f = float(T)
            f = self._find_segment(T_f)
            if f is None:
                return float("nan")
            x = T_f * 1e-3
            # G* = S - H/T, so G° = H - TS = -T * G*
            # Therefore G°/(RT) = -G* / R
            G_star = (
                f[0]
                + f[1] * math.log(x)
                + f[2] / (x * x)
                + f[3] / x
                + f[4] * x
                + f[5] * x**2
                + f[6] * x**3
            )
            return -G_star / R

        T_arr = np.asanyarray(T, dtype=float)
        return self.gibbs_free_energy(T_arr) / (R * T_arr)


class SpeciesDatabase:
    """
    Loads thermodynamic species from multiple database formats and provides
    element-based filtering and deduplication.

    When the same species (same composition and phase) exists in multiple
    enabled databases, a priority-based selection is performed.

    Priority (highest wins, default):
        NASA-9 > NASA-7 > JANAF > TERRA > AFCESIC

    Users can override this with ``source_priority`` in ``__init__`` or
    ``load()``.
    """

    _DEFAULT_PRIORITY_ORDER = ("NASA-9", "NASA-7", "JANAF", "TERRA", "AFCESIC")

    def __init__(
        self,
        nasa7_path: Optional[str] = None,
        nasa9_path: Optional[str] = None,
        janaf_path: Optional[str] = None,
        shomate_path: Optional[str] = None,
        afcesic_path: Optional[str] = None,
        terra_path: Optional[str] = None,
        source_priority: Optional[Union[Sequence[str], Dict[str, int]]] = None,
    ):
        self.nasa7_path = nasa7_path
        self.nasa9_path = nasa9_path
        self.janaf_path = janaf_path
        self.shomate_path = shomate_path
        self.afcesic_path = afcesic_path
        self.terra_path = terra_path
        self._source_priority_map = self._build_priority_map(source_priority)

        # We store every species ever loaded to allow dynamic filtering
        self._all_species: List[Species] = []

        # Cached "best" species mapping (unique by ID), built on load() or on demand
        self.species: Dict[str, Species] = {}

    @classmethod
    def _build_priority_map(
        cls,
        source_priority: Optional[Union[Sequence[str], Dict[str, int]]],
    ) -> Dict[str, int]:
        """Build a normalized source-priority map.

        Args:
            source_priority: Either an ordered sequence (highest first) or a
                dict mapping source name to integer priority.

        Returns:
            Dict mapping source name to integer priority (higher is better).
        """
        default_order = list(cls._DEFAULT_PRIORITY_ORDER)
        default_map = {src: len(default_order) - i for i, src in enumerate(default_order)}

        if source_priority is None:
            return default_map

        if isinstance(source_priority, dict):
            custom_map = dict(default_map)
            for src, value in source_priority.items():
                custom_map[str(src)] = int(value)
            return custom_map

        ordered: List[str] = []
        for src in source_priority:
            s = str(src)
            if s not in ordered:
                ordered.append(s)
        for src in default_order:
            if src not in ordered:
                ordered.append(src)

        return {src: len(ordered) - i for i, src in enumerate(ordered)}

    def set_source_priority(
        self,
        source_priority: Optional[Union[Sequence[str], Dict[str, int]]],
    ) -> None:
        """Set source-priority rules used during deduplication.

        Args:
            source_priority: Either an ordered sequence (highest first) or a
                dict mapping source name to integer priority.
        """
        self._source_priority_map = self._build_priority_map(source_priority)

    def __repr__(self) -> str:
        status = (
            f"{len(self._all_species)} total entries"
            if self._all_species
            else "Not loaded"
        )
        return f"<{self.__class__.__name__}: {status}>"

    def __str__(self) -> str:
        if not self._all_species:
            return "SpeciesDatabase (Empty - call .load() to populate)"

        from collections import Counter

        counts = Counter(sp.source for sp in self._all_species)

        lines = [f"SpeciesDatabase containing {len(self._all_species)} entries:"]
        for src, count in counts.most_common():
            lines.append(f"  • {src}: {count}")

        return "\n".join(lines)

    def __len__(self) -> int:
        return len(self.species)

    def load(
        self,
        include_nasa7: bool = True,
        include_nasa9: bool = True,
        include_afcesic: bool = True,
        include_janaf: bool = False,
        include_shomate: bool = True,
        include_terra: bool = True,
        source_priority: Optional[Union[Sequence[str], Dict[str, int]]] = None,
    ) -> None:
        """Load thermodynamic databases into the raw species list."""
        if source_priority is not None:
            self.set_source_priority(source_priority)

        self._all_species.clear()
        self.species.clear()

        # Load all requested databases
        if include_afcesic and self.afcesic_path:
            self._load_afcesic()
        if include_terra and self.terra_path:
            self._load_terra()
        if include_nasa7:
            self._load_nasa7()
        if include_nasa9:
            self._load_nasa9()
        if include_janaf:
            self._load_janaf()
        if include_shomate and self.shomate_path:
            self._load_shomate()

        # Build the default 'species' dict (deduplicated across ALL loaded)
        self.species = self._deduplicate(self._all_species)
        logging.info(
            "SpeciesDatabase loaded: %d entries, %d unique species.",
            len(self._all_species),
            len(self.species),
        )

    def _deduplicate(self, species_list: List[Species]) -> Dict[str, Species]:
        """Deduplicate a list of species based on priority and temperature range."""

        def _t_range(sp: Species) -> float:
            if isinstance(sp, (NASANineCoeff, ShomateCoeff)):
                return sp.temperatures[-1] - sp.temperatures[0]
            if isinstance(sp, NASASevenCoeff):
                return sp.T_high - sp.T_low
            try:
                t = sp._JANAF__temperature  # type: ignore
                return float(t[-1] - t[0])
            except AttributeError:
                return 0.0

        def _priority(sp: Species) -> int:
            """Priority for deduplication. Higher is better."""
            return self._source_priority_map.get(sp.source, 0)

        best: Dict[tuple, Species] = {}
        for sp in species_list:
            key = (tuple(sorted(sp.elements.items())), sp.state)
            existing = best.get(key)
            if existing is None:
                best[key] = sp
            else:
                # Prefer higher source priority, then wider T range
                if (_priority(sp), _t_range(sp)) > (
                    _priority(existing),
                    _t_range(existing),
                ):
                    best[key] = sp

        # Return dict keyed by canonical ID (e.g. H2O_G)
        return {f"{sp.formula}_{sp.state}": sp for sp in best.values()}

    def get_species(
        self,
        elements: set,
        max_atoms: Optional[int] = None,
        enabled_databases: Optional[List[str]] = None,
    ) -> List[Species]:
        """Return unique product species from enabled databases for specified elements."""
        if not self._all_species:
            raise RuntimeError("No species loaded — call SpeciesDatabase.load() first.")

        # 1. Filter raw list by source and elements
        candidates = []
        for sp in self._all_species:
            if enabled_databases is not None and sp.source not in enabled_databases:
                continue

            chemical_elements = {e for e in sp.elements if e != "e-"}
            if not chemical_elements.issubset(elements):
                continue

            if max_atoms is not None:
                total_atoms = sum(
                    int(round(v)) for k, v in sp.elements.items() if k != "e-"
                )
                if total_atoms > max_atoms:
                    continue
            candidates.append(sp)

        # 2. Deduplicate the filtered set
        deduped = self._deduplicate(candidates)
        return list(deduped.values())

    def get_species_containing(
        self,
        element: str,
        enabled_databases: Optional[List[str]] = None,
    ) -> List[Species]:
        """Return unique species containing a specific element from enabled databases."""
        if not self._all_species:
            raise RuntimeError("No species loaded — call SpeciesDatabase.load() first.")

        candidates = []
        for sp in self._all_species:
            if enabled_databases is not None and sp.source not in enabled_databases:
                continue
            if element in sp.elements:
                candidates.append(sp)

        deduped = self._deduplicate(candidates)
        return list(deduped.values())

    def find(self, formula: str, phase: str = "G") -> Species:
        """Find the best matching species across all loaded databases."""
        if not self._all_species:
            raise RuntimeError("No species loaded.")

        phase_up = phase.upper()
        canonical = f"{formula}_{phase_up}"
        if canonical in self.species:
            return self.species[canonical]

        # Hill-normalised canonical ID.
        hill = self._hill_from_string(formula)
        if hill and hill != formula:
            hill_id = f"{hill}_{phase_up}"
            if hill_id in self.species:
                return self.species[hill_id]

        # Alias/Full search
        formula_norm = formula.lstrip("*").upper()
        candidates = []
        for sp in self._all_species:
            alias = getattr(sp, "alias", None)
            if (alias and alias.lstrip("*").upper() == formula_norm) or (
                sp.formula.upper() == formula_norm
            ):
                candidates.append(sp)

        if not candidates:
            raise KeyError(f"Species {formula!r} (phase={phase!r}) not found.")

        # Deduplicate candidates to pick the best one
        best_of_match = self._deduplicate(candidates)
        # Try to find the one with the right phase
        for sp in best_of_match.values():
            if sp.state.upper() == phase_up:
                return sp

        return list(best_of_match.values())[0]

    @staticmethod
    def _hill_from_string(formula: str) -> Optional[str]:
        """Re-sort a molecular formula string into Hill order."""
        import re as _re

        tokens = _re.findall(r"([A-Z][a-z]?)(\d*)", formula)
        if not tokens:
            return None
        counts: Dict[str, float] = {}
        for sym, num in tokens:
            if not sym:
                continue
            counts[sym] = counts.get(sym, 0.0) + (float(num) if num else 1.0)
        order = []
        for sym in ("C", "H"):
            if sym in counts:
                order.append(sym)
        for sym in sorted(counts.keys()):
            if sym not in ("C", "H"):
                order.append(sym)
        parts = []
        for sym in order:
            n = int(counts[sym])
            parts.append(sym if n == 1 else f"{sym}{n}")
        return "".join(parts)

    @staticmethod
    def _phase_to_state(phase: str) -> Literal["S", "L", "G"]:
        p = phase.strip().upper()
        if p.startswith("G"):
            return "G"
        if p.startswith("L"):
            return "L"
        return "S"

    def _load_nasa7(self) -> None:
        with open(self.nasa7_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for sp_id, rec in data.items():
            try:
                sp = NASASevenCoeff(
                    elements=rec["elements"],
                    state=self._phase_to_state(rec.get("phase", "G")),
                    temperature=(rec["t_low"], rec["t_mid"], rec["t_high"]),
                    coefficients=(rec["coeffs"]["low"], rec["coeffs"]["high"]),
                    phase=rec.get("phase"),
                )
                setattr(sp, "source_attribution", "NASA-7")
                self._all_species.append(sp)
            except Exception as e:
                logging.debug("NASA-7 skip %s: %s", sp_id, e)

    def _load_nasa9(self, path: Optional[str] = None) -> None:
        with open(path or self.nasa9_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for sp_id, rec in data.items():
            try:
                segs = rec.get("segments", [])
                temps = []
                for seg in segs:
                    if not temps:
                        temps.append(seg["t_low"])
                    temps.append(seg["t_high"])
                exponents = tuple(tuple(seg["exponents"]) for seg in segs)
                coefficients = tuple(
                    tuple(seg["coeffs"]) + (seg["b1"], seg["b2"]) for seg in segs
                )
                sp = NASANineCoeff(
                    elements=rec["elements"],
                    state=self._phase_to_state(rec.get("phase", "G")),
                    temperatures=tuple(temps),
                    exponents=exponents,
                    coefficients=coefficients,
                    phase=rec.get("phase"),
                    alias=rec.get("alias"),
                    source_attribution="NASA-9",
                )
                self._all_species.append(sp)
            except Exception as e:
                logging.debug("NASA-9 skip %s: %s", sp_id, e)

    def _load_janaf(self) -> None:
        with open(self.janaf_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            next(reader)
            rows_by_id = {}
            meta_by_id = {}
            for row in reader:
                if len(row) < 9:
                    continue
                sp_id = row[0]
                if sp_id not in meta_by_id:
                    meta_by_id[sp_id] = (json.loads(row[1]), row[2])
                rows_by_id.setdefault(sp_id, []).append(row[3:])

        for sp_id, rows in rows_by_id.items():
            try:
                elements, phase = meta_by_id[sp_id]
                p_upper = phase.strip().upper()
                # "REF" entries are standard-state reference elements (e.g. H2, O2, N2)
                # which are gas-phase species.  All other non-gas phases are skipped.
                if p_upper != "REF" and self._phase_to_state(phase) != "G":
                    continue
                temps, cps, hs, ss = [], [], [], []
                h_f = 0.0
                for r in rows:
                    T, cp, s, h = (
                        float(r[0]),
                        float(r[1]),
                        float(r[2]),
                        float(r[4]) * 1000,
                    )
                    if abs(T - 298.15) < 0.01:
                        h_f = float(r[5]) * 1000
                    temps.append(T)
                    cps.append(cp)
                    hs.append(h)
                    ss.append(s)
                sp = JANAF(
                    elements,
                    "G",
                    tuple(temps),
                    tuple(cps),
                    tuple(hs),
                    tuple(ss),
                    phase,
                    h_f,
                )
                setattr(sp, "source_attribution", "JANAF")
                self._all_species.append(sp)
            except Exception as e:
                logging.debug("JANAF skip %s: %s", sp_id, e)

    def _load_shomate(self) -> None:
        with open(self.shomate_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for sp_id, rec in data.items():
            if sp_id.startswith("_"):
                continue
            try:
                segs = rec.get("segments", [])
                temps = []
                for seg in segs:
                    if not temps:
                        temps.append(seg["t_low"])
                    temps.append(seg["t_high"])
                sp = ShomateCoeff(
                    rec["elements"],
                    self._phase_to_state(rec.get("phase", "S")),
                    tuple(temps),
                    tuple(tuple(s["coefficients"]) for s in segs),
                    rec.get("phase"),
                    rec.get("alias"),
                )
                setattr(sp, "source_attribution", "NASA-9")
                self._all_species.append(sp)
            except Exception as e:
                logging.debug("Shomate skip %s: %s", sp_id, e)

    def _load_afcesic(self) -> None:
        with open(self.afcesic_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for sp_id, rec in data.items():
            if sp_id.startswith("_"):
                continue
            if (
                rec.get("calibration_source") == "TERRA_STOICH"
                and rec.get("phase") in ("S", "L")
            ):
                continue
            try:
                sp = AFCESICCoeff(
                    rec["elements"],
                    self._phase_to_state(rec.get("phase", "G")),
                    (300.0, 1200.0, 6000.0),
                    tuple(rec["low_coefficients"]),
                    tuple(rec["high_coefficients"]),
                    rec["rf"],
                    rec["ch"],
                    rec.get("phase"),
                )
                setattr(sp, "source_attribution", "AFCESIC")
                self._all_species.append(sp)
            except Exception as e:
                logging.debug("AFCESIC skip %s: %s", sp_id, e)

    def _load_terra(self) -> None:
        with open(self.terra_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for sp_id, rec in data.items():
            if sp_id.startswith("_"):
                continue
            try:
                segs = rec.get("segments", [])
                temps = []
                for seg in segs:
                    if not temps:
                        temps.append(seg["t_low"])
                    temps.append(seg["t_high"])
                sp = ShomateCoeff(
                    rec["elements"],
                    self._phase_to_state(rec.get("phase", "S")),
                    tuple(temps),
                    tuple(tuple(s["coefficients"]) for s in segs),
                    rec.get("phase"),
                    rec.get("alias"),
                )
                setattr(sp, "source_attribution", "TERRA")
                self._all_species.append(sp)
            except Exception as e:
                logging.debug("TERRA skip %s: %s", sp_id, e)


if __name__ == "__main__":
    import timeit

    print("Initializing dummy species for benchmarking...")

    janaf = JANAF(
        elements={"H": 2, "O": 1},
        state="G",
        temperature=(298.15, 1000.0, 2000.0),
        specific_heat_capacity=(33.0, 40.0, 50.0),
        enthalpy=(0.0, 30000.0, 80000.0),
        entropy=(188.0, 230.0, 260.0),
        h_formation=-241826.0,
    )

    nasa7 = NASASevenCoeff(
        elements={"H": 2, "O": 1},
        state="G",
        temperature=(300.0, 1000.0, 5000.0),
        coefficients=(
            (4.19, -0.002, 0.00001, -2e-8, 9e-12, -29000.0, 0.5),
            (3.0, 0.001, 1e-6, -1e-10, 1e-14, -28000.0, 1.0),
        ),
    )

    nasa9 = NASANineCoeff(
        elements={"H": 2, "O": 1},
        state="G",
        temperatures=(200.0, 1000.0, 6000.0),
        exponents=((-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0),) * 2,
        coefficients=(
            (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, -29000.0, 1.0),
            (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, -28000.0, 1.0),
        ),
    )

    shomate = ShomateCoeff(
        elements={"H": 2, "O": 1},
        state="G",
        temperatures=(298.0, 1000.0, 6000.0),
        coefficients=(
            (30.0, 10.0, 0.1, -0.01, 0.001, -241.8, 188.0, -241.8),
            (30.0, 10.0, 0.1, -0.01, 0.001, -241.8, 188.0, -241.8),
        ),
    )

    afcesic = AFCESICCoeff(
        elements={"H": 2, "O": 1},
        state="G",
        temperature=(300.0, 1000.0, 5000.0),
        low_coefficients=(8.0, 1.0, 0.1, 0.01, 0.001),
        high_coefficients=(10.0, 1.0, -0.1, -0.01, -0.001),
        rf=-57.8,
        ch=45.0,
    )

    species_dict = {
        "JANAF": janaf,
        "NASA-7": nasa7,
        "NASA-9": nasa9,
        "Shomate": shomate,
        "AFCESIC": afcesic,
    }

    T_test = 850.0  # Scalar hot-path temperature
    n_iters = 100000

    print(f"Benchmarking scalar hot path at T={T_test} K ({n_iters:,} iterations)")
    print("-" * 50)

    for name, sp in species_dict.items():
        # Warm up JIT kernels if available
        _ = sp.reduced_gibbs(T_test)
        _ = sp.reduced_enthalpy(T_test)
        _ = sp.specific_heat_capacity(T_test)

        t_gibbs = timeit.timeit(lambda: sp.reduced_gibbs(T_test), number=n_iters)
        t_enth = timeit.timeit(lambda: sp.reduced_enthalpy(T_test), number=n_iters)
        t_cp = timeit.timeit(lambda: sp.specific_heat_capacity(T_test), number=n_iters)

        print(f"{name:.<15}")
        print(f"  reduced_gibbs      : {t_gibbs:.5f} s")
        print(f"  reduced_enthalpy   : {t_enth:.5f} s")
        print(f"  specific_heat_cap  : {t_cp:.5f} s")
        print()
