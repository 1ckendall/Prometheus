"""
Mixture — a collection of Species with associated mole amounts.

This is the central thermodynamic state object passed between the equilibrium
solver components.  It holds the current estimate of the equilibrium
composition and computes weighted-sum mixture properties needed at each
Newton iteration.

Species ordering convention
---------------------------
Gas-phase species are stored first (indices 0 … n_gas-1), followed by
condensed-phase species (indices n_gas … n_species-1).  This matches the
block structure of the Jacobian in the Gordon-McBride method (RP-1311 §2).

Units
-----
- Temperature : K
- Pressure    : Pa  (standard reference P° = 1e5 Pa = 1 bar)
- Enthalpy    : J/mol
- Entropy     : J/(mol·K)
- Moles       : mol
"""

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

from __future__ import annotations

from typing import Dict, List

import numpy as np

from prometheus_equilibrium.core.constants import REFERENCE_PRESSURE as P_REF
from prometheus_equilibrium.core.constants import UNIVERSAL_GAS_CONSTANT as R
from prometheus_equilibrium.equilibrium.species import Species


class Mixture:
    """A collection of thermodynamic species and their mole amounts.

    Parameters
    ----------
    species : list of Species
        All species in the mixture, ordered gas-first then condensed.
        The ordering must be consistent with the *moles* array.
    moles : array-like, shape (n_species,)
        Mole amounts nⱼ [mol] for each species.  Must be non-negative.

    Notes
    -----
    *moles* is stored as a mutable NumPy array so the solver can update it
    in-place during the Newton iteration without constructing new objects.
    Use :py:meth:`copy` to take a snapshot of the current state.
    """

    def __init__(self, species: List[Species], moles: np.ndarray) -> None:
        if len(species) != len(moles):
            raise ValueError(
                f"species ({len(species)}) and moles ({len(moles)}) must have the same length."
            )
        # Gas-phase species first, then condensed — enforced on construction.
        gas = [(sp, n) for sp, n in zip(species, moles) if sp.condensed == 0]
        cnd = [(sp, n) for sp, n in zip(species, moles) if sp.condensed != 0]
        ordered = gas + cnd
        self._species: List[Species] = [sp for sp, _ in ordered]
        self._moles: np.ndarray = np.array([n for _, n in ordered], dtype=float)
        self._n_gas: int = len(gas)

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_dict(cls, species_moles: Dict[Species, float]) -> "Mixture":
        """Construct from a ``{Species: moles}`` mapping.

        The dict can be in any order; gas species are automatically placed
        before condensed species in the internal representation.
        """
        species = list(species_moles.keys())
        moles = np.array(list(species_moles.values()), dtype=float)
        return cls(species, moles)

    def copy(self) -> "Mixture":
        """Return a deep copy with an independent moles array."""
        return Mixture(list(self._species), self._moles.copy())

    # ------------------------------------------------------------------
    # Species accessors
    # ------------------------------------------------------------------

    @property
    def species(self) -> List[Species]:
        """All species in gas-first order."""
        return self._species

    @property
    def moles(self) -> np.ndarray:
        """Mole amounts nⱼ [mol], shape (n_species,).  Mutable."""
        return self._moles

    @moles.setter
    def moles(self, value: np.ndarray) -> None:
        value = np.asarray(value, dtype=float)
        if value.shape != self._moles.shape:
            raise ValueError("Cannot change the number of species via moles setter.")
        self._moles[:] = value

    @property
    def n_species(self) -> int:
        return len(self._species)

    @property
    def n_gas(self) -> int:
        """Number of gas-phase species."""
        return self._n_gas

    @property
    def n_condensed(self) -> int:
        """Number of condensed-phase species."""
        return self.n_species - self.n_gas

    @property
    def gas_species(self) -> List[Species]:
        """Gas-phase species (condensed == 0)."""
        return [sp for sp in self._species if sp.condensed == 0]

    @property
    def condensed_species(self) -> List[Species]:
        """Condensed-phase species (condensed != 0)."""
        return [sp for sp in self._species if sp.condensed != 0]

    def gas_moles(self) -> np.ndarray:
        """Mole amounts for gas-phase species only."""
        return self._moles[: self.n_gas]

    def condensed_moles(self) -> np.ndarray:
        """Mole amounts for condensed-phase species only."""
        return self._moles[self.n_gas :]

    # ------------------------------------------------------------------
    # Composition
    # ------------------------------------------------------------------

    @property
    def total_moles(self) -> float:
        """Total moles n = Σⱼ nⱼ."""
        return float(self._moles.sum())

    @property
    def total_gas_moles(self) -> float:
        """Total gas-phase moles n_gas = Σⱼ∈gas nⱼ."""
        return float(self.gas_moles().sum())

    @property
    def mole_fractions(self) -> np.ndarray:
        """Mole fraction xⱼ = nⱼ / Σⱼ nⱼ, shape (n_species,).

        Returns zeros if total_moles is zero (avoids divide-by-zero).
        """
        n_total = self.total_moles
        if n_total == 0.0:
            return np.zeros_like(self._moles)
        return self._moles / n_total

    @property
    def mean_molar_mass(self) -> float:
        """Mean molar mass of the total mixture M̄ = Σⱼ xⱼ·Mⱼ [kg/mol].

        Uses mole fractions over all species (gas + condensed).  For solid
        rocket motors where condensed products (e.g. Al₂O₃) are present, use
        :py:attr:`gas_mean_molar_mass` for performance calculations.
        """
        m_bar = sum(
            x * sp.molar_mass() for x, sp in zip(self.mole_fractions, self._species)
        )
        return float(m_bar)

    @property
    def gas_mean_molar_mass(self) -> float:
        """Gas-phase mean molar mass M̄_gas = Σⱼ∈gas xⱼ·Mⱼ [kg/mol].

        Uses gas-phase mole fractions only (``xⱼ = nⱼ / n_gas``).

        For solid rocket motors with condensed products (e.g. Al₂O₃ from
        aluminised propellants), the condensed phase does not contribute to
        the pressure-generating gas and must be excluded when computing the
        speed of sound, specific impulse, and nozzle flow properties.
        Returns 0.0 if there are no gas-phase species.
        """
        n_gas = self.total_gas_moles
        if n_gas == 0.0:
            return 0.0
        return float(
            sum(
                (n / n_gas) * sp.molar_mass()
                for sp, n in zip(self._species, self._moles)
                if sp.condensed == 0
            )
        )

    @property
    def mass_fractions(self) -> np.ndarray:
        """Mass fraction Yⱼ = nⱼ·Mⱼ / Σₖ nₖ·Mₖ, shape (n_species,).

        Requires Species.molar_mass() to be implemented.
        """
        masses = np.array([sp.molar_mass() for sp in self._species]) * self._moles
        total_mass = masses.sum()
        if total_mass == 0.0:
            return np.zeros_like(self._moles)
        return masses / total_mass

    # ------------------------------------------------------------------
    # Mixture thermodynamic properties
    # ------------------------------------------------------------------

    def cp(self, T: float) -> float:
        """Mixture molar heat capacity at constant pressure [J/(mol·K)].

        Computed as the mole-fraction–weighted sum over all species::

            Cp_mix(T) = Σⱼ xⱼ · Cpⱼ(T)

        where xⱼ is the mole fraction and Cpⱼ(T) is the molar heat capacity
        of species j from its :py:class:`~Prometheus.chemical.Species` object.

        For condensed species Cp is included directly (no pressure-mixing term).
        """
        n_total = self.total_moles
        if n_total == 0.0:
            return 0.0
        cp_total = 0.0
        for sp, n_i in zip(self._species, self._moles):
            if n_i <= 0.0:
                continue
            cp_total += n_i * sp.specific_heat_capacity(T)
        return float(cp_total / n_total)

    def enthalpy(self, T: float) -> float:
        """Mixture molar enthalpy [J/mol].

        ::

            H_mix(T) = Σⱼ xⱼ · Hⱼ°(T)

        Note: Hⱼ°(T) must be the *absolute* standard enthalpy (formation +
        sensible), not the sensible-only H−H(298.15) stored in JANAF data.
        See :py:attr:`Prometheus.chemical.JANAF` for the current limitation.
        """
        n_total = self.total_moles
        if n_total == 0.0:
            return 0.0
        h_total = 0.0
        for sp, n_i in zip(self._species, self._moles):
            if n_i <= 0.0:
                continue
            h_total += n_i * sp.enthalpy(T)
        return float(h_total / n_total)

    def entropy(self, T: float, P: float = P_REF) -> float:
        """Mixture molar entropy [J/(mol·K)] at temperature T and pressure P.

        For a two-phase (gas + condensed) mixture the total entropy is::

            S_total(T, P) = Σⱼ∈gas nⱼ · [Sⱼ°(T) − R·ln(xⱼ) − R·ln(P/P°)]
                          + Σⱼ∈cnd nⱼ · Sⱼ°(T)

        where ``xⱼ = nⱼ / n_gas`` is the *gas-phase* mole fraction (condensed
        species do not contribute to partial pressure and have no mixing or
        pressure term).

        The returned intensive quantity is ``S_total / n_total``, i.e. entropy
        per mole of total mixture (gas + condensed).  For solid rocket motor
        performance calculations that are based on the gas phase only, use
        :py:meth:`gas_entropy` instead.
        """
        n_total = self.total_moles
        if n_total == 0.0:
            return 0.0
        return self.total_entropy(T, P) / n_total

    def gibbs(self, T: float, P: float = P_REF) -> float:
        """Mixture Gibbs free energy G = H − T·S [J/mol].

        At thermodynamic equilibrium this is minimised subject to the
        element-balance constraints.
        """
        return self.enthalpy(T) - T * self.entropy(T, P)

    def gas_entropy(self, T: float, P: float = P_REF) -> float:
        """Gas-phase molar entropy [J/(mol·K)] at temperature T and pressure P.

        Computes the entropy per mole of *gas-phase* species only::

            S_gas(T, P) = Σⱼ∈gas xⱼ · [Sⱼ°(T) − R·ln(xⱼ) − R·ln(P/P°)]

        where ``xⱼ = nⱼ / n_gas`` is the gas-phase mole fraction.  Condensed
        species are excluded entirely.

        This is the quantity needed for nozzle performance calculations (e.g.
        frozen or shifting isentropic expansion) where only the gas phase
        produces thrust.  Returns 0.0 if there are no gas-phase species.
        """
        n_gas = self.total_gas_moles
        if n_gas == 0.0:
            return 0.0
        s_gas = 0.0
        for sp, n_i in zip(self._species, self._moles):
            if sp.condensed != 0 or n_i <= 0.0:
                continue
            x_i = n_i / n_gas
            s_gas += x_i * (sp.entropy(T) - R * np.log(x_i) - R * np.log(P / P_REF))
        return float(s_gas)

    def total_gas_entropy(self, T: float, P: float = P_REF) -> float:
        """Total gas-phase entropy Σⱼ∈gas nⱼ·Sⱼ_mix(T,P) [J/K].

        This is the extensive gas-only entropy used for nozzle SP constraints
        when condensed-phase entropy is intentionally excluded from isentropic
        expansion modeling.
        """
        n_gas = self.total_gas_moles
        if n_gas == 0.0:
            return 0.0
        return float(n_gas * self.gas_entropy(T, P))

    # ------------------------------------------------------------------
    # Extensive properties (used in energy-balance rows of the Jacobian)
    # ------------------------------------------------------------------

    def total_enthalpy(self, T: float) -> float:
        """Total mixture enthalpy H_total = Σⱼ nⱼ·Hⱼ°(T) [J].

        Only species with positive mole amounts contribute, so zero-mole
        species whose thermo polynomials have no coverage at T (returning
        NaN) do not poison the sum via 0 × NaN = NaN.
        """
        h_total = 0.0
        for sp, n_i in zip(self._species, self._moles):
            if n_i <= 0.0:
                continue
            h_total += n_i * sp.enthalpy(T)
        return float(h_total)

    def total_entropy(self, T: float, P: float = P_REF) -> float:
        """Total mixture entropy S_total = Σⱼ nⱼ·Sⱼ_mix(T,P) [J/K]."""
        s_total = 0.0
        n_gas = self.total_gas_moles

        for i, sp in enumerate(self._species):
            n_i = self._moles[i]
            if n_i <= 0.0:
                continue

            s_std = sp.entropy(T)

            if sp.condensed == 0:
                x_i = n_i / n_gas if n_gas > 0 else 0.0
                if x_i > 0:
                    s_total += n_i * (s_std - R * np.log(x_i) - R * np.log(P / P_REF))
            else:
                s_total += n_i * s_std

        return float(s_total)

    def total_cp(self, T: float) -> float:
        """Total mixture heat capacity Cp_total = Σⱼ nⱼ·Cpⱼ(T) [J/K].

        Used in the energy constraint row of the Jacobian (RP-1311 eq. 2.27).
        Only positive-mole species are evaluated to avoid 0 × NaN propagation.
        """
        cp_total = 0.0
        for sp, n_i in zip(self._species, self._moles):
            if n_i <= 0.0:
                continue
            cp_total += n_i * sp.specific_heat_capacity(T)
        return float(cp_total)

    def total_gas_cp(self, T: float) -> float:
        """Total gas-phase heat capacity Σⱼ∈gas nⱼ·Cpⱼ(T) [J/K]."""
        cp_total = 0.0
        for sp, n_i in zip(self._species[: self.n_gas], self._moles[: self.n_gas]):
            if n_i <= 0.0:
                continue
            cp_total += n_i * sp.specific_heat_capacity(T)
        return float(cp_total)

    # ------------------------------------------------------------------
    # Log-space helpers (used by the Newton iteration)
    # ------------------------------------------------------------------

    def log_moles(self) -> np.ndarray:
        """Natural logarithm of mole amounts ln(nⱼ), shape (n_species,).

        Species with nⱼ ≤ 0 are given ln(nⱼ) = -∞ (represented as a very
        large negative number) and are treated as trace/inactive.
        """
        with np.errstate(divide="ignore"):
            return np.log(np.maximum(self._moles, 0.0))

    def set_from_log_moles(self, ln_n: np.ndarray) -> None:
        """Update mole amounts from a log-space array.

        Inverse of :py:meth:`log_moles`.  Called after each Newton step to
        convert the updated ln(nⱼ) back to physical moles.
        """
        self._moles[:] = np.exp(ln_n)

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:  # pragma: no cover
        lines = [
            f"Mixture({self.n_species} species, n_total={self.total_moles:.4g} mol)"
        ]
        for sp, n in zip(self._species, self._moles):
            tag = "gas" if sp.condensed == 0 else "cnd"
            lines.append(f"  [{tag}] {sp.elements}  n={n:.4g}")
        return "\n".join(lines)
