"""
PropellantDatabase — load propellant ingredients and formulations from TOML.

The TOML file contains two table arrays:

  [[ingredient]]  — a pure chemical species that enters the combustion chamber.
  [[formulation]] — a named blend of ingredients by mass fraction.

Each ingredient either references an entry in SpeciesDatabase via ``thermo_id``
(the canonical ``{HillFormula}_{PHASE}`` key) or defines its own element
composition and thermodynamic properties inline.  Inline ingredients are
represented as :class:`SyntheticSpecies` objects with a constant Cp model.

The key output of this module is :class:`PropellantMixture`, a lightweight
dataclass holding the ``{Species: moles}`` dict, total reactant enthalpy, and
the element set needed to query :py:meth:`SpeciesDatabase.get_species`.
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

import dataclasses
import logging
import math
import tomllib
from typing import Dict, FrozenSet, List, Optional, Set, Tuple, Union

import numpy as np

from prometheus_equilibrium.core.constants import UNIVERSAL_GAS_CONSTANT as R
from prometheus_equilibrium.equilibrium.species import Species, SpeciesDatabase

_SCALAR_TYPES = (int, float, np.floating, np.integer)
_T_REF = 298.15  # K — standard reference temperature


def _elements_to_hill(elements: dict) -> str:
    """Return the Hill-canonical formula string for an element-count dict.

    Hill order: C first, H second, then remaining elements alphabetically.
    Count of 1 is omitted (e.g. ``O`` not ``O1``).

    Args:
        elements: Mapping of element symbol → atom count (may be float).

    Returns:
        Hill-canonical formula string, e.g. ``"Al2O3"``.
    """
    counts = {sym: int(round(n)) for sym, n in elements.items() if n > 0}
    order: list[str] = []
    for sym in ("C", "H"):
        if sym in counts:
            order.append(sym)
    for sym in sorted(counts):
        if sym not in ("C", "H"):
            order.append(sym)
    return "".join(sym if counts[sym] == 1 else f"{sym}{counts[sym]}" for sym in order)


# ---------------------------------------------------------------------------
# SyntheticSpecies
# ---------------------------------------------------------------------------


class SyntheticSpecies(Species):
    """Constant-Cp species for propellant ingredients not in the thermo database.

    Used for polymer binders (HTPB, CTPB, PBAN), kerosene-type fuels (RP-1),
    and other complex propellants whose thermo data is not available as a
    polynomial fit.  The thermodynamic model is:

    .. code-block:: text

        Cp°(T) = cp                               [J/mol/K]
        H°(T)  = dHf298 + cp × (T − 298.15)      [J/mol]
        S°(T)  = cp × ln(T / 298.15)              [J/mol/K]

    The entropy is anchored at S°(298.15 K) = 0 — a deliberate simplification
    valid because synthetic species only appear as *reactants*, never as
    equilibrium products.  Only :py:meth:`enthalpy` is evaluated by the
    solver's HP/SP energy constraint; Gibbs minimisation never uses these
    objects.

    Attributes:
        dHf298: Standard enthalpy of formation at 298.15 K [J/mol].
        cp: Molar heat capacity (constant approximation) [J/mol/K].
        alias: Optional human-readable name.
    """

    def __init__(
        self,
        elements: dict,
        state: str,
        dHf298: float,
        cp: float,
        phase: Optional[str] = None,
        alias: Optional[str] = None,
        molar_mass_g_mol: Optional[float] = None,
    ) -> None:
        super().__init__(elements, state, phase)
        self.dHf298 = float(dHf298)
        self.cp = float(cp)
        self.alias = alias
        self._molar_mass_kg = (
            float(molar_mass_g_mol) / 1000.0 if molar_mass_g_mol is not None else None
        )

    def molar_mass(self) -> float:
        """Return the molar mass in kg/mol.

        Uses the explicitly supplied value when provided; otherwise computes
        from the element dictionary (inherited behaviour).
        """
        if self._molar_mass_kg is not None:
            return self._molar_mass_kg
        return super().molar_mass()

    def specific_heat_capacity(
        self, T: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        """Cp°(T) = cp  [J/mol/K]  (constant)."""
        if isinstance(T, _SCALAR_TYPES):
            return self.cp
        return np.full_like(np.asanyarray(T, dtype=float), self.cp)

    def enthalpy(self, T: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """H°(T) = dHf298 + cp × (T − 298.15)  [J/mol]."""
        if isinstance(T, _SCALAR_TYPES):
            return self.dHf298 + self.cp * (float(T) - _T_REF)
        T_arr = np.asanyarray(T, dtype=float)
        return self.dHf298 + self.cp * (T_arr - _T_REF)

    def entropy(self, T: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """S°(T) = cp × ln(T / 298.15)  [J/mol/K]  (S°₂₉₈ = 0 approximation)."""
        if isinstance(T, _SCALAR_TYPES):
            T_f = float(T)
            return self.cp * math.log(T_f / _T_REF) if T_f > 0 else float("nan")
        T_arr = np.asanyarray(T, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(T_arr > 0, self.cp * np.log(T_arr / _T_REF), float("nan"))

    def reduced_gibbs(self, T: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """G°/(RT) computed directly for scalars."""
        if isinstance(T, _SCALAR_TYPES):
            T_f = float(T)
            H = self.dHf298 + self.cp * (T_f - _T_REF)
            S = self.cp * math.log(T_f / _T_REF) if T_f > 0 else 0.0
            return (H - S * T_f) / (R * T_f)
        T_arr = np.asanyarray(T, dtype=float)
        return self.gibbs_free_energy(T_arr) / (R * T_arr)

    def reduced_enthalpy(self, T: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """H°/(RT)."""
        if isinstance(T, _SCALAR_TYPES):
            T_f = float(T)
            return self.enthalpy(T_f) / (R * T_f)
        T_arr = np.asanyarray(T, dtype=float)
        return self.enthalpy(T_arr) / (R * T_arr)


# ---------------------------------------------------------------------------
# PropellantMixture
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class PropellantMixture:
    """Resolved propellant mixture ready to pass to EquilibriumProblem.

    All quantities are *per 1 kg of total propellant mixture*.

    Attributes:
        reactants: ``{Species: moles}`` per kg of propellant.  Pass directly
            as the ``reactants`` argument of :class:`EquilibriumProblem`.
        enthalpy: Total reactant enthalpy H₀ [J/kg] evaluated at each
            ingredient's supply temperature.  Use as ``constraint1`` for HP
            problems.
        elements: Frozenset of element symbols present in the mixture (``e-``
            excluded).  Pass to :py:meth:`SpeciesDatabase.get_species` to
            build the product candidate list.
    """

    reactants: Dict[Species, float]
    enthalpy: float
    elements: FrozenSet[str]


# ---------------------------------------------------------------------------
# PropellantDatabase
# ---------------------------------------------------------------------------


class PropellantDatabase:
    """Load propellant ingredient and formulation data from a TOML file.

    Parameters
    ----------
    path:
        Path to a ``propellants.toml`` file.
    species_db:
        A loaded :class:`~Prometheus.equilibrium.species.SpeciesDatabase`.
        Required for ingredients that use ``thermo_id``; not needed if all
        ingredients define their composition inline.

    Example
    -------
    ::

        prop_db = PropellantDatabase(
            "prometheus/propellants/propellants.toml",
            species_db=db,
        )
        prop_db.load()

        # O/F = 6 by mass
        mixture = prop_db.mix([("LOX", 6.0), ("LH2", 1.0)])
        products = db.get_species(mixture.elements)

        problem = EquilibriumProblem(
            reactants=mixture.reactants,
            products=products,
            problem_type=ProblemType.HP,
            constraint1=mixture.enthalpy,
            constraint2=6.894757e6,   # 1000 psia in Pa
        )
    """

    def __init__(
        self,
        path: str,
        species_db: Optional[SpeciesDatabase] = None,
    ) -> None:
        self._path = path
        self._sdb = species_db
        self._ingredients: Dict[str, dict] = {}
        self._formulations: Dict[str, dict] = {}

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Parse the TOML file and resolve all ingredient Species objects."""
        with open(self._path, "rb") as f:
            data = tomllib.load(f)

        self._ingredients.clear()
        self._formulations.clear()

        for rec in data.get("ingredient", []):
            self._ingredients[rec["id"]] = self._resolve_ingredient(rec)

        for rec in data.get("formulation", []):
            self._formulations[rec["id"]] = rec

        logging.info(
            "PropellantDatabase loaded: %d ingredients, %d formulations.",
            len(self._ingredients),
            len(self._formulations),
        )

    def _resolve_ingredient(self, rec: dict) -> dict:
        """Attach a Species object to an ingredient record."""
        if "thermo_id" in rec:
            if self._sdb is None:
                raise RuntimeError(
                    f"Ingredient {rec['id']!r} requires thermo_id={rec['thermo_id']!r} "
                    "but no SpeciesDatabase was provided to PropellantDatabase."
                )
            try:
                # Use robust find() method to resolve differences like _C vs _S and non-Hill formulas
                tid = rec["thermo_id"]
                if "_" in tid:
                    formula, phase = tid.rsplit("_", 1)
                    if phase.upper() == "C":
                        phase = "S"
                    sp = self._sdb.find(formula, phase)
                else:
                    sp = self._sdb.find(tid)
            except KeyError:
                raise KeyError(
                    f"Ingredient {rec['id']!r}: thermo_id={rec['thermo_id']!r} "
                    f"not found in SpeciesDatabase.  Available keys (sample): "
                    f"{list(self._sdb.species.keys())[:5]}"
                )
        else:
            sp = self._make_synthetic(rec)

        return {**rec, "_species": sp}

    @staticmethod
    def _make_synthetic(rec: dict) -> SyntheticSpecies:
        """Create a constant-Cp SyntheticSpecies from an inline-composition record."""
        required = {"elements", "dHf298"}
        missing = required - rec.keys()
        if missing:
            raise ValueError(
                f"Ingredient {rec['id']!r}: inline-composition entry must provide "
                f"{sorted(missing)} (or add thermo_id to use the thermo database)."
            )
        state_map = {"S": "S", "L": "L", "G": "G"}
        state = state_map.get(rec.get("phase", "S").upper(), "S")
        return SyntheticSpecies(
            elements=dict(rec["elements"]),
            state=state,
            dHf298=float(rec["dHf298"]),
            cp=float(rec.get("cp", 0.0)),
            phase=rec.get("phase"),
            alias=rec.get("name") or rec.get("alias"),
        )

    # ------------------------------------------------------------------
    # Mixture building
    # ------------------------------------------------------------------

    def expand(self, formulation_id: str) -> PropellantMixture:
        """Resolve a named formulation to a :class:`PropellantMixture`.

        The mass fractions stored in the TOML are normalised to sum to exactly
        1.0, so the result is always per 1 kg of total propellant.

        Parameters
        ----------
        formulation_id:
            Key of a ``[[formulation]]`` entry in the TOML file.

        Returns
        -------
        PropellantMixture

        Raises
        ------
        KeyError
            If *formulation_id* is not found.
        """
        if formulation_id not in self._formulations:
            raise KeyError(
                f"Formulation {formulation_id!r} not found. "
                f"Available: {self.formulation_ids}"
            )
        form = self._formulations[formulation_id]
        components = [
            (c["ingredient"], float(c["mass_fraction"]))
            for c in form["components"]
            if float(c.get("mass_fraction", 0.0)) > 0.0
        ]
        return self._build_mixture(components)

    def mix(self, components: List[Tuple[str, float]]) -> PropellantMixture:
        """Build a :class:`PropellantMixture` from ``(ingredient_id, mass_amount)`` pairs.

        The amounts are in any consistent mass units (kg, g, proportional) —
        only their ratios matter.  They are normalised so the result is always
        per 1 kg of total mixture.

        Parameters
        ----------
        components:
            ``[(ingredient_id, mass_amount), ...]``

        Returns
        -------
        PropellantMixture

        Example
        -------
        ::

            mixture = db.mix([("LOX", 6.0), ("LH2", 1.0)])  # O/F = 6 by mass
            mixture = db.mix([("AP", 0.68), ("Al", 0.18), ("HTPB", 0.14)])
        """
        return self._build_mixture(components)

    def _build_mixture(self, components: List[Tuple[str, float]]) -> PropellantMixture:
        total_mass = sum(m for _, m in components)
        if total_mass <= 0.0:
            raise ValueError("Total component mass must be positive.")

        reactants: Dict[Species, float] = {}
        enthalpy = 0.0
        all_elements: Set[str] = set()

        for ingredient_id, mass_amount in components:
            if ingredient_id not in self._ingredients:
                raise KeyError(
                    f"Ingredient {ingredient_id!r} not found. "
                    f"Available: {self.ingredient_ids}"
                )
            ingr = self._ingredients[ingredient_id]
            sp: Species = ingr["_species"]
            t_supply = float(ingr.get("t_supply", _T_REF))
            mass_frac = mass_amount / total_mass  # kg per kg total
            moles = mass_frac / sp.molar_mass()  # mol per kg total

            # Merge identical species (e.g. two ingredients sharing a thermo_id)
            reactants[sp] = reactants.get(sp, 0.0) + moles
            enthalpy += moles * sp.enthalpy(t_supply)
            all_elements.update(e for e in sp.elements if e != "e-")

        return PropellantMixture(
            reactants=reactants,
            enthalpy=enthalpy,
            elements=frozenset(all_elements),
        )

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def find_ingredient(self, ingredient_id: str) -> dict:
        """Return the resolved record for *ingredient_id* (includes ``_species`` key)."""
        if ingredient_id not in self._ingredients:
            raise KeyError(
                f"Ingredient {ingredient_id!r} not found. "
                f"Available: {self.ingredient_ids}"
            )
        return self._ingredients[ingredient_id]

    def find_formulation(self, formulation_id: str) -> dict:
        """Return the raw TOML record for *formulation_id*."""
        if formulation_id not in self._formulations:
            raise KeyError(
                f"Formulation {formulation_id!r} not found. "
                f"Available: {self.formulation_ids}"
            )
        return self._formulations[formulation_id]

    def search_items(self) -> List[dict]:
        """Return a list of search records for use with the GUI search dialog.

        Each record has the keys ``id``, ``display``, and ``search_text``.
        ``display`` is the human-readable name shown in the UI.
        ``search_text`` is a pre-lowercased concatenation of all searchable
        fields (ID, name, CAS number, aliases) so the dialog only needs a
        single ``in`` check per keystroke.

        Returns:
            List of ``{id, display, search_text}`` dicts, one per ingredient,
            in sorted ID order.
        """
        items = []
        for ing_id in sorted(self._ingredients):
            rec = self._ingredients[ing_id]
            name = rec.get("name", ing_id)
            cas = rec.get("cas", "")
            aliases = rec.get("aliases", [])
            # Prefer inline elements dict; fall back to the resolved species object
            # (always present after load()) so thermo_id-based entries also get a formula.
            elements = rec.get("elements") or getattr(
                rec.get("_species"), "elements", {}
            )
            formula = _elements_to_hill(elements) if elements else ""
            source = rec.get("source", "")
            display = name
            parts = [ing_id, name, cas, formula, source] + list(aliases)
            search_text = " ".join(p for p in parts if p).lower()
            items.append({"id": ing_id, "display": display, "search_text": search_text})
        return items

    @property
    def ingredient_ids(self) -> List[str]:
        """Sorted list of all loaded ingredient IDs."""
        return sorted(self._ingredients)

    @property
    def formulation_ids(self) -> List[str]:
        """Sorted list of all loaded formulation IDs."""
        return sorted(self._formulations)
