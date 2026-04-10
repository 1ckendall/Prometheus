"""
ElementMatrix — stoichiometric matrix relating species to elements.

The element matrix A is the core linear-algebra object shared by all three
equilibrium solvers.  Entry A[i, k] gives the number of atoms of element k
in species i (from the species' ``elements`` dict).

Mathematical role
-----------------
The equilibrium constraints are::

    A^T · n = b₀

where n is the vector of species mole amounts and b₀ is the vector of total
element abundances (fixed by the reactant composition).  The element residual

    Δb = b₀ − A^T · n

appears in the right-hand side of the element-balance equations in all three
solvers.

Basis selection (Browne's method — used by PEP and Hybrid solvers)
------------------------------------------------------------------
A *basis* is a subset of S species (S = number of elements) whose S×S
composition sub-matrix B is non-singular.  The *optimised* basis (Browne,
1960) picks the S species with the largest mole amounts that satisfy this
condition, found by sorting species descending by mole amount and accepting
candidates via the Gram-Schmidt orthogonality test.

The reaction-coefficient matrix ν = C · B⁻¹ expresses every non-basis
species as a stoichiometric combination of basis species.  Together B and ν
are the foundation of the Villars reaction-adjustment iteration and the
Hybrid solver's compressed Newton system.

Charge balance
--------------
The electron pseudo-element ``"e-"`` is included as an ordinary column so
that ionic species are balanced automatically.

Singular-system handling (Gordon-McBride solver)
------------------------------------------------
If the reactants do not supply every element represented in the product
species list, the corresponding column of A has no matching b₀ entry and
the Jacobian will be singular.  :py:meth:`independent_elements` identifies
the linearly independent subset of elements via QR decomposition, allowing
the G-McB solver to drop redundant constraints (RP-1311 §3.2).
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

from typing import List, Optional, Tuple

import numpy as np

from prometheus_equilibrium.equilibrium.mixture import Mixture
from prometheus_equilibrium.equilibrium.species import Species


class ElementMatrix:
    """Stoichiometric matrix A mapping species to element abundances.

    Parameters
    ----------
    species : list of Species
        All species in the mixture (gas first, condensed second).
    elements : list of str
        Ordered list of element symbols that define the columns of A.
        Typically derived from the union of all elements present in *species*.
    """

    def __init__(self, species: List[Species], elements: List[str]) -> None:
        self._species = species
        self._elements = list(elements)
        self._matrix = self._build(species, self._elements)

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_mixture(cls, mixture: Mixture) -> "ElementMatrix":
        """Build from a :py:class:`~Prometheus.equilibrium.mixture.Mixture`.

        The element list is the union of all elements across all species,
        sorted alphabetically (with ``"e-"`` placed last if present).
        """
        all_elements: set[str] = set()
        for sp in mixture.species:
            all_elements.update(sp.elements.keys())

        regular = sorted(el for el in all_elements if el != "e-")
        has_electron = "e-" in all_elements
        elements = regular + (["e-"] if has_electron else [])

        return cls(mixture.species, elements)

    @staticmethod
    def _build(species: List[Species], elements: List[str]) -> np.ndarray:
        """Populate the A matrix from species.elements dicts.

        A[i, k] = stoichiometric coefficient of element k in species i.
        Missing elements default to 0.
        """
        n_sp = len(species)
        n_el = len(elements)
        A = np.zeros((n_sp, n_el), dtype=float)
        el_index = {el: k for k, el in enumerate(elements)}
        for i, sp in enumerate(species):
            for el, coeff in sp.elements.items():
                k = el_index.get(el)
                if k is not None:
                    A[i, k] = coeff
        return A

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def matrix(self) -> np.ndarray:
        """The raw stoichiometric matrix A, shape (n_species, n_elements)."""
        return self._matrix

    @property
    def elements(self) -> List[str]:
        """Ordered element symbol list (defines the column ordering of A)."""
        return self._elements

    @property
    def species(self) -> List[Species]:
        """Species list (defines the row ordering of A)."""
        return self._species

    @property
    def n_species(self) -> int:
        return self._matrix.shape[0]

    @property
    def n_elements(self) -> int:
        return self._matrix.shape[1]

    # ------------------------------------------------------------------
    # Element abundance arithmetic
    # ------------------------------------------------------------------

    def element_abundances(self, moles: np.ndarray) -> np.ndarray:
        """Compute element abundances b = A^T · n [mol of element k].

        Parameters
        ----------
        moles : array-like, shape (n_species,)
            Species mole amounts nⱼ.

        Returns
        -------
        np.ndarray, shape (n_elements,)
            Total moles of each element in the mixture.
        """
        return self._matrix.T @ np.asarray(moles, dtype=float)

    def element_residuals(self, moles: np.ndarray, b0: np.ndarray) -> np.ndarray:
        """Compute the element-balance residual Δb = b₀ − A^T · n.

        This vector must be zero at equilibrium.  It forms the right-hand
        side of the element-balance equations in all three solvers.

        Args:
            moles: Current species mole amounts, shape (n_species,).
            b0: Target element abundances fixed by the reactant composition,
                shape (n_elements,).

        Returns:
            Residual vector b₀ − A^T·n, shape (n_elements,).
        """
        return np.asarray(b0, dtype=float) - self.element_abundances(moles)

    # ------------------------------------------------------------------
    # Basis selection (Browne's method — shared by PEP and Hybrid solvers)
    # ------------------------------------------------------------------

    def select_basis(self, moles: np.ndarray) -> Tuple[List[int], List[int]]:
        """Select the optimised basis using Browne's method (Gram-Schmidt).

        The optimised basis is the set of S = n_elements species with the
        largest mole amounts whose S×S composition sub-matrix B is
        non-singular.  It is found by sorting species in descending mole-
        amount order and accepting each candidate via the Gram-Schmidt
        orthogonality test (Cruise, NWC TP 6037, eq. 5).

        At the start of each outer iteration the basis is re-optimised
        (since the composition has changed).  This automatic re-selection
        serves two purposes:

        1. It keeps the basis well-conditioned (dominant species are always
           in the basis, so B⁻¹ is numerically stable).
        2. It relieves the user from choosing a basis manually.

        Parameters
        ----------
        moles : array-like, shape (n_species,)
            Current mole amounts nⱼ.  Species with nⱼ ≤ 0 are skipped.

        Returns
        -------
        basis_indices : list of int
            Indices (into ``self.species``) of the S basis species, in the
            order they were accepted.
        nonbasis_indices : list of int
            All remaining species indices, in original species order.

        Raises
        ------
        RuntimeError
            If fewer than S linearly independent species are found (the
            system is under-determined — reactant elements are not all
            representable).

        Notes
        -----
        The Gram-Schmidt test replaces the more expensive determinant test.
        A candidate row cₘ is orthogonalised against the current incomplete
        basis rows; if the result is the zero vector, the candidate is
        linearly dependent and rejected (Cruise eq. 5).  The PEP paper also
        describes updating the basis via a linear-programming tableau pivot
        (Smith & Missen, 1968) to avoid recomputing B⁻¹ from scratch each
        iteration; this optimisation can be added once the basic loop works.
        """
        n_basis = self.n_elements
        sort_order = np.argsort(moles)[::-1]  # descending mole amount

        basis_indices: List[int] = []
        B_rows: List[np.ndarray] = []

        for m in sort_order:
            # Allow zero-mole gas species as basis candidates — they appear last
            # in the descending sort and are only selected when positive-mole
            # species cannot complete the basis.  This prevents failures when the
            # Newton damping drives a whole element's species to exactly zero in
            # an early iteration (e.g. C-species in CH4/O2 at T_init).
            # Condensed-phase species with zero moles are absent from the system
            # and are still skipped.
            if moles[m] < 0.0:
                continue
            if moles[m] == 0.0 and getattr(self._species[m], "condensed", 0) != 0:
                continue
            candidate = self._matrix[m, :].copy().astype(float)
            # Gram-Schmidt: orthogonalise against already-accepted rows
            v = candidate.copy()
            for b in B_rows:
                v -= (np.dot(v, b) / np.dot(b, b)) * b
            if np.linalg.norm(v) > 1e-10:
                basis_indices.append(m)
                B_rows.append(v)
            if len(basis_indices) == n_basis:
                break

        if len(basis_indices) < n_basis:
            raise RuntimeError(
                f"select_basis: found only {len(basis_indices)} linearly independent "
                f"species for {n_basis} elements.  Check that all elements in b₀ are "
                f"representable by the product species list."
            )

        basis_set = set(basis_indices)
        nonbasis_indices = [i for i in range(self.n_species) if i not in basis_set]
        return basis_indices, nonbasis_indices

    def basis_matrix(self, basis_indices: List[int]) -> np.ndarray:
        """Return B = A[basis_indices, :], the S×S basis composition matrix.

        B[j, k] = atoms of element k in the j-th basis species.

        Parameters
        ----------
        basis_indices : list of int, length S
            Row indices of the basis species in A.

        Returns
        -------
        np.ndarray, shape (S, S) where S = n_elements
        """
        return self._matrix[basis_indices, :]

    def reaction_coefficients(self, basis_indices: List[int]) -> np.ndarray:
        """Compute ν = C · B⁻¹, the reaction-coefficient matrix.

        ν[i, j] is the stoichiometric coefficient of basis species j consumed
        when one mole of (non-basis) species i is formed from the basis.  The
        chemical reaction for species i is::

            Σⱼ ν[i,j] · basis_species[j]  →  species[i]

        Used to:

        - Compute equilibrium constants (PEP solver):
          ``ln Kᵢ = −Σⱼ ν[i,j] · gⱼ°/RT``
        - Correct basis mole amounts after each stoichiometric adjustment
          (PEP eq. 10).

        Parameters
        ----------
        basis_indices : list of int, length S

        Returns
        -------
        np.ndarray, shape (n_species, n_elements)
            Full ν matrix (rows for all species, including basis species
            themselves — their rows will be identity-like by construction).
        """
        B = self.basis_matrix(basis_indices)
        B_inv = np.linalg.inv(B)  # S×S inversion — tiny for rocket problems
        return self._matrix @ B_inv  # (N×S) · (S×S) = N×S

    # ------------------------------------------------------------------
    # Rank / independence analysis (used by Gordon-McBride solver)
    # ------------------------------------------------------------------

    def rank(self) -> int:
        """Numerical rank of A (number of independent element constraints).

        Computed via singular-value decomposition.  A rank less than
        ``n_elements`` indicates redundant elements that will make the G-McB
        Jacobian singular if included naively.
        """
        return int(np.linalg.matrix_rank(self._matrix, tol=1e-10))

    def independent_elements(self) -> List[str]:
        """Return the subset of element names forming a full-rank system.

        Uses QR decomposition with column pivoting on Aᵀ to identify a
        maximal linearly independent set of element columns.  Used by the
        G-McB solver to drop redundant element rows from the Jacobian
        (RP-1311 §3.2).

        Returns
        -------
        list of str
            Element symbols in a linearly independent ordering (most
            influential elements first, as determined by QR pivoting).
        """
        import scipy.linalg

        # QR with column pivoting on A (n_species × n_elements) selects the
        # most linearly independent element COLUMNS (indices into self._elements).
        _, _, pivots = scipy.linalg.qr(self._matrix, pivoting=True)
        r = self.rank()
        return [self._elements[p] for p in pivots[:r]]

    def reduced(self, active_elements: Optional[List[str]] = None) -> "ElementMatrix":
        """Return a new ElementMatrix restricted to *active_elements*.

        If *active_elements* is None, :py:meth:`independent_elements` is
        called automatically.  Used by the G-McB solver to remove redundant
        constraints before building the Newton system.

        Parameters
        ----------
        active_elements : list of str, optional
            Element symbols to keep.  Must be a subset of ``self.elements``.

        Returns
        -------
        ElementMatrix
            A new instance with the same species but only the specified
            element columns.
        """
        if active_elements is None:
            active_elements = self.independent_elements()
        col_idx = [self._elements.index(e) for e in active_elements]
        new_em = ElementMatrix.__new__(ElementMatrix)
        new_em._species = self._species
        new_em._elements = list(active_elements)
        new_em._matrix = self._matrix[:, col_idx]
        return new_em

    # ------------------------------------------------------------------
    # Jacobian sub-blocks (used by Gordon-McBride and Hybrid solvers)
    # ------------------------------------------------------------------

    def gas_rows(self) -> np.ndarray:
        """Sub-matrix of A for gas-phase species only, shape (n_gas, n_elements)."""
        n_gas = sum(1 for sp in self._species if sp.condensed == 0)
        return self._matrix[:n_gas, :]

    def condensed_rows(self) -> np.ndarray:
        """Sub-matrix of A for condensed-phase species, shape (n_cnd, n_elements)."""
        n_gas = sum(1 for sp in self._species if sp.condensed == 0)
        return self._matrix[n_gas:, :]

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"ElementMatrix({self.n_species} species × {self.n_elements} elements)\n"
            f"  elements: {self._elements}\n"
            f"  A =\n{self._matrix}"
        )
