"""Tests for ElementMatrix — stoichiometric matrix and basis selection."""

import numpy as np
import pytest

from prometheus_equilibrium.equilibrium.element_matrix import ElementMatrix

# ---------------------------------------------------------------------------
# Minimal mock that satisfies ElementMatrix's duck-typed species contract
# ---------------------------------------------------------------------------


class _Sp:
    """Tiny mock: only elements and condensed are needed by ElementMatrix."""

    def __init__(self, elements: dict, condensed: int = 0):
        self.elements = elements
        self.condensed = condensed


# ---------------------------------------------------------------------------
# Fixtures — H/O system with three species
# ---------------------------------------------------------------------------


@pytest.fixture
def ho_species():
    h2o = _Sp({"H": 2, "O": 1})  # most abundant
    h2 = _Sp({"H": 2})
    o2 = _Sp({"O": 2})
    return [h2o, h2, o2]


@pytest.fixture
def ho_elements():
    return ["H", "O"]


@pytest.fixture
def ho_em(ho_species, ho_elements):
    return ElementMatrix(ho_species, ho_elements)


# ---------------------------------------------------------------------------
# 1. Matrix assembly
# ---------------------------------------------------------------------------


def test_matrix_shape(ho_em):
    assert ho_em.matrix.shape == (3, 2)  # 3 species × 2 elements


def test_matrix_values(ho_em):
    # A[0]=H₂O: H=2, O=1; A[1]=H₂: H=2, O=0; A[2]=O₂: H=0, O=2
    expected = np.array([[2, 1], [2, 0], [0, 2]], dtype=float)
    np.testing.assert_array_equal(ho_em.matrix, expected)


def test_from_mixture_builds_elements():
    """ElementMatrix.from_mixture sets elements from union of species elements."""

    class _Mix:
        species = [_Sp({"H": 2}), _Sp({"H": 2, "O": 1})]

    em = ElementMatrix.from_mixture(_Mix())
    assert set(em.elements) == {"H", "O"}


# ---------------------------------------------------------------------------
# 2. Element abundance arithmetic
# ---------------------------------------------------------------------------


def test_element_abundances(ho_em):
    """A^T · n computes element totals correctly."""
    moles = np.array([1.0, 0.5, 0.25])
    # H: 2*1.0 + 2*0.5 + 0*0.25 = 3.0
    # O: 1*1.0 + 0*0.5 + 2*0.25 = 1.5
    expected = np.array([3.0, 1.5])
    np.testing.assert_allclose(ho_em.element_abundances(moles), expected)


def test_element_residuals_zero_at_balance(ho_em):
    """Residuals are zero when abundances match b₀."""
    moles = np.array([1.0, 0.5, 0.25])
    b0 = ho_em.element_abundances(moles)
    residuals = ho_em.element_residuals(moles, b0)
    np.testing.assert_allclose(residuals, [0.0, 0.0], atol=1e-15)


def test_element_residuals_nonzero(ho_em):
    """Residuals are non-zero when composition doesn't match b₀."""
    moles = np.array([0.5, 0.5, 0.5])
    b0 = np.array([2.0, 1.0])  # intentionally different
    residuals = ho_em.element_residuals(moles, b0)
    assert not np.allclose(residuals, 0.0)


# ---------------------------------------------------------------------------
# 3. Basis selection (Browne's method)
# ---------------------------------------------------------------------------


def test_select_basis_size(ho_em):
    """select_basis returns exactly S = n_elements basis species."""
    moles = np.array([0.5, 0.3, 0.2])
    basis, nonbasis = ho_em.select_basis(moles)
    assert len(basis) == ho_em.n_elements  # S = 2
    assert len(nonbasis) == ho_em.n_species - ho_em.n_elements  # 1


def test_select_basis_nonsingular(ho_em):
    """Basis sub-matrix B must be non-singular (det ≠ 0)."""
    moles = np.array([0.5, 0.3, 0.2])
    basis, _ = ho_em.select_basis(moles)
    B = ho_em.basis_matrix(basis)
    assert abs(np.linalg.det(B)) > 1e-10


def test_select_basis_prefers_largest_moles(ho_em):
    """The most abundant species is always included in the basis."""
    # H₂O (index 0) is most abundant — it must be in the basis
    moles = np.array([0.5, 0.3, 0.2])
    basis, _ = ho_em.select_basis(moles)
    assert 0 in basis


def test_select_basis_complete_partition(ho_em):
    """Basis and non-basis together cover every species index exactly once."""
    moles = np.array([0.5, 0.3, 0.2])
    basis, nonbasis = ho_em.select_basis(moles)
    all_indices = sorted(basis + nonbasis)
    assert all_indices == list(range(ho_em.n_species))


def test_select_basis_skips_zero_moles():
    """Species with zero moles must not be chosen as basis."""
    # 3 species, 2 elements; the first species has 0 moles
    sp = [_Sp({"H": 2, "O": 1}), _Sp({"H": 2}), _Sp({"O": 2})]
    em = ElementMatrix(sp, ["H", "O"])
    moles = np.array([0.0, 0.5, 0.4])  # H₂O has 0 moles
    basis, _ = em.select_basis(moles)
    assert 0 not in basis


def test_select_basis_raises_if_underdetermined():
    """RuntimeError if fewer than S linearly independent species exist."""
    # Two H₂ species — both have the same element row → rank 1 but S = 2.
    sp = [_Sp({"H": 2}), _Sp({"H": 2}), _Sp({"H": 4})]
    em = ElementMatrix(sp, ["H", "O"])
    moles = np.array([1.0, 1.0, 1.0])
    with pytest.raises(RuntimeError, match="linearly independent"):
        em.select_basis(moles)


# ---------------------------------------------------------------------------
# 4. Basis matrix and reaction coefficients
# ---------------------------------------------------------------------------


def test_basis_matrix_shape(ho_em):
    moles = np.array([0.5, 0.3, 0.2])
    basis, _ = ho_em.select_basis(moles)
    B = ho_em.basis_matrix(basis)
    S = ho_em.n_elements
    assert B.shape == (S, S)


def test_reaction_coefficients_shape(ho_em):
    moles = np.array([0.5, 0.3, 0.2])
    basis, _ = ho_em.select_basis(moles)
    nu = ho_em.reaction_coefficients(basis)
    assert nu.shape == (ho_em.n_species, ho_em.n_elements)


def test_reaction_coefficients_basis_rows_are_identity():
    """ν[basis_j, :] recovers the identity when ν = C·B⁻¹."""
    # For basis species themselves: C[j,:] is the j-th row of A restricted to
    # basis columns, and B = those S rows. So ν[basis_j, :] = e_j (unit vector).
    sp = [_Sp({"H": 2, "O": 1}), _Sp({"H": 2}), _Sp({"O": 2})]
    em = ElementMatrix(sp, ["H", "O"])
    moles = np.array([0.5, 0.3, 0.2])
    basis, _ = em.select_basis(moles)
    nu = em.reaction_coefficients(basis)
    B_nu = nu[basis, :]  # basis rows of ν
    # B_nu = B · B⁻¹ = I
    np.testing.assert_allclose(B_nu, np.eye(em.n_elements), atol=1e-12)


# ---------------------------------------------------------------------------
# 5. Rank analysis
# ---------------------------------------------------------------------------


def test_rank_full(ho_em):
    assert ho_em.rank() == 2  # H and O are independent


def test_rank_deficient():
    """Duplicate element column gives rank 1."""
    sp = [_Sp({"H": 2}), _Sp({"H": 4})]
    em = ElementMatrix(sp, ["H"])
    assert em.rank() == 1


def test_independent_elements_full_rank(ho_em):
    indep = ho_em.independent_elements()
    assert len(indep) == 2
    assert set(indep) == {"H", "O"}


# ---------------------------------------------------------------------------
# 6. Reduced sub-matrix
# ---------------------------------------------------------------------------


def test_reduced_drops_elements(ho_em):
    """reduced(['H']) should drop the O column."""
    em_h = ho_em.reduced(["H"])
    assert em_h.elements == ["H"]
    assert em_h.matrix.shape == (3, 1)
    np.testing.assert_array_equal(em_h.matrix[:, 0], [2.0, 2.0, 0.0])


def test_reduced_auto_uses_independent(ho_em):
    """reduced() with no argument calls independent_elements()."""
    em_r = ho_em.reduced()
    assert len(em_r.elements) == ho_em.rank()


# ---------------------------------------------------------------------------
# 7. Gas / condensed row partitioning
# ---------------------------------------------------------------------------


def test_gas_rows_shape():
    sp = [_Sp({"H": 2}), _Sp({"O": 2}), _Sp({"H": 2, "O": 1}, condensed=1)]
    em = ElementMatrix(sp, ["H", "O"])
    assert em.gas_rows().shape == (2, 2)


def test_condensed_rows_shape():
    sp = [_Sp({"H": 2}), _Sp({"O": 2}), _Sp({"H": 2, "O": 1}, condensed=1)]
    em = ElementMatrix(sp, ["H", "O"])
    assert em.condensed_rows().shape == (1, 2)
