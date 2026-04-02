"""Sphinx configuration for Prometheus documentation."""

import os
import sys

# Make the package importable from the repo root.
sys.path.insert(0, os.path.abspath(".."))

# -- Project information -------------------------------------------------------
project = "Prometheus"
copyright = "2026, Prometheus contributors"
author = "Prometheus contributors"
release = "0.0.1"

# -- General configuration -----------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    # "sphinx.ext.viewcode",  # temporarily disabled to diagnose duplicate warnings
    "sphinx.ext.intersphinx",
]

# Napoleon: only Google style, disable NumPy style.
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True

# autosummary: generate stub .rst files automatically for every module/class.
# All module stub files are committed in docs/api/ and are kept up to date
# manually.  Disabling autosummary_generate stops Sphinx from running an
# in-memory generation pass over every module, which was the source of the
# "duplicate object description" warnings for dataclass fields (the generation
# pass documented EquilibriumSolution/PropellantMixture a second time alongside
# the automodule:: directive in each stub file).
autosummary_generate = False
autosummary_generate_overwrite = False
autosummary_imported_members = False

# autodoc defaults.
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "private-members": False,
    # Excluding __init__ from special-members avoids duplicate field descriptions
    # on dataclasses: autodoc would otherwise document each field once as a class
    # variable and once as an __init__ parameter, registering the same qualified
    # name (e.g. EquilibriumSolution.mixture) twice in the Python domain.
    "inherited-members": False,
    "show-inheritance": True,
}
autodoc_member_order = "bysource"

# Mock heavy system-library imports so autodoc can run on ReadTheDocs without
# Qt or Numba being installed.
autodoc_mock_imports = ["PySide6", "numba"]
# "description" mode moves typehints into the description body; combined with
# dataclass __init__ auto-generation this caused duplicate object registrations.
# "signature" keeps types in the function signature where they belong.
autodoc_typehints = "signature"

# intersphinx: link to NumPy, SciPy, Python standard library docs.
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Suppress warnings that are structural consequences of __init__.py re-exports:
# - app.add_directive: duplicate object descriptions arise when a symbol is
#   documented in both prometheus.equilibrium and prometheus.equilibrium.solution
#   (etc.) because __init__.py re-exports from sub-modules.
# - ref.python: "more than one target" for cross-references to re-exported names
#   (e.g. Mixture appears as both prometheus.equilibrium.Mixture and
#   prometheus.equilibrium.mixture.Mixture).
suppress_warnings = [
    # Ambiguous cross-references arise when the same symbol is accessible at
    # both the package level (prometheus.equilibrium.Mixture) and the submodule
    # level (prometheus.equilibrium.mixture.Mixture) due to __init__.py re-exports.
    # The canonical target is the submodule; this suppresses the noise.
    "ref.python",
]

# -- HTML output ---------------------------------------------------------------
html_theme = "renku"
html_static_path = ["_static"]
