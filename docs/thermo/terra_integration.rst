TERRA Database Integration
==========================

This document describes the integration of the TERRA thermodynamic database
(developed by Boris Trusov) into Prometheus.

Database Overview
-----------------

The TERRA database consists of binary files (``terra.bas`` for metadata and
``terra_a.bas`` for coefficients) containing approximately 2,600 chemical
species.  It uses a unique polynomial form for the reduced Gibbs energy
(Φ\*).

Conversion Logic
----------------

TERRA uses a 7-coefficient polynomial for the reduced Gibbs energy G\*(x),
where x = T / 10000 is the normalised temperature:

.. math::

   G^*(x) = f_1 + f_2 \ln x + f_3 x^{-2} + f_4 x^{-1}
             + f_5 x + f_6 x^2 + f_7 x^3

To integrate this into Prometheus these coefficients are converted into the
standard NIST Shomate form (with t = T / 1000):

.. math::

   C_p^\circ(t) = A + Bt + Ct^2 + Dt^3 + E/t^2

Coefficient Mapping
~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 10 45 35

   * - Shomate
     - TERRA Relation
     - Notes
   * - **A**
     - f₂
     - Constant Cₚ term
   * - **B**
     - 0.2 · f₅
     - Linear T term
   * - **C**
     - 0.06 · f₆
     - Quadratic T term
   * - **D**
     - 0.012 · f₇
     - Cubic T term
   * - **E**
     - 200.0 · f₃
     - Inverse quadratic term
   * - **F**
     - (H_meta − 10000·f₄ − shift) / 1000
     - Enthalpy constant (with reference shift)
   * - **G**
     - f₁ + f₂ − f₂ ln(10)
     - Entropy constant

Enthalpy Reference Shift
------------------------

TERRA uses H(0) as its reference point for enthalpy.  NASA and Prometheus use
H(298.15 K).  A **dynamic stoichiometric back-substitution** is applied:

1. **Dynamic discovery** — for each of the 79 unique elements and the electron
   (e⁻) in the database, the parser identifies the most stable reference
   substance (lowest raw Gibbs energy at 298.15 K).
2. **Reference calculation** — the raw enthalpy at 298.15 K is computed for
   each reference substance using the TERRA polynomials and metadata offsets.
3. **Universal shift** — for every species in the database (including ions),
   the weighted sum of element-specific reference enthalpies is subtracted
   based on the species' full elemental composition.
4. **Consistency** — all standard elements have H_f°(298.15) = 0, and all
   species (neutral and ionic) are correctly aligned with the NASA-9/JANAF
   reference scale used throughout Prometheus.

Accuracy Verification
---------------------

A comparison of over 1,400 overlapping species at 298.15 K shows excellent
agreement with NASA-9/CEA:

.. list-table::
   :header-rows: 1
   :widths: 40 50

   * - Property
     - Typical Deviation
   * - Enthalpy (H)
     - < 0.1 kJ/mol (common species)
   * - Entropy (S)
     - < 0.1 J/(mol·K)
   * - Heat Capacity (Cₚ)
     - < 0.05 J/(mol·K)
   * - Reduced Gibbs (G/RT)
     - < 0.05

.. note::

   Differences in large hydrocarbons or isomers are expected due to varying
   standard-state definitions between databases.

Usage in Prometheus
-------------------

The database is available as ``prometheus/thermo_data/terra.json``.  Load it
via :class:`~prometheus.equilibrium.species.SpeciesDatabase` by passing the
``terra_path`` argument.  Species originating from this database carry the
``[TERRA]`` source attribution.
