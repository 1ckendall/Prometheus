Thermodynamic Database Guide
=============================

This document describes the thermodynamic databases integrated into Prometheus,
their polynomial formats, and the reference-state calibration logic used to
ensure cross-database consistency.

Overview
--------

.. list-table::
   :header-rows: 1
   :widths: 15 25 22 20 28

   * - Database
     - Primary Source
     - Polynomial Format
     - Reference State
     - Calibration
   * - NASA-9
     - NASA CEA (NRL 2002)
     - 9-coefficient NASA
     - H_f°(298.15) = 0
     - Standard
   * - TERRA
     - Bauman MSTU (Trusov)
     - Shomate (converted)
     - H(0) = 0 (raw)
     - Dynamic stoichiometric shift
   * - AFCESIC
     - US Air Force
     - θ-based AFCESIC
     - Mixed / arbitrary
     - Hybrid NASA/TERRA calibration

NASA-9 / NASA-7
---------------

The gold standard for Prometheus.  These databases use piecewise polynomials
for Cₚ/R, H/RT, and S/R.

- **Reference point:** elements in their standard states have H_f°(298.15) = 0.
- **Format:** multi-segment (usually 2 or 3) polynomials covering
  200 K – 6000 K.

TERRA
-----

Contains approximately 2,600 species.  Originally uses a reduced Gibbs energy
(Φ\*) polynomial; converted to NIST Shomate form (t = T/1000) internally:

.. math::

   C_p^\circ(t) = A + Bt + Ct^2 + Dt^3 + E/t^2

**Calibration:** TERRA raw data is referenced to 0 K.  A dynamic stoichiometric
shift is applied:

1. The parser identifies the most stable state for every element in the TERRA
   binary.
2. For every species, the weighted sum of its constituent elements' raw
   enthalpies at 298.15 K is subtracted.
3. This aligns TERRA enthalpies exactly with the NASA-9 formation enthalpy
   scale.

See :doc:`terra_integration` for the full derivation and accuracy tables.

AFCESIC
-------

Contains specialised ionic and condensed species for metalized propellant
studies.

- **Format:** θ = T/1000 based polynomials with forward powers (low-T range)
  and inverse powers (high-T range).
- **Calibration (hybrid):**

  1. **NASA point-match** — if a species exists in NASA-9/7, the AFCESIC
     integration constants (RF and CH) are adjusted to match NASA H and S
     at 1000 K.
  2. **TERRA-stoich calibration** — for species unique to AFCESIC, the
     stoichiometric shift is calculated using TERRA element reference
     enthalpies.  This ensures that even unique AFCESIC species (e.g.
     Al₂MgO₄(s)) are correctly referenced to the same elemental baseline
     as the rest of the project.

Reference Enthalpies
--------------------

Prometheus uses a unified set of element reference enthalpies (H_raw at
298.15 K) derived from the TERRA database to normalise any database that does
not follow the H_f°(298.15) = 0 convention.

**Key element references:**

- H₂ (gas): 7.45 kJ/mol
- O₂ (gas): −2.33 kJ/mol
- C (graphite): 0.00 kJ/mol
- e⁻ (electron): 0.00 kJ/mol

Loading and Priority
--------------------

Source priority
~~~~~~~~~~~~~~~

When multiple databases cover the same species Prometheus uses a priority-based
selection (highest to lowest):

1. NASA-9
2. NASA-7
3. JANAF
4. TERRA
5. AFCESIC

Override the priority at load time:

.. code-block:: python

   db.load(source_priority=["JANAF", "NASA-9", "NASA-7", "TERRA", "AFCESIC"])

Rebuilding from raw data
~~~~~~~~~~~~~~~~~~~~~~~~

Regenerate the translated TERRA/AFCESIC databases from raw binaries:

.. code-block:: bash

   uv run prometheus-build-legacy all

Regenerate all thermo databases (NASA/JANAF + TERRA/AFCESIC) in one step:

.. code-block:: bash

   uv run prometheus-build-all-thermo
