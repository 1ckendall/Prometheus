Frozen Nozzle Expansion — Condensed-Phase Handling
====================================================

Background
----------

Rocket performance calculations require a thermodynamic model for how the
combustion gas expands through the nozzle.  Two limiting assumptions are in
common use:

**Shifting equilibrium** — the gas re-equilibrates at every cross-section.
Composition changes continuously from chamber to exit.  Gives an upper bound
on specific impulse.

**Frozen equilibrium** — composition is fixed at the chamber value.  All
species (gas and condensed) are carried unchanged into the nozzle, and only
temperature and pressure evolve.  Gives a lower bound on specific impulse.

For propellants that produce condensed products — most notably aluminised
propellants such as APCP, which produce Al₂O₃ — the frozen assumption requires
special care.  The equilibrium solver selects condensed phases based on their
Gibbs criterion at the chamber temperature (~3000–3500 K).  Aluminium oxide
is liquid at those conditions.  As the flow expands and cools, a strict frozen
model must carry that liquid phase all the way to the nozzle exit.  In
practice, the thermo data for the liquid species (Al₂O₃(L)) has valid
polynomial coefficients only above its melting point (~2327 K); below that
threshold the coefficients are uninformative placeholders (typically all-zero),
making the entropy and Cp evaluations unreliable and causing the frozen solver
to fail.

This document surveys the available options and explains the chosen strategy.

.. _frozen-condensed-options:

Options for Condensed-Phase Handling in Frozen Expansion
---------------------------------------------------------

Five approaches were analysed.  They are ordered roughly from simplest to most
physically rigorous.

Option 1 — Strict Frozen with Clamped Thermo Evaluation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Copy the chamber composition exactly (as cpropep/ProPEP does), but handle
out-of-range temperatures by clamping ``T`` to the species' declared
``[T_low, T_high]`` before evaluating enthalpy, entropy, and Cp.

**Pros**

* Matches cpropep reference behaviour directly — easiest to validate.
* Conceptually pure: "frozen" means nothing changes.
* Single-line fix in ``_solve_frozen_at_p``.

**Cons**

* Al₂O₃(L) below 2327 K does not exist physically; clamping gives wrong
  Cp, H, and S values.
* For APCP the condensed phase carries ~25–30 % of the total mixture enthalpy;
  errors here are not negligible.
* The NASA-7 liquid Al₂O₃ entry has all-zero low-range coefficients, so even
  with clamping the polynomial evaluates to zero Cp — the same failure mode,
  just deferred.

**Verdict** — Insufficient.  Does not address the zero-coefficient data
quality issue and is physically indefensible below the melting point.

Option 2 — Drop Condensed Species from Frozen Expansion
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Zero all condensed moles in the frozen mixture and solve the expansion as a
gas-only calculation.

**Pros**

* No temperature-range issues whatsoever.
* Trivially simple.

**Cons**

* Discards 2.78 mol Al₂O₃ (per basis kg in the APCP case) and its associated
  Cp (~120 J mol⁻¹ K⁻¹).  The gas mixture has too little heat capacity;
  predicted exit temperature and speed of sound are over-estimated.
* Results cannot be compared against any reference code (cpropep, CEA,
  ProPEP all retain condensed species).
* Isp error is roughly 10–15 % for 18 % Al loadings.

**Verdict** — Not acceptable for quantitative work.

Option 3 — Frozen Gas + Condensed Phase Transitions (Recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Hold the gas-phase mole fractions fixed (classical "frozen" assumption) but
allow condensed species to transition between phase forms as temperature
changes.  When the exit temperature falls below the melting point of a
condensed species, substitute the equivalent solid-phase species (same
chemical formula, condensed = 1) while conserving the total mole count.

**Physical justification** — In a rocket nozzle, Al₂O₃ droplets cool
rapidly and solidify once the gas temperature drops below the melting point.
The kinetics of solidification are orders of magnitude faster than any
appreciable chemical reaction.  "Frozen flow" in engineering usage means the
*gas* composition is kinetically frozen; condensed particles are free to
exchange heat with and re-phase the surrounding droplet population.  This
interpretation is consistent with the NASA CEA manual and standard rocket
propulsion textbooks.

**Pros**

* Physically correct: thermo data stays within its valid temperature range
  throughout the expansion.
* Latent heat of solidification is handled automatically because the NASA
  polynomial integration constants encode the formation enthalpy — the
  enthalpy difference between Al₂O₃(L) and Al₂O₃(S) at the melting point is
  the latent heat, which is released into the gas stream as Δ*T*.
* Gas composition is strictly frozen; element inventories are conserved.
* Matches the implicit behaviour of cpropep, which uses contiguous TERRA-style
  thermo data that bridges the phase transition without a discontinuity.
* Straightforward to implement once the species database properly labels liquid
  and solid condensed entries.

**Cons**

* Condensed *phase* changes, which is a conceptual departure from the strict
  academic definition of "frozen composition".
* Requires the database to correctly distinguish liquid from solid condensed
  entries for each species (see *Database Changes* below).
* Phase-pairing logic needs a lookup mechanism in ``SpeciesDatabase``.
* Latent heat is accounted for implicitly; a phase transition mid-expansion
  produces a small discontinuity in temperature profile (physically real, but
  may surprise users).

**Verdict** — Recommended.  Best balance of physical correctness, numerical
robustness, and compatibility with reference codes.

Option 4 — Frozen Gas + Re-Equilibrated Condensed Phases
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Fix gas moles exactly; at each temperature step during expansion, run a
condensed-only Gibbs minimisation to find the optimal condensed form for the
current element inventory from the condensed phase.

**Pros**

* Thermodynamically rigorous: condensed Gibbs energy is minimised at each
  cross-section.
* Handles multi-step transition sequences automatically (e.g.
  α → γ → liquid Al₂O₃).

**Cons**

* Requires a mini condensed-phase equilibrium solver, adding implementation
  complexity.
* For APCP there is only one condensed species (Al₂O₃), so Options 3 and 4
  are equivalent in practice.
* Generalisation is not needed for any current test case.

**Verdict** — Deferred; no benefit over Option 3 for the current propellant
set.

Option 5 — Gas-Entropy-Only Isentropic Target
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Replace ``s_target = chamber.entropy`` (total mixture) with
``s_target = chamber.gas_entropy(T_c, P_c)`` and solve the Newton loop using
``mix.gas_entropy(T, P_target)`` only.  Condensed species still contribute to
Cp and H through the ``mix.cp`` call.

**Pros**

* Gas isentropic path is exact.
* Avoids the out-of-range entropy evaluation entirely.

**Cons**

* Does not resolve the Cp evaluation issue: ``mix.cp(T)`` still calls
  ``sp.specific_heat_capacity(T)`` on the liquid Al₂O₃ species at exit
  temperatures, returning zero and making ``ds_dT = cp/T`` vanish — the
  same INVALID_THERMO_PROPERTIES failure.
* Introduces a thermodynamic inconsistency: condensed entropy is not conserved.
  For 2.78 mol Al₂O₃ vs ~30 mol gas, this introduces a ~50–100 K error
  in exit temperature.

**Verdict** — Does not fix the root cause; introduces a new inconsistency.

Comparison Table
----------------

.. list-table::
   :header-rows: 1
   :widths: 32 12 12 12 12 12 14

   * - Criterion
     - Opt 1
     - Opt 2
     - Opt 3
     - Opt 4
     - Opt 5
     - Notes
   * - No T-range nan/zero
     - No
     - Yes
     - Yes
     - Yes
     - No
     - Root cause addressed?
   * - Physically correct thermo
     - No
     - No
     - Yes
     - Yes
     - Partial
     -
   * - Gas composition frozen
     - Yes
     - Yes
     - Yes
     - Yes
     - Yes
     -
   * - Condensed Cp/H retained
     - Partial
     - No
     - Yes
     - Yes
     - Yes
     -
   * - Matches cpropep intent
     - Yes
     - No
     - Yes
     - Yes
     - No
     -
   * - Implementation effort
     - Low
     - Low
     - Medium
     - High
     - Low
     -

Chosen Approach: Option 3
-------------------------

Option 3 is implemented.  The key changes are described in the sections below.

TERRA Databases
~~~~~~~~~~~~~~~

The TERRA thermodynamic database (and its derived ``terra.json`` / Shomate
JSON format) does not distinguish between liquid and solid condensed phases.
All condensed species are assigned ``state = "S"`` by the TERRA binary parser.
The underlying Shomate polynomials, however, are calibrated to be continuous
across the melting point — TERRA stitches solid and liquid intervals into a
single entry.  As a result, TERRA-sourced condensed species evaluate Cp, H,
and S correctly at any temperature within their declared range without any
phase-transition logic.

When ``SpeciesDatabase`` is loaded with TERRA as the lowest-priority source
and NASA-9 / NASA-7 at higher priority, TERRA condensed species are used only
as a fallback when no NASA polynomial exists for a formula.  In that case the
TERRA species already bridges the phase transition implicitly, so no
substitution is needed during frozen expansion.

Database Changes
~~~~~~~~~~~~~~~~

The NASA-7 entry ``Al2O3_C`` in ``nasa7.json`` originates from the Burcat
library and represents liquid Al₂O₃.  It carries:

* ``t_low  = 327 K``, ``t_mid  = 2327 K`` (melting point), ``t_high = 6000 K``
* Low-range coefficients (T ∈ [327, 2327]): **all zeros** — the original
  data source only provided high-range coefficients.
* High-range coefficients (T ∈ [2327, 6000]): valid liquid Al₂O₃ data.

Two changes are required:

1. **Phase code mapping**: The Burcat parser maps phase code ``"C"``
   (condensed/liquid) to ``"G"`` because ``"C"`` is not in the explicit
   ``{"G", "S", "L"}`` allow-list.  This must be corrected: ``"C"`` and
   ``"B"`` (liquid in some JANAF-derived formats) should both map to ``"L"``.
   After the fix, ``Al2O3_C`` is loaded with ``state = "L"``, distinguished
   from the solid ``Al2O3_S`` / ``Al2O3_S_2`` entries.

2. **T-range tightening**: The declared ``t_low = 327 K`` is misleading — the
   species only has meaningful data above 2327 K.  Changing ``t_low`` to
   ``t_mid`` (2327 K) in the JSON means the ``NASASevenCoeff`` object will
   correctly return ``nan`` for T < 2327 K, triggering the phase-transition
   fallback rather than silently returning zero Cp.

These changes require rebuilding ``nasa7.json`` from the raw ``burcat7.thr``
source (``uv run prometheus-build-all-thermo``).

Solver Changes
~~~~~~~~~~~~~~

``PerformanceSolver._solve_frozen_at_p`` is extended with a
``_replace_out_of_range_condensed(mixture, T, db)`` step called at the start
of each Newton iteration.  For each condensed species in the mixture:

1. Evaluate ``sp.specific_heat_capacity(T)``.  If the result is ``nan``, the
   species polynomial is out of range at the current temperature.
2. Call ``db.condensed_phase_partner(sp, T)`` to find a species with the same
   ``elements`` and ``condensed == 1`` whose temperature range contains ``T``.
3. If a valid partner is found, replace the species object in the mixture while
   conserving its mole count.  The partner is sourced from the same database
   instance that was used to build the equilibrium problem.

A new ``SpeciesDatabase.condensed_phase_partner(species, T)`` method performs
the lookup: it iterates over ``_all_species`` filtering by matching elements,
``condensed == 1``, and a valid T-range check, returning the highest-priority
match.

This approach is robust to TERRA species (their Shomate polynomials cover the
full range without returning nan) and to future additions of multi-step
condensed phase sequences.

References
----------

* Gordon, S. and McBride, B.J., *NASA Technical Paper 1311* (1994) — frozen
  and shifting equilibrium nozzle performance formulation.
* Cruise, D.R., *NWC Technical Publication 6037* (1973/1979) — ProPEP
  frozen expansion by composition copy.
* cpropep source: ``libcpropep/src/performance.c:frozen_performance`` —
  chamber composition copied unchanged to throat and exit states.
