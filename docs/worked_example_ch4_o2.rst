.. _worked-example-ch4-o2:

Worked Example: CH\ :sub:`4` / O\ :sub:`2` Combustion
=======================================================

This page walks through a complete equilibrium calculation for a methane–oxygen
rocket combustion chamber, using real numbers produced by Prometheus.  If you
are new to the idea of a "chemical equilibrium solver" the :ref:`background`
section below gives a plain-English introduction; if you already understand
what Gibbs energy minimisation does, skip straight to :ref:`problem-setup`.

.. _background:

Background: what the solver is trying to do
--------------------------------------------

When a fuel and oxidiser burn together, the hot products react further among
themselves before the gas leaves the nozzle.  At any given temperature and
pressure there is one particular mixture composition — the
**chemical equilibrium** — at which no further net reaction occurs.  Finding
that composition is the solver's job.

The solver does not model the rates of individual reactions.  It asks a
simpler question: *given a fixed amount of energy and a fixed pressure, what
mixture of product species minimises the total Gibbs free energy?*  The minimum
of Gibbs energy corresponds exactly to the state the gas would reach if left
alone at that temperature and pressure long enough for all reactions to
complete.  Gibbs energy is a thermodynamic potential that accounts for both
the enthalpy (heat content) of each species and the entropy (disorder) of the
mixture; the equilibrium mixture balances both.

A combustion problem also involves finding the temperature.  The constraint
is that the **enthalpy must be conserved**: the products must contain the same
total enthalpy as the cold reactants (an adiabatic, constant-pressure process —
the **HP problem**).  This couples the temperature and composition together —
changing the temperature changes which species are stable, which in turn
changes the enthalpy balance.

The two solvers in Prometheus — **MajorSpeciesSolver** and
**GordonMcBrideSolver** — take different numerical routes to the same answer.
Both are explained below through the same example calculation.

.. _problem-setup:

Problem definition
------------------

Propellant pair
~~~~~~~~~~~~~~~

**Methane / oxygen (CH**\ :sub:`4` **/ O**\ :sub:`2`\ **)** is the combination
used in the SpaceX Raptor engine.  It is an important test case because:

* Three elements are present (C, H, O), so the solver must track carbon species
  as well as the water and oxygen species that dominate simpler H\ :sub:`2`/O\ :sub:`2`
  problems.
* At stoichiometric mixture ratio the flame temperature exceeds 3600 K, where
  many species are partially dissociated — there is significant CO, OH, atomic
  H and O present alongside the expected CO\ :sub:`2` and H\ :sub:`2`\ O
  products.  Capturing this dissociation correctly is the key challenge.

Mixture ratio
~~~~~~~~~~~~~

The **oxidiser-to-fuel mass ratio O/F = 4.0** is used, which is exactly
stoichiometric for methane:

.. code-block:: text

   CH4  +  2 O2  -->  CO2  +  2 H2O          (complete combustion, balanced)

   Molar masses: CH4 = 16 g/mol,  O2 = 32 g/mol
   O/F (mass)  = (2 × 32) / (1 × 16) = 64 / 16 = 4.0   ✓

In the solver one mole of CH\ :sub:`4` is used as the basis, giving
**n**\ :sub:`CH4` = 1.0 mol and **n**\ :sub:`O2` = 2.0 mol.

Chamber pressure
~~~~~~~~~~~~~~~~

**P = 1 000 psia (6.895 MPa)**, a representative high-pressure rocket chamber
value.  This is above the critical pressure of water, which is why several
normally-liquid products appear as gases in the equilibrium composition.

Enthalpy constraint
~~~~~~~~~~~~~~~~~~~

For an HP (adiabatic, constant-pressure) problem the reactant enthalpy is
computed at the standard reference temperature of 298.15 K:

.. code-block:: text

   H0(CH4, 298.15 K) = -74 600 J/mol    (negative because CH4 is exothermic
                                          relative to the elements — its
                                          standard enthalpy of formation)
   H0(O2,  298.15 K) =       0 J/mol    (reference state: 0 by definition)

   H0_total = 1.0 × (-74 600) + 2.0 × 0 = -74 600 J

This negative value is the target: however the product mixture distributes its
atoms, its total enthalpy must equal −74 600 J.  Since the products include many
species with strong exothermic formation enthalpies (H\ :sub:`2`\ O, CO\ :sub:`2`),
they release far more energy than this, driving the temperature up to around
3 600 K.

Initial guess
~~~~~~~~~~~~~

The solver is started at **T\ :sub:`init` = 3 500 K** with the product moles
distributed equally across all gas-phase species as a starting guess.
**376 candidate product species** [#f1]_ are considered (all molecules and
atoms that can be formed from C, H, O and have thermodynamic data valid at
3 500 K).

.. [#f1] 391 species in the database contain only C, H and O; 15 of those have
   polynomial fits that do not extend to 3 500 K and are excluded for this
   calculation.

.. _major-species-solver:

MajorSpeciesSolver: inner and outer loops
-----------------------------------------

The MajorSpeciesSolver divides the problem into two nested loops.

The **inner loop** (``_tp_equilibrium``) fixes the temperature and solves for
the composition that is in chemical equilibrium at that T.  It is a Newton
iteration on a small (3×3 for three-element C/H/O) linear system.

The **outer loop** (``_temperature_search``) adjusts T until the computed
product enthalpy matches the reactant enthalpy H\ :sub:`0`.  It uses a
Newton step with interval-halving fallback.

Why is the inner matrix only 3×3?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

With 376 species and 3 elements (C, H, O), the stoichiometry is fully
determined by just 3 independent "basis" species — one per element.  Prometheus
selects these automatically using Browne's algorithm (implemented in
:py:meth:`~prometheus_equilibrium.equilibrium.element_matrix.ElementMatrix.select_basis`).
At any temperature guess the basis species are typically the three dominant
gas species (for example H\ :sub:`2`\ O, CO\ :sub:`2`, CO at stoichiometric
mixture ratio).

Every other species is either:

* **Major non-basis** — present in significant amounts, included in the Newton
  system by adding rows/columns; or
* **Minor** — its mole amount is set analytically from the solution without
  adding to the matrix size.

In practice at T = 3 500 K the solver classifies roughly **8 species as
major** and the remaining **221+ as minor**, so the Newton system is only
around 4×4 throughout.

First outer step (T = 3 500 K → inner solve)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

At the initial temperature guess of 3 500 K the inner loop runs for 50
iterations.  The table below shows selected steps; ``el_res`` is the maximum
element-balance residual (how far the mole amounts are from satisfying
conservation of C, H, and O atoms), and ``dlnn`` is the step size for the
total gas moles (a measure of how much the overall scale of the composition is
changing).

.. code-block:: text

   Inner loop at T = 3500.0 K
   ──────────────────────────────────────────────────────────────────
   iter   el_res      dlnn        n_major   n_minor    notes
   ────   ────────    ────────    ───────   ───────    ─────
      1   1.05e+01    1.36e+00      228         1      many species active
      2   1.55e+01    2.19e+00      187        42      species winnowed down
      3   2.48e+01    1.14e+00       83       146
      4   8.10e+01   -8.20e-02       32       197      residual spikes as
      5   5.14e+01   -8.92e-01        8       221      minor species branch off
      6   3.20e+01   -8.47e-01        8       221
      7   1.93e+01   -7.78e-01        9       220      ← start of quadratic phase
      8   1.43e+01   -6.78e-01        8       221
     13   2.80e+00   -5.72e-01        7       222
     14   8.98e-01   -3.83e-01        8       221
     15   1.29e-01   -1.48e-01        8       221
     16   1.97e-03   -2.93e-02        8       221      ← rapid convergence here
     17   2.69e-07   -7.39e-04        8       221
     18   1.14e-09   -2.80e-04        8       221      ← machine precision
     19   1.53e-12   -2.80e-04        8       221
   ──────────────────────────────────────────────────────────────────

The residual drops from ~80 to ~10\ :sup:`−12` in roughly 15 meaningful
iterations, with the steep quadratic drop (factor-of-1000 per step) starting
around iteration 15.  Newton's method has a characteristic
**quadratic convergence** signature: once the iterate is close to the solution,
the number of correct decimal digits roughly doubles each step.  This is the
"17 → 7 → 3 → 2.7 → 0.9 → 0.0013 → …" drop visible in the ``el_res`` column.

After iteration 18 the residual is limited by floating-point rounding
(≈10\ :sup:`−14`), but the total-moles step ``dlnn`` is still −2.8×10\ :sup:`−4`
— slightly above the 5×10\ :sup:`−6` convergence tolerance.  The composition
is correctly balanced but the total *scale* has not quite settled, because
3 500 K is the wrong temperature.  The inner loop therefore exits without
declaring convergence (flag ``inner_conv=False``).

Outer loop: finding the adiabatic flame temperature
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

With the composition converged at 3 500 K, the outer loop evaluates the
**energy residual** f(T):

.. code-block:: text

   f(T) = H_products(T, composition) − H0_reactants
        = H_products(3500 K) − (−74 600 J)

At 3 500 K the equilibrium mixture is too cool — the actual adiabatic flame
temperature must be higher.  The outer loop performs a Newton step to find the
temperature where f(T) = 0:

.. code-block:: text

   T_new = T − f(T) / f'(T) = T + ΔT

   where f'(T) = Cp (total heat capacity of the mixture).

The table below shows every outer step.  The composition snapshot at each
step is summarised by the three largest mole fractions.

.. code-block:: text

   Outer temperature search
   ──────────────────────────────────────────────────────────────────────────
   step   T [K]      |dlnT|      Top-3 species
   ────   ────────   ─────────   ────────────────────────────────────────────
      0   3500.00    1.20e-01    H2O=0.490  CO2=0.166  CO=0.126
      1   3919.05    2.83e-01    H2O=0.362  CO=0.170   OH=0.133   ← overshoot
      2   3709.53    8.03e-02    H2O=0.429  CO=0.151   CO2=0.127
      3   3604.76    2.01e-02    H2O=0.461  CO2=0.146  CO=0.139
      4   3677.17    4.92e-02    H2O=0.439  CO=0.148   CO2=0.133
      5   3640.97    1.45e-02    H2O=0.450  CO=0.144   CO2=0.139
      6   3622.87    2.80e-03    H2O=0.455  CO2=0.143  CO=0.141
      7   3633.01    6.90e-03    H2O=0.452  CO=0.143   CO2=0.141
      8   3627.94    2.05e-03    H2O=0.454  CO=0.142   CO2=0.142
      9   3625.40    3.76e-04    H2O=0.455  CO2=0.142  CO=0.142
     10   3626.76    9.28e-04    H2O=0.454  CO=0.142   CO2=0.142
     11   3626.08    2.76e-04    H2O=0.454  CO2=0.142  CO=0.142
     12   3625.74    5.03e-05    H2O=0.455  CO2=0.142  CO=0.142  ← CONVERGED
   ──────────────────────────────────────────────────────────────────────────

Several features are visible:

* **Step 0→1 overshoot**: the Newton step from 3 500 K predicts 3 919 K, which
  is too hot.  The sign of f(T) flips, so the bisection bracket captures the
  solution in the interval [3 500, 3 919].
* **Interval narrowing**: subsequent steps alternate Newton (inside the
  bracket) and midpoint (bisection fallback) to narrow down on the answer.
* **Composition change with temperature**: as T rises from 3 500 K to 3 625 K,
  CO\ :sub:`2` and H\ :sub:`2`\ O fractions fall slightly as dissociation
  increases (more OH, CO, H, O appear).  This is the solver self-consistently
  tracking the shifting chemical equilibrium.
* **Convergence criterion**: the iteration stops when
  ``abs(f(T)) / (T · Cp) < 1e-4``, which corresponds to a relative
  temperature error of ``abs(dlnT) < 1e-4`` — at 3 626 K this is a
  temperature accuracy of about 0.4 K.

Final result
~~~~~~~~~~~~

After 13 outer iterations the solver reports:

.. code-block:: text

   T_equilibrium = 3 625.74 K
   P             = 6.895 MPa  (1 000 psia)
   M_gas         = 22.715 g/mol (mean molecular weight of gas mixture)
   gamma (frozen)= 1.198
   Cp  (frozen)  = 50.31 J/(mol·K)
   a   (frozen)  = 1 260.9 m/s  (frozen speed of sound)

The **equilibrium composition** at this state (mole fractions ≥ 0.01 %):

.. code-block:: text

   H2O   0.4545   (45.45 %)   ← dominant product; water vapour
   CO2   0.1420   (14.20 %)
   CO    0.1418   (14.18 %)   ← significant CO due to dissociation
   OH    0.0967   ( 9.67 %)   ← hydroxyl radical; large fraction at this T
   O2    0.0682   ( 6.82 %)   ← unreacted oxygen
   H2    0.0536   ( 5.36 %)
   H     0.0217   ( 2.17 %)   ← atomic hydrogen; high T dissociation
   O     0.0210   ( 2.10 %)   ← atomic oxygen
   HO2   0.0003   ( 0.03 %)

The mixture is far from the simple "complete combustion" picture of just
CO\ :sub:`2` and H\ :sub:`2`\ O.  At 3 625 K, roughly **23 % of the gas is
dissociated radical or intermediate species** (OH + H + O + CO + H\ :sub:`2`).
This is physically important: these species carry stored chemical potential
that would be released if the gas were cooled, which is why the frozen speed
of sound (calculated assuming composition *does not* change) is lower than the
equilibrium speed of sound would be.

The **mean molecular weight of 22.7 g/mol** (compare with air at 29 g/mol)
reflects the large fraction of light radicals and diatomics.  The **adiabatic
flame temperature of 3 626 K** is one of the highest of any practical
hydrocarbon propellant pair.

.. _gordon-mcbride-solver:

GordonMcBrideSolver: coupled Newton iteration
---------------------------------------------

The Gordon-McBride solver takes a fundamentally different approach: it solves
for temperature *and* composition in a **single Newton system**.  At each
iteration the unknowns are updated simultaneously:

* The element potentials π\ :sub:`k` (one per element — C, H, O)
* The correction to total gas moles Δln(n)
* The temperature correction Δln(T)

This gives a **5×5 matrix** for an HP problem with three elements and no
condensed phases.  Gas-phase species mole corrections are then computed
analytically from the updated π.

Convergence history
~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   GordonMcBrideSolver iteration log
   ──────────────────────────────────────────────────────────────────────────
   iter   T [K]      lam       dlnT        dlnn        notes
   ────   ────────   ───────   ─────────   ─────────   ──────────────────────
      0   3500.00    0.120     +1.46e-01   +9.44e-01   initial (heavily damped)
      1   3562.10    0.139     -3.53e-02   +2.59e+00   large dlnn, off-scale
      2   3544.68    0.084     -1.63e-01   +4.75e+00   ← very large steps
      3   3496.30    0.120     -1.14e-01   +3.35e+00
      4   3448.93    0.147     -1.24e-01   +2.72e+00
      5   3386.73    0.397     -1.22e-01   +1.01e+00
      6   3226.20    1.000     -4.93e-02   -3.09e-01   ← undershoot, T drops
      7   3071.15    0.538     +1.14e-01   -7.44e-01   exploring fuel-rich
      8   3265.00    0.476     +1.18e-01   -6.06e-01   recovering
      9   3453.91    0.043     +1.02e-01   -4.71e-01   heavy damping
     10   3469.17    0.006     +1.08e-01   -5.01e-01   λ = 0.006: very small step
     11   3471.37    0.006     +1.15e-01   -6.05e-01
     12   3473.82    0.010     +1.11e-01   -6.47e-01
     13   3477.78    0.024     +9.23e-02   -6.57e-01
     14   3485.46    0.073     +7.60e-02   -6.57e-01
     15   3504.89    0.166     +1.51e-01   -6.49e-01
     16   3593.83    0.443     +1.18e-01   -6.34e-01
     17   3786.94    0.663     +1.68e-02   -6.03e-01   ← past the answer
     18   3829.39    0.746     -1.10e-02   -5.37e-01   overshooting back
     19   3797.99    1.000     -2.58e-02   -3.71e-01
     20   3701.17    1.000     -1.36e-02   -2.04e-01
     21   3651.30    1.000     -5.00e-03   -5.26e-02
     22   3633.08    1.000     -1.94e-03   -1.51e-03   ← rapid convergence
     23   3626.02    1.000     -1.15e-04   +7.06e-04
     24   3625.61    1.000     +2.56e-05   +2.58e-05   CONVERGED
   ──────────────────────────────────────────────────────────────────────────

Key observations:

* **Damping (λ column)**: In the first 10 iterations the damping factor is much
  less than 1, meaning the solver accepts only a small fraction of the raw
  Newton step.  This prevents the composition from going negative or the
  temperature from jumping to nonphysical values.  The raw steps are *extremely*
  large (``dlnn`` = 5 in step 2 means a factor-of-148 change in total moles).
* **Exploration phase (steps 0–18)**: T wanders from 3 500 K down to 3 071 K
  and back up to 3 830 K before settling.  During this phase the composition
  snapshot shows physically unrealistic mixtures (H\ :sub:`2`-dominated at
  3 071 K — pure fuel-side products — and CO-dominated at later steps) because
  the coupled system is far from equilibrium.
* **Quadratic convergence (steps 21–24)**: once the iterate is inside the
  basin of attraction (around step 20) the error drops by two orders of
  magnitude every step — the hallmark of Newton's method.  Steps 21→22→23→24
  show ``abs(dlnT)`` = 5×10\ :sup:`−3` → 2×10\ :sup:`−3` → 1×10\ :sup:`−4` →
  3×10\ :sup:`−5`.

Final result
~~~~~~~~~~~~

.. code-block:: text

   T_equilibrium = 3 625.70 K
   P             = 6.895 MPa
   M_gas         = 22.715 g/mol
   gamma (frozen)= 1.198
   Cp  (frozen)  = 50.31 J/(mol·K)
   a   (frozen)  = 1 260.9 m/s

The composition is essentially identical to the MajorSpeciesSolver result.  The
element-balance residual is **2.5×10\ :sup:`−9`** (near machine precision) compared
to **1.4×10\ :sup:`−3`** for the MajorSpeciesSolver.  This difference occurs because
the MajorSpeciesSolver applies a "final exact update" step at the end that sets
every species from the converged element potentials, which slightly disturbs
the element balance without an additional inner iteration.

.. _element-potentials:

Element potentials in depth: the H\ :sub:`2`/O\ :sub:`2` case
--------------------------------------------------------------

The element potentials π\ :sub:`k` are the heart of the Newton system.  This
section works through their meaning and computation in detail using
**hydrogen–oxygen combustion** (2 mol H\ :sub:`2` + 1 mol O\ :sub:`2`,
stoichiometric, 1 000 psia), where only two elements (H and O) are active.
The Newton matrix is then 3×3 — small enough to trace every number.

What is π\ :sub:`k`?
~~~~~~~~~~~~~~~~~~~~~

At chemical equilibrium, every species must satisfy a condition of the form:

.. code-block:: text

   mu_j / (R*T) = sum_k  a[j,k] * pi_k          (equilibrium condition)

where:

* ``mu_j / (R*T)`` is the **dimensionless chemical potential** of species j —
  its "preference" to exist, combining its internal energy, entropy, and
  concentration.
* ``a[j,k]`` is the number of atoms of element k in species j (the
  stoichiometric matrix).
* ``pi_k`` is the **element potential** for element k — effectively a
  "price per atom" for element k, determined self-consistently so that
  all equilibrium conditions hold simultaneously.

Because every species is constrained by the same set of π values, this is an
extremely compressed representation: instead of tracking N species separately,
the solver finds just S values (one per element) from which every species mole
amount can be recovered analytically.

The π values carry information about how scarce or abundant each element is at
equilibrium.  A very negative π means the element is "cheap" — there are many
atoms of it relative to species that want to incorporate it.  A less negative
π means it is "expensive" and mostly tied up in stable molecules.

The Newton matrix (H\ :sub:`2`/O\ :sub:`2`, iteration 0)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The H\ :sub:`2`/O\ :sub:`2` pool contains 14 gas species: O, O\ :sub:`2`,
H, H\ :sub:`2`, OH, HO\ :sub:`2`, H\ :sub:`2`\ O, H\ :sub:`2`\ O\ :sub:`2`,
O\ :sub:`3`, HO\ :sub:`3`, H\ :sub:`2`\ O\ :sub:`3`, H\ :sub:`3`\ O,
H\ :sub:`6`\ O, O\ :sub:`4`.  With 2 elements (O and H) the Newton system has
S + 0 + 1 = **3 unknowns**: π\ :sub:`O`, π\ :sub:`H`, and δln(n).

The system is started at T = 3 500 K with all 14 species at equal mole
amounts (n\ :sub:`j` = 0.0769 mol, n\ :sub:`gas,total` = 1.077 mol).  The
reduced chemical potential for each species at this state is:

.. code-block:: text

   Species  formula    n_j       mu_j / (RT)
   ─────────────────────────────────────────
   O        1O         0.0769    -13.149
   O2       2O         0.0769    -29.309
   H        1H         0.0769     -8.582
   H2       2H         0.0769    -19.809
   OH       1O+1H      0.0769    -24.990
   HO2      2O+1H      0.0769    -33.898
   H2O      1O+2H      0.0769    -36.924
   H2O2     2O+2H      0.0769    -42.068
   (6 more species, not listed)

The assembled 3×3 Newton matrix G and right-hand side (RHS) at this step are:

.. code-block:: text

   Row/col label      pi_O     pi_H     d_ln_n    RHS
   ────────────────────────────────────────────────────
   O (el. balance)    4.615    2.077     1.846    -63.32
   H (el. balance)    2.077    5.000     1.615    -52.95
   n_gas (tot.moles)  1.846    1.615     0        -32.58

Reading across the O-row: the (O, O) entry is ``sum_j  a[j,O]^2 * n_j``
(the weighted sum of squared oxygen-atom counts over all major species) and
the O-row RHS is ``sum_j a[j,O]*n_j*mu_j + b0[O] - current_O_atoms``.

Solving gives:

.. code-block:: text

   pi_O   = -11.903    (oxygen   "price" per atom in units of RT)
   pi_H   =  -6.566    (hydrogen "price" per atom in units of RT)
   d_ln_n =  +2.849    (predicted scale-up in total moles)

The positive δln(n) = +2.85 means the solver expects n\ :sub:`gas,total` to
multiply by e\ :sup:`2.85` ≈ 17× — the initial equal-distribution starting
guess (dominated by dozens of nearly-empty species) is drastically off scale.
The damping factor λ = 0.136 prevents the full step from being taken.

After updating with the damped step and rerunning at T = 3 500 K, the basis
shifts to **H**\ :sub:`2`\ **O and H**\ :sub:`2` (the two species that have
grown most).  Iteration 1 gives:

.. code-block:: text

   Row/col label      pi_O     pi_H     d_ln_n    RHS
   ────────────────────────────────────────────────────
   O (el. balance)    6.133    3.832     3.171   -111.3
   H (el. balance)    3.832    6.736     3.348   -107.3
   n_gas (tot.moles)  3.171    3.348     0        -71.8

   Solution: pi_O = -13.081,  pi_H = -9.058,  d_ln_n = +1.132

Note how the matrix diagonal entries are increasing — this reflects a growing
total gas moles (n\ :sub:`gas,total` = 2.40 after iteration 1, up from 1.08)
as the composition converges toward the true equilibrium.

How a species is placed from π (the minor-species formula)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

By iteration 2 (n\ :sub:`gas,total` = 7.52) species such as H\ :sub:`2`\ O\ :sub:`3`
and H\ :sub:`3`\ O are small enough to become **minor** — their mole amounts
are set directly from π rather than entering the Newton matrix.  The
analytical formula is:

.. code-block:: text

   ln(n_j) = sum_k  a[j,k] * pi_k  -  g°_j/RT  -  ln(P/P°)  +  ln(n_gas_total)

where ``g°_j/RT`` is the species' reduced standard Gibbs energy at T.  For
H\ :sub:`2`\ O\ :sub:`3` (formula: 3O + 2H) at iteration 2
(T = 3 500 K, n\ :sub:`gas,total` = 7.52, π\ :sub:`O` = −14.149,
π\ :sub:`H` = −10.168):

.. code-block:: text

   ln(n_H2O3) = 3*pi_O + 2*pi_H  -  g°/RT  -  ln(P/P°)  +  ln(n)
              = 3*(-14.149) + 2*(-10.168) - (-48.458) - 4.233 + 2.017
              = -62.783 + 48.458 - 4.233 + 2.017
              = -16.541
   n_H2O3 = exp(-16.541) = 6.5e-8 mol   (effectively zero)

For H\ :sub:`3`\ O (formula: 1O + 3H):

.. code-block:: text

   ln(n_H3O) = pi_O + 3*pi_H  -  g°/RT  -  ln(P/P°)  +  ln(n)
             = (-14.149) + 3*(-10.168) - (-28.002) - 4.233 + 2.017
             = -44.653 + 28.002 - 4.233 + 2.017
             = -18.867
   n_H3O = exp(-18.867) = 6.4e-9 mol

These two species contribute negligible atoms, so excluding them from the
Newton matrix causes no meaningful error.  The 3×3 matrix stays 3×3 regardless
of whether there are 14 species or 14 000 in the database.

For comparison, by iteration 3 the HO\ :sub:`2` mole fraction has grown large
enough to be reclassified as major (it enters the Newton matrix), while
H\ :sub:`2`\ O\ :sub:`2` moves into the minor category:

.. code-block:: text

   Iteration 3 minor species (from pi_O = -14.466, pi_H = -10.077):
     H2O2   ln(n) = 2*pi_O + 2*pi_H - g°/RT - ln(P/P°) + ln(n)
                  = -48.754 + 43.662 - 4.233 + 1.722 = -7.936  ->  n = 3.6e-4
     O3     ln(n) = 3*pi_O          - g°/RT - ln(P/P°) + ln(n)
                  = -43.399 + 33.855 - 4.233 + 1.722 = -12.055  ->  n = 5.8e-6

H\ :sub:`2`/O\ :sub:`2` final element potentials
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

After 15 outer temperature iterations the H\ :sub:`2`/O\ :sub:`2` solve
converges at T = 3 674 K with:

.. code-block:: text

   pi_O (converged) = -15.103
   pi_H (converged) =  -9.719

These final values encode the entire equilibrium composition.  Any species
not explicitly tracked can be placed:

.. code-block:: text

   H2O  (1O+2H):  ln(n) = pi_O + 2*pi_H - g°/RT - ln(P/P°) + ln(n)
   H2   (2H):     ln(n) = 2*pi_H        - g°/RT - ln(P/P°) + ln(n)
   OH   (1O+1H):  ln(n) = pi_O + pi_H  - g°/RT - ln(P/P°) + ln(n)
   H    (1H):     ln(n) = pi_H          - g°/RT - ln(P/P°) + ln(n)
   O    (1O):     ln(n) = pi_O          - g°/RT - ln(P/P°) + ln(n)

Every one of the 14 candidate species is set by this formula.  The final
composition (mole fractions ≥ 0.1 %):

.. code-block:: text

   H2O   67.9 %    OH    10.8 %    H2   12.4 %
   H      3.7 %    O2     3.6 %    O     1.7 %

.. _element-potentials-ch4:

Extending to three elements: CH\ :sub:`4`/O\ :sub:`2`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Adding carbon adds one element (C) and one unknown (π\ :sub:`C`), making the
system **4×4**.  The target element abundances are:

.. code-block:: text

   b0_H = 4.0  (4 H atoms from 1 mol CH4)
   b0_C = 1.0  (1 C atom  from 1 mol CH4)
   b0_O = 4.0  (4 O atoms from 2 mol O2)

At iteration 0 of the inner Newton solve (T = 3 500 K, 229 gas species, all at
equal small moles), the assembled 4×4 matrix is:

.. code-block:: text

   Row/col     pi_H     pi_C     pi_O    d_ln_n    RHS
   ─────────────────────────────────────────────────────
   H          74.84    54.96     9.83     9.89     -821.3
   C          54.96    61.22     7.08     9.08     -677.0
   O           9.83     7.08     4.53     1.88     -154.8
   n_gas       9.89     9.08     1.88     0.0      -124.4

The large off-diagonal entries (e.g. H-C = 54.96) come from the many
hydrocarbon species (CH\ :sub:`4`, C\ :sub:`2`\ H\ :sub:`2`,
C\ :sub:`3`\ H\ :sub:`8`, …) which contain both hydrogen and carbon, so
their contribution lands in both rows.

Solving this system gives:

.. code-block:: text

   pi_H =  -6.396   pi_C = -3.687   pi_O = -14.755   d_ln_n = +0.519

The small positive δln(n) = 0.52 here (compared to 2.85 for H\ :sub:`2`/O\ :sub:`2`)
reflects that the 229-species initial mixture contains many species that
remain sizeable throughout, so the total moles does not need to change as
dramatically.  The damping factor is λ = 0.117 — similar to the H\ :sub:`2`/O\ :sub:`2`
case, as the same λ₁ logic applies.

After convergence at T = 3 626 K:

.. code-block:: text

   pi_H (converged) = -10.113
   pi_C (converged) = -16.519   (carbon is very scarce relative to H and O)
   pi_O (converged) = -14.749

The very negative π\ :sub:`C` = −16.5 reflects that the equilibrium mixture has
only ~14 % each of CO and CO\ :sub:`2` — carbon is a "limiting resource" in
this mixture.  π\ :sub:`H` = −10.1 and π\ :sub:`O` = −14.7 are similar in
magnitude to the H\ :sub:`2`/O\ :sub:`2` values because H and O dominate the
mixture (45 % H\ :sub:`2`\ O).

.. _gmcb-damping:

Gordon-McBride damping and the exploration phase
-------------------------------------------------

The G-McB solver updates temperature and composition simultaneously in a single
Newton loop.  This is algorithmically elegant but requires careful damping to
avoid the iteration diverging before it finds the basin of attraction.

How the damping factor λ is computed
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The raw Newton step produces updates Δln(n\ :sub:`j`) for every gas species.
These updates are in *log space*: a step of Δln(n) = +2 means the species
would grow by a factor of e\ :sup:`2` ≈ 7.4×.  Very large steps occur in
early iterations because the composition is far from equilibrium and the
Jacobian is poorly conditioned.

Two limits are applied (matching NASA RP-1311 / CEA):

**λ₁ — step-size cap:**

.. code-block:: text

   l1_denom = max(|d_ln_T|, |d_ln_n|) * 5.0
   For each non-floor gas species j with d_ln_nj > 0:
       l1_denom = max(l1_denom, d_ln_nj)

   lam1 = 2.0 / l1_denom  if l1_denom > 2.0
          1.0              otherwise

This ensures that no species grows by more than e\ :sup:`2` ≈ 7.4× in a
single step.  The factor ``5.0 * max(|dlnT|, |dlnn|)`` protects the
temperature step too.

**λ₂ — floor protection:**

If a species is currently at (or near) the concentration floor (mole fraction
below ~10\ :sup:`−8`) and its log-space step is positive, a separate limit
prevents it from jumping too far above the floor in one step.  Only positive
corrections are considered — negative ones (driving a species toward zero) are
always allowed.

The final damping is:

.. code-block:: text

   lam = min(1.0, lam1, lam2)

Tracing the early G-McB iterations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For CH\ :sub:`4`/O\ :sub:`2` at iteration 0 (T = 3 500 K), the raw Newton
step has δln(n) = 0.944 and δln(T) = 0.146.  Among the 234 gas species,
several have Δln(n\ :sub:`j`) > 2 (e.g. CO and H\ :sub:`2`\ O are predicted
to grow rapidly).  With l1_denom ≈ 16.6, λ₁ = 2.0/16.6 = 0.120, so
only 12 % of the Newton step is accepted.

.. code-block:: text

   iter   T [K]    lam    d_ln_T    d_ln_n    notes
   ─────────────────────────────────────────────────────────────────
      0   3500    0.120   +0.146    +0.945    initial, heavily damped
      1   3562    0.139   -0.035    +2.592    d_ln_n still large → lam stays small
      2   3545    0.084   -0.163    +4.748    huge d_ln_n → lam1 = 2/47 = 0.04
      3   3496    0.120   -0.114    +3.345    composition still far off
      4   3449    0.147   -0.124    +2.721
      5   3387    0.397   -0.122    +1.008    d_ln_n approaching 1 → lam grows
      6   3226    1.000   -0.049   -0.309     lam1 = 1.0 (full step)
      7   3071    0.538   +0.114   -0.744
   ─────────────────────────────────────────────────────────────────

Why does T drop to 3 071 K before recovering?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The exploration is not random — it is Newton's method faithfully following the
gradient of the Jacobian.  The reason T overshoots is that in the early
iterations the composition (dominated by CO and H\ :sub:`2` because the solver
hasn't balanced the oxygen yet) represents a very **fuel-rich** mixture.  A
fuel-rich mixture has a much lower adiabatic flame temperature, so the energy
constraint pushes T downward.  As more oxygen-bearing species grow (H\ :sub:`2`\ O,
CO\ :sub:`2`) and the oxygen balance improves, the energy constraint gradually
pulls T back up toward the true stoichiometric value.

The key point is that the G-McB Newton step is not directly controlling
**convergence** in these early iterations — λ is so small (0.084–0.147) that
the step is essentially a cautious **search** for a region where the Jacobian
is well-conditioned.  Once the composition enters the physically realistic
neighbourhood (around iteration 20, where λ = 1.0 first appears), the
quadratic Newton convergence takes over and the temperature homes in rapidly:

.. code-block:: text

   iter 21:  T = 3651.30 K,  |d_ln_T| = 5.0e-3
   iter 22:  T = 3633.08 K,  |d_ln_T| = 1.9e-3
   iter 23:  T = 3626.02 K,  |d_ln_T| = 1.2e-4
   iter 24:  T = 3625.61 K,  |d_ln_T| = 2.6e-5  ← CONVERGED

The MajorSpeciesSolver avoids this exploration entirely because the inner loop
fully converges the composition at each temperature guess **before** the outer
loop attempts a temperature step.  This means the outer loop always sees a
physically correct (element-balanced) composition gradient, and the Newton
temperature step is reliable from the very first outer iteration.

.. _solver-comparison-ch4-o2:

Side-by-side comparison
-----------------------

.. list-table:: CH\ :sub:`4`/O\ :sub:`2` result comparison (O/F=4.0, 1 000 psia)
   :header-rows: 1
   :widths: 30 25 25

   * - Property
     - MajorSpeciesSolver
     - GordonMcBrideSolver
   * - T\ :sub:`adiabatic` [K]
     - 3625.74
     - 3625.70
   * - M\ :sub:`gas` [g/mol]
     - 22.715
     - 22.715
   * - γ (frozen)
     - 1.1980
     - 1.1980
   * - C\ :sub:`p` [J/(mol·K)]
     - 50.315
     - 50.314
   * - a\ :sub:`frozen` [m/s]
     - 1260.9
     - 1260.9
   * - Outer iterations
     - 13
     - 25 (single loop)
   * - Converged?
     - Yes
     - Yes
   * - Element balance error
     - 1.4 × 10\ :sup:`−3`
     - 2.5 × 10\ :sup:`−9`

Both solvers agree to better than 0.1 K in temperature and are identical to
four significant figures in all other thermodynamic properties.  The
temperature difference (0.04 K) is within the convergence tolerance.

The MajorSpeciesSolver uses **fewer outer iterations** because its separated
inner/outer structure allows tight composition convergence before committing to
a temperature step.  The GordonMcBrideSolver requires more iterations overall
but achieves a tighter element-balance residual because it does not apply the
final-step shortcut.

.. _running-the-example:

Running this calculation
------------------------

The following code reproduces the result:

.. code-block:: python

   from prometheus_equilibrium.equilibrium.problem import EquilibriumProblem, ProblemType
   from prometheus_equilibrium.equilibrium.solver import MajorSpeciesSolver
   from prometheus_equilibrium.equilibrium.species import SpeciesDatabase

   db = SpeciesDatabase(
       nasa7_path="prometheus_equilibrium/thermo_data/nasa7.json",
       nasa9_path="prometheus_equilibrium/thermo_data/nasa9.json",
       janaf_path="prometheus_equilibrium/thermo_data/janaf.csv",
   )
   db.load(include_janaf=False)

   ch4 = db.find("CH4")
   o2  = db.find("O2")

   # Stoichiometric CH4/O2: 1 mol CH4 + 2 mol O2
   n_ch4, n_o2 = 1.0, 2.0
   T_ref = 298.15   # K — standard reference temperature

   H0 = ch4.enthalpy(T_ref) * n_ch4 + o2.enthalpy(T_ref) * n_o2

   products = db.get_species({"C", "H", "O"}, max_atoms=20)

   problem = EquilibriumProblem(
       reactants={ch4: n_ch4, o2: n_o2},
       products=products,
       problem_type=ProblemType.HP,
       constraint1=H0,
       constraint2=1000 * 6894.757,   # 1000 psia in Pa
       t_init=3500.0,
   )

   solver = MajorSpeciesSolver()
   solution = solver.solve(problem)
   print(solution.summary())

Expected output:

.. code-block:: text

   EquilibriumSolution
     T          = 3625.74 K
     P          = 6.895e+06 Pa
     converged  = True  (13 iterations)
     M_gas      = 22.7150 g/mol
     Cp         = 50.3149 J/(mol·K)
     gamma      = 1.19796
     a          = 1260.90 m/s
     Major species (xj >= 1e-4):
       H2O                             0.454536
       CO2                             0.142030
       CO                              0.141763
       HO                              0.096749
       O2                              0.068180
       H2                              0.053640
       H                               0.021716
       O                               0.021048
       HO2                             0.000294

The full trace script used to generate this page is available at
``docs/scripts/ch4_o2_worked_example.py``.
