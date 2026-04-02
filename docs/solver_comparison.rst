Solver Algorithm Comparison
============================

Three equilibrium solver algorithms were analysed before deciding on the
implementation strategy for Prometheus.  This document records the analysis,
explains why **Gordon-McBride (G-McB)** is the production default today, and
documents the specific differences between Prometheus's implementation and the
two reference codes: ``pep-for-zos`` (Villars-Browne PEP) and
``cpropep``/RocketCEA (Gordon-McBride).

The Three Candidate Algorithms
-------------------------------

Gordon-McBride (G-McB)
~~~~~~~~~~~~~~~~~~~~~~

**Source:** NASA RP-1311, Gordon & McBride (1994).

**Reference implementation:** ``cpropep/libcpropep/src/equilibrium.c``,
RocketCEA.

The G-McB method solves a *modified Newton system* where the primary unknowns
are the **Lagrange multipliers** πₖ (one per element), the condensed-species
mole corrections Δnc, the correction to total gas moles Δln(n), and — for
HP/SP problems — the temperature correction Δln(T).  Gas-phase species mole
corrections are computed *analytically* after each Newton step:

.. code-block:: text

   Δln(nⱼ) = Σₖ aₖⱼ·πₖ  +  Δln(n)  −  μⱼ   [+ H°ⱼ/RT·Δln(T) for HP/SP]

This analytical substitution keeps the Newton matrix small: **(S + nc + 1)**
for TP problems and **(S + nc + 2)** for HP/SP, regardless of the total
number of product species.  For H₂/O₂ with no condensed phases this is a
**3×3** system; for a solid-motor problem with one condensed phase (Al₂O₃)
it becomes **4×4**.

**Temperature handling:** Δln(T) is an unknown in the same Newton system as
the composition.  T and composition converge simultaneously.  This is elegant
but means the iteration can diverge if the starting guess is far from the
adiabatic temperature.

**Condensed phases:** managed by a restart strategy — a condensed species is
added or removed and the Newton loop is restarted with a counter that prevents
cycling.

Villars-Browne PEP
~~~~~~~~~~~~~~~~~~

**Source:** Cruise, NWC Technical Publication 6037 (1973/1979).

**Reference implementation:** ``pep-for-zos/PEP`` (Fortran 77/IBM z/OS port).

PEP selects an optimised *basis* of S species (one per element) using Browne's
Gram-Schmidt rank test.  All non-basis species are expressed as stoichiometric
combinations of the basis via the reaction-coefficient matrix ν = C·B⁻¹.  The
algorithm then iterates on **reaction extents** Δζᵢ:

.. code-block:: text

   ln Kᵢ  = Σⱼ ν[i,j]·g°ⱼ/RT  −  g°ᵢ/RT        (log equilibrium constant)
   ln Qᵢ  = γᵢ·ln(P·nᵢ/n)  −  Σⱼ γⱼ·ν[i,j]·ln(P·nⱼ/n) (log reaction quotient)
   Δζᵢ    = (ln Kᵢ − ln Qᵢ) / (γᵢ/nᵢ + Σⱼ γⱼ·ν[i,j]²/nⱼ)

(γⱼ = 1 for gas, 0 for condensed — pure condensed species have unit activity.)

Basis species moles are corrected after each non-basis adjustment to maintain
element balance.

**Temperature handling:** separate outer Newton + interval-halving loop that
calls the TP inner loop at each trial temperature.  This guarantees convergence
as long as the solution lies in [298 K, 6000 K].

**Convergence rate:** each iteration is a first-order (linear) correction —
typically 5–15 inner iterations per temperature step.

**pep-for-zos specifics:** VLNK is computed using the *full* chemical potential
(H − TS including mixing terms), ensuring VLNK → 0 at equilibrium regardless
of sign convention.  Basis swaps use a TABLO linear-algebra tableau pivot.
Damping is adaptive: VQQ = max(0.05, 0.5 − (JC−1)/20), starting at 50% and
tightening with iteration count.

**Why PEP is strictly inferior to Hybrid:**

- Same S×S matrix size as Hybrid.
- Linear rather than quadratic convergence.
- For H₂/O₂ at 3000 K, PEP typically requires 10–20 iterations; Hybrid
  converges in 4–6.
- No simplicity advantage: the reaction-coefficient machinery (ν = C·B⁻¹,
  basis selection) is needed by Hybrid anyway.

PEP is therefore **not implemented** as a production solver.

Hybrid Solver
~~~~~~~~~~~~~

**Source:** Prometheus (original design).

**Concept:** combines PEP's reaction-adjustment architecture with G-McB's
Newton-step quadratic convergence.

The key insight is that *any* set of S linearly independent species can serve
as the primary Newton unknowns — the remaining species follow analytically.
Hybrid selects the S "major" species (highest mole amounts, Browne's test) and
forms a compressed Newton system in those S unknowns.  Minor species are
updated analytically from the element potentials π extracted from the Newton
solve, identically to the G-McB gas-correction formula:

.. code-block:: text

   Minor-species update (no iteration needed):
     ln(nⱼ) = Σₖ πₖ·A[j,k]  −  g°ⱼ/RT  −  ln(P/P°)  +  ln(n_gas)

- **Matrix size:** S×S (identical to PEP).
- **Convergence rate:** quadratic (identical to G-McB).
- **Temperature handling:** outer Newton + interval-halving (identical to PEP,
  more robust than G-McB's embedded-T approach).
- **Condensed phases:** phase-deletion rule (same as PEP, no restart required).

Comparison Table
----------------

.. list-table::
   :header-rows: 1
   :widths: 30 22 22 26

   * - Feature
     - G-McB
     - PEP
     - Hybrid
   * - Primary unknowns
     - π (Lagrange mult.)
     - reaction extents Δζ
     - π for major species
   * - Matrix size (TP)
     - (S + nc + 1)²
     - S×S
     - S×S
   * - Matrix size (HP/SP)
     - (S + nc + 2)²
     - S×S + outer loop
     - S×S + outer loop
   * - Convergence rate
     - quadratic
     - **linear**
     - quadratic
   * - Temperature
     - embedded in Newton
     - outer interval-halving
     - outer interval-halving
   * - T convergence guarantee
     - none
     - yes
     - yes
   * - Condensed handling
     - restart Newton
     - phase deletion
     - phase deletion
   * - Reference code
     - cpropep, RocketCEA
     - pep-for-zos
     - —
   * - Implementation status
     - complete
     - not implemented
     - complete

For H₂/O₂ combustion (S = 2, nc = 0):

- G-McB: **3×3** matrix, no outer loop.
- Hybrid: **2×2** matrix, outer loop (fast).
- PEP: **2×2** matrix, outer loop (slow — linear convergence).

For a solid-motor problem (S = 3 elements: Al, H, O; nc = 1: Al₂O₃(s)):

- G-McB: **5×5** matrix.
- Hybrid: **3×3** matrix.
- PEP: **3×3** matrix (linear convergence).

Why G-McB Is The Preferred Default
----------------------------------

G-McB is both the preferred and fastest solver in the current Prometheus
implementation for three practical reasons:

1. **Reference code available.** ``cpropep/libcpropep/src/equilibrium.c``
   implements G-McB with the exact same RP-1311 formulas, providing a
   function-by-function reference for debugging.

2. **Used by RocketCEA.** The torture-test benchmark compares Prometheus
   against RocketCEA output.  Implementing G-McB first ensures we can
   validate composition, temperature, and performance numbers against the
   same underlying algorithm.

3. **Current runtime profile.** In the present codebase, G-McB delivers the
   best end-to-end runtime for the standard benchmark workloads while
   preserving robust convergence behaviour.

Differences: Prometheus G-McB vs cpropep
-----------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Aspect
     - cpropep
     - Prometheus G-McB
   * - Language
     - C
     - Python / NumPy
   * - Rank analysis
     - simple ``rank()`` check
     - QR with column pivoting
   * - Independent elements
     - drop elements with zero b₀
     - QR pivot gives maximally independent subset
   * - Condensed inclusion
     - most-negative μ°c/RT − Σπa
     - same criterion, same π source
   * - Convergence criterion
     - 5×10⁻⁶ (matches RP-1311)
     - 5×10⁻⁶ (same)
   * - Temperature handling
     - embedded in Newton (HP/SP)
     - same: Δln(T) as Newton unknown
   * - Initial moles
     - equal among gas species
     - equal among gas species
   * - Damping
     - λ₁ caps Δln to ≤ 2; λ₂ trace floor
     - same two-step strategy

.. note::

   ``fill_matrix`` in cpropep builds the composition-search system (what
   Prometheus calls ``_assemble_jacobian``).  ``fill_equilibrium_matrix`` is a
   *separate* function called only for HP/SP that adds the Δln(T) row and
   column.  Prometheus merges both into a single ``_assemble_jacobian`` call
   that is aware of ``problem_type``.

Differences: Prometheus Hybrid vs pep-for-zos PEP
---------------------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Aspect
     - pep-for-zos
     - Prometheus Hybrid
   * - Inner loop
     - reaction-extent Δζ (linear)
     - compressed Newton on major species (quadratic)
   * - Minor species update
     - K/Q ratio, every 4th pass
     - analytical from π, every pass
   * - Basis swap
     - TABLO tableau pivot
     - Gram-Schmidt re-selection
   * - Damping
     - adaptive VQQ
     - RP-1311-style λ₁/λ₂
   * - Mole space
     - linear
     - log-space for stability
   * - Temperature
     - outer HBAL/SBAL loop
     - outer ``_temperature_search`` Newton + halving
   * - VLNK convention
     - full μ (includes mixing)
     - not applicable

Summary
-------

.. list-table::
   :header-rows: 1
   :widths: 30 40 30

   * - Solver
     - Role
     - Status
   * - :class:`~prometheus_equilibrium.equilibrium.solver.GordonMcBrideSolver`
     - Production default / RocketCEA comparison
     - Complete
   * - :class:`~prometheus_equilibrium.equilibrium.solver.HybridSolver`
     - Alternative solver
     - Complete
   * - :class:`~prometheus_equilibrium.equilibrium.solver.PEPSolver`
     - Not implemented (strictly inferior to Hybrid)
     - Deferred indefinitely

:class:`~prometheus_equilibrium.equilibrium.solver.GordonMcBrideSolver` is the recommended
solver for production use and is the current default throughout Prometheus.
It is the best-performing solver in the current implementation and matches the
CEA algorithm used by RocketCEA, enabling direct comparison of composition,
temperature, and performance metrics.
:class:`~prometheus_equilibrium.equilibrium.solver.HybridSolver` remains available as an
alternative implementation for algorithmic comparison and future optimisation work.
