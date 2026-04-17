Propellant Optimizer
====================

Prometheus includes a gradient-based propellant formulation optimizer that
searches composition space for the mixture maximising a user-defined figure of
merit — typically a weighted combination of specific impulse and bulk density.

.. contents:: Contents
   :local:
   :depth: 2


Algorithm overview
------------------

The optimizer wraps `scipy.optimize.minimize` (SLSQP) with a **multi-start**
strategy: ``n_starts`` independent SLSQP runs are launched from random feasible
starting points, and the best result across all starts is returned.  Gradients
are estimated by forward finite differences with step size ``fd_step``
(default 1 × 10⁻⁴ in mass-fraction units).

Objective
~~~~~~~~~

The optimizer maximises the log-scale figure of merit

.. math::

   \ln \text{FoM} = \ln I_{sp} + n \cdot \ln \rho

where :math:`I_{sp}` is the selected Isp variant (actual, vacuum, or sea-level),
:math:`\rho` is the bulk mixture density, and :math:`n` is the user-supplied
density exponent (typically 0 for pure Isp, 0.25 for volumetric propulsion
index, or 1.0 for density-weighted Isp).

Constraints
~~~~~~~~~~~

All constraints are linear and are enforced directly by SLSQP:

- **Mass balance** (equality): :math:`\sum x_i = 1`.
- **Ratio locks** (equality, fixed-proportion groups): internal ingredient
  ratios are held constant while their combined mass fraction varies.
- **Sum bounds** (inequality, sum-to-total groups): a sub-set of ingredients
  is constrained to lie within a specified total range.
- **Box bounds**: each ingredient has independent lower and upper bounds.

Convergence tolerance
~~~~~~~~~~~~~~~~~~~~~

SLSQP stops when the absolute change in the objective falls below ``ftol``
(default 1 × 10⁻⁴).  Tighter values (e.g. 1 × 10⁻⁶) improve accuracy at the
cost of more iterations; looser values (e.g. 1 × 10⁻³ or 1 × 10⁻²) exit
earlier and are useful for broad exploratory sweeps or when the objective
landscape is flat near the optimum.

For high-fidelity final runs after an exploratory staged pass, tightening to
1 × 10⁻⁵ or 1 × 10⁻⁶ is recommended.


Expansion modes
---------------

Three nozzle expansion modes are available, controlling both the Isp
evaluation and (in staged mode) the overall optimization strategy.

Frozen
~~~~~~

The nozzle expansion uses *frozen* composition: the product mixture is fixed at
the chamber equilibrium state throughout expansion.  This is a fast
approximation — each performance evaluation requires only a single equilibrium
solve.

Suitable for quick screening runs or propellant families where the frozen/
shifting difference is small.

Shifting
~~~~~~~~

The nozzle expansion uses *shifting* composition: the product mixture
re-equilibrates at each pressure step along the nozzle.  This is physically
accurate (the optimizer targets the quantity that actually matters) but
roughly 10× slower than frozen because each performance evaluation requires
many sequential equilibrium solves along the expansion path.

Shifting (Staged)
~~~~~~~~~~~~~~~~~

A two-stage strategy that combines the speed of frozen exploration with the
accuracy of shifting refinement:

**Stage 1 — Frozen exploration**
  ``n_starts`` random starting compositions are optimised with frozen
  expansion using the full iteration budget (``max_iter / start``).  Because
  frozen evaluation is ~10× cheaper, this stage surveys the composition
  landscape at low cost and produces ``n_starts`` locally optimal frozen
  compositions.

**Stage 2 — Shifting refinement**
  The top ``n_refine`` stage-1 results (ranked by frozen objective) are
  warm-started into SLSQP with full shifting expansion and a separate
  iteration budget (``iter/start (stage 2)``).  Because the warm starts are
  already close to the optimum, far fewer shifting iterations are needed to
  converge.

Benchmark result on a 5-ingredient Bi₂O₃/APCP formulation:

+------------------------------------------+--------+---------+-----------+
| Scenario                                 | Time   | Isp×ρ   | Converged |
+==========================================+========+=========+===========+
| Shifting — 12 starts × 10 iter          | 1547 s | 482 824 | 8 / 12    |
+------------------------------------------+--------+---------+-----------+
| Staged — 12 frozen × 10 + 4 shifting×10 |  624 s | 482 824 | 4 / 4     |
+------------------------------------------+--------+---------+-----------+

The staged approach delivered **2.5× speedup** with identical solution quality
and a higher convergence rate (4/4 vs 8/12) because the frozen stage
pre-identifies the feasible basins.

**Recommended defaults** for staged mode: ``n_starts = 12``,
``max_iter / start = 10`` (stage 1), ``n_refine = 4``,
``iter/start (stage 2) = 10``.  Increasing ``n_starts`` beyond ~16 with the
same ``n_refine = 4`` rarely improves quality and increases stage-1 cost
proportionally.

Progress display
~~~~~~~~~~~~~~~~

In staged mode the GUI shows **two separate graphs**: one for the frozen stage-1
traces and one for the shifting stage-2 traces.  The progress bar spans the
total start count across both stages, and the status label shows
``[stage 1]`` / ``[stage 2]`` tags to indicate which phase is active.


API reference
-------------

.. autoclass:: prometheus_equilibrium.optimization.MultiStartGradientOptimizer
   :members: optimize
   :undoc-members:

.. autoclass:: prometheus_equilibrium.optimization.StagedGradientOptimizer
   :members: optimize
   :undoc-members:

.. autoclass:: prometheus_equilibrium.optimization.OptimizationProblem
   :members:
   :undoc-members:

.. autoclass:: prometheus_equilibrium.optimization.ObjectiveSpec
   :members:
   :undoc-members:

.. autoclass:: prometheus_equilibrium.optimization.OperatingPoint
   :members:
   :undoc-members:

.. autoclass:: prometheus_equilibrium.optimization.OptimizationResult
   :members:
   :undoc-members:


Configuration file format
--------------------------

Optimization runs can be saved and reloaded as ``.prop-opt.json`` files.
All mass-fraction values are stored as **percentages (0–100)**.

.. code-block:: json

   {
     "schema_version": 1,
     "problem": {
       "variables": [
         {"ingredient_id": "AP", "minimum": 50.0, "maximum": 75.0, "pinned": false},
         {"ingredient_id": "AL", "minimum": 5.0,  "maximum": 20.0, "pinned": false},
         {"ingredient_id": "HTPB", "minimum": 8.0, "maximum": 16.0, "pinned": false},
         {"ingredient_id": "IPDI", "minimum": 1.0, "maximum": 4.0, "pinned": false}
       ],
       "fixed_proportion_groups": [
         {"group_id": "binder", "members": ["HTPB", "IPDI"], "ratios": [0.12, 0.04]}
       ],
       "sum_to_total_groups": [
         {"group_id": "solids", "members": ["AP", "AL"],
          "minimum_total": 80.0, "maximum_total": 88.0}
       ],
       "total_mass_fraction": 100.0
     },
     "objective": {"isp_variant": "isp_actual", "rho_exponent": 0.25},
     "operating_point": {
       "chamber_pressure_pa": 6894757.0,
       "expansion_type": "area_ratio",
       "expansion_value": 40.0,
       "ambient_pressure_pa": 101325.0,
       "shifting": true
     },
     "run": {
       "n_starts": 12,
       "max_iter_per_start": 10,
       "fd_step": 0.0001,
       "ftol": 0.0001,
       "n_workers": 0,
       "seed": 42
     },
     "staged": {
       "enabled": true,
       "n_refine": 4,
       "max_iter_stage2": 10
     },
     "solver": {
       "type": "gmcb",
       "enabled_databases": ["NASA-7", "NASA-9", "TERRA"],
       "max_atoms": 6
     }
   }

Key fields:

``run.ftol``
  SLSQP convergence tolerance.  Omit to use the default (1 × 10⁻⁴).

``staged.enabled``
  Set to ``true`` to activate the two-stage frozen→shifting strategy.
  Requires ``operating_point.shifting = true``; ignored otherwise.

``staged.n_refine``
  Number of stage-1 optima to carry into stage-2 refinement.

``staged.max_iter_stage2``
  SLSQP iteration budget per start in stage 2.


Headless runner
---------------

Run an optimization from a saved config without launching the GUI::

    prometheus-optimize my_propellant.prop-opt.json
    prometheus-optimize my_propellant.prop-opt.json --output results.json
    prometheus-optimize my_propellant.prop-opt.json --n-starts 20 --seed 0

Results are written as JSON to stdout (or ``--output``) and a human-readable
summary is printed to stderr.
