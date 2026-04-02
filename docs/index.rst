Prometheus Documentation
========================

Prometheus is an open-source combustion equilibrium solver for Python.  It
parses thermodynamic data from multiple sources (NASA-7, NASA-9, JANAF, CEA,
TERRA, AFCESIC), builds a unified species database, and implements equilibrium
solvers targeting H₂/O₂ and general-hydrocarbon combustion validation against
NASA CEA / RocketCEA.

.. toctree::
   :maxdepth: 1
   :caption: User Guide

   thermo/index
   solver_comparison

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/prometheus

Quick Start
-----------

.. code-block:: python

   from prometheus_equilibrium.equilibrium import (
       SpeciesDatabase,
       EquilibriumProblem,
       ProblemType,
       GordonMcBrideSolver,
   )

   db = SpeciesDatabase(
       nasa7_path="prometheus/thermo_data/nasa7.json",
       nasa9_path="prometheus/thermo_data/nasa9.json",
       janaf_path="prometheus/thermo_data/janaf.csv",
   )
   db.load()

   h2 = db["H2_G"]
   o2 = db["O2_G"]
   products = db.get_species({"H", "O"}, max_atoms=20)

   T_react = 298.15
   H0 = sum(n * sp.enthalpy(T_react) for sp, n in {h2: 2.0, o2: 1.0}.items())

   problem = EquilibriumProblem(
       reactants={h2: 2.0, o2: 1.0},
       products=products,
       problem_type=ProblemType.HP,
       constraint1=H0,
       constraint2=30e5,   # 30 bar
       t_init=3500.0,
   )

   solution = GordonMcBrideSolver().solve(problem)
   print(solution.summary())

Indices and Tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
