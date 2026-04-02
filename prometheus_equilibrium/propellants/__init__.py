"""
Prometheus propellant ingredient and formulation database.

Usage::

    from prometheus_equilibrium.propellants import PropellantDatabase, PropellantMixture

    prop_db = PropellantDatabase(
        "prometheus/propellants/propellants.toml",
        species_db=db,   # loaded SpeciesDatabase
    )
    prop_db.load()

    # Named formulation from TOML
    mixture = prop_db.expand("LOX_LH2_OF6")

    # Custom O/F ratio (mass amounts, any consistent unit)
    mixture = prop_db.mix([("LOX", 6.0), ("LH2", 1.0)])

    # Use with EquilibriumProblem
    from prometheus_equilibrium.equilibrium.problem import EquilibriumProblem, ProblemType
    problem = EquilibriumProblem(
        reactants=mixture.reactants,
        products=db.get_species(mixture.elements),
        problem_type=ProblemType.HP,
        constraint1=mixture.enthalpy,
        constraint2=pressure_pa,
    )
"""

from prometheus_equilibrium.propellants.loader import (
    PropellantDatabase,
    PropellantMixture,
    SyntheticSpecies,
)

__all__ = ["PropellantDatabase", "PropellantMixture", "SyntheticSpecies"]
