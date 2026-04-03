# Prometheus

Prometheus is an open-source Python package for chemical equilibrium and
rocket-combustion analysis. It parses multiple thermo-data formats, builds a
unified species database, and solves equilibrium problems such as adiabatic
chamber combustion (HP) and isentropic nozzle expansion (SP).

## Project Status

Pre-1.0 (`v0.1.0`), active development. Core solver infrastructure and all
thermo-data parsers are complete; the recommended solver is `GordonMcBrideSolver`.
Expect breaking changes to the API and data formats!

## TO-IMPLEMENT Roadmap

Nice-to-have items for upcoming releases, grouped by impact area.

### Core Solver and Data

- [x] Add structured non-convergence diagnostics to `EquilibriumSolution` (failure reason, residuals, element-balance error, last step norm).
- [ ] Implement `PEPSolver._tp_equilibrium` and add regression coverage for TP/HP/SP behavior.
- [ ] Add optional numerical-stability fallback modes for difficult edge cases (adaptive damping / tighter line search controls).
- [ ] Add lightweight profiling hooks to report per-iteration timing and major matrix-solve costs.
- [ ] Investigate if key species from the species database can also be used as propellant ingredients e.g. methane, oxygen

### GUI

- [ ] Implement full report export from the GUI (TXT/CSV/JSON) including sweep metadata and final plots.
- [ ] Add progress + cancel support for long sweep runs in `PerformanceWorker`.
- [ ] Persist user session settings (units, selected species databases, sweep mode, and recent inputs).

### CLI and Tooling

- [ ] Add a non-interactive solve CLI for HP/TP/SP runs with machine-readable output (`--json` / `--csv`).
- [ ] Add benchmark baseline comparison mode in `tests/benchmark.py` with threshold-based pass/fail output for CI.
- [ ] Add strict non-interactive build options in thermo tooling (`--no-prompt`, `--fail-on-missing-label`) for reproducible automation.

## Features

**Thermodynamic databases**

- NASA-7 and NASA-9 (including CEA-derived polynomials)
- JANAF tables
- TERRA / Shomate-equivalent (translated from binary)
- AFCESIC ionic and condensed species (translated and calibrated)

**Equilibrium problem types:** `TP`, `HP`, `SP`, `TV`, `UV`, `SV`

**Solvers**

| Solver | Role | Notes                                                                    |
|---|---|--------------------------------------------------------------------------|
| `GordonMcBrideSolver` | Production (default) | Fastest in current implementation; matches NASA CEA / RocketCEA algorithm |
| `MajorSpeciesSolver` | Alternative | S×S Newton, quadratic convergence                                        |
| `PEPSolver` | Reference | Linear convergence, not recommended, currently unstable solve properties |

**Rocket performance:** frozen and shifting isentropic nozzle expansion with
c\*, Isp (vacuum and sea-level), and area ratio.

For shifting nozzle expansion, Prometheus supports configurable SP entropy
bases: full-mixture entropy (default, best ProPEP/CEA parity), gas-only
entropy, and an auto mode that retries gas-only if full-mixture SP does not
converge.

## Installation

### Install from PyPI (recommended)

```bash
python -m pip install prometheus-equilibrium
python -m pip install "prometheus-equilibrium[gui]"
```

### Install from source (development workflow)

This repository uses [uv](https://docs.astral.sh/uv/).

```bash
uv sync --group dev
pip install -e .
pip install -e ".[gui]"
```

## Quick Start

> **Note:** `SpeciesDatabase()` now resolves package-relative thermo data paths
> automatically. You can still override any source path in the constructor.

```python
from prometheus_equilibrium.equilibrium import (
    SpeciesDatabase,
    EquilibriumProblem,
    ProblemType,
    GordonMcBrideSolver,
)

db = SpeciesDatabase()
db.load(include_nasa9=True, include_nasa7=True, include_terra=True)

h2 = db.find("H2", phase="G")
o2 = db.find("O2", phase="G")
products = db.get_species({"H", "O"}, max_atoms=6)

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
```
Or access the GUI:

```bash
prometheus-gui
```

If you are running from source without installing scripts, use:

```bash
uv run prometheus-gui
```

For a longer walkthrough, open [`demonstration.ipynb`](demonstration.ipynb).

## Propellant Database

Prometheus ships a TOML propellant database covering hundreds of ingredients
(AP, HTPB, aluminium, MMH, UDMH, N₂O₄, kerosene, and many more) that has been directly
converted from PROPEP. This database is in active development and may change in the future
as propellants are added / removed / consolidated.
`PropellantDatabase.mix()` returns a `PropellantMixture` with the element set,
reactant moles, and total H₀ pre-computed so you don't need to calculate them
manually.  Pass the mixture directly to
`PerformanceSolver.solve_from_mixture()` to get paired frozen and shifting
results in one call:

```python
from prometheus_equilibrium.equilibrium import SpeciesDatabase, PerformanceSolver
from prometheus_equilibrium.propellants import PropellantDatabase

db = SpeciesDatabase()
db.load(include_nasa9=True, include_nasa7=True, include_terra=True)

prop_db = PropellantDatabase(
    "prometheus_equilibrium/propellants/propellants.toml"
)
prop_db.load()

# AP/Al/HTPB composite solid propellant at O/F ≈ 2.74 (68/18/14 by mass)
mixture = prop_db.mix([
    ("AMMONIUM_PERCHLORATE",    0.68),
    ("ALUMINUM_PURE_CRYSTALINE", 0.18),
    ("HTPB_R_45HT",              0.14),
])

result = PerformanceSolver().solve_from_mixture(
    mixture, db,
    p_chamber=70e5,    # 70 bar
    area_ratio=10.0,
)

print(f"T_chamber = {result.shifting.chamber.temperature:.0f} K")
print(f"c*        = {result.shifting.cstar:.1f} m/s")
print(f"Isp (vac, shifting) = {result.shifting.isp_vac:.1f} m/s")
print(f"Isp (vac, frozen)   = {result.frozen.isp_vac:.1f} m/s")
```

The `mix()` call accepts amounts in any consistent mass unit — only the ratios
matter.  For liquid bipropellants simply name the two ingredients:

```python
mixture = prop_db.mix([("HYDRAZINE", 1.0), ("NITROGEN_TETROXIDE_LIQ", 1.33)])
```

Use `prop_db.ingredient_ids` and `prop_db.formulation_ids` to browse what is
available.  Named formulations stored in the TOML can be loaded with
`prop_db.expand("formulation_id")` instead of `mix()`.

## Documentation

Online: https://prometheus-equilibrium.readthedocs.io/en/latest/

Build the Sphinx docs locally:

```bash
uv run --group docs sphinx-build -b html docs docs/_build/html
```

The docs include:

- **API reference** — every public class and function with docstrings
- **Thermodynamic database guide** — polynomial formats and reference-state
  calibration across NASA-9, TERRA, and AFCESIC
- **Solver comparison** — algorithm analysis and implementation notes for
  MajorSpeciesSolver, GordonMcBrideSolver, and PEP

## Development

```bash
uv sync --group dev            # Install dev extras (pytest, black, rocketcea …)
uv run pytest                  # Run the test suite
uv run python tests/benchmark.py  # Benchmark 170 cases vs RocketCEA
uv run black prometheus_equilibrium tests
uv run isort prometheus_equilibrium tests
uv run prometheus-build-all-thermo   # Re-generate all thermo databases
uv run prometheus-build-legacy all   # Re-generate TERRA + AFCESIC from binaries
```

## Package Layout

```
prometheus_equilibrium/
  core/constants.py              element masses, R, P° = 1×10⁵ Pa
  equilibrium/
    species.py                   Chemical, Species hierarchy, SpeciesDatabase
    mixture.py                   Mixture (mutable moles array + all properties)
    element_matrix.py            ElementMatrix, basis selection
    problem.py                   EquilibriumProblem, ProblemType
    solution.py                  EquilibriumSolution + derived properties
    solver.py                    MajorSpeciesSolver, GordonMcBrideSolver, PEPSolver
    performance.py               Frozen / shifting nozzle expansion
  propellants/loader.py          PropellantDatabase, TOML-based definitions
  thermo_data/                   Compiled JSON/CSV databases + raw source files
  tools/                         CLI build scripts (prometheus-build-*)
  gui/                           PySide6 desktop interface
```

## Validation

Prometheus has been developed by cross-referencing multiple existing solver implementations:

- [NASA CEA](https://github.com/nasa/cea)
- [PyProPEP / cpropep](https://github.com/jonnydyer/pypropep)
- [pep-for-zos](https://github.com/haynieresearch/pep-for-zos)

The `tests/benchmark.py` script runs 170 cases (H₂/O₂, CH₄/O₂, N₂H₄/N₂O₄,
NH₃/O₂ across 11 O/F ratios × 5 pressures) and reports temperature, molar
mass, γ, Cₚ, and speed-of-sound error against RocketCEA.

## Contributing

- Open an issue first when possible to discuss the change.
- Add or update tests for any behaviour change.
- Keep docstrings in Google style (`Args:`, `Returns:`, etc.).
- Install the pre-commit hooks after cloning so that black and isort run
  automatically on every commit:

```bash
uv sync --group dev
uv run pre-commit install
```

## License

GPL-3.0-or-later. See [`LICENSE.txt`](LICENSE.txt).

## Acknowledgements

- Bonnie J. McBride and Sanford Gordon ([NASA CEA](https://github.com/nasa/cea))
- Boris Trusov (TERRA thermodynamic database)
- Alexander Burcat ([BURCAT thermochemical database](https://respecth.elte.hu/burcat.php))
