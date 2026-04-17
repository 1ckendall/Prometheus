# Prometheus

Prometheus is an open-source Python package for chemical equilibrium and
rocket-combustion analysis. It parses multiple thermo-data formats, builds a
unified species database, and solves equilibrium problems such as adiabatic
chamber combustion (HP) and isentropic nozzle expansion (SP).

## Project Status

Pre-1.0 (`v0.1`), active development. Core solver infrastructure, all
thermo-data parsers, and the desktop GUI are complete. The recommended solver
is `GordonMcBrideSolver`. Expect breaking changes to the API and data formats
before 1.0.

## Roadmap

### Core Solver

- [ ] Implement `PEPSolver._tp_equilibrium` (currently raises `NotImplementedError`).
- [ ] Add optional numerical-stability fallback modes for difficult edge cases.
- [ ] Add lightweight profiling hooks to report per-iteration timing and major matrix-solve costs.
- [ ] Investigate using species-database entries directly as propellant ingredients (e.g. methane, oxygen).

### GUI

- [ ] Full report export (TXT / CSV / JSON) including sweep metadata and plots.
- [ ] Cancel support for long sweep runs in `PerformanceWorker`.
- [ ] Persist user session settings (units, database selection, sweep mode, recent inputs).

### CLI and Tooling

- [ ] Non-interactive solve CLI for HP/TP/SP runs with machine-readable output (`--json` / `--csv`).
- [ ] Benchmark baseline mode in `tests/benchmark.py` with threshold-based pass/fail for CI.

## Features

**Thermodynamic databases**

- NASA-7 and NASA-9 (including CEA-derived polynomials)
- JANAF tables
- TERRA / Shomate-equivalent (translated from binary)
- AFCESIC ionic and condensed species (translated and calibrated)

**Equilibrium problem types:** `TP`, `HP`, `SP`, `TV`, `UV`, `SV`

**Solvers**

| Solver | Role | Convergence | Mean T error vs RocketCEA | Speed |
|---|---|---|---|---|
| `GordonMcBrideSolver` | **Recommended** | 100 % | 0.017 % | ~14 ms/case |
| `MajorSpeciesSolver` | Alternative | 95.9 % | 0.023 % | ~235 ms/case |
| `PEPSolver` | Not yet implemented | — | — | — |

**Rocket performance:** frozen and shifting isentropic nozzle expansion with
c\*, Isp (vacuum, sea-level, and actual), and area ratio.

Note: ``c*`` is currently computed from the converged throat state, while
``Isp`` is computed mainly from chamber-to-exit energy/pressure terms. In
shifting mode, two solvers can therefore report nearly identical ``Isp`` but
slightly different ``c*`` if their throat SP root converges to slightly
different states.

Shifting nozzle expansion uses full-mixture entropy as the SP isentrope
constraint.

**Desktop GUI** (`prometheus-gui`): PySide6 interface for propellant
composition, database selection, sweep runs (O/F or Pc), and result
visualisation including solver convergence plots, nozzle expansion curves, and
per-station performance charts.

## Installation

### Install from PyPI

```bash
pip install prometheus-equilibrium           # solver + propellant database
pip install "prometheus-equilibrium[gui]"    # + PySide6 desktop GUI
```

### Install from source (development)

Requires [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/1ckendall/Prometheus
cd Prometheus
uv sync          # installs all dependencies including dev tools and GUI
```

## Quick Start

```python
from prometheus_equilibrium.equilibrium import (
    SpeciesDatabase,
    EquilibriumProblem,
    ProblemType,
    GordonMcBrideSolver,
)

db = SpeciesDatabase()
db.load()   # NASA-7, NASA-9, TERRA loaded by default

h2 = db.find("H2", phase="G")
o2 = db.find("O2", phase="G")
products = db.get_species({"H", "O"}, max_atoms=6)

T_react = 298.15
H0 = h2.enthalpy(T_react) * 2.0 + o2.enthalpy(T_react) * 1.0

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

Launch the desktop GUI:

```bash
uv run prometheus-gui
```

### GUI File Formats (Breaking Change)

- Composition snapshots now use `*.prop.json` (previously `*.prop`).
- Optimizer configs now use `*.prop-opt.json` (previously generic `*.json`).

The GUI enforces these extensions when saving/loading via the File menu.


## Propellant Database

Prometheus ships a TOML propellant database covering hundreds of ingredients
(AP, HTPB, aluminium, MMH, UDMH, N₂O₄, kerosene, and many more) converted
from PROPEP. `PropellantDatabase.mix()` returns a `PropellantMixture` with the
element set, reactant moles, and total H₀ pre-computed. Pass the mixture
directly to `PerformanceSolver.solve_from_mixture()` to get paired frozen and
shifting results in one call:

```python
from prometheus_equilibrium.equilibrium import SpeciesDatabase, PerformanceSolver
from prometheus_equilibrium.propellants import PropellantDatabase

db = SpeciesDatabase()
db.load()

prop_db = PropellantDatabase()
prop_db.load()

# AP/Al/HTPB composite solid propellant (68/18/14 by mass)
mixture = prop_db.mix([
    ("AMMONIUM_PERCHLORATE",     0.68),
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

For liquid bipropellants:

```python
mixture = prop_db.mix([("HYDRAZINE", 1.0), ("NITROGEN_TETROXIDE_LIQ", 1.33)])
```

Use `prop_db.ingredient_ids` and `prop_db.formulation_ids` to browse available
ingredients. Named formulations stored in the TOML can be loaded with
`prop_db.expand("formulation_id")`.

## Documentation

Online: https://prometheus-equilibrium.readthedocs.io/en/latest/

Build locally:

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
uv sync                              # install everything (dev tools, GUI deps, test deps)
uv run pytest                        # run the test suite
uv run pytest -m "not integration"   # fast suite (no RocketCEA required)
uv run python tests/benchmark.py     # benchmark 170 cases vs RocketCEA
uv run prometheus-build-all-thermo   # re-generate all compiled thermo databases
uv run prometheus-build-legacy all   # re-generate TERRA + AFCESIC from binaries
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

Prometheus is cross-referenced against multiple existing solver implementations:

- [NASA CEA](https://github.com/nasa/cea)
- [PyProPEP / cpropep](https://github.com/jonnydyer/pypropep)
- [pep-for-zos](https://github.com/haynieresearch/pep-for-zos)

`tests/benchmark.py` runs 170 cases (H₂/O₂, CH₄/O₂, N₂H₄/N₂O₄, NH₃/O₂
across 11 O/F ratios × 5 pressures) and reports temperature, molar mass, γ,
Cₚ, and speed-of-sound error against RocketCEA.

## Contributing

- Open an issue first to discuss the change.
- Add or update tests for any behaviour change.
- Keep docstrings in Google style (`Args:`, `Returns:`, etc.).
- Pre-commit hooks (black + isort) run automatically on every commit after
  the one-time setup:

```bash
uv sync
uv run pre-commit install
```

## License

GPL-3.0-or-later. See [`LICENSE.txt`](LICENSE.txt).

## Acknowledgements

- Bonnie J. McBride and Sanford Gordon ([NASA CEA](https://github.com/nasa/cea))
- Boris Trusov (TERRA thermodynamic database)
- Alexander Burcat ([BURCAT thermochemical database](https://respecth.elte.hu/burcat.php))
