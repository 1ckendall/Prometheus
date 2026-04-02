# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Prometheus is an open-source combustion equilibrium solver for Python. It parses thermodynamic data from multiple sources (BURCAT/NASA-7, NASA-9, JANAF, CEA), builds a species database, and implements equilibrium solvers. The long-term goal is H₂/O₂ combustion validation against RocketCEA.

## Development Commands

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
uv sync                                                    # Install dependencies
uv run pytest                                              # Run the test suite
uv run pytest tests/test_solver_gmcb.py                   # Run a single test file
uv run pytest tests/test_solver_gmcb.py::test_hp_h2o2     # Run a single test
uv run python tests/benchmark.py                           # Full solver vs. RocketCEA benchmark (170 cases)
uv run python compare_rocketcea.py                         # G-McB vs. CEA single-case comparison
uv run python prometheus_equilibrium/thermo_data/parsers/cea.py           # Re-convert CEA thermo.inp → cea_nasa9.json
uv run prometheus-build-all-thermo                            # Re-generate all thermo databases in one command
uv run prometheus-build-legacy all                            # Re-generate terra.json + afcesic.json (calibrated)
uv run black prometheus_equilibrium/ tests/                               # Format code
uv run isort prometheus_equilibrium/ tests/                               # Sort imports
```

`tests/benchmark.py` is the primary regression benchmark. It tests H2/O2, CH4/O2, N2H4/N2O4, and NH3/O2 across 11 O/F ratios and 5 pressures (170 cases) against RocketCEA and reports temperature error, mean molar mass, γ, Cp, and speed of sound.

## Implementation Status

Current state:

- **Complete**: all thermodynamic data parsing; `Chemical`, `Species`, `JANAF`, `NASASevenCoeff`, `NASANineCoeff`, `SpeciesDatabase`; `Mixture` (all properties); `ElementMatrix` (all methods including `select_basis`); `EquilibriumProblem`; `EquilibriumSolution` (all derived properties); `GordonMcBrideSolver`; `MajorSpeciesSolver`.
- **Not yet implemented**: `PEPSolver._tp_equilibrium` (raises `NotImplementedError`).

## Architecture

### Module layout

```
prometheus_equilibrium/
  core/constants.py          — element molar masses, R, P° = 1e5 Pa
  equilibrium/
    species.py               — Chemical, Species, JANAF, NASASevenCoeff, NASANineCoeff, SpeciesDatabase
    _thermo_kernels.py       — optional Numba JIT kernels for NASASevenCoeff/NASANineCoeff hot path
    mixture.py               — Mixture (species list + mutable moles array)
    element_matrix.py        — ElementMatrix (stoichiometric matrix A)
    problem.py               — EquilibriumProblem, ProblemType enum
    solution.py              — EquilibriumSolution dataclass + derived properties
    solver.py                — EquilibriumSolver ABC, _ReactionAdjustmentBase, all three solver classes
    performance.py           — PerformanceSolver (frozen + shifting nozzle expansion)
  gui/                       — PySide6 desktop interface
  propellants/
    loader.py                — PropellantDatabase, SyntheticSpecies, TOML-based propellant definitions
  thermo_data/
    compiler.py              — thermo data compiler
    parsers/                 — individual format parsers
      burcat.py              — BURCAT/NASA-7 format
      cea.py                 — CEA thermo.inp format (also converts thermo.inp → cea_nasa9.json)
      janaf.py               — JANAF CSV format
      shomate.py             — Shomate polynomial format
      terra.py               — TERRA binary format
      _common.py             — shared parsing utilities
  tools/build_db.py          — parse raw .thr/.jnf files → nasa7.json, nasa9.json, janaf.csv
  tools/build_legacy_thermo.py — parse TERRA/AFCESIC binaries + calibrate AFCESIC references
tests/                       — pytest unit tests (mock species, no external DB dependency)
tests/benchmark.py           — integration benchmark vs. RocketCEA
```

### Species and database

`Chemical` → `Species` (abstract) → `JANAF`, `NASASevenCoeff`, `NASANineCoeff`. All species carry `elements: dict[str, float]`, `state`, `condensed` (0=gas, 1=condensed), and `alias`. Thermodynamic methods accept scalar float or `np.ndarray`; scalar calls take a fast path that skips NumPy overhead (and optionally uses Numba from `_thermo_kernels.py`).

`SpeciesDatabase` loads up to five sources; call `load()` before use:

```python
db = SpeciesDatabase(
    nasa7_path="prometheus_equilibrium/thermo_data/nasa7.json",
    nasa9_path="prometheus_equilibrium/thermo_data/nasa9.json",
    janaf_path="prometheus_equilibrium/thermo_data/janaf.csv",
    cea_nasa9_path="prometheus_equilibrium/thermo_data/cea_nasa9.json",  # optional, CEA-sourced polynomials
)
db.load(include_janaf=False)  # benchmark.py uses include_janaf=False for stability
```

Priority when the same species ID appears in multiple sources (default): NASA-9 > NASA-7 > JANAF > TERRA > AFCESIC. This can be overridden via `SpeciesDatabase(..., source_priority=...)` or `db.load(source_priority=...)`. `get_species(elements, max_atoms=N)` returns all species whose element set is a subset of `elements`; `max_atoms` filters out large molecules (benchmark.py uses `max_atoms=20`).

Canonical species ID: `{HillFormula}_{PHASE}` — e.g. `CO2_G`, `H2O_L`, `Al2O3_S`. Hill order: C first, H second, then alphabetical.

### Equilibrium data flow

```
EquilibriumProblem  →  Solver.solve()  →  EquilibriumSolution
      ↑                     ↑
  reactants+products    ElementMatrix
  (ProblemType.HP etc)  (from mixture.species)
```

`EquilibriumProblem` is purely declarative. `initial_mixture()` distributes moles equally across all gas-phase products as a starting guess. `b0_array(elements)` returns the element-abundance vector from reactants.

`Mixture` always keeps gas species before condensed (enforced at construction). The solver mutates `mixture.moles` in-place during Newton iteration.

`ElementMatrix` stores `A[i,k]` = atoms of element k in species i. Key methods: `element_abundances(n)` → `Aᵀ·n`, `select_basis(moles)` → Browne/Gram-Schmidt basis indices, `reaction_coefficients(basis)` → ν = C·B⁻¹ (used by Hybrid).

### Solver hierarchy

```
EquilibriumSolver (ABC)
  ├── _ReactionAdjustmentBase
  │     ├── MajorSpeciesSolver  ← recommended default
  │     └── PEPSolver     ← not yet implemented
  └── GordonMcBrideSolver  ← reference / validation
```

**`MajorSpeciesSolver`** (96.5% convergence, 0.013% mean T error vs RocketCEA):
- Inner loop (`_tp_equilibrium`): compressed Newton on major gas species only — matrix is always S×S regardless of species count. Minor species set analytically from element potentials π after each Newton step.
- Outer loop (`_temperature_search`): Newton + interval-halving to satisfy energy constraint (HP/SP).
- `_tp_equilibrium` returns a 4-tuple `(mixture, pi, n_iters, converged_bool)`; `_temperature_search` returns a 5-tuple `(mixture, T, pi, n_outer, converged_bool)`. `solve()` uses the returned `converged` flag directly — it does NOT recheck element residuals, as the "final exact update" in `_temperature_search` temporarily disturbs element balance.

**`GordonMcBrideSolver`** (72.9% convergence, 0.82% mean T error — reference/validation):
- Single Newton loop solving π, Δnc, Δln(n), Δln(T) simultaneously. Matrix size (S+nc+1)² for TP or (S+nc+2)² for HP/SP.
- Stability fix: tracks `n_var` as the *previous iteration's* `n_gas_total`. This gives `G[idx_n, idx_n] = n_gas_total_current − n_gas_total_prev ≠ 0`, preventing runaway `delta_ln_n` on lean mixtures. `n_var` is stored before `_apply_update`, so the next iteration sees the change. `mu_gas` always uses the actual current `n_gas_total`, not `n_var`.

**Shared** (`_ReactionAdjustmentBase`): `_temperature_search`, `_split_major_minor`, `_manage_condensed_phases`, `_initialise` (validates, filters ionic species, removes products not coverable from reactant elements).

### Key design constants

- All temperatures in Kelvin, energies in J/mol, pressures in Pa (P° = 1e5 Pa = 1 bar)
- `_LOG_CONC_TOL = math.log(1e-8)` — concentration floor; species with `ln(nⱼ) - ln(n) ≤ _LOG_CONC_TOL` are zeroed
- Convergence tolerance: `5e-6` (step size criterion) for both solvers
- `_ReactionAdjustmentBase.__init__` defaults: `max_iterations=50`, `tolerance=5e-6`, `minor_threshold=1e-2`

### Reference material and submodules

- `cea/` — git submodule: NASA CEA Fortran source (Gordon-McBride reference implementation, especially `cea/source/`)
- `pypropep/` — git submodule: C implementation of Propep (PEP reference); uses NASA-format thermo data — **not relevant to AFCESIC**
- `pep-for-zos/` — git submodule: historical PEP Fortran source for z/OS (1970s Naval Weapons Center origin)
- `terra/` — external equilibrium solver (BASIC source + compiled binaries); not integrated into Prometheus
- `papers/` — Villars-Browne PEP paper and NASA RP-1311

We use Google-style docstrings throughout. **All new docstrings must use Google style** (`Args:`, `Returns:`, `Raises:`, `Attributes:`, `Example:`, `Note:` sections with 4-space-indented content). NumPy/reStructuredText docstring styles are not used in this project.

## Documentation

Sphinx docs live in `docs/`. Build with:

```bash
uv run --group docs sphinx-build -b html docs docs/_build/html
# or, on Unix/macOS:
cd docs && make html
```

The docs use `sphinx.ext.autodoc` + `sphinx.ext.autosummary` (with `:recursive:`) to automatically discover and document every module, class, and function in `prometheus_equilibrium/`. The Napoleon extension translates Google-style docstrings. Theme: `renku` (package: `renku-sphinx-theme`).

The `docs/` directory structure:
- `index.rst` — landing page + quick-start
- `thermo/` — thermodynamic database guide and TERRA integration notes
- `solver_comparison.rst` — algorithm analysis for MajorSpeciesSolver, G-McB, PEP
- `api/` — auto-generated API reference stubs
