# Optuna Propellant Optimizer Implementation Plan

## Scope and goals

Build an Optuna-driven optimizer that searches propellant compositions to maximize:

- `FoM = Isp * rho**n`
- with `0 <= n <= 1`

The design must support:

- hard formulation relations (fixed proportions and sum-to-total groups),
- extensible external metrics (for example viscosity or pourability simulations),
- future expansion to multi-objective optimization and constrained optimization.

## Why this shape fits Prometheus

- Existing flow already computes performance from compositions (`PropellantDatabase.mix` -> `EquilibriumProblem` -> `PerformanceSolver`).
- Existing GUI already has long-running workers in background threads (`QThread` in `gui/engine.py`).
- `optuna` is already present in the `dev` dependency group in `pyproject.toml`.

## Proposed architecture

Add a new package:

- `prometheus_equilibrium/optimization/__init__.py`
- `prometheus_equilibrium/optimization/problem.py`
- `prometheus_equilibrium/optimization/constraints.py`
- `prometheus_equilibrium/optimization/metrics.py`
- `prometheus_equilibrium/optimization/evaluator.py`
- `prometheus_equilibrium/optimization/engine.py`
- `prometheus_equilibrium/optimization/study_io.py` (optional persistence helpers)

### Core objects

1. `OptimizationProblem`
   - Declarative spec of decision variables, relations, operating conditions, and objective settings.
   - Contains all data needed to evaluate one trial.

2. `FormulationConstraintCompiler`
   - Converts relation declarations into executable transforms/checks.
   - Handles:
     - **fixed-proportion groups** (`x_i = alpha_i * g`),
     - **sum-to-total groups** (`sum(x_i) = T`).

3. `TrialEvaluator`
   - Samples free variables from Optuna trial,
   - builds a feasible composition,
   - runs solver/performance,
   - computes metrics and final objective,
   - returns objective plus optional constraint residuals.

4. `MetricPlugin` interface
   - Extension point for external metrics.
   - Example: viscosity surrogate or external simulator wrapper.

5. `OptimizationEngine`
   - Owns the Optuna study lifecycle and reproducibility settings.
   - Supports callbacks, progress, and checkpointing.

## Data model and schema (v1)

Use a JSON/TOML-serializable schema (can later be integrated into propellant files or a separate optimization config file).

```toml
[objective]
isp_variant = "isp_actual"   # or isp_vac / isp_sl
rho_exponent = 0.6           # n, validated 0..1
mode = "maximize"

[operating_point]
pc_pa = 6_894_757.0
expansion_type = "pressure"  # pressure | area_ratio
pe_pa = 101_325.0
ambient_pa = 101_325.0
shifting = true

[[variables]]
id = "AP"
kind = "mass_fraction"
min = 0.55
max = 0.75

[[variables]]
id = "AL"
kind = "mass_fraction"
min = 0.05
max = 0.25

[[variables]]
id = "BINDER"
kind = "mass_fraction"
min = 0.05
max = 0.25

[[relations.fixed_proportion_groups]]
id = "binder_system"
members = ["HTPB", "IPDI"]
ratios = [100.0, 8.0]
# Internally normalized to alpha_i so x_i = alpha_i * g

[[relations.sum_to_total_groups]]
id = "solids"
members = ["AP", "AL"]
total = 0.84

[normalization]
require_total_mass_fraction = 1.0
```

## Constraint handling math

Let composition vector be `x` over ingredient mass fractions.

### 1) Fixed proportion groups

For group members `i in G` with user ratios `r_i > 0`:

- normalize: `alpha_i = r_i / sum(r_j)`
- introduce one group scalar `g_G`
- enforce: `x_i = alpha_i * g_G`

This reduces dimensionality and guarantees feasibility by construction.

### 2) Sum-to-total groups

For group `H` with target `T_H`:

- enforce: `sum(x_i for i in H) = T_H`

Recommended v1 implementation:

- pick `|H|-1` free variables, compute final one by closure,
- reject/penalize if closure value violates bounds.

Alternative (future): soft constraints using Optuna constrained sampler residuals.

### 3) Global mass-fraction closure

- enforce `sum(x_i) = 1.0` (or configured total)
- include a consistency validator to detect overlapping/conflicting relation graphs.

## Objective and metric pipeline

### Base objective

- compute selected Isp (`isp_actual`, `isp_vac`, or `isp_sl`),
- compute mixture density `rho` from ingredient densities in `PropellantDatabase`,
- compute `FoM = Isp * rho**n`.

For numerical robustness in optimization:

- optimize `log(FoM) = log(Isp) + n * log(rho)` internally,
- store both raw and log values in trial user attrs.

### Metric plugin system

Define plugin protocol:

- `name: str`
- `evaluate(context) -> MetricResult`

`context` includes composition, operating point, chamber/performance outputs, and optional cache handle.

Each `MetricResult` can provide:

- `value` (float),
- `status` (`ok`, `invalid`, `error`),
- optional `constraint_residuals` (for hard/soft enforceable bounds),
- metadata (runtime, warnings, external tool output digest).

This makes it straightforward to add viscosity/pourability constraints later:

- hard filter (`viscosity <= limit`),
- penalty in scalar objective,
- or separate objective dimension (future MO).

## Additional useful elements to include now

1. **Infeasibility diagnostics**
   - Return structured reasons when a trial is rejected (bound violation, invalid thermo solve, external metric failure).
   - Essential for tuning search spaces.

2. **Caching of expensive solver evaluations**
   - Hash canonicalized composition + operating conditions.
   - Avoid re-solving near-duplicate trials during TPE startup/noisy phases.

3. **Deterministic/reproducible runs**
   - Seed all stochastic components (`optuna`, optional numpy use).
   - Record seed, git commit hash, and database options in study metadata.

4. **Graceful failure policy**
   - On solver non-convergence or plugin timeout: mark trial failed with consistent penalty rather than crashing the study.

5. **Pluggable samplers/pruners**
   - Start with TPE; allow CMA-ES/NSGA-II later.
   - Keep settings in `OptimizationProblem`.

6. **Result artifact model**
   - Persist best composition, top-N Pareto candidates (future), and full trial table for GUI plotting/export.

## Integration points

### Backend integration

- Use existing `PropellantDatabase.mix` for reactant building.
- Use existing `SpeciesDatabase.get_species` + `EquilibriumProblem` + `PerformanceSolver.solve_pair` to evaluate each candidate.
- Keep optimization package solver-agnostic by injecting configured solver instance (G-McB default).

### GUI integration (phase 2)

- Add `OptimizationWorker(QThread)` parallel to existing `PerformanceWorker` in `gui/engine.py`.
- Add optimizer panel/page for:
  - variable bounds,
  - relation groups,
  - objective parameters (`n`, Isp variant),
  - study controls (trials, timeout, seed).
- Reuse existing results page infrastructure for progress and charts.

### CLI integration (phase 2)

- Add `prometheus_equilibrium/tools/optimize_propellant.py`.
- Register script in `pyproject.toml`, e.g. `prometheus-optimize`.
- Input: optimization config file + optional resume database path.

## Validation and testing strategy

Add tests:

- `tests/test_optimization_constraints.py`
  - fixed-proportion expansion correctness,
  - sum-to-total closure,
  - conflict/overlap detection,
  - bounds propagation.

- `tests/test_optimization_evaluator.py`
  - objective computation,
  - penalty behavior for invalid trials,
  - plugin hook execution and error handling.

- `tests/test_optimization_engine.py`
  - deterministic result with fixed seed on mocked evaluator,
  - study resume/checkpoint behavior.

- GUI worker smoke tests near `tests/test_gui_engine_dock.py`
  - start/stop/progress/report wiring.

Use mocked solver outputs for unit speed; reserve small integration tests for real solver calls.

## Phased delivery plan

### Milestone M1: Core optimizer MVP (backend only)

- Implement optimization package (`problem`, `constraints`, `evaluator`, `engine`, `metrics`).
- Support scalar objective `Isp * rho**n` with `0 <= n <= 1`.
- Implement fixed-proportion and sum-to-total relation support.
- Add unit tests for constraints and evaluator.

Exit criteria:

- Run optimization from a Python script,
- produce best formulation and objective with reproducible seed,
- all new tests pass.

### Milestone M2: UX surfaces (CLI + GUI worker)

- CLI entry point for study execution and result export.
- GUI worker + basic optimizer configuration widgets.
- Show top candidates and per-trial convergence plots.

Exit criteria:

- User can run optimizer without writing code,
- cancellation and progress reporting are reliable.

### Milestone M3: External metric hooks + constrained/MO readiness

- Add first external metric plugin skeleton (viscosity placeholder adapter).
- Expose hard/soft constraint channels in evaluator output.
- Add optional Optuna constrained optimization path and MO scaffolding.

Exit criteria:

- external metric can influence acceptance/ranking,
- architecture supports adding a second objective with minimal refactor.

## Risks and mitigations

- **Solver failures for exotic compositions**: robust penalty policy + narrow validated bounds templates.
- **Constraint graph complexity**: pre-compile and validate relation graph before study starts.
- **Runtime cost**: caching + warm starts + optional lower-fidelity pre-screen plugins.
- **Data quality (ingredient density, cp, etc.)**: track source/confidence and surface warnings in reports.

## Recommended first implementation order (concrete)

1. Implement `OptimizationProblem` and relation schema validation.
2. Implement `FormulationConstraintCompiler` and unit tests.
3. Implement `TrialEvaluator` with base FoM objective.
4. Implement `OptimizationEngine` with Optuna study orchestration.
5. Add CLI tool to exercise the full path.
6. Add GUI worker after backend APIs stabilize.

