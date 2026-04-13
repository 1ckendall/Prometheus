# Prometheus Equilibrium Audit Report (2026-04-13)

## Scope and Method

This audit focused on:

- solver correctness and numerical behavior (`equilibrium/solver.py`, `equilibrium/performance.py`, `equilibrium/solution.py`),
- API/contract consistency (`equilibrium/problem.py`, docs, README),
- hygiene and robustness issues (warnings, exception handling, test coverage around new optimizer stack).

Evidence used:

- test runs: non-integration suite and integration suite,
- targeted runtime probes for numerical invariance and problem-type handling,
- code and documentation cross-checks.

## Executive Summary

The codebase is in substantially better shape than many scientific Python projects (strong test pass rate, structured solver architecture), but there are several serious contract/correctness issues:

1. `specific_impulse()` currently depends on mixture scaling (critical correctness bug).
2. `TV`/`UV`/`SV` are exposed as supported problem types, but solver internals treat their second constraint as pressure.
3. Validation and docs claim checks that are not actually implemented.
4. Warning behavior in production paths is too noisy and can hide real failures.

---

## Findings (ordered by severity)

### 1) CRITICAL - `specific_impulse()` is not basis/scale invariant

**Location**

- `prometheus_equilibrium/equilibrium/solution.py:347-351`

**Issue**

- `dh_j_kg` is computed from `total_enthalpy` (extensive J) but treated as J/kg.
- The implementation relies on an implicit assumption that mixtures always represent 1 kg basis.
- That assumption is not guaranteed by API contracts and is violated by generic `Mixture` usage.

**Evidence**

- Direct runtime probe produced different Isp for thermodynamically identical states scaled in moles:
  - scale 1.0: `41.647 s`
  - scale 10.0: `131.700 s`

**Impact**

- Performance metrics can be physically wrong when callers do not normalize to one specific basis.

**Fix**

- Introduce explicit mass-basis properties (e.g., total mass, specific enthalpy J/kg).
- Compute Isp from true specific quantities; enforce scale invariance with tests.

---

### 2) HIGH - `TV`/`UV`/`SV` are advertised but not implemented correctly

**Location**

- Type exposure: `prometheus_equilibrium/equilibrium/problem.py:59-67`
- Solver assumption: `prometheus_equilibrium/equilibrium/solver.py:495`, `prometheus_equilibrium/equilibrium/solver.py:2294`

**Issue**

- For all solve paths, code sets `P = problem.constraint2` and uses it as pressure.
- In `TV`/`UV`/`SV`, `constraint2` is documented as volume.

**Evidence**

- Runtime probe with `ProblemType.TV` logged `P=1.00e+00 Pa, type=TV`, then singular matrix failure.

**Impact**

- Silent semantic mismatch, solver instability, and invalid results for volume-constrained modes.

**Fix**

- Immediate: reject `TV`/`UV`/`SV` explicitly with clear error.
- Longer term: implement proper volume-based formulation before re-enabling.

---

### 3) HIGH - Validation and docs are out of sync; missing physical guards

**Location**

- `prometheus_equilibrium/equilibrium/problem.py:267-316`

**Issue**

- Docstring claims sign checks for HP/SP/UV/SV constraints.
- Actual checks only enforce `t_init > 0` and positive fixed temperature.
- No explicit positivity checks for pressure (TP/HP/SP) or volume (TV/UV/SV).

**Impact**

- Invalid inputs pass validation and fail later in numeric kernels.

**Fix**

- Add explicit per-problem-type constraint checks.
- Keep validation docs strictly aligned with implementation.

---

### 4) MEDIUM - Repeated divide-by-zero warnings in hot loop

**Location**

- `prometheus_equilibrium/equilibrium/solver.py:2060`

**Issue**

- `np.where(current > 0.0, np.log(current), ...)` evaluates both branches; logs zero entries and warns.

**Impact**

- Warning noise in integration and continuation runs; diagnostics become noisy.

**Fix**

- Use masked log evaluation to avoid calling `log` on non-positive entries.

---

### 5) MEDIUM - Warning flood from thermo-range filtering in continuation paths

**Location**

- Warning emission: `prometheus_equilibrium/equilibrium/solver.py:417-423`
- Frequent call site during expansion march: `prometheus_equilibrium/equilibrium/performance.py:722`

**Issue**

- Species-drop warning is emitted repeatedly for iterative SP solves in nozzle continuation.

**Impact**

- CI/user logs can be overwhelmed, reducing signal-to-noise.

**Fix**

- Warn once per top-level solve (or once per species set change), downgrade repeated events to debug counters.

---

### 6) MEDIUM - Docs/README implementation status drift

**Location**

- `README.md:19`, `README.md:52`
- `docs/solver_comparison.rst:92`, `docs/solver_comparison.rst:165`

**Issue**

- Documentation still states PEP TP solver is not implemented.
- Code contains a non-stub `PEPSolver._tp_equilibrium` and dedicated tests (`tests/test_solver_pep.py`).

**Impact**

- Misleads users and contributors; reduces confidence in docs.

**Fix**

- Update implementation status sections in README and solver docs.

---

### 7) LOW - Broad exception handling hides defects

**Location**

- bare `except` in `prometheus_equilibrium/tools/compare_databases.py:66-67`
- several bare `except` in GUI parsing paths (`prometheus_equilibrium/gui/pages/simulator.py`)

**Issue**

- Catch-all exception swallowing masks data and parsing errors.

**Impact**

- Harder debugging and latent behavior bugs.

**Fix**

- Catch concrete exception types, add minimal debug logging or user feedback.

---

### 8) LOW - Test coverage gap around optimizer engine/CLI paths

**Location**

- New optimizer stack under `prometheus_equilibrium/optimization/` and `prometheus_equilibrium/tools/run_optimizer.py`

**Issue**

- Constraint compiler has tests; engine and headless CLI path appear under-tested.

**Impact**

- Operational regressions possible despite green suite.

**Fix**

- Add smoke tests for config load/dump, optimizer engine with minimal trial count, and CLI run path.

---

## Prioritized Remediation Plan

### P0 (must fix first)

1. Fix `specific_impulse()` to use explicit mass-basis thermodynamics.
2. Disable/reject `TV`/`UV`/`SV` in solver entry points until truly implemented.
3. Harden `EquilibriumProblem.validate()` with physically correct per-type checks.

### P1

4. Remove divide-by-zero warning source in log-space updates.
5. De-duplicate species-drop warnings during continuation solves.
6. Align README/docs implementation status and supported-problem matrix.

### P2

7. Replace broad exception catches with narrow handling and diagnostics.
8. Add optimizer engine and CLI smoke/regression tests.

---

## What is lost if `TV`/`UV`/`SV` are not supported?

If only `TP`/`HP`/`SP` are supported, the project loses entire classes of physically important scenarios:

- **Closed, rigid-system combustion/explosion modeling** (`UV`): cannot directly model constant-volume energy release (pressure vessel, bomb calorimetry style analyses).
- **Constant-volume equilibrium studies** (`TV`): cannot study equilibrium composition at fixed geometric volume and temperature.
- **Isentropic compression/expansion at fixed volume constraints** (`SV`): loses access to some compression process analyses and related benchmark parity with CEA mode set.
- **Reference-coverage parity**: harder one-to-one comparisons against tools/papers that report these modes.
- **API expectations**: users expecting NASA-CEA-style six-mode support will face reduced functional scope.

For many rocket-nozzle workflows this is acceptable (they mostly rely on `HP` and `SP`), but for general combustion thermodynamics and closed-system studies it is a meaningful capability gap.

