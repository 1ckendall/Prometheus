# GUI Optimizer Integration Spec (MVP)

## Purpose

Integrate Optuna-based propellant optimization into the GUI with minimal risk by:

- preserving existing Simulator/EngineDock workflows,
- running optimization in a background thread,
- exposing constrained composition search controls,
- allowing one-click application of best composition back to Simulator.

## Scope (implemented in this MVP)

- New navigation page: `Optimizer`.
- New GUI page: `prometheus_equilibrium/gui/pages/optimizer.py`.
- New optimization backend package:
  - `prometheus_equilibrium/optimization/problem.py`
  - `prometheus_equilibrium/optimization/constraints.py`
  - `prometheus_equilibrium/optimization/engine.py`
- Objective support:
  - maximize `Isp * rho^n` with `0 <= n <= 1`.
  - selectable Isp variant: `isp_actual`, `isp_vac`, `isp_sl`.
- Constraint support:
  - fixed-proportion groups,
  - sum-to-total groups,
  - optional closure ingredient for global mass-fraction balance.
- Background execution support:
  - trial progress updates,
  - cooperative cancellation at trial boundaries.

## UX workflow

1. Open `Optimizer` page.
2. Click `Load from Simulator (Solid)` to preload variable bounds from current solid formulation.
3. Assign optional `Group Label` values to ingredients in the variable table.
4. Define one rule per label in `Group Rules` (fixed-proportion or sum-to-total).
5. Set objective controls (`n`, Isp variant) and study controls (trials, timeout, seed).
6. Click `Start Optimization`.
7. Monitor trial progress and best objective plot.
8. Click `Apply Best to Simulator` to populate solid composition and return to Simulator.

## Data contract

### Variable bounds

- Table columns: `Ingredient`, `Min`, `Max`, `Group Label`.
- One row per ingredient participating in optimization.

### Group rules (label-based)

- Table columns: `Group Label`, `Type`, `Min Total`, `Max Total`.
- Labels are assigned at ingredient-row level and compiled to members automatically.
- `Type = fixed_proportion`:
  - Ratios are auto-derived from the starting formulation quantities of members in the label.
  - If a baseline value is unavailable, the UI falls back to midpoint of member min/max bounds.
- `Type = sum_to_total`:
  - Set `Min Total`, `Max Total`, or both as an inequality bound.
  - Setting both to the same value is equivalent to an exact total.
  - Example: label `solids` used by `AP` and `AL` with `Min Total = 0.82`, `Max Total = 0.86`.

### Popup helper

- `Group Rule Helper...` opens a small dialog to add one rule row from existing labels.
- Inline table editing remains the primary workflow; popup is an accelerator.

### Objective and operating controls

- Objective from optimizer page.
- Operating point read from `EngineDock` at run start:
  - chamber pressure,
  - expansion mode/value,
  - ambient pressure,
  - solver/database settings.

## Threading and cancellation

- `OptimizationWorker(QThread)` runs study off the UI thread.
- Progress signal emits trial status payload.
- Cancel button sets a stop flag; engine stops on next trial boundary.
- Runtime non-convergence during a trial is treated as `PRUNED` with diagnostics (not a hard study crash).

## Robustness defaults

- After loading from Simulator, closure ingredient defaults to the first ingredient row.
- Progress text handles "no complete trial yet" when early trials are all pruned.

## Known limitations in MVP

- Preload helper currently supports only solid/monopropellant simulator snapshot.
- Cross-membership between fixed-proportion and sum-to-total groups is intentionally rejected in v1.
- Each non-empty label must have exactly one corresponding rule row.
- No persistence/resume to Optuna SQL storage yet.
- External metric plugins are not wired into the GUI yet.

## Next increments

1. Add Optuna storage path + resume controls.
2. Add bipropellant preload and grouped fuel/oxidizer optimization templates.
3. Add external metric plugin controls (viscosity/pourability) with hard/soft constraints.
4. Add dedicated optimization plots in `Analysis` page (best-so-far, rejection breakdown, top-N candidates).

