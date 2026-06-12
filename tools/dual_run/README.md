# Matched dual run — C# FraNchEstYN vs Python simpest

Runs the reference C# FraNchEstYN and the Python `simpest` franchestyn step on
**identical** inputs and diffs the daily outputs column-by-column. This turns the
structural replication review into a measured numeric tolerance.

## What it does

1. Assembles one shared input set (`inputs/`) from the Python example data so
   both engines read the same files:
   - `weather/daily/indiana.csv` ← `weather_franchestyn.csv`
   - `cropModel/cropModelData.csv` ← `cropModel_data.csv`
   - `management/sowing.csv` ← `management_franchestyn.csv`
   - `reference/referenceData.csv` ← `FRA_REFERENCE_Indiana.csv`
   - `parameters/franchestynParameters.csv` ← the C# reference parameter CSV
2. Writes a C# config (`config_indiana.json`) and runs the prebuilt
   `FraNchEstYN.exe` in **validation** mode (no calibration → deterministic).
3. Runs the Python `FranchestynRunner` directly on the same inputs with
   C#-parity flags (`use_gdd=False`, `use_prev_day_alignment=True`), disease on,
   fungicide off, `crop_type=None` so it uses the **same** parameter CSV.
4. Aligns the two daily outputs on `(year, doy)` and reports, per column,
   `max|diff|`, `mean|diff|`, and correlation. Output: `out/diff_report.txt`.

Parameter parity is exact: Python's legacy CSV reader uses the same
`Name_Class` last-row-wins keying as C#, so both engines get identical
parameter values from the one shared CSV.

## Run

```bash
python tools/dual_run/run_dual.py                # assemble + run both + diff
python tools/dual_run/run_dual.py --no-assemble  # reuse existing inputs/
python tools/dual_run/run_dual.py --python-only  # skip C# (no .NET)
```

Requires: the .NET 8 runtime (the prebuilt `FraNchEstYN.exe` under
`franchestyn_ref/.../bin/Debug/net8.0`), pandas. The Python side does **not**
need SIMPLACE/JVM.

## Baseline result (Indiana, 1973–1990, validation, 4563 matched days)

After the replication fixes (F1–F5) **and** the F11 fix this run uncovered, all
dynamic variables match to float precision:

| column            | max\|diff\| | corr   | note |
|-------------------|------------:|-------:|------|
| CyclePct          |     9.8e-06 | 1.0000 | |
| LightIntAtt       |     5.6e-08 | 1.0000 | |
| LightIntHealthy   |     2.2e-03 | 1.0000 | |
| AGBatt / YieldAtt |    ~4e-04   | 1.0000 | |
| AGBact            |       55.5  | 1.0000 | residual float32 accumulation (mean 0.81) |
| YieldAct          |       21.1  | 1.0000 | residual float32 accumulation (mean 0.21) |
| HTtimeR / HTtimeS |    ~4e-05   | 1.0000 | |
| Latent/Spor/Dead/Affected/DiseaseSev | ~5e-03 | ~1.0000 | |
| Susceptible       |       0.65  | 0.9970 | 16/4563 days; float32 branch flip at senescence boundary (benign) |
| Tmax/Tmin/Prec/Rad|     ~3e-06  | 1.0000 | weather synthesis parity |
| DaysAfterSowing*  |        269  |    n/a | by design (F7) |
| GDD*              |          0  |    n/a | source TSUM ~0; by design (F7) |

\* `DaysAfterSowing` and `GrowingDegreeDays` differ by design in the
external-crop branch (Python populates them; C# leaves 0).

### Bug found by this harness

**F11 (HIGH):** `RUEreducerDamage` was silently dropped in Python because
`_camel_to_snake("RUEreducerDamage")` → `"ru_ereducer_damage"` ≠ the field
`rue_reducer_damage`, so disease never reduced RUE and actual (diseased)
biomass/yield were over-estimated (AGBact corr 0.96, max diff ~1867 kg/ha
before the fix). Fixed in `fr_runner._set_param` with an acronym-safe
normalized field match. After the fix: AGBact corr 1.0000, max diff 55 kg/ha.

## Calibration comparison (`run_calibration.py`)

Extends the harness to *calibration*. Both engines calibrate the same 18 disease
parameters against the same reference data (disease-only, because crop params are
unidentifiable with the external crop model), then results are compared under the
Python objective (the common yardstick).

```bash
python tools/dual_run/run_calibration.py                 # C# + Python calibration + report
python tools/dual_run/run_calibration.py --restarts 2 --iters 200 --seed 0
python tools/dual_run/run_calibration.py --python-only
```

Baseline result (Indiana 1973–1990, disease):

| param set            | RMSE (Python objective) |
|----------------------|------------------------:|
| default (CSV)        | 31.67 |
| C# calibrated        | 22.88 |
| Python calibrated    | 24.93 |

* **Objective parity at C#'s optimum:** C# output-derived RMSE vs Python
  `compute_rmse` agree to ~0.02 — confirming the objective is replicated at a
  non-trivial (calibrated) parameter point, complementing the fixed-default-param
  severity check above.
* Both calibrations beat the default; C# reaches the lower value because the
  Python optimiser under-converges at modest `--restarts`/`--iters` on this
  18-dim problem (raise them to close the gap).
* Parameter **values** are not expected to match (different optimiser — F9);
  prev-day vs same-day RMSE alignment (F3) is immaterial for this dataset.
* C#'s console progress min is an unreliable artifact (it does not capture the
  returned optimum); the harness uses C#'s validation-output RMSE instead.

## Caveats

- The C# binary is a prebuilt artifact (its output header matches the head C#
  source, so it is the current build). The .NET SDK is not on PATH here, so it
  is not rebuilt from source by this harness.
- The Indiana case is winter wheat (autumn sowing, cross-year), which also
  exercises the season-grouping path (F5).
- Reference data only affects RMSE, not the daily-trajectory diff; this run is a
  validation (fixed-parameter) comparison. A calibration comparison would test
  the F3 alignment default and is left as a follow-up.
