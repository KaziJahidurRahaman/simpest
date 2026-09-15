# Usage

```python
import simpest
```

simpest couples two stages: a SIMPLACE/Lintul5 crop growth run
([`simpest.models.simplace`](models/simplace.md)) and a disease/fungicide
simulation ([`simpest.models.franchestyn`](models/franchestyn.md)). You can
run the full pipeline, or run the disease/fungicide stage on its own against
crop-model output produced elsewhere.

## Stage 1 — Crop growth (SIMPLACE)

```python
from pathlib import Path
import simpest.models.simplace as simplace

sp_cfg = simplace.SimplaceConfig(
    install_dir="<SIMPLACE_INSTALL>",
    work_dir="<SIMPLACE_WORK>",
    output_dir="<OUTPUT_DIR>",
    solution_path="<SOLUTION_PATH>",
    project_path="<PROJECT_PATH>",
)

shell = simplace.init_simplace(sp_cfg)
simplace.run_simplace(shell, sp_cfg, project_lines=[1])

dirs = simplace.get_simplace_directories(shell)
work_root = Path(dirs["_WORKDIR_"])
output_root = Path(dirs["_OUTPUTDIR_"])
project_row = simplace.get_project_row(work_root, selected_line=1)
```

`project_row` carries the selected project's metadata (location, start/end
dates, sowing option) plus a `yearly_sowing_doy` mapping extracted from the
SIMPLACE project data.

## Converting SIMPLACE output for the disease/fungicide stage

```python
import pandas as pd

crop_model_path = simplace.export_crop_model_data(output_root, project_row)
weather_path = simplace.convert_weather(work_root, output_root, project_row["location"])
management_path = simplace.build_management(output_root, project_row)

crop_model_df = pd.read_csv(crop_model_path)
weather_df = pd.read_csv(weather_path)
management_df = pd.read_csv(management_path)
ref_df = pd.read_csv("<REFERENCE_CSV>")
```

- `export_crop_model_data` writes the daily light interception, biomass,
  yield, and thermal-time series the crop damage mechanisms need.
- `convert_weather` reshapes the SIMPLACE weather file (units, humidity
  derived from vapour pressure) into the expected columns.
- `build_management` writes one row per simulated year with the sowing
  day-of-year (and any fungicide treatment columns you add).

## Stage 2 — Disease, pest, and fungicide simulation

```python
from simpest.models.franchestyn import FranchestynConfig, run_franchestyn

fr_cfg = FranchestynConfig(
    crop_type="wheat",
    disease_type="septoria",
    fungicide_type=None,       # or e.g. "protectant" if treatments are scheduled
    site="indiana",
    variety="Generic",
    disease="thisDisease",     # disease column name in the reference CSV
    is_calibration=True,
    calibration_variable="all",  # "crop", "disease", or "all"
)

start_year = int(project_row["startdate"].split(".")[-1])
end_year = int(project_row["enddate"].split(".")[-1])

result = run_franchestyn(
    start_year=start_year,
    end_year=end_year,
    config=fr_cfg,
    weather_df=weather_df,
    management_df=management_df,
    crop_model_df=crop_model_df,
    ref_df=ref_df,
)

summary = result["outputs"]["summary"]      # rmse, calibration flags, best_params
simulation = result["outputs"]["simulation"]  # list of daily record dicts
```

`weather_df`, `management_df`, `crop_model_df`, and `ref_df` are plain
DataFrames — `run_franchestyn` writes them to temporary CSVs internally, so
any source that produces DataFrames with the expected columns works, not
just the SIMPLACE conversion helpers above.

### Daily output fields

Each record in `result["outputs"]["simulation"]` has one row per simulated
day, including `Date`, `GrowingSeason`, `DaysAfterSowing`,
`GrowingDegreeDays`, `CycleCompletionPercentage`, attainable/actual light
interception (`LightInterception`, `LightIntHealthy`), attainable/actual
biomass and yield (`AGBattainable`, `AGBactual`, `YieldAttainable`,
`YieldActual`), the SEIR compartments (`Susceptible`, `Latent`,
`Sporulating`, `Affected`, `Dead`), `DiseaseSeverity`, `FungicideEfficacy`,
and the day's weather (`Tmax`, `Tmin`, `RHx`, `RHn`, `TotalPrec`, `TotalRad`,
`TotalLW`).

### Season summary and saved outputs

```python
simulation_df = pd.DataFrame(simulation)
season_summary = build_season_summary(simulation_df, site=fr_cfg.site, variety=fr_cfg.variety)
```

`build_season_summary` groups the daily output by growing season and reports
AUDPC (trapezoidal integral of disease severity over time), peak disease
severity, peak attainable/actual yield and biomass, absolute and percentage
yield loss, and season-level weather aggregates.

Convenience writers save these to CSV under
`<output_root>/SimulationExperimentTemplate/`:

```python
from simpest.models.franchestyn import (
    save_simulation_results_csv,
    save_season_summary_csv,
    save_calibrated_parameters_csv,
)

save_simulation_results_csv(simulation, output_root)
save_season_summary_csv(season_summary, output_root)
save_calibrated_parameters_csv(
    summary.get("best_params", {}), output_root,
    site=fr_cfg.site, variety=fr_cfg.variety, config=fr_cfg,
)
```

## Calibration

Setting `is_calibration=True` (the default) runs a multi-start Nelder–Mead
search over every parameter flagged for calibration in the crop/disease
parameter JSON files, scoped by `calibration_variable`. `n_restarts` and
`max_iter` control the search; `crop_disabled_params` /
`disease_disabled_params` (or `deactivate_calibration`) exclude specific
parameters:

```python
from simpest.models.franchestyn import deactivate_calibration

fr_cfg = FranchestynConfig(is_calibration=True, n_restarts=5, max_iter=200)
fr_cfg.crop_parameters = deactivate_calibration(
    fr_cfg.crop_parameters, {"TbaseCrop", "TmaxCrop"}
)
```

For a fixed-parameter run (no search), set `is_calibration=False`; the model
runs once with the JSON defaults (or with `crop_parameters` /
`disease_parameters` values you've edited in place).

### Choosing the optimizer

`optimizer` selects the search strategy:

- `"nelder-mead"` (default) — the built-in multi-start Nelder–Mead simplex
  search described above, controlled by `n_restarts`, `max_iter`, and `ftol`.
- `"scipy-nelder-mead"` — the same algorithm run through
  [`scipy.optimize.minimize`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html),
  also multi-start and sharing the `n_restarts` / `max_iter` / `ftol` knobs.
  Useful as a maintained reference to cross-check the built-in optimizer.
- `"optuna"` — an [Optuna](https://optuna.org/) TPE (Bayesian) search,
  controlled by `n_trials` (number of parameter sets to evaluate) and an
  optional `optuna_timeout` (seconds). It keeps every candidate inside the
  parameter bounds and tends to handle the noisy objective surface with fewer
  evaluations than restart-heavy Nelder–Mead.

`seed` makes any of the searches reproducible.

```python
fr_cfg = FranchestynConfig(
    is_calibration=True,
    optimizer="optuna",
    n_trials=200,
    seed=0,
)
```

### Screening parameters before calibration

When many parameters are flagged for calibration it is worth checking which
ones actually move the objective. `run_morris_sensitivity` runs a Morris
elementary-effects screen over the same parameters calibration would use
(flagged for `calibration`, within `calibration_variable` scope, minus the
disabled ones):

```python
from simpest.models import run_morris_sensitivity
from simpest.models.franchestyn import deactivate_calibration

sa = run_morris_sensitivity(
    start_year, end_year, config=fr_cfg,
    weather_df=weather_df, management_df=management_df,
    crop_model_df=crop_model_df, ref_df=ref_df,
    n_trajectories=20, seed=0,
)
table = sa["table"]   # ranked by mu_star
```

The table has one row per parameter with `mu_star` (mean absolute effect on the
run's RMSE — higher is more influential), `sigma` (spread — high relative to
`mu_star` means non-linear or interacting), and `mu_star_conf` (bootstrap CI on
`mu_star`). The screen costs about `n_trajectories * (k + 1)` model runs for `k`
parameters.

Then freeze the parameters whose effect is indistinguishable from zero and
calibrate the rest:

```python
inert = table.loc[table["mu_star"] <= 2 * table["mu_star_conf"], "parameter"]
deactivate_calibration(fr_cfg.disease_parameters, set(inert))
# ... now run calibration as usual over the remaining parameters
```

## Crop-only or disease-only runs

- `disease_type=None` skips the epidemiological model entirely (disease
  severity stays 0), giving a crop-only run.
- `fungicide_type=None` skips the fungicide step; only set it when
  treatments are actually scheduled in the management data, otherwise they
  are ignored.

## Notes on cycle progress and day alignment

- `use_gdd` controls how crop cycle completion is derived from the external
  crop-model series: `False` (default) interpolates linearly over calendar
  days within a cycle; `True` scales by accumulated thermal time (GDD)
  instead. Use `True` when the reference crop-model run itself completes its
  cycle based on thermal time rather than a fixed calendar length, otherwise
  the two can drift out of sync.
- `use_prev_day_alignment` (default `True`) compares each simulated day to
  the *next* day's reference observation during calibration/RMSE scoring
  (`sim[d-1]` vs `ref[d]`); set `False` for same-day alignment.

See the [API Reference](simpest.md) for the full parameter and return-value
documentation, and
[`examples/1_Run Simpest.ipynb`](examples/1_Run%20Simpest.ipynb) /
[`examples/2_Plot.ipynb`](examples/2_Plot.ipynb) for a complete, runnable
walkthrough.
