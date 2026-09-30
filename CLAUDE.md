# CLAUDE.md

This file gives Claude Code guidance for working in this repository.

## What this is

`simpest` is a Python package that couples a crop growth trajectory with a daily
plant disease (SEIR) and fungicide model, and calibrates disease and crop
parameters against field observations. The disease, fungicide and damage logic
is a Python port of the C# **FraNchEstYN** model. Parity with the C# model is
an explicit goal (see `tools/dual_run/`).

The crop trajectory can come from three places:
- **SIMPLACE/Lintul5**, through a JVM bridge (`simpest/models/simplace.py`).
- **torchcrop**, a differentiable Lintul5 port. See
  `docs/examples/torchcrop-integration/`.
- **The internal crop model** in `fr_crop_model.py`, used when no external
  crop-model series is given.

## Commands

```bash
# Tests. CI runs unittest, not pytest, and pytest is not installed in .venv.
.venv/bin/python -m unittest discover tests
.venv/bin/python -m unittest tests.test_optimizer              # one module
.venv/bin/python -m unittest tests.test_optimizer.TestX.test_y # one test

# Docs site (MkDocs Material and mkdocstrings)
mkdocs serve

# Parity and regression tools
python tools/dual_run/run_dual.py --python-only   # Python side only, no .NET needed
python tools/dual_run/run_calibration.py          # C# vs Python calibration comparison
python tools/check_parameters.py <franchestynParametersForPackage.csv>  # JSON params vs canonical source
python tools/fetch_torchcrop_dataset.py --lat .. --lon .. --start-year .. --end-year .. --location-name ..
```

On this Linux box, importing `optuna` (and therefore the notebooks and
calibration) fails with a `CXXABI_1.3.15` sqlite3 error unless you prefix the
command with `LD_LIBRARY_PATH=/home/rahaman/miniforge3/lib`.

Linting uses flake8 with `max-line-length = 88` (set in `pyproject.toml`) and
black. Dependencies come from `requirements.txt`, which `pyproject.toml` reads
dynamically. Keep new runtime dependencies there.

## Architecture (`simpest/models/`)

The public API is exported from `simpest/models/__init__.py`: `FranchestynConfig`,
`run_franchestyn`, `run_morris_sensitivity` and `SimplaceConfig`.

**Entry layer: `franchestyn.py`**
- `FranchestynConfig` is a dataclass that holds every run and calibration
  setting. `__post_init__` loads the editable parameter spec dicts
  (`crop_parameters`, `disease_parameters`, `fungicide_parameters`) from the
  JSON files. Setting `disease_type=None` or `fungicide_type=None` turns that
  sub-model off completely.
  `weather_time_step` (`"daily"` by default, or `"hourly"`) and `latitude`
  are passed through to the runner. Daily weather gets its hourly RH and leaf
  wetness synthesised; without `rhx`/`rhn` columns, RH comes from Tmax/Tmin
  alone. Hourly weather (`date, hour, temp, rh, prec, rad`) uses measured
  values.
- `run_franchestyn(...)` takes in-memory DataFrames (weather, management,
  crop_model, reference). It writes them to temporary CSVs because the readers
  are file-based, builds a `FranchestynRunner`, optionally calibrates, and
  returns `{"outputs": {"simulation": [records], "summary": {rmse, best_params, ...}}}`.
- `build_runner(...)` returns a runner that you own, with its weather already
  cached. Pass it back as `run_franchestyn(..., runner=r)` to avoid re-parsing
  inputs in calibration loops. When you do, only the config's parameter specs
  are reapplied, through `runner.update_param_specs`.
- `deactivate_calibration(specs, names)` turns off the `calibration` flag on
  the named parameters, in place.
- It also provides `build_season_summary` and the `save_*_csv` output helpers.

**Simulation core: `fr_runner.py` (`FranchestynRunner`)**
- The runner loops over **hourly** weather. On each hour it runs the hourly
  disease step. At hour 23 it builds `InputsDaily` from the hourly accumulators
  and runs the crop, fungicide and disease daily steps in that order. Outputs
  are keyed by the hour-23 `datetime`.
- It keeps two state objects: `output` (previous day) and `output_t1` (today).
  Season-persistent fields are carried over by hand at the day swap. If you add
  a stateful field, carry it over there too.
- A season resets on the sowing DOY and ends at 100% cycle completion or after
  `_MAX_DAS` days.
- `compute_rmse` is **the** calibration objective. It uses scaled squared errors
  for AGB (/200), yield (/100), fInt (×100) and severity (%), with ×1000
  penalties. By default it aligns days as sim[d-1] vs ref[d]
  (`use_prev_day_alignment`) to match C#.
- Parameter loading has three precedence levels:
  1. In-memory spec dicts from the config.
  2. JSON file plus type.
  3. Legacy CSV (`param_file`).
- Parameter keys look like `"<class>_<Name>"`, for example `disease_PathogenSpread`.
  `_set_param` maps each CamelCase name onto a snake_case dataclass field in
  `fr_data.py`. It includes an acronym-safe fallback, because the regex turns
  `RUEreducerDamage` into the wrong field name. A parameter that matches no
  field is **silently ignored**. When you add a parameter, make sure a
  `ParCrop`, `ParDisease` or `ParFungicide` field with the matching name exists.

**Sub-models**
- `fr_crop_model.py`: daily crop growth and the four damage mechanisms: light
  stealers, RUE reducers, assimilate sappers and senescence accelerators. It
  has an internal branch (thermal time, logistic fInt, RUE) and an external
  branch (attainable series from the crop model, with damage applied to the
  daily increments).
- `fr_disease_model.py`: `DiseaseModel`, with `run_hourly` for infection and
  wetness, and `run_daily` for SEIR tissue compartments and severity.
- `fr_fungicide_model.py`: fungicide efficacy decay since the last treatment.
- `fr_data.py`: all dataclasses for inputs, outputs, parameters and the
  simulation unit.
- `fr_utilities.py`: shared response functions.

**Readers**
- `fr_weather_reader.py`: reads daily or hourly weather. It synthesises hourly
  values from daily ones (temperature, RH, leaf wetness, day length).
- `fr_reference_reader.py`: reads sowing and management, reference
  observations, and the external crop-model series. Column matching goes
  through alias lookup.
- `fr_param_reader.py`: reads the JSON spec files and the legacy CSV.

**Parameter data** is stored in `fr_{crop,disease,fungicide}_parameters.json`,
keyed by type (crops: `wheat`, `rice`; diseases: `septoria`, `brown_rust`,
`black_rust`, `yellow_rust`, `wheat_blast`). Each entry has
`min/max/value/calibration/unit/description`. These files should match the
canonical FraNchEstYN CSV; `tools/check_parameters.py` checks this.

**Calibration and sensitivity** use a pluggable design:
- `fr_optimizer_base.py`: `BaseFranchestynOptimizer` owns parameter selection
  (flagged, in scope, not disabled, not boolean), bounds, and `_evaluate_rmse`.
  A failed run returns 1e300. The `OPTIMIZERS` registry maps a name to
  `"module:Class"`, and `build_optimizer()` imports the class lazily.
- `fr_optimizer.py`: `NelderMeadOptimizer`, a multi-start scipy Nelder–Mead
  that runs restarts in parallel through joblib when `n_jobs` is above 1.
- `fr_optimizer_optuna.py`: `OptunaOptimizer`, using TPE. When `n_jobs` is
  above 1, it runs batched ask/tell against a persistent process pool, where
  each worker receives the optimizer once. Results are reproducible for a given
  `seed` and `n_jobs`.
- `fr_sensitivity.py`: `MorrisSensitivity` and `run_morris_sensitivity`
  (SALib). They reuse the base selection and objective, then return a ranked
  `mu_star`/`sigma` table.
- To add an optimizer, subclass the base, implement `calibrate()` and
  `from_config()`, and register it in `OPTIMIZERS`. Then add a docs page (see
  below).

**Other modules**
- `simplace.py`: SIMPLACE init and run, plus converters from SIMPLACE output
  to the FraNchEstYN input CSVs.
- `simpest/simpest.py` and `common.py` are legacy placeholders.
- `pyproject.toml` declares a `simpest.cli:main` console script, but
  `simpest/cli.py` does not exist.

## Conventions and gotchas

- **Parity with C# matters.** Several defaults exist only to match C#:
  `use_gdd=False`, `use_prev_day_alignment=True` and
  `all_row_includes_end_year=False`. Don't change them without a reason. Check
  numeric changes with `tools/dual_run/run_dual.py`; `tools/dual_run/README.md`
  lists the baseline tolerances. The C# binary (`franchestyn_ref/`) is not in
  this repo, so only `--python-only` works without it.
- Tests use stubs such as `_FakeRunner` and `FranchestynRunner.__new__` to skip
  file I/O. Follow that pattern rather than adding data fixtures.
- Docstrings are Google-style and are rendered by mkdocstrings. Every module
  has a page in `docs/models/<module>.md` and an entry in the `mkdocs.yml` nav.
  When you add, rename or remove a module, update both.
- The example notebooks in `docs/examples/` are rendered into the docs through
  mkdocs-jupyter. The torchcrop notebook is the main end-to-end example of
  torchcrop crop growth plus septoria, with calibration and Morris screening.
