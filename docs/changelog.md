# Changelog

## v0.0.4

**Crop and disease/pest simulation**

- SIMPLACE/Lintul5 crop growth integration (`simpest.models.simplace`):
  configuration, run orchestration, project-row and yearly sowing-day
  extraction, and conversion of SIMPLACE weather/crop-model/management
  output into the disease/fungicide simulation's input format.
- Coupled crop growth, SEIR disease, and fungicide simulation
  (`simpest.models.franchestyn`), including four disease damage mechanisms
  (light stealers, RUE reducers, assimilate sappers, senescence
  accelerators) and optional fungicide efficacy.
- Support for running the crop model internally (thermal-time based) or from
  an external daily crop-model series, with calendar-day or thermal-time
  (GDD) based cycle-completion.
- Multi-start Nelder–Mead calibration (`simpest.models.fr_optimizer`) against
  reference field observations, with per-parameter calibration flags and
  exclusion lists.
- The disease/fungicide simulation runner (`run_franchestyn`) takes in-memory
  pandas DataFrames for weather, management, crop-model, and reference
  inputs.
- Daily simulation records and per-season summaries (AUDPC, yield loss, peak
  severity, weather aggregates), with CSV export helpers.

See the [Usage guide](usage.md) and the
[API Reference](simpest.md) for details, and the
[GitHub releases](https://github.com/KaziJahidurRahaman/simpest/releases)
for the exact set of changes in each published version.
