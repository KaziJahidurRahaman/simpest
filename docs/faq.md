# FAQ

### Do I need SIMPLACE installed to use simpest?

Only for the crop growth stage (`simpest.models.simplace`). The
disease/fungicide simulation stage (`simpest.models.franchestyn`) takes plain
pandas DataFrames as input, so you can run it on its own — using crop-model
output produced elsewhere — without a SIMPLACE installation. See
[Installation](installation.md).

### Why is my disease severity always zero?

Check that `disease_type` is set on `FranchestynConfig` (or the runner's
`disease_type` argument). Passing `None` intentionally disables the
epidemiological model and gives a crop-only run. Also confirm the crop-model
series covers the sowing-to-maturity window: the disease step only runs on
days after sowing.

### My calibration and simulation results don't match a reference run

Two settings commonly explain a mismatch:

- **`use_gdd`.** If the reference crop-model series ties its own cycle
  completion to accumulated thermal time rather than a fixed calendar
  length, set `use_gdd=True` on `FranchestynConfig` so simpest's cycle
  completion is computed the same way. Leaving it at the calendar-day
  default in that case makes the two cycle percentages drift apart over the
  season.
- **`use_prev_day_alignment`.** This controls whether the calibration
  objective compares each simulated day to the reference on the same date or
  the next date (`sim[d-1]` vs `ref[d]`, the default). Confirm it matches the
  convention used by whatever reference implementation you're comparing
  against.

### Calibration is slow — how do I speed it up?

Reduce `n_restarts` and/or `max_iter` on `FranchestynConfig`, narrow
`calibration_variable` to `"crop"` or `"disease"` instead of `"all"`, or
exclude parameters you don't want searched with `crop_disabled_params` /
`disease_disabled_params` (or `deactivate_calibration`). Fewer free
parameters and fewer restarts both reduce the number of model evaluations.

### Why did a fungicide treatment have no effect?

`fungicide_type` must be set (not `None`) for the fungicide step to run at
all, and the management data must include a scheduled treatment date. A
missing treatment date is a silent no-op rather than an error, so check the
management CSV's treatment columns if efficacy stays at zero.

### Where do the crop, disease, and fungicide parameter defaults come from?

From the JSON files bundled with the package
(`simpest/models/fr_crop_parameters.json`, `fr_disease_parameters.json`,
`fr_fungicide_parameters.json`), keyed by crop/disease/fungicide type. Each
parameter carries a default value, calibration bounds, and a `calibration`
flag; see [Usage](usage.md#calibration) for how to override or exclude them.
