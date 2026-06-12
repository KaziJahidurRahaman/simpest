# Replication Review Prompt for simpest

Use this document as the review brief for an AI that will assess whether the Python package `simpest` faithfully replicates the behavior of the original FraNchEstYN C# implementation.

## Goal

Check whether the current Python pipeline reproduces the original FraNchEstYN model logic, inputs, outputs, and summary calculations closely enough that results are materially comparable.

The main question is not whether the code runs, but whether it matches the reference implementation in structure, assumptions, transformations, and numerical behavior.

## Project Context

`simpest` combines two modeling steps:

1. A crop growth model is run first.
2. The crop outputs are then reformatted and passed into FraNchEstYN-style pest and disease simulation logic.

In this repository, the native FraNchEstYN crop-model step has been replaced with SIMPLACE / Lintul5 outputs, and those outputs are transformed into the format expected by the FraNchEstYN runner.

The reference implementation to compare against is the original FraNchEstYN repository at https://github.com/GeoModelLab/FraNchEstYN.

Important scope note:

- The FraNchEstYN upstream repository contains the original C# backbone and R-based usage/implementation patterns.
- The FraNchEstYN upstream repository does not include a SIMPLACE implementation.
- The SIMPLACE integration layer being reviewed is specific to this Python project.

The Python implementation under review is this repository: https://github.com/KaziJahidurRahaman/simpest.

## What To Compare

Review the Python implementation against the C# implementation for the following areas:

### 1. Pipeline structure

Verify that the execution order matches the reference model as closely as possible:

- define inputs
- run crop simulation
- convert crop outputs to FraNchEstYN-compatible records
- run disease and pest simulation
- compute seasonal and daily outputs
- optionally run calibration

### 2. Input contract

Check that the Python package accepts and interprets the same conceptual inputs as the C# model, including:

- weather data
- management / sowing data
- crop type
- disease type
- fungicide type
- site
- variety
- reference data
- calibration flags and bounds

### 3. Crop model replacement logic

Verify that the SIMPLACE / Lintul5 substitution preserves the semantics required by FraNchEstYN:

- crop output fields are mapped correctly
- units are converted correctly
- date handling is consistent
- zero-value or missing-value rows are filtered the same way or with documented intent
- the resulting crop-model CSV is structurally compatible with the FraNchEstYN runner

### 4. Weather conversion logic

Check that the weather conversion from SIMPLACE format to FraNchEstYN format matches the reference assumptions:

- date parsing
- precipitation units
- radiation units
- vapour pressure to relative humidity conversion
- clipping or normalization rules
- site naming and station mapping

### 5. Management and sowing date logic

Verify that yearly sowing-day handling is equivalent or intentionally compatible with the reference behavior.

Pay special attention to:

- per-year sowing day extraction
- `ISOW` precedence when present
- fallback behavior when yearly sowing-day values are missing
- the relationship between `IDEM`, sowing day, and project rows

### 6. Disease and pest dynamics

Check that the disease / pest state evolution, severity calculations, and any fungicide effects align with the original model logic:

- state transitions
- severity accumulation
- latent / sporulating / affected / dead compartments
- fungicide efficacy application
- daily update order

### 7. Calibration behavior

If calibration is enabled, compare the Python optimizer behavior against the original implementation:

- calibration objective definition
- RMSE computation
- parameter exclusion rules
- restart strategy
- stopping conditions
- caching / rounding behavior

### 8. Output parity

Compare the Python outputs against the C# outputs at both daily and seasonal levels:

- daily record schema
- field names
- seasonal summary fields
- aggregation rules
- yield loss calculations
- AUDPC calculation
- rounding / floating-point tolerance
### 9. Input output units parity, the standard should be the units of variables in franchestyn
### 10. Used equations for different agricultural processes should be same. In the documentation make sure that the equations are explained.



## Existing Python Implementation Notes

The current Python package already exposes these public entry points:

- `simpest.models.simplace.SimplaceConfig`
- `simpest.models.simplace.init_simplace`
- `simpest.models.simplace.run_simplace`
- `simpest.models.simplace.get_project_row`
- `simpest.models.simplace.extract_yearly_sowing_doy`
- `simpest.models.simplace.export_crop_model_data`
- `simpest.models.simplace.convert_weather`
- `simpest.models.franchestyn.FranchestynConfig`
- `simpest.models.franchestyn.run_franchestyn`
- `simpest.models.franchestyn.build_season_summary`

The review should also note that the Python implementation currently uses local JSON parameter files for crop, disease, and fungicide configuration.

## Evidence To Inspect

The AI reviewer should inspect:

- FraNchEstYN reference repository: https://github.com/GeoModelLab/FraNchEstYN
- simpest Python repository: https://github.com/KaziJahidurRahaman/simpest
- the Python source under `simpest/models`
- any shipped example data under `docs/examples/data`
- the original C# FraNchEstYN implementation and executable artifacts available in the FraNchEstYN repository
- the R-based scripts/workflows in FraNchEstYN that expose intended model behavior and expected outputs
- the original parameter files and reference CSV files used by FraNchEstYN
- example outputs from both implementations for the same inputs

## Required Review Method

For each model component, determine:

1. Whether the Python behavior is equivalent to the reference behavior.
2. Whether any differences are intentional and documented.
3. Whether any differences are likely to change output materially.
4. Whether the difference is structural, numerical, or only a naming / formatting difference.

If the exact C# behavior is unknown, the reviewer should mark the result as uncertain rather than guessing.

## Acceptance Criteria

Treat the Python implementation as a faithful replication only if all of the following hold:

- the execution flow matches the FraNchEstYN pipeline at a meaningful level
- input and output schemas are compatible
- unit conversions are correct
- management and sowing-date handling is consistent
- daily state updates produce similar trajectories
- seasonal summaries differ only within an acceptable floating-point tolerance
- any deviations are explicitly documented and justified

## What The Reviewer Should Return

Ask the reviewer to return a concise report with this structure:

### Overall verdict

One of:

- faithful replication
- partially equivalent
- not equivalent
- cannot determine from available evidence

### Findings

List each discrepancy with:

- file or function name
- what differs from the C# reference
- why it matters
- severity: low, medium, high
- recommended fix or follow-up check

### Output comparison

Summarize whether daily and seasonal outputs are close enough to be considered equivalent, and state the observed tolerance or mismatch.

### Missing evidence

List any reference files, test cases, or C# outputs that are still needed before the review can be completed confidently.

## Suggested Verification Cases

Use a small set of matched inputs to compare the two implementations:

- one single-site, single-season run
- one multi-year run with varying sowing dates
- one calibration run
- one run with fungicide enabled
- one run with missing or partially specified management data

For each case, compare:

- daily outputs row by row
- season summary outputs
- calibrated parameter results
- any warnings or fallback behavior

## Notes For The Reviewer

- Do not assume that matching field names means matching behavior.
- Do not assume that a working run means the model is equivalent.
- Pay special attention to date conventions, unit conversions, and fallback logic.
- Flag any places where the Python implementation introduces new behavior that is not present in the reference model.
