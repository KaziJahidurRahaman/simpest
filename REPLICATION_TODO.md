# simpest ↔ FraNchEstYN Replication To-Do

Action items from the replication review of `simpest` (Python) against the
FraNchEstYN C# reference (https://github.com/GeoModelLab/FraNchEstYN).

**Verdict:** Partially equivalent. Core physics (SEIR disease, fungicide, crop
damage, external-crop branch, hourly→daily loop, RMSE objective, AUDPC) are
faithful ports. The items below are the divergences to resolve or document.

Severity legend: 🔴 high · 🟠 medium · 🟡 low · ❓ uncertain

---

## 🔴 High priority

- [x] **F11 — `RUEreducerDamage` silently dropped (found by the matched dual run).** ✅ Done.
  `simpest/models/fr_runner.py` `_set_param` / `_camel_to_snake`.
  `_camel_to_snake("RUEreducerDamage")` → `"ru_ereducer_damage"`, which does not
  match the dataclass field `rue_reducer_damage`, so the parameter was never set
  (stayed `0.0`) and **disease never reduced RUE** → actual (diseased)
  biomass/yield over-estimated. Affected BOTH the legacy-CSV and modular-JSON
  parameter paths (production bug). Fixed with an acronym-safe normalized field
  match (strip underscores + lowercase). Verified by the dual run: AGBactual
  correlation 0.96 → 1.0000, max diff ~1867 → 55 kg/ha; YieldActual 0.95 →
  1.0000.

- [x] **F1 — Fix dew-point sign error.** ✅ Done.
  `simpest/models/fr_weather_reader.py:381` `_dew_point()`.
  Python computed `0.38*tmax - 0.018*tmax² - 1.4*tmin + 5`; C# is
  `0.38*tmax - 0.018*tmax² + 1.4*tmin - 5`. The `1.4*tmin` and `5` signs were
  flipped (misplaced parenthesis).
  - Applied fix: `return 0.38*tmax - 0.018*(tmax**2) + 1.4*tmin - 5.0`
  - Impact: corrected RH→leaf-wetness→infection when `RHx/RHn` are absent
    (daily fallback + hourly fallback). Was latent for the SIMPLACE path because
    `convert_weather()` always emits `rhx/rhn`, but a real bug for RH-less input.

- [x] **F2 — GDD-based vs calendar cycle percentage.** ✅ Done (C#-parity default).
  `simpest/models/fr_reference_reader.py` `read_crop_model_data`,
  `FranchestynConfig.use_gdd`, `FranchestynRunner.use_gdd`.
  Default flipped to `use_gdd=False` → calendar interpolation
  `(d-start)/total*100`, matching C#. GDD/TSUM-based cycle % remains available
  via `use_gdd=True` (more physical, opt-in). Docstrings updated.

## 🟠 Medium priority

- [x] **F3 — RMSE day alignment (off-by-one).** ✅ Done (C#-parity default).
  Resolved via new `use_prev_day_alignment` flag (`FranchestynConfig`,
  `FranchestynRunner`, `run_franchestyn`). Default `True` replicates the C#
  `ObjfuncVal` alignment (`ref_key = sim_day + 1` → `sim[d-1] vs ref[d]`); set
  `False` for same-day alignment (`sim[d] vs ref[d]`). Name parallels `use_gdd`.
  Original analysis below:
  `simpest/models/fr_runner.py:301` `compute_rmse` vs C# `optimizer.cs:ObjfuncVal`.
  C# objective evaluates at `hour==0` against the *previous* day's `outputT1`
  → `sim(d-1)↔ref(d)`. Python evaluates at `hour==23` → `sim(d)↔ref(d)`.
  (Note: the C# *output writer* aligns `sim(d)↔ref(d)`, so Python matches the
  writer but not the C# optimizer — C# is internally inconsistent.)
  - [x] Document the chosen alignment (done — both modes documented in docstrings).
  - [x] (Follow-up) Confirmed via `tools/dual_run/run_calibration.py`: objective
        parity holds at C#'s calibrated optimum (C# output-derived RMSE vs Python
        `compute_rmse` agree to ~0.02), and prev-day vs same-day alignment is
        immaterial for this dataset (22.88 vs 23.01). Param *values* differ by
        optimiser (F9), as expected.

- [x] **F4 — Don't override measured radiation in daily→hourly synthesis.** ✅ Done.
  `simpest/models/fr_weather_reader.py` `_estimate_hourly` now uses uniform
  `rad_daily / 24` (C# `estimateHourly` parity), removing the solar-geometry
  override that made the hourly sum equal the Hargreaves re-estimate instead of
  the measured daily total. Verified: hourly rad re-sums to the measured 9.38
  MJ/m²/d on the Indiana example. Solar-geometry distribution is retained where
  C# also uses it (`read_hourly` backfill of missing hourly radiation). Diurnal
  shape is irrelevant downstream (runner re-sums hourly rad to the daily total).
  Original analysis below:
  `simpest/models/fr_weather_reader.py:468` `_estimate_hourly`.
  When latitude is present, Python replaces measured daily `rad` with a
  Hargreaves clear-sky re-estimate; C# uses uniform `rad/24` (preserves the
  measured total). Immaterial for simpest (radiation unused by external-crop +
  disease) but wrong for the internal crop model and changes reported `TotalRad`.
  - Fix: distribute via solar geometry only when `rad` is missing; otherwise
    split the measured daily total.

- [x] **F5 — Group season summary by growing season, not calendar year.** ✅ Done.
  `simpest/models/franchestyn.py` `build_season_summary` + `_outputs_to_records`.
  Records previously omitted `GrowingSeason`, so grouping fell back to
  `Date.dt.year`, which splits seasons crossing 1 Jan (winter crops). R/C# group
  by `crop.growingSeason` (= sowing year).
  - [x] Emit `out.crop.growing_season` in `_outputs_to_records`.
  - [x] Group on it in `build_season_summary` (coerce unset `0`/non-numeric
        labels back to the calendar year so valid seasons are not split).

- [x] **F9 - Optimizer reimplementation.** Done (documented).
  `simpest/models/fr_optimizer.py`. Rewrote the module docstring into a
  "Replication note": it is a reimplementation (pure-Python multi-start
  Nelder-Mead), not a port of UNIMI `MultiStartSimplex`. Control knobs
  (`n_restarts`/`max_iter`/`ftol`) and the objective (incl. the `1e300`
  out-of-bounds penalty) match C#; the random search trajectory and exact
  calibrated params will not match bit-for-bit. Added an optional `seed` for
  reproducible restarts (aids the F3 calibration-output check). The C#
  `if Coefficient[j]==0 break;` bounds quirk is documented, not reproduced
  (measure-zero for continuous floats). Fixed a stale `scipy.optimize`
  reference in the `_objective` docstring. For deterministic logic
  comparison use the fixed-parameter dual run, not calibration.

## 🟡 Low priority

- [x] **F6 — `"All"` sowing-row year range.** ✅ Done (C#-parity default).
  New `all_row_includes_end_year` flag (`read_sowing`, `FranchestynRunner`,
  `FranchestynConfig`, `run_franchestyn`). Default `False` reproduces C#
  (`for y < endYear` → `"All"` row covers `[start, end-1]`, omitting the final
  year). Set `True` for the inclusive (more correct) behaviour. Verified both
  modes on a synthetic `"All"` CSV (2010–2015: False→2010–2014, True→2010–2015).
  Note: the C# default is almost certainly an off-by-one bug — it silently drops
  the final season for an `"All"` row — so prefer `True` unless you specifically
  need byte-for-byte C# replication. Latent for simpest (`build_management`
  writes explicit per-year rows, which are always honoured).

- [x] **F7 — Document FIX 1 / FIX 2 deviations.** ✅ Done.
  Rewrote the `fr_crop_model.py` module docstring (so it publishes via
  mkdocstrings) into a "Replication note" framing both as intentional
  external-branch deviations, not "bug fixes", and updated the inline comments.
  Key clarification surfaced by the dual run: DAS is **not** cosmetic — the
  runner's maturity/safety stop and `compute_rmse`'s `is_planted` reconstruction
  derive from `day_after_sowing > 0`, which reproduces C#'s loop-variable
  `isPlanted` fInt penalty in the objective; leaving DAS at 0 would silently
  disable it. GDD is reporting-only when `use_gdd=False` (default) and drives
  cycle % when `use_gdd=True`. Output expectation: `DaysAfterSowing` /
  `GrowingDegreeDays` differ from C#'s raw 0 in the external branch; all dynamic
  variables match (see `tools/dual_run/`).

- [x] **F8 — Document optional disease/fungicide modules.** ✅ Done.
  Documented in the `FranchestynRunner` and `FranchestynConfig` docstrings and at
  the gating site in `fr_runner.run()`. `disease_type`/`fungicide_type` double as
  on/off switches: `None` skips that model (crop-only / no-fungicide); set them
  to match C# (which always runs both). When both are set the behavior matches C#
  (as in the dual run). Caveat noted: skipping fungicide is equivalent to C# only
  when there are no scheduled treatments — set `fungicide_type` whenever
  treatments exist, or they are ignored.

- [x] **F10 — Calibration bounds/defaults vs reference.** ✅ Done — no
  reconciliation needed; the JSONs already match the canonical source.
  Resolution: the simpest JSONs match `franchestynParametersForPackage.csv`
  (the labeled `Model,Set,Parameter,min,max,value,calibration` source that builds
  the R datasets) **exactly — 140/140 parameters** (min, max, value, calibration)
  across all crop/disease/fungicide sets. Verified by `tools/check_parameters.py`
  (regression guard; exits non-zero on drift).
  The `AssimilateSappersDamage` 20-vs-300 note compared against the wrong file:
  the C# *runtime* `franchestynParameters.csv` (concatenated, no `Set` column,
  last-block-wins) uses a blanket bound (0-300) across all disease blocks and is
  not the per-disease canonical source. The JSONs correctly use the per-disease
  values. For strict parity with a specific C# run that used the runtime CSV,
  feed simpest that same CSV via the legacy-CSV path (`crop_type=None`,
  `param_file=<csv>`), as the `tools/dual_run/` harness does.

## ❓ Verification / missing evidence

- [x] **Matched dual run.** ✅ Done — harness at `tools/dual_run/` (see its
      README). Indiana 1973–1990, validation, 4563 matched days. After F1–F5 +
      F11, all dynamic variables match to float precision (corr 1.0000):
      cycle %, light interception (att + healthy), AGB/Yield (att + act),
      hydro-thermal time, all disease compartments + severity, and weather
      synthesis. Residuals: AGBact/YieldAct float32 accumulation (mean 0.8 /
      0.2 kg/ha); Susceptible float32 branch-flip on 16/4563 days (benign).
      `DaysAfterSowing`/`GDD` differ by design (F7). This run found and
      validated the F11 fix.
- [x] **C# calibration output** for Indiana/Generic — done via
      `tools/dual_run/run_calibration.py` (disease calibration; crop params are
      unidentifiable with the external crop model). Both engines calibrate the
      same 18 disease params against the same reference; both beat the default
      (31.67 → C# 22.88 / Python 24.93 under the common Python objective).
      Objective parity confirmed at C#'s optimum (RMSE agree to ~0.02). F3
      alignment immaterial for this dataset. Optimiser-level parameter
      differences are expected (F9).
- [x] **SIMPLACE `VapourPressure` units** confirmed **kPa across all datasets**.
      VP-median / median-RH(if kPa) / %rows-plausible(kPa vs hPa): indiana
      0.972 kPa / 94.9% / 64.6% vs 0.2%; indiana_sim 0.972 / 91.9% / 76.2% vs 0.0%;
      sevilla 1.566 / 113.1% / 33.0% vs 2.2%; wageningen 0.922 / 104.5% / 40.7% vs 0.8%.
      hPa is implausible everywhere (0-2% in range) -> kPa confirmed; matches the
      Indiana es(Tmin) spot-check (77.2%). Note: sevilla/wageningen have median
      unclamped RH > 100% (vp > es(Tmin) on most days), so `rhx` clamps to 100
      frequently at those humid sites -> more leaf-wetness hours / disease pressure.
      The clamp in `convert_weather` handles it correctly (no C# counterpart; the
      VP->RH step is simpest-specific). Not a code bug.
- [ ] **Winter-crop (cross-year) case** to confirm F5 impact.

---

## Confirmed faithful (no action needed)

- SEIR disease engine incl. inoculum release shapes, hourly/daily order, and the
  `output ↔ output1` swap (`fr_disease_model.py` ↔ `disease.cs`).
- Crop damage mechanisms + external-crop-model branch (`fr_crop_model.py` ↔
  `crop.cs`).
- Fungicide degradation/tenacity/efficacy + 30-day hard stop
  (`fr_fungicide_model.py` ↔ `fungicide.cs`).
- `t_response` / `rain_detachment` (`fr_utilities.py` ↔ `utilities.cs`); only
  defensive zero-guards added.
- Hourly→daily loop, sowing reset, treatment scheduling (`fr_runner.py` ↔
  `optimizer.cs:modelCall/oneShot`).
- RMSE objective formula incl. `/200`, `/100`, and `×1000` penalties.
- AUDPC (trapezoid of `DiseaseSeverity·100` over unit-spaced daily axis).
- Parameter names and units vs `franchestynParameters.csv`.
- VapourPressure→RH conversion internally consistent (kPa, Tetens es).
