"""Main simulation runner for simpest's crop-disease-fungicide model.

The runner orchestrates a full simulation: it loads parameters, weather, the
sowing schedule, reference observations, and any external crop-model series,
then drives the hourly–daily model loop across the configured years. Each
simulated day couples the crop, disease, and fungicide sub-models and records a
complete output bundle. The runner also exposes the calibration objective used
to score a run against reference data.

Example:
    ```python
    from simpest.models.fr_runner import FranchestynRunner

    runner = FranchestynRunner(...)
    date_outputs = runner.run(param_values={})  # optional {param_key: value} overrides
    ```
"""

from __future__ import annotations

import os
from copy import deepcopy
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from .fr_data import (
    CropModelData, InputsDaily, InputsHourly,
    Outputs, Parameters, ParCrop, ParDisease, ParFungicide,
    Parameter, SimulationUnit,
)
from .fr_crop_model import run as crop_run
from .fr_disease_model import DiseaseModel
from .fr_fungicide_model import run as fungicide_run
from .fr_param_reader import (
    read as param_read, calibrated_read, build_params_from_specs,
    read_crop_parameters, read_disease_parameters, read_fungicide_parameters
)
from .fr_weather_reader import read_daily, read_hourly
from .fr_reference_reader import read_sowing, read_reference, read_crop_model_data


# Maximum days-after-sowing safety stop (≈ 11 months)
_MAX_DAS = 11 * 30


class FranchestynRunner:
    """
    End-to-end crop-disease-fungicide simulation runner.

    Args:
        weather_dir (str): Directory containing weather data.
        param_file (str): Path to parameter file.
        sowing_file (str): Path to sowing or management CSV.
        ref_dir (str): Directory or file path for reference data.
        crop_model_dir (Optional[str]): Directory or file path for external
            crop model data. If ``None``, the internal crop model is used.
        site (str): Site identifier.
        variety (str): Variety identifier.
        disease (str): Disease column/key name used in reference data.
        start_year (int): First simulation year (inclusive).
        end_year (int): Last simulation year (inclusive).
        weather_time_step (str): Weather frequency, ``daily`` or ``hourly``.
        calibration_variable (str): Calibration target scope: ``crop``,
            ``disease``, or ``all``.
        is_calibration (bool): Whether the run is used for calibration.
        latitude (float): Site latitude in decimal degrees.
        crop_type (Optional[str]): Crop type for modular parameter loading.
        crop_param_file (Optional[str]): Crop parameter JSON path.
        disease_param_file (Optional[str]): Disease parameter JSON path.
        disease_type (Optional[str]): Disease type key for modular loading. Also
            acts as the on/off switch for the disease model: when ``None`` the
            SEIR disease step (hourly and daily) is skipped entirely, so disease
            severity stays 0 and there is no disease impact (a crop-only run).
            Provide a ``disease_type`` to enable the epidemiological model.
        fungicide_param_file (Optional[str]): Fungicide parameter JSON path.
        fungicide_type (Optional[str]): Fungicide type key for modular loading.
            Also the on/off switch for the fungicide model: when ``None`` the
            fungicide step is skipped. The step is a no-op unless a treatment has
            been scheduled, so it only needs to be enabled when fungicide
            treatments are present (otherwise they would be ignored).
        use_gdd (bool): How crop cycle completion is derived. Defaults to
            ``False`` (calendar-day interpolation of the external crop-model
            series); set ``True`` to use the thermal-time (GDD) based cycle
            percentage, which ties phenology more directly to accumulated heat.
        use_prev_day_alignment (bool): Day alignment used by the calibration
            objective. Defaults to ``True``, which compares the previous day's
            simulated output to the current day's reference observation
            (sim[d-1] vs ref[d]). Set ``False`` for same-day alignment
            (sim[d] vs ref[d]), which matches the daily output table.
        all_row_includes_end_year (bool): How far an ``"All"`` sowing row is
            applied. Defaults to ``False`` (the final year is omitted); set
            ``True`` to apply it through ``end_year`` inclusive. Has no effect
            for per-year sowing CSVs.
    """

    def __init__(
        self,
        weather_dir: str,
        param_file: str,
        sowing_file: str,
        ref_dir: str,
        crop_model_dir: Optional[str],
        site: str,
        variety: str,
        disease: str,
        start_year: int,
        end_year: int,
        weather_time_step: str = "daily",
        calibration_variable: str = "all",
        is_calibration: bool = False,
        latitude: float = 0.0,
        crop_type: Optional[str] = None,
        crop_param_file: Optional[str] = None,
        disease_param_file: Optional[str] = None,
        disease_type: Optional[str] = None,
        fungicide_param_file: Optional[str] = None,
        fungicide_type: Optional[str] = None,
        crop_param_specs: Optional[Dict[str, dict]] = None,
        disease_param_specs: Optional[Dict[str, dict]] = None,
        fungicide_param_specs: Optional[Dict[str, dict]] = None,
        use_gdd: bool = False,
        use_prev_day_alignment: bool = True,
        all_row_includes_end_year: bool = False,
    ) -> None:
        self.weather_dir = weather_dir
        self.param_file = param_file
        self.sowing_file = sowing_file
        self.ref_dir = ref_dir
        self.crop_model_dir = crop_model_dir
        self.site = site
        self.variety = variety
        self.disease = disease
        self.start_year = start_year
        self.end_year = end_year
        self.weather_time_step = weather_time_step.lower()
        self.calibration_variable = calibration_variable
        self.is_calibration = is_calibration
        self.latitude = latitude
        self.disease_type = disease_type
        self.fungicide_type = fungicide_type
        self.use_gdd = use_gdd
        self.use_prev_day_alignment = use_prev_day_alignment
        self.all_row_includes_end_year = all_row_includes_end_year

        # Read parameter definitions (bounds, defaults)
        # Three loading scenarios, in precedence order:
        # 1. In-memory spec dicts (crop_param_specs/…) → build directly. These
        #    are the (possibly edited) dicts held by FranchestynConfig and take
        #    priority so caller-side edits (e.g. deactivate_calibration) apply.
        # 2. crop_param_file + crop_type → modular loading from JSON files.
        # 3. Only param_file → legacy CSV.

        self.name_param: Dict[str, Parameter] = {}

        if crop_param_specs is not None:
            # Modular loading from in-memory spec dicts
            self.name_param.update(build_params_from_specs(crop_param_specs, "crop"))
            if disease_param_specs and disease_type:
                self.name_param.update(build_params_from_specs(disease_param_specs, "disease"))
            if fungicide_param_specs and fungicide_type:
                self.name_param.update(build_params_from_specs(fungicide_param_specs, "fungicide"))
        elif crop_param_file and crop_type:
            # Modular loading: always load crop
            self.name_param.update(read_crop_parameters(crop_param_file, crop_type))

            # Optionally load disease if both file and type provided
            if disease_param_file and disease_type:
                disease_catalog = read_disease_parameters(disease_param_file)
                if disease_type not in disease_catalog:
                    raise ValueError(
                        f"Disease type '{disease_type}' not found. "
                        f"Available: {list(disease_catalog.keys())}"
                    )
                self.name_param.update(disease_catalog[disease_type])

            # Optionally load fungicide if both file and type provided
            if fungicide_param_file and fungicide_type:
                self.name_param.update(read_fungicide_parameters(fungicide_param_file, fungicide_type))
        else:
            # Legacy CSV loader
            self.name_param = param_read(param_file, calibration_variable)

        # Read calibrated values (override file, empty if not present)
        self.param_out_calibration: Dict[str, float] = {}

        # Read sowing + reference data
        self.sim_unit: SimulationUnit = read_sowing(
            sowing_file, site, variety, start_year, end_year,
            all_row_includes_end_year=all_row_includes_end_year,
        )
        self.sim_unit = read_reference(
            ref_dir, sowing_file, site, variety,
            start_year, end_year, self.sim_unit, disease
        )

        # Read external crop model data (None → internal model used)
        self.crop_model_data: Optional[CropModelData] = None
        if crop_model_dir:
            self.crop_model_data = read_crop_model_data(crop_model_dir, use_gdd=self.use_gdd)

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def run(
        self,
        param_values: Optional[Dict[str, float]] = None,
    ) -> Dict[datetime, Outputs]:
        """
        Run the model for all configured years and return daily outputs.

        Args:
            param_values (Optional[Dict[str, float]]): Optional override values
                keyed by ``class_ParamName``.

        Returns:
            Dict[datetime, Outputs]: Mapping of end-of-day timestamp to model
            outputs.
        """
        parameters = self._build_parameters(param_values or {})
        weather_data = self._load_weather()

        date_outputs: Dict[datetime, Outputs] = {}

        # Per-hour accumulators aggregated into the daily input each day
        hourly_temps:  List[float] = []
        hourly_rads:   List[float] = []
        hourly_precip: List[float] = []
        hourly_rhs:    List[float] = []
        hourly_lw:     List[float] = []

        output    = Outputs()
        output_t1 = Outputs()
        disease_model = DiseaseModel()

        is_planted = False
        last_treatment_date = datetime(1900, 1, 1)

        for hour_dt in sorted(weather_data.keys()):
            year = hour_dt.year
            if year < self.start_year or year > self.end_year:
                output = Outputs()
                output_t1 = Outputs()
                continue

            hourly_rec = weather_data[hour_dt]
            hourly_rec.date = hour_dt
            hourly_rec.latitude = self.latitude

            # Sowing: reset on the correct DOY
            sow_doy = self.sim_unit.year_sowing_doy.get(year)
            if sow_doy and hour_dt.timetuple().tm_yday == sow_doy:
                is_planted = True
                output = Outputs()
                output_t1 = Outputs()
                disease_model = DiseaseModel()
                output_t1.crop.growing_season = year

            if is_planted:
                # Track fungicide treatments
                sched = self.sim_unit.fungicide_treatment_schedule
                treatment_dates = {t.date() for t in sched.treatments}
                if hour_dt.date() in treatment_dates:
                    last_treatment_date = datetime.combine(
                        hour_dt.date(), datetime.min.time()
                    )
                hourly_rec.date_treatment_last = last_treatment_date

                # Accumulate hourly observations
                hourly_temps.append(hourly_rec.air_temperature)
                hourly_rads.append(hourly_rec.rad)
                hourly_precip.append(hourly_rec.precipitation)
                hourly_rhs.append(hourly_rec.relative_humidity)
                hourly_lw.append(hourly_rec.leaf_wetness)

                # Skip the hourly disease step when disease is disabled, and also
                # for crop-only calibration runs where disease plays no role.
                if self.disease_type and (self.calibration_variable != "crop" or not self.is_calibration):
                    disease_model.run_hourly(hourly_rec, parameters, output, output_t1)

                if hour_dt.hour == 23:
                    # ----------------------------------------------------------
                    # End of day: swap, build daily input, run daily models
                    # ----------------------------------------------------------
                    output = output_t1

                    # Fresh daily output with season-persistent fields carried over
                    output_t1 = Outputs()
                    output_t1.crop.growing_season = output.crop.growing_season
                    output_t1.crop.f_int_peak = output.crop.f_int_peak
                    output_t1.disease.is_primary_inoculum_started = (
                        output.disease.is_primary_inoculum_started
                    )
                    output_t1.disease.first_seasonal_infection = (
                        output.disease.first_seasonal_infection
                    )
                    output_t1.disease.cycle_percentage_first_infection = (
                        output.disease.cycle_percentage_first_infection
                    )

                    # Assemble daily input from hourly lists
                    input_daily = InputsDaily()
                    input_daily.tmax = max(hourly_temps)
                    input_daily.tmin = min(hourly_temps)
                    input_daily.rad  = sum(hourly_rads)
                    input_daily.precipitation = sum(hourly_precip)
                    input_daily.rhx = max(hourly_rhs)
                    input_daily.rhn = min(hourly_rhs)
                    input_daily.leaf_wetness = sum(hourly_lw)
                    # Date of this day = hour 23 minus 23 hours
                    input_daily.date = hour_dt - timedelta(hours=23)
                    input_daily.date_treatment_last = hourly_rec.date_treatment_last
                    input_daily.crop_model_data = self.crop_model_data

                    # Run the daily sub-models. The crop step always runs; the
                    # fungicide and disease steps are optional and gated on their
                    # respective *_type (None skips the step, giving a crop-only
                    # or no-fungicide run).
                    crop_run(input_daily, parameters, output, output_t1)
                    if self.fungicide_type:
                        fungicide_run(input_daily, parameters, output, output_t1)
                    if self.disease_type:
                        disease_model.run_daily(input_daily, parameters, output, output_t1)

                    # Store daily output
                    output_t1.inputs_daily = input_daily
                    date_outputs[hour_dt] = output_t1

                    # Clear hourly accumulators
                    hourly_temps.clear()
                    hourly_rads.clear()
                    hourly_precip.clear()
                    hourly_rhs.clear()
                    hourly_lw.clear()

                    # Safety stop or maturity check
                    if (output_t1.crop.cycle_completion_percentage >= 100
                            or output_t1.crop.day_after_sowing >= _MAX_DAS):
                        is_planted = False

            else:
                is_planted = False
                output = Outputs()
                output_t1 = Outputs()

        return date_outputs

    def compute_rmse(
        self,
        date_outputs: Dict[datetime, Outputs],
        include_crop: bool = True,
        include_disease: bool = True,
    ) -> float:
        """Compute the root-mean-square error against reference data.

        This is the calibration objective. For each simulated day that has a
        matching reference observation, it accumulates the squared, scaled
        errors of the selected variables — above-ground biomass, attainable and
        actual yield, light interception, and disease severity — and returns the
        root mean of those per-day error sums. Two penalty rules sharpen the
        objective: a zero simulated yield at maturity where the reference is
        positive, and a near-zero simulated light interception during the
        growing season, are both heavily penalised.

        Args:
            date_outputs (Dict[datetime, Outputs]): Daily outputs produced by
                :meth:`run`.
            include_crop (bool): Include the crop-related error terms (biomass,
                attainable yield, light interception).
            include_disease (bool): Include the disease-related error terms
                (severity and actual yield).

        Returns:
            float: The root-mean-square error, rounded to three decimals; ``0.0``
            when no reference observations overlap the simulated days.
        """
        errors: List[float] = []
        ref = self.sim_unit.reference_data

        from datetime import date as date_type
        for hour_dt, out in date_outputs.items():
            if hour_dt.hour != 23:
                continue
            # Reference keys are datetime.date; convert for lookup.
            sim_day: date_type = hour_dt.date()
            # Day alignment between simulated output and reference data.
            #   Previous-day alignment (default): compare this simulated day to
            #   the next day's reference, i.e. sim[d] vs ref[d+1], which is
            #   equivalent to sim[d-1] vs ref[d].
            #   Same-day alignment: sim[d] vs ref[d], matching the daily table.
            ref_key: date_type = (
                sim_day + timedelta(days=1)
                if self.use_prev_day_alignment else sim_day
            )
            total_err = 0.0
            has_ref = False

            # Infer crop state for the penalty multipliers below
            is_matured = out.crop.cycle_completion_percentage >= 100.0
            is_planted = out.crop.day_after_sowing > 0 and not is_matured

            if include_crop:
                agb_err = 0.0
                if ref_key in ref.date_agb:
                    sim_agb = out.crop.agb_attainable
                    agb_err = ((ref.date_agb[ref_key] - sim_agb) / 200.0) ** 2
                    has_ref = True

                yield_err = 0.0
                if ref_key in ref.date_yield_attainable:
                    sim_y = out.crop.yield_attainable
                    yield_ref = ref.date_yield_attainable[ref_key]
                    yield_err = ((yield_ref - sim_y) / 100.0) ** 2
                    # Penalty: heavily penalise zero yield at maturity when ref > 0
                    if sim_y == 0.0 and yield_ref > 0.0 and is_matured:
                        yield_err *= 1000.0
                    has_ref = True

                fint_err = 0.0
                if ref_key in ref.date_fint:
                    sim_fi = out.crop.light_interception_attainable * 100.0
                    fint_err = (ref.date_fint[ref_key] * 100.0 - sim_fi) ** 2
                    # Penalty: heavily penalise near-zero interception during the season
                    if sim_fi < 0.1 and is_planted:
                        fint_err *= 1000.0
                    has_ref = True

                total_err += agb_err + yield_err + fint_err

            if include_disease:
                dis_err = 0.0
                by_date = ref.disease_date_disease_sev.get(self.disease, {})
                if ref_key in by_date:
                    sim_ds = out.disease.disease_severity * 100.0
                    dis_err = (by_date[ref_key] - sim_ds) ** 2
                    has_ref = True

                yield_err2 = 0.0
                if ref_key in ref.date_yield_actual:
                    sim_ya = out.crop.yield_actual
                    yield_ref2 = ref.date_yield_actual[ref_key]
                    yield_err2 = ((yield_ref2 - sim_ya) / 100.0) ** 2
                    # Penalty: heavily penalise zero actual yield at maturity when ref > 0
                    if sim_ya == 0.0 and yield_ref2 > 0.0 and is_matured:
                        yield_err2 *= 1000.0
                    has_ref = True

                total_err += dis_err + yield_err2
            if has_ref:
                errors.append(total_err)

        if not errors:
            return 0.0
        import math
        return round(math.sqrt(sum(errors) / len(errors)), 3)

    # -----------------------------------------------------------------------
    # Private helpers
    # -----------------------------------------------------------------------

    def _build_parameters(
        self, param_values: Dict[str, float]
    ) -> Parameters:
        """Assemble a Parameters object from CSV defaults + overrides."""
        parameters = Parameters()

        for key, p in self.name_param.items():
            # key = "class_ParamName", e.g. "crop_TbaseCrop"
            parts = key.split("_", 1)
            if len(parts) != 2:
                continue
            param_class, param_name = parts

            # Use override value if supplied, else CSV default
            if key in param_values:
                value = param_values[key]
            elif key in self.param_out_calibration:
                value = self.param_out_calibration[key]
            else:
                value = p.value_bool if p.is_boolean else p.value

            _set_param(parameters, param_class, param_name, value)

        return parameters

    def _load_weather(self) -> Dict[datetime, InputsHourly]:
        """Load weather data for the configured site and time step.

        If ``self.weather_dir`` is itself an existing CSV file, it is used
        directly (i.e. the caller already resolved the full path).
        """
        wd = self.weather_dir
        if os.path.isfile(wd):
            weather_file = wd
        else:
            weather_file = os.path.join(
                wd, self.weather_time_step, f"{self.site}.csv"
            )
        if self.weather_time_step == "hourly":
            return read_hourly(
                weather_file, self.start_year, self.end_year,
                site=self.site, latitude=self.latitude
            )
        else:
            return read_daily(
                weather_file, self.start_year, self.end_year,
                latitude=self.latitude
            )


# ---------------------------------------------------------------------------
# Helper: set a named parameter on the correct sub-object of Parameters
# ---------------------------------------------------------------------------

def _set_param(parameters: Parameters, param_class: str, param_name: str, value) -> None:
    """Find the matching field on par_crop, par_disease, or par_fungicide and set it."""
    # Map CSV class names (case-insensitive) to Parameters sub-objects
    sub_map = {
        "crop":      parameters.par_crop,
        "disease":   parameters.par_disease,
        "fungicide": parameters.par_fungicide,
    }
    sub_obj = sub_map.get(param_class.lower())
    if sub_obj is None:
        return

    # Convert CamelCase param_name → snake_case field name. The regex-based
    # conversion mishandles embedded acronyms (e.g. "RUEreducerDamage" →
    # "ru_ereducer_damage"), so fall back to a normalized match that strips
    # underscores and lowercases both sides ("ruereducerdamage" ==
    # "rue_reducer_damage"). Without this, such parameters are silently dropped.
    snake = _camel_to_snake(param_name)
    if not hasattr(sub_obj, snake):
        target = param_name.replace("_", "").lower()
        snake = next(
            (f for f in vars(sub_obj) if f.replace("_", "").lower() == target),
            snake,
        )
    if hasattr(sub_obj, snake):
        field_type = type(getattr(sub_obj, snake))
        try:
            setattr(sub_obj, snake, field_type(value))
        except (TypeError, ValueError):
            setattr(sub_obj, snake, value)


def _camel_to_snake(name: str) -> str:
    """Convert CamelCase to snake_case (handles runs of uppercase too)."""
    import re
    s1 = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1_\2', name)
    result = re.sub(r'([a-z\d])([A-Z])', r'\1_\2', s1).lower()
    return result
