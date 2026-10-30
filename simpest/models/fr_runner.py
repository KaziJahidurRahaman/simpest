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
        rain_limited_primary_inoculum (bool): Scale primary (external) infection
            by rain detachment for splash-borne diseases. Defaults to ``False``
            (C# behaviour). See :class:`~simpest.models.fr_data.Parameters`.
        htt_state_dry_decay (float): Fraction of accumulated hydro-thermal time
            surviving a dry-interrupted day. Defaults to ``1.0`` (C# behaviour,
            no decay). See :class:`~simpest.models.fr_data.Parameters`.
        carry_dry_counter (bool): Carry the dry-hour counter across midnight.
            Defaults to ``False`` (C# behaviour), where the counter resets with
            each daily output and so never exceeds 23, making
            ``DryCriticalInterruption`` values above 23 inert.
        upper_leaf_pool (bool): Track a second tissue pool for the upper
            leaves and use its severity in the calibration objective. Defaults
            to ``False`` (C# behaviour). See
            :class:`~simpest.models.fr_data.Parameters`.
        upper_leaf_cycle_start (float): Crop cycle percentage at which the
            upper leaves start to emerge.
        upper_leaf_cycle_full (float): Crop cycle percentage at which the
            upper leaves are fully emerged.
        score_harvest_after_maturity (bool): Score a yield observation dated
            after the simulated crop has matured against the season's final
            simulated yield. Defaults to ``False`` (C# behaviour), where such an
            observation is compared with zero output and, being a zero yield
            after maturity, gets the ×1000 penalty. See :meth:`compute_rmse`.
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
        rain_limited_primary_inoculum: bool = False,
        htt_state_dry_decay: float = 1.0,
        carry_dry_counter: bool = False,
        upper_leaf_pool: bool = False,
        upper_leaf_cycle_start: float = 36.0,
        upper_leaf_cycle_full: float = 52.0,
        score_harvest_after_maturity: bool = False,
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
        self.rain_limited_primary_inoculum = rain_limited_primary_inoculum
        self.htt_state_dry_decay = htt_state_dry_decay
        self.carry_dry_counter = carry_dry_counter
        self.upper_leaf_pool = upper_leaf_pool
        self.upper_leaf_cycle_start = upper_leaf_cycle_start
        self.upper_leaf_cycle_full = upper_leaf_cycle_full
        self.score_harvest_after_maturity = score_harvest_after_maturity
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

        # Populated lazily by _load_weather() and reused for the runner's
        # lifetime — weather_dir/site/start_year/end_year/latitude never
        # change after construction, so re-parsing on every run() call (as
        # calibration does, once per objective evaluation) is pure waste.
        self._weather_cache: Optional[Dict[datetime, InputsHourly]] = None
        self._weather_hours_cache: Optional[List[datetime]] = None

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
        weather_hours = self._weather_hours()

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

        for hour_dt in weather_hours:
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
                    if self.carry_dry_counter:
                        # Optional: let a dry spell span midnight instead of
                        # resetting with the daily output (see Parameters).
                        output_t1.disease.counter_dry = output.disease.counter_dry
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

        This is the calibration objective, a port of the C#
        ``Optimizer.ObjfuncVal``. Every reference observation dated within the
        simulated years is scored: the squared, scaled errors of the selected
        variables (above-ground biomass, attainable and actual yield, light
        interception and disease severity) are summed per date, and the
        result is the root mean of those per-date sums.

        The simulated state an observation on day ``d`` is compared with is
        the C# state at hour 0 of ``d`` (see :meth:`_state_at`): the output of
        ``d - 1`` while the crop grows (previous-day alignment), and all-zero
        output outside a season, i.e. before sowing and from the day after
        maturity until the next sowing. An observation dated after maturity is
        therefore compared with zero, not left out. Two penalty rules sharpen
        the objective, as in C#: a zero simulated yield where the reference
        is positive once the crop has matured, and a near-zero simulated light
        interception while the crop is planted, are multiplied by 1000.

        With ``score_harvest_after_maturity`` on (a simpest option, not in
        C#), a yield observation between maturity and the next sowing is
        compared with the season's final simulated yield instead of zero.

        Args:
            date_outputs (Dict[datetime, Outputs]): Daily outputs produced by
                :meth:`run`.
            include_crop (bool): Include the crop-related error terms (biomass,
                attainable yield, light interception).
            include_disease (bool): Include the disease-related error terms
                (severity and actual yield).

        Returns:
            float: The root-mean-square error, rounded to three decimals; ``0.0``
            when there are no reference observations in the simulated years.
        """
        ref = self.sim_unit.reference_data
        by_date = ref.disease_date_disease_sev.get(self.disease, {})
        series = []
        if include_crop:
            series += [ref.date_agb, ref.date_yield_attainable, ref.date_fint]
        if include_disease:
            series += [by_date, ref.date_yield_actual]
        ref_days = sorted({
            d for s in series for d in s
            if self.start_year <= d.year <= self.end_year
        })

        days = sorted(
            (dt.date(), out) for dt, out in date_outputs.items() if dt.hour == 23
        )
        sim_dates = [d for d, _ in days]

        errors: List[float] = []
        for ref_key in ref_days:
            out, is_planted, is_matured, last_out = self._state_at(ref_key, days, sim_dates)
            total_err = 0.0

            def yield_error(yield_ref: float, sim_y: float, final_y: Optional[float]) -> float:
                if final_y is not None:
                    sim_y = final_y
                err = ((yield_ref - sim_y) / 100.0) ** 2
                # Penalty: heavily penalise zero yield at maturity when ref > 0
                if sim_y == 0.0 and yield_ref > 0.0 and is_matured:
                    err *= 1000.0
                return err

            # The season's final yield, for score_harvest_after_maturity.
            harvest = last_out if (self.score_harvest_after_maturity and last_out is not None) else None

            if include_crop:
                if ref_key in ref.date_agb:
                    total_err += ((ref.date_agb[ref_key] - out.crop.agb_attainable) / 200.0) ** 2
                if ref_key in ref.date_yield_attainable:
                    total_err += yield_error(
                        ref.date_yield_attainable[ref_key], out.crop.yield_attainable,
                        harvest.crop.yield_attainable if harvest else None)
                if ref_key in ref.date_fint:
                    sim_fi = out.crop.light_interception_attainable * 100.0
                    fint_err = (ref.date_fint[ref_key] * 100.0 - sim_fi) ** 2
                    # Penalty: heavily penalise near-zero interception during the season
                    if sim_fi < 0.1 and is_planted:
                        fint_err *= 1000.0
                    total_err += fint_err

            if include_disease:
                if ref_key in by_date:
                    # With the upper-leaf pool on, the reference severity is
                    # taken to describe the upper leaves.
                    if self.upper_leaf_pool:
                        sim_ds = out.disease.upper_leaf_severity * 100.0
                    else:
                        sim_ds = out.disease.disease_severity * 100.0
                    total_err += (by_date[ref_key] - sim_ds) ** 2
                if ref_key in ref.date_yield_actual:
                    total_err += yield_error(
                        ref.date_yield_actual[ref_key], out.crop.yield_actual,
                        harvest.crop.yield_actual if harvest else None)

            errors.append(total_err)

        if not errors:
            return 0.0
        import math
        return round(math.sqrt(sum(errors) / len(errors)), 3)

    # -----------------------------------------------------------------------
    # Private helpers
    # -----------------------------------------------------------------------

    def _state_at(self, ref_day, days, sim_dates):
        """Simulated state an observation on ``ref_day`` is compared with.

        Mirrors the C# loop, which scores an observation at hour 0 of its day,
        after the day's state update:

        - while the crop grows, the output of the previous day (or of the same
          day with ``use_prev_day_alignment=False``);
        - on a sowing day, zero output with the crop planted;
        - from the day after maturity until the next sowing, zero output with
          the crop matured;
        - before the first sowing, zero output.

        Returns:
            tuple: ``(outputs, is_planted, is_matured, last_matured)``, where
            ``last_matured`` is the matured output of the season (``None``
            unless the zero state follows a maturity).
        """
        from bisect import bisect_right

        sim_day = ref_day - timedelta(days=1) if self.use_prev_day_alignment else ref_day
        i = bisect_right(sim_dates, sim_day) - 1
        out = days[i][1] if i >= 0 and sim_dates[i] == sim_day else None
        if out is not None:
            matured = out.crop.cycle_completion_percentage >= 100.0
            if not (matured and self.use_prev_day_alignment):
                return out, out.crop.day_after_sowing > 0 and not matured, matured, None

        if self.sim_unit.year_sowing_doy.get(ref_day.year) == ref_day.timetuple().tm_yday:
            return Outputs(), True, False, None
        last = days[i][1] if i >= 0 else None
        if last is not None and last.crop.cycle_completion_percentage >= 100.0:
            return Outputs(), False, True, last
        return Outputs(), False, False, None

    def _build_parameters(
        self, param_values: Dict[str, float]
    ) -> Parameters:
        """Assemble a Parameters object from CSV defaults + overrides."""
        parameters = Parameters(
            rain_limited_primary_inoculum=self.rain_limited_primary_inoculum,
            htt_state_dry_decay=self.htt_state_dry_decay,
            upper_leaf_pool=self.upper_leaf_pool,
            upper_leaf_cycle_start=self.upper_leaf_cycle_start,
            upper_leaf_cycle_full=self.upper_leaf_cycle_full,
        )

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

        The parsed/synthesized hourly series is cached on the instance after
        the first call: ``weather_dir``/``site``/``start_year``/``end_year``/
        ``latitude`` are fixed for the runner's lifetime, so re-reading and
        re-synthesizing (daily -> hourly involves physically based diurnal
        curves, not just a CSV parse) on every ``run()`` call — as happens
        once per calibration objective evaluation — would be pure waste.
        """
        if self._weather_cache is not None:
            return self._weather_cache

        wd = self.weather_dir
        if os.path.isfile(wd):
            weather_file = wd
        else:
            weather_file = os.path.join(
                wd, self.weather_time_step, f"{self.site}.csv"
            )
        if self.weather_time_step == "hourly":
            weather_data = read_hourly(
                weather_file, self.start_year, self.end_year,
                site=self.site, latitude=self.latitude
            )
        else:
            weather_data = read_daily(
                weather_file, self.start_year, self.end_year,
                latitude=self.latitude
            )

        self._weather_cache = weather_data
        return weather_data

    def _weather_hours(self) -> List[datetime]:
        """Return the cached weather series' timestamps, sorted once."""
        if self._weather_hours_cache is None:
            self._weather_hours_cache = sorted(self._load_weather().keys())
        return self._weather_hours_cache

    def update_param_specs(
        self,
        crop_param_specs: Optional[Dict[str, dict]] = None,
        disease_param_specs: Optional[Dict[str, dict]] = None,
        fungicide_param_specs: Optional[Dict[str, dict]] = None,
    ) -> None:
        """Refresh calibration flags/values from updated spec dicts in place.

        Rebuilds only ``self.name_param`` (the cheap part) from the given
        spec dicts — leaving weather, sowing, reference and crop-model data
        untouched — so a single runner can be reused across an entire
        calibration session (staged sweeps, repeated forward checks, …)
        instead of being rebuilt from scratch for every call.

        Args:
            crop_param_specs: Updated crop parameter spec dict, or ``None`` to
                leave the crop parameters as they are.
            disease_param_specs: Updated disease parameter spec dict, or
                ``None`` to leave the disease parameters as they are.
            fungicide_param_specs: Updated fungicide parameter spec dict, or
                ``None`` to leave the fungicide parameters as they are.
        """
        if crop_param_specs is not None:
            self.name_param.update(build_params_from_specs(crop_param_specs, "crop"))
        if disease_param_specs is not None and self.disease_type:
            self.name_param.update(build_params_from_specs(disease_param_specs, "disease"))
        if fungicide_param_specs is not None and self.fungicide_type:
            self.name_param.update(build_params_from_specs(fungicide_param_specs, "fungicide"))


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
