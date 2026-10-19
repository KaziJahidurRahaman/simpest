"""Hourly infection accumulation and daily SEIR tissue progression.

This module implements the epidemiological core of the model. Infection
pressure is accumulated at an hourly time step from temperature and leaf-wetness
suitability; once per day the accumulated pressure drives new infections and the
progression of existing lesions through a compartmental SEIR scheme
(susceptible → latent → sporulating → dead).

The :class:`DiseaseModel` is stateful and must be instantiated once per growing
season (the runner creates a fresh instance at sowing). Within a day,
:meth:`DiseaseModel.run_hourly` is called for every hour; at the end of the day
the runner advances the daily state and calls :meth:`DiseaseModel.run_daily`
with the aggregated weather.
"""

from __future__ import annotations
import math
from datetime import datetime
from typing import List

from .fr_data import InputsHourly, InputsDaily, Parameters, Outputs, TissueState
from .fr_utilities import t_response, rain_detachment


class DiseaseModel:
    """Stateful SEIR disease model.

    Lifecycle
    ---------
    Create a fresh instance at the start of each growing season (sowing date).
    For each hour: call run_hourly().
    After hour 23 (handled by the runner): call run_daily().
    """

    def __init__(self) -> None:
        # Hourly accumulators – cleared at end of each day by run_hourly()
        self._temp: List[float] = []
        self._rh:   List[float] = []
        self._rain: List[float] = []
        self._lw:   List[float] = []

        # Tissue cohort list – persists across the entire season
        self.tissue_tracking: List[TissueState] = []

        # Cohorts of the upper-leaf pool (only filled when
        # Parameters.upper_leaf_pool is on)
        self.upper_tissue_tracking: List[TissueState] = []

    # -----------------------------------------------------------------------
    # Hourly step
    # -----------------------------------------------------------------------

    def run_hourly(
        self,
        input_: InputsHourly,
        parameters: Parameters,
        output: Outputs,
        output1: Outputs,
    ) -> None:
        """Process one hour of weather data.

        During hours 0–22 this accumulates the hydro-thermal time rate (the
        product of temperature and relative-humidity suitability) and tracks the
        dry-spell counter. At hour 23 it finalises the daily infection and
        sporulation efficiencies, normalises the accumulated rate, records the
        daily weather summary, and clears the hourly buffers.

        The same ``output1`` object is shared across all 24 hours of a day; the
        daily state advance happens only after the hour-23 call returns.

        Args:
            input_ (InputsHourly): The current hour's weather record.
            parameters (Parameters): Model parameters, including the disease group.
            output (Outputs): Previous day's output state, used to carry forward
                the accumulated hydro-thermal time state.
            output1 (Outputs): Current day's output state, updated in place.

        Returns:
            None: The function mutates ``output1`` in place.
        """
        par = parameters.par_disease

        # Accumulate hourly observations
        self._temp.append(input_.air_temperature)
        self._rh.append(input_.relative_humidity)
        self._rain.append(input_.precipitation)
        self._lw.append(input_.leaf_wetness)

        # Dry-spell counter (resets on any wet hour, otherwise increments)
        if input_.leaf_wetness == 1:
            output1.disease.counter_dry = 0.0
        else:
            output1.disease.counter_dry += 1.0

        # Temperature suitability (0–1)
        output1.disease.temp_function = t_response(
            input_.air_temperature, par.tmin, par.topt, par.tmax
        )

        hour = input_.date.hour

        if hour == 23:
            # ----------------------------------------------------------------
            # End-of-day: finalise infection metrics and daily weather summary
            # ----------------------------------------------------------------
            rate = output1.disease.hydro_thermal_time_rate  # accumulated hours 0-22

            # Sporulation efficiency = average hourly HTT per hour of the day
            output1.disease.sporulation_efficiency = rate / 24.0

            # Infection efficiency (before normalisation against WetnessDurationOptimum)
            if rate > par.wetness_duration_minimum:
                htt_inf = rate / par.wetness_duration_optimum
                if htt_inf >= 1.0:
                    htt_inf = 1.0
                output1.disease.hydro_thermal_time_infection = htt_inf
            else:
                output1.disease.hydro_thermal_time_infection = 0.0

            # Normalise daily rate (0–1)
            rate_norm = rate / par.wetness_duration_optimum
            if rate_norm >= 1.0:
                rate_norm = 1.0
            output1.disease.hydro_thermal_time_rate = rate_norm

            # Cumulate state (carried forward day to day via output)
            carried = output.disease.hydro_thermal_time_state

            # Dry-spell interruption overrides infection for this day, and
            # optionally decays the accumulated state (see Parameters).
            if output1.disease.counter_dry > par.dry_critical_interruption:
                output1.disease.hydro_thermal_time_infection = 0.0
                carried *= parameters.htt_state_dry_decay

            output1.disease.hydro_thermal_time_state = carried + rate_norm

            # Daily weather summary (for output only)
            output1.disease.tmax_daily  = max(self._temp)
            output1.disease.tmin_daily  = min(self._temp)
            output1.disease.rhmax_daily = max(self._rh)
            output1.disease.rhmin_daily = min(self._rh)
            output1.disease.rain_daily  = sum(self._rain)
            output1.disease.lw_daily    = sum(self._lw)

            # Clear buffers for the next day
            self._temp.clear()
            self._rh.clear()
            self._rain.clear()
            self._lw.clear()

        else:
            # ----------------------------------------------------------------
            # Within-day: accumulate HTT rate when within wet-period threshold
            # ----------------------------------------------------------------
            if output1.disease.counter_dry <= par.dry_critical_interruption:
                rh_fn = _rh_function(
                    input_.relative_humidity,
                    input_.leaf_wetness,
                    par.relative_humidity_not_limiting,
                    par.relative_humidity_critical,
                )
                output1.disease.rh_function = rh_fn
                output1.disease.hydro_thermal_time_rate += (
                    output1.disease.temp_function * rh_fn
                )
            else:
                output1.disease.rh_function = 0.0

    # -----------------------------------------------------------------------
    # Daily step (called AFTER the runner swaps output ↔ output1 at hour 23)
    # -----------------------------------------------------------------------

    def run_daily(
        self,
        input_: InputsDaily,
        parameters: Parameters,
        output: Outputs,
        output1: Outputs,
    ) -> None:
        """Run the daily SEIR tissue progression.

        Given the day's accumulated infection pressure, this step (1) creates a
        new latent tissue cohort when onset conditions are met, combining
        external (primary) and internal (secondary) infection sources scaled by
        susceptible tissue, host resistance, and fungicide efficacy; (2) advances
        every existing cohort through the latent → sporulating → dead transitions
        according to thermal time; and (3) aggregates the compartment fractions
        into the daily latent, sporulating, dead, affected, and severity outputs,
        and updates the susceptible fraction for the following day.

        With ``parameters.upper_leaf_pool`` on, a second tissue pool is tracked
        for the upper leaves, the leaves that emerge last and whose disease is
        what field assessments around heading usually score. The pool starts
        empty and opens linearly between the crop cycle percentages
        ``upper_leaf_cycle_start`` and ``upper_leaf_cycle_full``. Its tissue is
        infected by the same rule as the canopy: primary inoculum plus
        secondary spread at the ``PathogenSpread`` rate, where the sporulating
        source is the canopy's sporulating fraction plus the pool's own. New
        infection is that pressure times the emerged, unaffected fraction of
        the pool (no light-interception scaling, no loss to senescence), and
        the cohorts progress like the canopy's. The results are written to the
        ``upper_leaf_*`` outputs; the canopy epidemic, and so the damage to
        the crop, is not changed.

        At entry, ``output`` holds the hourly-accumulated state for the day and
        ``output1`` is a fresh daily output whose season-persistent fields
        (growing season, peak interception, and first-infection markers) have
        already been carried forward by the runner.

        Args:
            input_ (InputsDaily): The day's aggregated inputs.
            parameters (Parameters): Model parameters, including the disease and
                crop groups.
            output (Outputs): Previous (hourly-accumulated) state for the day.
            output1 (Outputs): Current day's output state, updated in place.

        Returns:
            None: The function mutates ``output1`` and the cohort list in place.
        """
        par = parameters.par_disease
        pc  = parameters.par_crop

        # ----------------------------------------------------------------
        # New infection – only once onset conditions are satisfied
        # ----------------------------------------------------------------
        if (output.disease.hydro_thermal_time_state >= par.hydro_thermal_time_onset
                and output.crop.cycle_completion_percentage >= par.cycle_percentage_onset):

            # Record first infection date (once per season)
            if not output1.disease.is_primary_inoculum_started:
                output1.disease.first_seasonal_infection = input_.date
                output1.disease.cycle_percentage_first_infection = (
                    output.crop.cycle_completion_percentage
                )
                output1.disease.is_primary_inoculum_started = True

            # Rescale existing tissue fractions for expanding green area
            # (growth phase only; skip during senescence)
            light_today = output1.crop.light_interception_attainable
            light_yesterday = output.crop.light_interception_attainable
            if light_today > 0.0:
                if (output.disease.affected_sum <= 1.0
                        and not output1.crop.senescence_started):
                    for tissue in self.tissue_tracking:
                        ratio = light_yesterday / light_today
                        tissue.latent_state      *= ratio
                        tissue.sporulating_state *= ratio
                        tissue.dead_state        *= ratio

            # Primary inoculum (reduced by fungicide efficacy)
            outer_ino = _inoculum_model(input_, parameters, output, output1) * (
                1.0 - output1.fungicide.efficacy
            )
            output1.disease.outer_inoculum = outer_ino

            # Splash-borne: spore detachment efficiency
            spo_detach = 0.0
            if par.is_splash_borne:
                spo_detach = 1.0 - rain_detachment(
                    input_.precipitation,
                    par.rain50_detachment,
                    output1.crop.light_interception_attainable,
                )

            # External infection (primary inoculum × HTT infection efficiency),
            # optionally requiring rain to detach the primary inoculum too.
            ext_inf = output.disease.hydro_thermal_time_infection * outer_ino
            if parameters.rain_limited_primary_inoculum and par.is_splash_borne:
                ext_inf *= (1.0 - spo_detach)

            # Internal infection (secondary spread from sporulating tissue)
            int_inf = (
                output.disease.sporulating_sum
                * output.disease.sporulation_efficiency
                * par.pathogen_spread
                * (1.0 - spo_detach)
            )

            # New infection cohort size
            latent_value = (
                (ext_inf + int_inf)
                * output.disease.susceptible_fraction
                * output1.crop.light_interception_attainable
                * (1.0 - pc.varietal_resistance)
                * (1.0 - output1.fungicide.efficacy)
            )

            if latent_value > 0.0:
                self.tissue_tracking.append(TissueState(latent_state=latent_value))

            # Upper-leaf pool: same infection pressure, with the pool's own
            # sporulating tissue added to the secondary source.
            if parameters.upper_leaf_pool:
                emerged = _upper_leaf_emerged(
                    output.crop.cycle_completion_percentage,
                    parameters.upper_leaf_cycle_start,
                    parameters.upper_leaf_cycle_full,
                )
                if emerged > 0.0:
                    upper_int_inf = (
                        output.disease.upper_leaf_sporulating / emerged
                        * output.disease.sporulation_efficiency
                        * par.pathogen_spread
                        * (1.0 - spo_detach)
                    )
                    upper_latent = (
                        (ext_inf + int_inf + upper_int_inf)
                        * max(0.0, emerged - output.disease.upper_leaf_affected)
                        * (1.0 - pc.varietal_resistance)
                        * (1.0 - output1.fungicide.efficacy)
                    )
                    if upper_latent > 0.0:
                        self.upper_tissue_tracking.append(
                            TissueState(latent_state=upper_latent)
                        )

        # ----------------------------------------------------------------
        # SEIR progression for all existing cohorts
        # ----------------------------------------------------------------
        t_ave = (input_.tmin + input_.tmax) / 2.0
        gdd_disease = t_response(t_ave, par.tmin, par.topt, par.tmax)

        lat_progress = (gdd_disease / par.latency_duration
                        if par.latency_duration > 0.0 else 0.0)
        spo_progress = (gdd_disease / par.sporulation_duration
                        if par.sporulation_duration > 0.0 else 0.0)

        efficacy = output1.fungicide.efficacy

        _advance_cohorts(self.tissue_tracking, lat_progress, spo_progress, efficacy)

        # ----------------------------------------------------------------
        # Aggregate tissue states (cap each fraction at 1)
        # ----------------------------------------------------------------
        latent_sum = min(sum(t.latent_state for t in self.tissue_tracking), 1.0)
        spor_sum   = min(sum(t.sporulating_state for t in self.tissue_tracking), 1.0)
        dead_sum   = min(sum(t.dead_state for t in self.tissue_tracking), 1.0)

        affected_sum     = min(latent_sum + spor_sum + dead_sum, 1.0)
        disease_severity = min(spor_sum + dead_sum, 1.0)

        output1.disease.latent_sum       = latent_sum
        output1.disease.sporulating_sum  = spor_sum
        output1.disease.dead_sum         = dead_sum
        output1.disease.affected_sum     = affected_sum
        output1.disease.disease_severity = disease_severity

        # ----------------------------------------------------------------
        # Susceptible tissue fraction
        # ----------------------------------------------------------------
        tissue_availability = 1.0 - affected_sum
        susceptible = max(0.0, min(tissue_availability, 1.0))

        light_att_today = output1.crop.light_interception_attainable
        light_att_prev  = output.crop.light_interception_attainable

        if (light_att_today >= light_att_prev
                and not output1.crop.senescence_started):
            # Growth phase: susceptible fraction = unaffected tissue
            output1.disease.susceptible_fraction = susceptible
        else:
            # Senescence phase: also subtract senesced green area
            f_int_peak = output1.crop.f_int_peak
            if f_int_peak > 0.0:
                sen_loss = (f_int_peak - light_att_today) / f_int_peak
            else:
                sen_loss = 0.0
            output1.disease.susceptible_fraction = susceptible - sen_loss

        output1.disease.susceptible_fraction = max(
            0.0, min(output1.disease.susceptible_fraction, 1.0)
        )

        # ----------------------------------------------------------------
        # Upper-leaf pool: progress the cohorts and aggregate
        # ----------------------------------------------------------------
        if parameters.upper_leaf_pool:
            _advance_cohorts(
                self.upper_tissue_tracking, lat_progress, spo_progress, efficacy
            )
            emerged = _upper_leaf_emerged(
                output.crop.cycle_completion_percentage,
                parameters.upper_leaf_cycle_start,
                parameters.upper_leaf_cycle_full,
            )
            upper_latent_sum = sum(t.latent_state for t in self.upper_tissue_tracking)
            upper_spor_sum = sum(
                t.sporulating_state for t in self.upper_tissue_tracking
            )
            upper_dead_sum = sum(t.dead_state for t in self.upper_tissue_tracking)

            output1.disease.upper_leaf_emerged = emerged
            output1.disease.upper_leaf_latent = upper_latent_sum
            output1.disease.upper_leaf_sporulating = upper_spor_sum
            output1.disease.upper_leaf_dead = upper_dead_sum
            output1.disease.upper_leaf_affected = min(
                upper_latent_sum + upper_spor_sum + upper_dead_sum, emerged
            )
            if emerged > 0.0:
                output1.disease.upper_leaf_severity = min(
                    (upper_spor_sum + upper_dead_sum) / emerged, 1.0
                )

        # ----------------------------------------------------------------
        # Reset aggregates when no green tissue remains
        # ----------------------------------------------------------------
        if light_att_today == 0.0:
            output1.disease.latent_sum      = 0.0
            output1.disease.sporulating_sum = 0.0
            output1.disease.dead_sum        = 0.0
            output1.disease.affected_sum    = 0.0


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _advance_cohorts(
    cohorts: List[TissueState],
    lat_progress: float,
    spo_progress: float,
    efficacy: float,
) -> None:
    """Advance tissue cohorts one day through latent → sporulating → dead."""
    for tissue in cohorts:
        if tissue.dead_state == 1.0:
            continue

        # Latent → sporulating
        tissue.latent_counter += lat_progress * (1.0 - efficacy)
        if tissue.latent_counter >= 1.0 and tissue.latent_state > 0.0:
            tissue.sporulating_state = tissue.latent_state
            tissue.latent_state = 0.0

        # Sporulating → dead
        if tissue.sporulating_state > 0.0:
            tissue.sporulating_counter += spo_progress * (1.0 - efficacy)
        if tissue.sporulating_counter >= 1.0 and tissue.sporulating_state > 0.0:
            tissue.dead_state = tissue.sporulating_state
            tissue.sporulating_state = 0.0


def _upper_leaf_emerged(cycle_pct: float, start: float, full: float) -> float:
    """Fraction (0–1) of the upper-leaf area emerged at a crop cycle percentage.

    Rises linearly from 0 at ``start`` to 1 at ``full``; a step at ``start``
    when ``full`` is not above it.
    """
    if full <= start:
        return 1.0 if cycle_pct >= start else 0.0
    return min(max((cycle_pct - start) / (full - start), 0.0), 1.0)


def _rh_function(
    rh: float, lw: float, rh_lw: float, rh_crit: float
) -> float:
    """Logistic RH suitability function (0–1).

    Returns 1 during leaf wetness, 0 below the critical threshold, and
    follows a logistic curve between critical and non-limiting thresholds.
    """
    if lw == 1:
        return 1.0
    if rh <= rh_crit:
        return 0.0
    if rh >= rh_lw:
        return 1.0
    rh_mid = (rh_crit + rh_lw) / 2.0
    k = 10.0 / (rh_lw - rh_crit)
    return 1.0 / (1.0 + math.exp(-k * (rh - rh_mid)))


def _inoculum_model(
    input_: InputsDaily,
    parameters: Parameters,
    output: Outputs,
    output1: Outputs,
) -> float:
    """Primary inoculum release model (three release shapes).

    Shape 0 – constant: always returns OuterInoculumMax.
    Shape 1 – bell-shaped: rises then falls symmetrically around crop midpoint.
    Shape 2 – logistic decrease: peaks at onset and falls logistically.
    """
    par = parameters.par_disease
    shape = int(par.outer_inoculum_shape_release)
    slope = par.outer_inoculum_shape_parameter
    cycle_pct = output.crop.cycle_completion_percentage
    pct_first = output1.disease.cycle_percentage_first_infection

    if shape == 0:
        # Constant release
        return par.outer_inoculum_max

    elif shape == 1:
        # Bell-shaped release
        ino_max = pct_first + (100.0 - pct_first) / 2.0
        ino_gro = 1.0 / (
            1.0 + math.exp(
                -slope * ((cycle_pct - pct_first)
                          - 0.5 * (ino_max - pct_first))
            )
        )
        ino_dec = 1.0 / (
            1.0 + math.exp(
                slope * (
                    (cycle_pct - (pct_first + (ino_max - pct_first)))
                    - 0.5 * pct_first
                )
            )
        )
        return min(ino_gro, ino_dec) * par.outer_inoculum_max

    else:
        # Logistic decrease (shape == 2 or any other value)
        ino_max = pct_first + (100.0 - pct_first) / 2.0
        ino_dec = 1.0 / (1.0 + math.exp(slope * (cycle_pct - ino_max)))
        return ino_dec * par.outer_inoculum_max
