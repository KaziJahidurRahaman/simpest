"""
disease_model.py – Hourly accumulation and daily SEIR tissue progression.

Translated from models/disease.cs.

The DiseaseModel class must be instantiated once per growing season (the runner
resets it at sowing).  Call run_hourly() for every hour, then at hour 23 the
runner swaps output/output1 and calls run_daily() with the daily aggregates.
"""

from __future__ import annotations
import math
from datetime import datetime
from typing import List

from .fr_data import InputsHourly, InputsDaily, Parameters, Outputs, TissueState
from .fr_utilities import t_response, rain_detachment


class DiseaseModel:
    """Stateful SEIR disease model for within-season foliar disease progression.

    Create a fresh instance at the start of each growing season (sowing date).
    For each hour, call ``run_hourly()``. After hour 23 (handled by the
    runner), call ``run_daily()``.
    """

    def __init__(self) -> None:
        # Hourly accumulators – cleared at end of each day by run_hourly()
        self._temp: List[float] = []
        self._rh:   List[float] = []
        self._rain: List[float] = []
        self._lw:   List[float] = []

        # Tissue cohort list – persists across the entire season
        self.tissue_tracking: List[TissueState] = []

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

        During hours 0–22 this accumulates the hydro-thermal time rate and the
        RH suitability function. At hour 23 it finalises the daily infection
        metrics and clears the hourly buffers.

        Note:
            ``output1`` is the same object across all 24 hours of a day; the
            runner swaps ``output ↔ output1`` only after hour 23 returns.

        Args:
            input_: Hourly weather inputs including temperature, RH,
                precipitation, and leaf wetness.
            parameters: Model parameters.
            output: Previous day's accumulated output state.
            output1: Current day's output, accumulated in-place across all
                24 hours.
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
            output1.disease.hydro_thermal_time_state = (
                output.disease.hydro_thermal_time_state + rate_norm
            )

            # Dry-spell interruption overrides infection for this day
            if output1.disease.counter_dry > par.dry_critical_interruption:
                output1.disease.hydro_thermal_time_infection = 0.0

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

        Args:
            input_: Daily aggregated weather inputs.
            parameters: Model parameters.
            output: Hourly-accumulated state from hours 0–23 of today (after
                the runner has swapped ``output ↔ output1``).
            output1: Fresh daily output with season-persistent fields pre-set
                by the runner (``growing_season``, ``f_int_peak``,
                ``is_primary_inoculum_started``, ``first_seasonal_infection``,
                ``cycle_percentage_first_infection``).
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

            # External infection (primary inoculum × HTT infection efficiency)
            ext_inf = output.disease.hydro_thermal_time_infection * outer_ino

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

        for tissue in self.tissue_tracking:
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

def _rh_function(
    rh: float, lw: float, rh_lw: float, rh_crit: float
) -> float:
    """Logistic relative humidity suitability function.

    Args:
        rh: Current relative humidity (%).
        lw: Leaf wetness flag (1 = wet, 0 = dry).
        rh_lw: RH threshold above which RH is non-limiting (%).
        rh_crit: Critical minimum RH threshold (%).

    Returns:
        Suitability index in [0, 1]: 1.0 during leaf wetness, 0.0 below the
        critical threshold, and a logistic value in between.
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
    """Primary inoculum release model supporting three release shapes.

    Shape 0 – constant: always returns ``OuterInoculumMax``.
    Shape 1 – bell-shaped: rises then falls symmetrically around crop midpoint.
    Shape 2 – logistic decrease: peaks at onset and falls logistically.

    Args:
        input_: Daily weather inputs (unused; kept for signature consistency).
        parameters: Model parameters.
        output: Previous day's output state.
        output1: Current day's output state.

    Returns:
        Primary inoculum quantity for today.
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
