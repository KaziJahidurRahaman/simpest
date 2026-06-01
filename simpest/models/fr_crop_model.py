"""
crop_model.py – Daily crop growth step and damage mechanisms.

Translated from models/crop.cs with two bug fixes applied to the external-model branch:
  FIX 1 – day_after_sowing is now incremented each day (was never set in C#)
  FIX 2 – growing_degree_days is now populated from the GDD column of the crop
           model CSV (was always left as 0 in C#)
"""

from __future__ import annotations
import math
from datetime import datetime, date, timedelta

from .fr_data import InputsDaily, Parameters, Outputs, CropOutputs
from .fr_utilities import t_response


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run(input_: InputsDaily, parameters: Parameters,
        output: Outputs, output1: Outputs) -> None:
    """
    Compute one daily crop growth step.

    Args:
        input_ (InputsDaily): Today's daily inputs, including optional external
            crop model data.
        parameters (Parameters): Model parameters.
        output (Outputs): Previous day's output state used as input state.
        output1 (Outputs): Today's output state, updated in place.

    Returns:
        None: This function updates ``output1`` in place.
    """
    pc = parameters.par_crop
    pd = parameters.par_disease

    # -----------------------------------------------------------------------
    # Damage mechanisms – always computed before selecting model branch
    # -----------------------------------------------------------------------
    severity = output.disease.disease_severity

    output1.disease.damage_mechanisms.light_stealers = (
        1.0 - _light_stealers_fn(severity, pd.light_stealer_damage)
    )
    output1.disease.damage_mechanisms.rue_reducers = (
        1.0 - _rue_reduction_fn(severity, pd.rue_reducer_damage)
    )
    output1.disease.damage_mechanisms.assimilate_sappers = (
        _assimilate_sappers_fn(severity, pd.assimilate_sappers_damage)
    )
    output1.disease.damage_mechanisms.senescence_accelerators = (
        _senescence_accelerator_fn(severity, pd.senescence_accelerator_damage)
    )

    dm = output1.disease.damage_mechanisms
    cmd = input_.crop_model_data

    # -----------------------------------------------------------------------
    # Branch A: FraNchEstYN internal crop model (no external crop model data)
    # -----------------------------------------------------------------------
    if cmd is None or len(cmd.f_int) == 0:
        t_ave = (input_.tmin + input_.tmax) / 2.0
        t_func = t_response(t_ave, pc.tbase_crop, pc.topt_crop, pc.tmax_crop)

        # Accumulate GDD
        output1.crop.growing_degree_days = (
            output.crop.growing_degree_days
            + t_func * (pc.topt_crop - pc.tbase_crop)
        )

        if output.crop.cycle_completion_percentage <= 100.0:
            gdd = output1.crop.growing_degree_days

            # Phenological code (1 = vegetative, 2 = reproductive)
            output1.crop.pheno_code = _pheno_code(
                gdd, pc.flowering_start / 100.0 * pc.cycle_length
            )

            # Attainable light interception
            f_int_att, senescence_started = _f_int_compute(
                pc.cycle_length, pc.slope_growth, pc.half_int_growth,
                pc.slope_senescence, pc.half_int_senescence, gdd
            )
            output1.crop.light_interception_attainable = f_int_att
            output1.crop.senescence_started = senescence_started

            # Save peak fInt at the onset of senescence
            if senescence_started and output1.crop.f_int_peak == 0.0:
                output1.crop.f_int_peak = output.crop.light_interception_attainable

            # Shift senescence half-point due to senescence accelerators
            half_int_sen_shifted = (
                pc.half_int_senescence - dm.senescence_accelerators * 100.0
            )

            # Actual light interception (with disease pressure)
            f_int_act, _ = _f_int_compute(
                pc.cycle_length, pc.slope_growth, pc.half_int_growth,
                pc.slope_senescence, half_int_sen_shifted, gdd
            )
            output1.crop.light_interception_actual = f_int_act

            # Apply light stealers
            output1.crop.light_interception_actual -= (
                output1.crop.light_interception_actual * dm.light_stealers
            )
            if output1.crop.light_interception_actual < 0.0:
                output1.crop.light_interception_actual = 0.0

            # Potential biomass accumulation
            carbon_rate_pot = _carbon_rate(
                pc.radiation_use_efficiency, input_.rad, t_func,
                output1.crop.light_interception_attainable
            ) * 10.0  # g m⁻² → kg ha⁻¹
            output1.crop.agb_attainable = output.crop.agb_attainable + carbon_rate_pot

            # Actual biomass accumulation
            carbon_rate_act = _carbon_rate(
                pc.radiation_use_efficiency
                - pc.radiation_use_efficiency * dm.rue_reducers,
                input_.rad, t_func,
                output1.crop.light_interception_actual
            ) * 10.0
            carbon_rate_act -= dm.assimilate_sappers
            output1.crop.agb_actual = output.crop.agb_actual + carbon_rate_act
            if output1.crop.agb_actual < 0.0:
                output1.crop.agb_actual = 0.0

            # Potential yield
            output1.crop.yield_attainable = output.crop.yield_attainable + _yield_rate(
                output1.crop.pheno_code, carbon_rate_pot, pc.partitioning_maximum
            )

            # Actual yield
            output1.crop.yield_actual = output.crop.yield_actual + _yield_rate(
                output1.crop.pheno_code, carbon_rate_act, pc.partitioning_maximum
            )
            if output1.crop.yield_actual < 0.0:
                output1.crop.yield_actual = 0.0

            # Days after sowing
            output1.crop.day_after_sowing = output.crop.day_after_sowing + 1

            # Cycle completion (clamped at 100 %)
            output1.crop.cycle_completion_percentage = min(
                gdd / pc.cycle_length * 100.0, 100.0
            )
        else:
            # Crop cycle complete → reset
            output1.crop = CropOutputs()
            output.crop = CropOutputs()

    # -----------------------------------------------------------------------
    # Branch B: External crop model supplies daily f_int, AGB, yield, GDD
    # -----------------------------------------------------------------------
    else:
        # Normalise date key (InputsDaily.date is datetime; dict keys are date)
        today: date = (
            input_.date.date() if isinstance(input_.date, datetime) else input_.date
        )

        if today in cmd.f_int:
            # --- Attainable values from external model ---
            output1.crop.light_interception_attainable = cmd.f_int[today]
            output1.crop.agb_attainable = cmd.agb[today]
            output1.crop.yield_attainable = cmd.yield_.get(today, 0.0)

            # --- Actual light interception ---
            f_int_att = output1.crop.light_interception_attainable
            f_int_act = (f_int_att - f_int_att * dm.light_stealers - dm.senescence_accelerators)
            if f_int_act < 0.0:
                f_int_act = 0.0
            output1.crop.light_interception_actual = f_int_act

            # Senescence flag: senescence starts when yield appears
            if cmd.yield_.get(today, 0.0) > 0.0:
                output1.crop.senescence_started = True

            # Peak fInt across entire season
            output1.crop.f_int_peak = max(cmd.f_int.values())

            # --- Daily rates (attainable) ---
            prev_day = today - timedelta(days=1)
            if prev_day in cmd.agb:
                pot_agb_rate = cmd.agb[today] - cmd.agb[prev_day]
                pot_yield_rate = (cmd.yield_.get(today, 0.0) - cmd.yield_.get(prev_day, 0.0))
            else:  # first day of season
                pot_agb_rate = cmd.agb[today]
                pot_yield_rate = cmd.yield_.get(today, 0.0)

            # --- Actual rates ---
            if f_int_att > 0.0:
                damage_ratio = (f_int_att - f_int_act) / f_int_att
                act_agb_rate = pot_agb_rate - pot_agb_rate * damage_ratio
                act_yield_rate = pot_yield_rate - pot_yield_rate * damage_ratio
            else:
                # First day, no light interception yet
                act_agb_rate = output.crop.agb_actual
                act_yield_rate = output.crop.yield_actual

            # Clamp negative rates before applying further reductions
            if act_agb_rate < 0.0:
                act_agb_rate = 0.0
            if act_yield_rate < 0.0:
                act_yield_rate = 0.0

            # Apply RUE reducers and assimilate sappers
            act_agb_rate = (
                act_agb_rate
                - act_agb_rate * dm.rue_reducers
                - dm.assimilate_sappers
            )
            act_yield_rate = (
                act_yield_rate
                - act_yield_rate * dm.rue_reducers
                - dm.assimilate_sappers
            )

            # Update state variables
            output1.crop.agb_actual = output.crop.agb_actual + act_agb_rate
            output1.crop.yield_actual = output.crop.yield_actual + act_yield_rate
            if output1.crop.agb_actual < 0.0:
                output1.crop.agb_actual = 0.0
            if output1.crop.yield_actual < 0.0:
                output1.crop.yield_actual = 0.0

            # Cycle completion from external model
            output1.crop.cycle_completion_percentage = cmd.cycle_percentage.get(today, 0.0)

            # --- FIX 1: increment day_after_sowing (never done in C# external branch) ---
            output1.crop.day_after_sowing = output.crop.day_after_sowing + 1

            # --- FIX 2: set GDD from external model (never done in C# external branch) ---
            output1.crop.growing_degree_days = cmd.gdd.get(today, 0.0)

        else:
            # Date not covered by external model → crop harvested, reset
            output1.crop = CropOutputs()
            output.crop = CropOutputs()


# ---------------------------------------------------------------------------
# Helper functions (translated directly from crop.cs)
# ---------------------------------------------------------------------------

def _light_stealers_fn(severity: float, damage: float) -> float:
    """
    Compute retained light interception fraction.

    Args:
        severity (float): Disease severity fraction.
        damage (float): Light-stealer damage coefficient.

    Returns:
        float: Retained light interception fraction, $(1 - severity)^{damage}$.
    """
    return (1.0 - severity) ** damage


def _rue_reduction_fn(severity: float, damage: float) -> float:
    """
    Compute retained radiation use efficiency (RUE) fraction.

    Args:
        severity (float): Disease severity fraction.
        damage (float): RUE reducer damage coefficient.

    Returns:
        float: Retained RUE fraction, $(1 - severity)^{damage}$.
    """
    return (1.0 - severity) ** damage


def _assimilate_sappers_fn(severity: float, damage_max: float) -> float:
    """
    Compute assimilate drain due to disease damage.

    Args:
        severity (float): Disease severity fraction.
        damage_max (float): Maximum assimilate drain.

    Returns:
        float: Assimilate drain in kg ha^-1, ``severity * damage_max``.
    """
    return severity * damage_max


def _senescence_accelerator_fn(severity: float, accel_max: float) -> float:
    """
    Compute senescence acceleration factor.

    Args:
        severity (float): Disease severity fraction.
        accel_max (float): Maximum senescence acceleration.

    Returns:
        float: Senescence acceleration fraction, ``severity * accel_max``.
    """
    return severity * accel_max


def _pheno_code(gdd: float, flowering_gdd: float) -> int:
    """
    Determine phenological phase code.

    Args:
        gdd (float): Accumulated growing degree days.
        flowering_gdd (float): GDD threshold for flowering.

    Returns:
        int: ``1`` for vegetative phase, ``2`` for reproductive phase.
    """
    return 1 if gdd < flowering_gdd else 2


def _f_int_compute(
    cycle_length: float, slope_growth: float, half_int_growth: float,
    slope_senescence: float, half_int_senescence: float, gdd: float
) -> tuple[float, bool]:
    """
    Compute light interception from logistic growth and senescence curves.

    Args:
        cycle_length (float): Crop cycle length in GDD.
        slope_growth (float): Logistic growth slope.
        half_int_growth (float): Percent of cycle length at half interception
            during growth.
        slope_senescence (float): Logistic senescence slope.
        half_int_senescence (float): Percent of cycle length at half
            interception during senescence.
        gdd (float): Current accumulated growing degree days.

    Returns:
        tuple[float, bool]:
            - Light interception fraction.
            - Whether senescence has started.
    """
    hig_gdd = cycle_length * half_int_growth / 100.0
    his_gdd = cycle_length * half_int_senescence / 100.0

    f_int_growth = 1.0 / (1.0 + math.exp(-slope_growth * (gdd - hig_gdd)))
    f_int_senescence = 1.0 / (1.0 + math.exp(slope_senescence * (gdd - his_gdd)))

    senescence_started = f_int_senescence < f_int_growth
    return min(f_int_growth, f_int_senescence), senescence_started


def _carbon_rate(rue: float, radiation: float, f_temp: float, f_int: float) -> float:
    """
    Compute daily carbon assimilation rate.

    Args:
        rue (float): Radiation use efficiency.
        radiation (float): Daily incoming radiation.
        f_temp (float): Temperature response factor.
        f_int (float): Intercepted radiation fraction.

    Returns:
        float: Carbon assimilation rate in g m^-2 d^-1. Multiply by 10 at the
        call site to convert to kg ha^-1.
    """
    return rue * f_int * radiation * 0.5 * f_temp


def _yield_rate(pheno_code: int, bio_rate_pot: float, partitioning_max: float) -> float:
    """
    Compute daily yield increment.

    Args:
        pheno_code (int): Phenological phase code.
        bio_rate_pot (float): Daily biomass accumulation rate.
        partitioning_max (float): Maximum partitioning to yield.

    Returns:
        float: Yield increment. Returns ``0.0`` outside reproductive phase.
    """
    return 0.0 if pheno_code != 2 else bio_rate_pot * partitioning_max
