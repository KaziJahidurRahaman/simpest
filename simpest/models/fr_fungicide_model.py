"""Daily fungicide degradation, tenacity, and efficacy.

This module advances the fungicide state by one day. After an application, the
active ingredient decays through two mechanisms: first-order chemical
degradation over time and rainfall-driven wash-off (tenacity). The resulting
effective dose is mapped to a protective efficacy through a logistic response,
and the treatment is retired after a fixed persistence window.
"""

from __future__ import annotations
import math
from datetime import datetime

from .fr_data import InputsDaily, Parameters, Outputs


_MAX_DAYS = 30  # hard stop: fungicide expires after 30 days


def run(
    input_: InputsDaily,
    parameters: Parameters,
    output: Outputs,
    output1: Outputs,
) -> None:
    """Advance the fungicide state by one day.

    Updates the concentration factor, tenacity, degradation, and protective
    efficacy for the current day. The calculation proceeds in stages:

    1. **Concentration.** On the application day the concentration factor is
       reset to 1; thereafter it decays exponentially with the previous day's
       effective degradation.
    2. **Potential degradation.** First-order decay of the initial dose,
       ``initial_dose · exp(-degradation_rate · days)``.
    3. **Tenacity.** Rainfall wash-off reduces the retained fraction by
       ``exp(-tenacity_factor · concentration · √precipitation)``, applied
       cumulatively across days.
    4. **Effective degradation.** The product of tenacity and potential
       degradation, i.e. the dose still active on the canopy.
    5. **Efficacy.** A logistic response of the effective degradation, scaled by
       the initial efficacy.

    If no treatment has yet been applied (``date_treatment_last.year <= 1``) the
    call is a no-op. All state is cleared once the persistence window of
    ``_MAX_DAYS`` days has elapsed.

    Args:
        input_ (InputsDaily): Daily inputs, including date, precipitation, and
            the last treatment date.
        parameters (Parameters): Model parameters, including the fungicide group.
        output (Outputs): Previous day's output state.
        output1 (Outputs): Current day's output state, updated in place.

    Returns:
        None: The function mutates ``output1`` in place.
    """
    pf = parameters.par_fungicide
    last = input_.date_treatment_last

    # Sentinel: DateTreatmentLast.year == 1 means no treatment yet
    if last.year <= 1:
        return

    # Days since last application
    days = (input_.date - last).total_seconds() / 86_400.0

    # -----------------------------------------------------------------------
    # Concentration factor
    # -----------------------------------------------------------------------
    if input_.date == last:
        # Application day: baseline values
        output1.fungicide.concentration_factor = 1.0
        output1.fungicide.tenacity_function = 1.0
        output.fungicide.tenacity = 1.0
    else:
        # After application: exponential decay based on yesterday's actual degradation
        output1.fungicide.concentration_factor = math.exp(
            (output.fungicide.actual_degradation - 1.0) * 3.0
        )

    # -----------------------------------------------------------------------
    # Potential degradation (first-order decay of initial dose)
    # -----------------------------------------------------------------------
    output1.fungicide.potential_degradation = pf.initial_dose * math.exp(
        -pf.degradation_rate * days
    )

    # -----------------------------------------------------------------------
    # Tenacity (rainfall-driven wash-off)
    # -----------------------------------------------------------------------
    output.fungicide.tenacity_function = math.exp(
        -pf.tenacity_factor
        * output1.fungicide.concentration_factor
        * math.sqrt(max(input_.precipitation, 0.0))
    )
    output1.fungicide.tenacity = (
        output.fungicide.tenacity * output.fungicide.tenacity_function
    )

    # -----------------------------------------------------------------------
    # Actual degradation
    # -----------------------------------------------------------------------
    output1.fungicide.actual_degradation = (
        output1.fungicide.tenacity * output1.fungicide.potential_degradation
    )

    # -----------------------------------------------------------------------
    # Efficacy (logistic response to actual degradation)
    # -----------------------------------------------------------------------
    output1.fungicide.efficacy = pf.initial_efficacy / (
        1.0 + math.exp(
            pf.a_shape_parameter
            - pf.b_shape_parameter * output1.fungicide.actual_degradation
        )
    )

    # -----------------------------------------------------------------------
    # Hard stop: zero everything after 30 days
    # -----------------------------------------------------------------------
    if days >= _MAX_DAYS:
        output1.fungicide.efficacy               = 0.0
        output1.fungicide.actual_degradation     = 0.0
        output1.fungicide.concentration_factor   = 0.0
        output1.fungicide.potential_degradation  = 0.0
        output1.fungicide.tenacity_function      = 0.0
