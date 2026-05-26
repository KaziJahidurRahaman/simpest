"""
fungicide_model.py – Daily fungicide degradation, tenacity, and efficacy.

Translated from models/fungicide.cs.
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
    """Compute one daily fungicide degradation, tenacity, and efficacy step.

    If no treatment has been applied yet (``DateTreatmentLast.year <= 1``)
    this function is a no-op and all fungicide output fields remain at their
    default value (0).

    Args:
        input_: Today's daily inputs, including ``date``,
            ``date_treatment_last``, and ``precipitation``.
        parameters: Model parameters (uses ``par_fungicide`` sub-record).
        output: Previous day's fungicide output state (tenacity, actual
            degradation, tenacity_function).
        output1: Current day's fungicide output, modified in-place.
    """"
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
