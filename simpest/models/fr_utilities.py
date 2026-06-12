"""Biophysical helper functions shared across the model.

This module provides small, stateless functions that encode reusable
biophysical relationships, such as the cardinal-temperature response used by
both the crop and disease sub-models and the rainfall-driven spore detachment
index used by the disease model.
"""

from __future__ import annotations
import math


def t_response(t_ave: float, t_base: float, t_opt: float, t_max: float) -> float:
    """Beta-shaped cardinal-temperature response function.

    Returns a dimensionless growth or development efficiency in the range
    ``[0, 1]`` that peaks at the optimum temperature and falls to zero at the
    base and maximum temperatures. For ``t_base < t_ave < t_max`` the response is

    $$
    f = \\frac{t_{max} - t_{ave}}{t_{max} - t_{opt}}
        \\left(\\frac{t_{ave} - t_{base}}{t_{opt} - t_{base}}\\right)
        ^{\\frac{t_{opt} - t_{base}}{t_{max} - t_{opt}}}
    $$

    and ``0`` outside the interval ``[t_base, t_max]``.

    Args:
        t_ave (float): Average temperature for the step (°C).
        t_base (float): Base (minimum) cardinal temperature (°C).
        t_opt (float): Optimum cardinal temperature (°C).
        t_max (float): Maximum cardinal temperature (°C).

    Returns:
        float: Temperature response factor in ``[0, 1]``.
    """
    if t_ave <= t_base or t_ave >= t_max:
        return 0.0

    first_term = (t_max - t_ave) / (t_max - t_opt)
    second_term = (t_ave - t_base) / (t_opt - t_base)
    exponent = (t_opt - t_base) / (t_max - t_opt)

    return first_term * math.pow(second_term, exponent)


def rain_detachment(rainfall: float, rain50: float, f_int: float) -> float:
    """Rain-driven spore detachment index.

    Returns a dimensionless index in ``[0, 1]`` that saturates as rainfall
    increases relative to the canopy-scaled half-saturation term:

    $$
    \\text{detachment} = \\frac{\\text{rainfall}}
                              {\\text{rain50} \\cdot f_{int} + \\text{rainfall}}
    $$

    Args:
        rainfall (float): Precipitation over the time step (mm).
        rain50 (float): Half-saturation parameter (mm); the rainfall that yields
            an index of about 0.5 at full canopy (``f_int = 1``).
        f_int (float): Light interception fraction in ``[0, 1]``.

    Returns:
        float: Detachment index in ``[0, 1]``; ``0.0`` when rainfall is zero or
        the denominator is zero.
    """
    denominator = (rain50 * f_int) + rainfall
    if denominator == 0.0:
        return 0.0
    return rainfall / denominator
