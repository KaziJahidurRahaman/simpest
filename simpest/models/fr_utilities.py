"""
utilities.py – Biophysical helper functions for the FraNchEstYN model.

Translated from utilities.cs.

ReadFileOrExitsParameters was intentionally skipped in utilities.py. The C# ReadFileOrExitsParameters 
in utilities.cs is just a thin wrapper: In the Python rewrite, calibrated_read() was translated 
directly into param_reader.py:87 as a standalone function. Since the wrapper added no logic 
(the variety argument was even noted as "currently unused"), there was no need to reproduce 
it in utilities.py.

"""

from __future__ import annotations
import math


def t_response(t_ave: float, t_base: float, t_opt: float, t_max: float) -> float:
    """Beta-shaped temperature response function.

    Returns a dimensionless efficiency in [0, 1] as a function of the average
    temperature relative to the cardinal temperatures.

    When ``t_base < t_ave < t_max`` the formula is:

    .. code-block:: none

        f = ((t_max - t_ave) / (t_max - t_opt))
            * ((t_ave - t_base) / (t_opt - t_base)) ** ((t_opt - t_base) / (t_max - t_opt))

    Args:
        t_ave: Average (or mean) daily temperature (\u00b0C).
        t_base: Minimum (base) cardinal temperature (\u00b0C).
        t_opt: Optimum cardinal temperature (\u00b0C).
        t_max: Maximum cardinal temperature (\u00b0C).

    Returns:
        Dimensionless temperature efficiency in [0, 1]; 0.0 outside
        ``[t_base, t_max]``.
    """
    if t_ave <= t_base or t_ave >= t_max:
        return 0.0

    first_term = (t_max - t_ave) / (t_max - t_opt)
    second_term = (t_ave - t_base) / (t_opt - t_base)
    exponent = (t_opt - t_base) / (t_max - t_opt)

    return first_term * math.pow(second_term, exponent)


def rain_detachment(rainfall: float, rain50: float, f_int: float) -> float:
    """Rain-driven spore detachment index (dimensionless, 0\u20131).

    Saturates as rainfall increases relative to the capacity term
    ``rain50 \u00d7 f_int``. Formula:

    .. code-block:: none

        detachment = rainfall / (rain50 * f_int + rainfall)

    Args:
        rainfall: Precipitation (mm) over the time step.
        rain50: Half-saturation parameter (mm) \u2014 rainfall giving ~0.5 when
            ``f_int = 1``.
        f_int: Light interception fraction [0, 1].

    Returns:
        Detachment index in [0, 1], or 0.0 when the denominator is zero.
    """
    denominator = (rain50 * f_int) + rainfall
    if denominator == 0.0:
        return 0.0
    return rainfall / denominator
