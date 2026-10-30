"""Build FraNchEstYN's input DataFrames from an external crop model's output.

`franchestyn.run_franchestyn` takes weather, management and crop-model
DataFrames. When crop growth comes from an external model run in Python, its
output arrives as per-season arrays; the builders here turn those into the
frames FraNchEstYN reads. They need only NumPy and pandas, so any crop model
can use them.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

WEATHER_MODES = ("daily", "daily_rh", "hourly")
"""Weather inputs `build_weather_df` can build for the disease model."""


def build_crop_model_df(
    agb,
    yield_,
    fint,
    gdd,
    dvs,
    sow_years: Sequence[int],
    sowing_doys: Sequence[int],
    include_dvs: bool = False,
) -> pd.DataFrame:
    """Build FraNchEstYN's external crop-model series from per-season arrays.

    Every array is ``[B, T]``: one row per season, one column per day, with
    day 0 on 1 January of that season's sowing year, so the sowing day sits
    at index ``sowing_doy - 1``. Each season runs from sowing to the first day
    ``dvs`` reaches 2 (maturity), and the seasons are concatenated into one
    multi-year series.

    Args:
        agb: Above-ground biomass, kg ha-1.
        yield_: Grain yield, kg ha-1.
        fint: Fraction of light intercepted, 0-1.
        gdd: Thermal time since emergence, °C d.
        dvs: Development stage (2 = maturity).
        sow_years: Sowing year of each season.
        sowing_doys: Sowing day of year of each season.
        include_dvs: Also write a ``dvs`` column.

    Returns:
        pd.DataFrame: Columns ``year, doy, agb, yield, fint, gdd[, dvs], season``,
        sorted by date.

    Raises:
        ValueError: If a season never reaches maturity.
    """
    agb, yield_, fint, gdd, dvs = (np.asarray(a) for a in (agb, yield_, fint, gdd, dvs))

    rows = []
    for b, (sow_year, sowing_doy) in enumerate(zip(sow_years, sowing_doys)):
        mature = np.where(dvs[b] >= 2.0)[0]
        if len(mature) == 0:
            raise ValueError(f"season {sow_year} never reaches maturity (DVS 2)")
        sow_idx, maturity_idx = int(sowing_doy) - 1, mature[0]
        span = slice(sow_idx, maturity_idx + 1)
        dates = pd.Timestamp(year=int(sow_year), month=1, day=1) + pd.to_timedelta(
            np.arange(sow_idx, maturity_idx + 1), unit="D")
        season = {
            "year": dates.year,
            "doy": dates.dayofyear,
            "agb": agb[b, span],
            "yield": yield_[b, span],
            "fint": fint[b, span],
            "gdd": gdd[b, span],
        }
        if include_dvs:
            season["dvs"] = dvs[b, span]
        season["season"] = int(sow_year)
        rows.append(pd.DataFrame(season))

    return pd.concat(rows, ignore_index=True).sort_values(["year", "doy"]).reset_index(drop=True)


def build_weather_df(
    weather_raw: pd.DataFrame,
    site: str,
    mode: str = "daily",
    weather_hourly: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build the weather input for FraNchEstYN.

    Args:
        weather_raw: Daily weather with ``Date``, ``TempMax``, ``TempMin``,
            ``Precipitation`` and ``Radiation`` (kJ m-2 d-1) columns, and
            optionally ``RHmax`` and ``RHmin`` [%].
        site: Site name written into the frame.
        mode: ``"daily"`` (Tmax/Tmin/rain/radiation only), ``"daily_rh"``
            (daily plus daily RH extremes) or ``"hourly"`` (the hourly
            records as they are). In ``"daily_rh"`` the extremes come from
            ``weather_raw``'s ``RHmax``/``RHmin`` columns, so every value
            comes from the same daily table; without those columns they are
            taken from the hourly records.
        weather_hourly: Hourly records with ``date``, ``rh`` and the other
            hourly columns FraNchEstYN reads. Required for ``"hourly"``, and
            for ``"daily_rh"`` when ``weather_raw`` has no RH columns.

    Returns:
        pd.DataFrame: The frame to pass as ``weather_df``. Set
        ``FranchestynConfig.weather_time_step`` to ``"hourly"`` for the
        ``"hourly"`` mode and to ``"daily"`` otherwise.

    Raises:
        ValueError: If ``mode`` is unknown or hourly records are missing.
    """
    if mode not in WEATHER_MODES:
        raise ValueError(f"unknown weather mode {mode!r}; expected one of {WEATHER_MODES}")
    has_rh = {"RHmax", "RHmin"} <= set(weather_raw.columns)
    if weather_hourly is None and (mode == "hourly" or (mode == "daily_rh" and not has_rh)):
        raise ValueError(f"weather mode {mode!r} needs weather_hourly or RHmax/RHmin columns")
    if mode == "hourly":
        return weather_hourly

    weather_daily = pd.DataFrame({
        "site":  site,
        "year":  weather_raw["Date"].dt.year,
        "month": weather_raw["Date"].dt.month,
        "day":   weather_raw["Date"].dt.day,
        "tx":    weather_raw["TempMax"],
        "tn":    weather_raw["TempMin"],
        "p":     weather_raw["Precipitation"],
        "rad":   weather_raw["Radiation"] / 1000.0,   # kJ -> MJ m-2 d-1
    })
    if mode == "daily":
        return weather_daily

    if has_rh:
        return weather_daily.assign(rhx=weather_raw["RHmax"].values, rhn=weather_raw["RHmin"].values)
    rh = weather_hourly.groupby("date")["rh"].agg(["max", "min"])
    rh.index = pd.to_datetime(rh.index)
    return weather_daily.assign(
        rhx=rh["max"].reindex(weather_raw["Date"]).values,
        rhn=rh["min"].reindex(weather_raw["Date"]).values,
    )


def build_management_df(
    sowing: pd.DataFrame,
    site: str,
    crop: str,
    variety: str,
    year_col: str = "location",
    sowing_doy_col: str = "IDPL",
) -> pd.DataFrame:
    """Build FraNchEstYN's management (sowing) input.

    Args:
        sowing: One row per season with the sowing year and sowing day of
            year, as columns or as the index (e.g. a ``site.csv`` table
            indexed by ``location``).
        site: Site name.
        crop: Crop type.
        variety: Variety name.
        year_col: Name of the sowing-year column or index.
        sowing_doy_col: Name of the sowing-day column.

    Returns:
        pd.DataFrame: Columns ``site, crop, variety, year, sowingDOY``.
    """
    management_df = sowing.reset_index() if year_col not in sowing.columns else sowing.copy()
    management_df = management_df.rename(columns={year_col: "year", sowing_doy_col: "sowingDOY"})
    management_df["site"] = site
    management_df["crop"] = crop
    management_df["variety"] = variety
    return management_df[["site", "crop", "variety", "year", "sowingDOY"]]
