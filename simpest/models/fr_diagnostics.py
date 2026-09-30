"""Diagnostics for FraNchEstYN runs against reference observations.

These helpers turn a `run_franchestyn` result into a DataFrame, match
reference observations to the simulated value on the same date, and
summarise the errors per season.

Seasons are keyed by sowing year. Reference observations are keyed by
calendar ``year`` and ``DOY``, and a season's observations fall in its harvest
year (sowing year + 1).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


def simulation_frame(result: dict, severity_percent: bool = True) -> pd.DataFrame:
    """Turn a `run_franchestyn` result into a daily DataFrame.

    Args:
        result: The dict returned by `run_franchestyn`.
        severity_percent: Convert ``DiseaseSeverity`` from a 0-1 fraction to
            percent. Build any `build_season_summary` from the fraction first:
            it expects 0-1.

    Returns:
        pd.DataFrame: One row per simulated day, with ``Date`` parsed.
    """
    sim = pd.DataFrame(result["outputs"]["simulation"])
    sim["Date"] = pd.to_datetime(sim["Date"], dayfirst=True)
    if severity_percent:
        sim["DiseaseSeverity"] = sim["DiseaseSeverity"] * 100.0
    return sim


def reference_dates(ref_df: pd.DataFrame) -> pd.Series:
    """Calendar dates of reference rows keyed by ``year`` and ``DOY``."""
    return pd.to_datetime(ref_df["year"].astype(str)) + pd.to_timedelta(ref_df["DOY"] - 1, unit="D")


def match_ref_to_sim(
    sim_df: pd.DataFrame,
    ref_df: pd.DataFrame,
    seasons: Sequence[int],
    ref_col: str,
    sim_col: str,
    sim_scale: float = 1.0,
) -> pd.DataFrame:
    """Match reference observations to same-date simulated values, per season.

    Observations on dates the season does not simulate are dropped.

    Args:
        sim_df: Daily simulation from `simulation_frame`.
        ref_df: Reference observations with ``year``, ``DOY`` and ``ref_col``.
        seasons: Sowing years to match.
        ref_col: Observed column in ``ref_df``.
        sim_col: Simulated column in ``sim_df``.
        sim_scale: Factor applied to the simulated values.

    Returns:
        pd.DataFrame: One row per matched observation with ``season``,
        ``date``, ``observed``, ``simulated`` and ``error``
        (simulated - observed).
    """
    rows = []
    for sow_year in seasons:
        harvest_yr = sow_year + 1
        sub = sim_df[sim_df["GrowingSeason"] == sow_year].set_index("Date")
        ref_sub = ref_df[(ref_df["year"] == harvest_yr) & ref_df[ref_col].notna()]
        if ref_sub.empty:
            continue
        for date, obs in zip(reference_dates(ref_sub), ref_sub[ref_col]):
            if date in sub.index:
                rows.append({
                    "season": sow_year,
                    "date": date,
                    "observed": obs,
                    "simulated": sub.loc[date, sim_col] * sim_scale,
                })
    out = pd.DataFrame(rows, columns=["season", "date", "observed", "simulated"])
    out["error"] = out["simulated"] - out["observed"]
    return out


def season_rmse(err_df: pd.DataFrame) -> pd.Series:
    """Root-mean-square error per season of a `match_ref_to_sim` table."""
    return err_df.groupby("season")["error"].apply(lambda e: np.sqrt((e ** 2).mean()))


def error_stats(err_df: pd.DataFrame) -> dict:
    """Overall ``n``, ``RMSE`` and ``MAE`` of a `match_ref_to_sim` table."""
    err = err_df["error"]
    return {"n": len(err), "RMSE": float(np.sqrt((err ** 2).mean())), "MAE": float(err.abs().mean())}


def first_date(dates, values, threshold: float) -> str:
    """Month and day (``"mm-dd"``) values first reach ``threshold``, or ``"-"``."""
    s = pd.Series(np.asarray(values), index=pd.DatetimeIndex(dates))
    d = s[s >= threshold].index.min()
    return d.strftime("%m-%d") if pd.notna(d) else "-"


def days_between(dates, values, lo: float, hi: float) -> float:
    """Days from values first reaching ``lo`` to first reaching ``hi``.

    Returns:
        float: The number of days, or ``NaN`` if either level is never reached.
    """
    s = pd.Series(np.asarray(values), index=pd.DatetimeIndex(dates))
    a, b = s[s >= lo].index.min(), s[s >= hi].index.min()
    return (b - a).days if pd.notna(a) and pd.notna(b) else np.nan


def season_diagnostics(
    sim_df: pd.DataFrame,
    ref_df: pd.DataFrame,
    severity_err: pd.DataFrame,
    seasons: Sequence[int],
    disease_col: str,
) -> pd.DataFrame:
    """Per-season fit and weather summary of a disease simulation.

    ``n_obs_after_season`` counts observations after the simulated season
    ends; the calibration objective never sees them. Fully-wet days (24 wet
    hours) are mostly an artifact of daily weather, where a day's rain is
    spread evenly over 24 h.

    Args:
        sim_df: Daily simulation from `simulation_frame` (severity in percent).
        ref_df: Reference observations.
        severity_err: `match_ref_to_sim` table for disease severity.
        seasons: Sowing years to summarise.
        disease_col: Observed severity column in ``ref_df``.

    Returns:
        pd.DataFrame: One row per season (``"1972/73"`` style index) with
        ``rmse``, ``bias``, ``peak_sim``, ``peak_obs``, ``sim_end``,
        ``last_obs``, ``n_obs``, ``n_obs_after_season``, ``wet_hours_AMJ``
        and ``fully_wet_days_AMJ`` (April-June).
    """
    ref_sev = ref_df[ref_df[disease_col].notna()].copy()
    ref_sev["Date"] = reference_dates(ref_sev)

    rows = []
    for sow_year in seasons:
        sim = sim_df[sim_df["GrowingSeason"] == sow_year]
        obs = ref_sev[ref_sev["year"] == sow_year + 1]
        err = severity_err.loc[severity_err["season"] == sow_year, "error"]
        amj = sim[sim["Date"].dt.month.isin([4, 5, 6])]
        sim_end = sim["Date"].max()
        rows.append({
            "season": f"{sow_year}/{str(sow_year + 1)[2:]}",
            "rmse": np.sqrt((err ** 2).mean()),
            "bias": err.mean(),
            "peak_sim": sim["DiseaseSeverity"].max(),
            "peak_obs": obs[disease_col].max(),
            "sim_end": sim_end.date(),
            "last_obs": obs["Date"].max().date(),
            "n_obs": len(obs),
            "n_obs_after_season": int((obs["Date"] > sim_end).sum()),
            "wet_hours_AMJ": amj["TotalLW"].sum(),
            "fully_wet_days_AMJ": int((amj["TotalLW"] >= 24).sum()),
        })
    return pd.DataFrame(rows).set_index("season")
