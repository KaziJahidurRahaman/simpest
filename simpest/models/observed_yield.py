"""Observed yield statistics: reading, trends, detrending and comparison.

Observed county yields rise over the decades as varieties, fertiliser and
disease control improve. A crop model run with one fixed variety and
management has no such trend, so observed yields must be detrended before
they are compared with, or used to calibrate, simulated yields.

The functions here read USDA NASS yield exports, fit a linear or two-segment
linear trend, express observed yields at one technology level and line them
up with simulated yields season by season. `measured_disease_loss` turns
paired fungicide-treated and untreated plot yields into a measured disease
loss, to check a simulated relative yield loss.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

BU_ACRE_TO_KG_HA = 60 * 0.45359237 / 0.40468564
"""Wheat bushels per acre to kg per hectare (60 lb per bushel, ≈ 67.25)."""

GRAIN_DRY_FRACTION = 1 - 0.135
"""Dry-matter fraction of grain at the 13.5 % moisture NASS reports yield at."""


def read_nass_yield(path) -> pd.DataFrame:
    """Read a USDA NASS QuickStats yield export.

    The export has one row per year with ``Year`` (the harvest year) and
    ``Value`` (yield in bu/acre) columns.

    Args:
        path: Path to the CSV export.

    Returns:
        pd.DataFrame: Columns ``harvest_year``, ``yield_bu_acre`` and
        ``yield_kg_ha``, sorted by harvest year.
    """
    raw = pd.read_csv(path)
    out = raw[["Year", "Value"]].rename(columns={"Year": "harvest_year", "Value": "yield_bu_acre"})
    out["yield_kg_ha"] = out["yield_bu_acre"] * BU_ACRE_TO_KG_HA
    return out.sort_values("harvest_year").reset_index(drop=True)


@dataclass
class TwoSegmentTrend:
    """A continuous two-segment ("hinge") linear trend.

    The trend has one slope before ``breakpoint`` and another after it, and
    the two segments meet at the breakpoint.

    Attributes:
        breakpoint: Year where the slope changes.
        slope_before: Slope before the breakpoint, per year.
        slope_after: Slope after the breakpoint, per year.
        level_at_breakpoint: Trend value at the breakpoint.
        rmse: Root-mean-square error of the fit.
        rmse_single_line: Root-mean-square error of a single straight line
            fitted to the same data. The breakpoint is only worth reading into
            if ``rmse`` is clearly lower.
    """

    breakpoint: int
    slope_before: float
    slope_after: float
    level_at_breakpoint: float
    rmse: float
    rmse_single_line: float

    def __call__(self, years) -> np.ndarray:
        """Evaluate the trend at ``years``."""
        dx = np.asarray(years, dtype=float) - self.breakpoint
        return (self.level_at_breakpoint + self.slope_before * dx
                + (self.slope_after - self.slope_before) * np.clip(dx, 0, None))


@dataclass
class LinearTrend:
    """A single straight-line trend.

    Use it when the record is too short for `fit_two_segment_trend`.

    Attributes:
        slope: Slope, per year.
        intercept: Trend value in year 0.
        rmse: Root-mean-square error of the fit.
    """

    slope: float
    intercept: float
    rmse: float

    def __call__(self, years) -> np.ndarray:
        """Evaluate the trend at ``years``."""
        return self.intercept + self.slope * np.asarray(years, dtype=float)


def fit_linear_trend(years, values) -> LinearTrend:
    """Fit a straight-line trend by least squares.

    Args:
        years: 1-D array of years.
        values: 1-D array of values, same length as ``years``.

    Returns:
        LinearTrend: The fitted trend.
    """
    x = np.asarray(years, dtype=float)
    y = np.asarray(values, dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    return LinearTrend(
        slope=float(slope),
        intercept=float(intercept),
        rmse=float(np.sqrt(np.mean((y - (intercept + slope * x)) ** 2))),
    )


def fit_two_segment_trend(years, values, min_segment_years: int = 10) -> TwoSegmentTrend:
    """Fit a two-segment linear trend by breakpoint grid search.

    Every whole year that leaves at least ``min_segment_years`` years on each
    side is tried as the breakpoint; the one with the lowest sum of squared
    errors wins.

    Args:
        years: 1-D array of years.
        values: 1-D array of values, same length as ``years``.
        min_segment_years: Minimum number of years on each side of the
            breakpoint.

    Returns:
        TwoSegmentTrend: The fitted trend.

    Raises:
        ValueError: If the record is too short for two segments.
    """
    x = np.asarray(years, dtype=float)
    y = np.asarray(values, dtype=float)
    candidates = range(int(x.min()) + min_segment_years, int(x.max()) - min_segment_years + 1)
    if len(candidates) == 0:
        raise ValueError(
            f"{int(x.min())}-{int(x.max())} is too short for two segments of "
            f"{min_segment_years} years")

    best = None
    for bp in candidates:
        X = np.column_stack([np.ones_like(x), x - bp, np.clip(x - bp, 0, None)])
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        sse = np.sum((y - X @ coef) ** 2)
        if best is None or sse < best[0]:
            best = (sse, bp, coef)
    sse, bp, (b0, b1, b2) = best

    line = np.polyval(np.polyfit(x, y, 1), x)
    return TwoSegmentTrend(
        breakpoint=int(bp),
        slope_before=float(b1),
        slope_after=float(b1 + b2),
        level_at_breakpoint=float(b0),
        rmse=float(np.sqrt(sse / len(y))),
        rmse_single_line=float(np.sqrt(np.mean((y - line) ** 2))),
    )


def compare_with_potential(
    obs_yield: pd.DataFrame,
    sow_years: Sequence[int],
    potential_kg_ha,
    trend: TwoSegmentTrend | LinearTrend,
    ref_harvest_year: int | None = None,
    dry_fraction: float = GRAIN_DRY_FRACTION,
) -> pd.DataFrame:
    """Line up observed and simulated yields per season.

    Each season is matched to its harvest year (sowing year + 1). Observed
    yields are converted to dry matter and detrended to the technology level
    of ``ref_harvest_year``::

        detrended = observed - trend(harvest_year) + trend(ref_harvest_year)

    with ``observed`` and ``trend`` both in dry matter.

    Args:
        obs_yield: Observed yields from `read_nass_yield`.
        sow_years: Sowing year of each simulated season.
        potential_kg_ha: Simulated yield per season, kg ha-1 dry matter.
        trend: Trend fitted to ``obs_yield`` (at reported moisture).
        ref_harvest_year: Technology level to detrend to. Defaults to the last
            simulated harvest year.
        dry_fraction: Dry-matter fraction of the reported yields.

    Returns:
        pd.DataFrame: One row per season with ``sow_year``, ``harvest_year``,
        ``potential``, ``observed`` (dry matter, as reported),
        ``observed_detrended`` and ``yield_gap_ratio``
        (``observed_detrended / potential``). Seasons without an observation
        have ``NaN`` observed values.
    """
    sow_years = np.asarray(sow_years, dtype=int)
    if ref_harvest_year is None:
        ref_harvest_year = int(sow_years.max()) + 1

    cmp_df = pd.DataFrame({"sow_year": sow_years, "harvest_year": sow_years + 1})
    cmp_df["potential"] = np.asarray(potential_kg_ha, dtype=float)
    obs_by_year = obs_yield.set_index("harvest_year")["yield_kg_ha"] * dry_fraction
    cmp_df["observed"] = cmp_df["harvest_year"].map(obs_by_year)
    cmp_df["observed_detrended"] = (
        cmp_df["observed"] - trend(cmp_df["harvest_year"]) * dry_fraction
        + trend([ref_harvest_year])[0] * dry_fraction
    )
    cmp_df["yield_gap_ratio"] = cmp_df["observed_detrended"] / cmp_df["potential"]
    return cmp_df


def fit_stats(simulated, observed) -> dict:
    """Goodness of fit of simulated against observed values.

    Args:
        simulated: Simulated values.
        observed: Observed values, same length.

    Returns:
        dict: ``RMSE``, ``bias`` (mean of simulated - observed) and Pearson ``r``.
    """
    sim = np.asarray(simulated, dtype=float)
    obs = np.asarray(observed, dtype=float)
    err = sim - obs
    return {"RMSE": float(np.sqrt(np.mean(err ** 2))), "bias": float(err.mean()),
            "r": float(np.corrcoef(sim, obs)[0, 1])}


def measured_disease_loss(
    plots: pd.DataFrame,
    treatment_col: str,
    treated: str,
    untreated: str,
    by: str = "n_rate_kg_ha",
    yield_col: str = "yield_kg_ha",
) -> pd.DataFrame:
    """Measured disease yield loss from paired treated and untreated plots.

    Within each group (for example each nitrogen rate), the loss is the drop
    in mean yield from the disease-protected (e.g. fungicide-treated) plots
    to the untreated plots::

        loss_pct = 100 * (1 - untreated_mean / treated_mean)

    The ratio cancels the yield level of the plots, so it can be compared with
    a simulated relative yield loss whatever the simulated attainable yield.
    Pass plots of one cultivar and one season, at one grain moisture. If the
    protected plots still had some disease, the loss is a lower bound.

    Args:
        plots: Plot yields, one row per plot.
        treatment_col: Column that marks the disease treatment of each plot.
        treated: Value of ``treatment_col`` for the protected plots.
        untreated: Value of ``treatment_col`` for the untreated plots.
        by: Column to group the plots by before pairing.
        yield_col: Column with the plot yield.

    Returns:
        pd.DataFrame: One row per group with ``by``, ``treated_mean``,
        ``untreated_mean``, ``n_treated``, ``n_untreated`` and ``loss_pct``.
        Groups missing either treatment are left out.
    """
    stats = (
        plots[plots[treatment_col].isin([treated, untreated])]
        .groupby([by, treatment_col])[yield_col]
        .agg(["mean", "count"])
        .unstack(treatment_col)
        .dropna()
    )
    out = pd.DataFrame({
        "treated_mean": stats[("mean", treated)],
        "untreated_mean": stats[("mean", untreated)],
        "n_treated": stats[("count", treated)].astype(int),
        "n_untreated": stats[("count", untreated)].astype(int),
    })
    out["loss_pct"] = 100.0 * (1.0 - out["untreated_mean"] / out["treated_mean"])
    return out.reset_index()
