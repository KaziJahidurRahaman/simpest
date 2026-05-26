"""FraNchEstYN model orchestration for the simpest pipeline.

This module exposes the public configuration dataclass, the main simulation
entrypoint, and CSV output helpers for season summary, simulation results,
and calibrated parameters.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from .fr_optimizer import FranchestynOptimizer
from .fr_runner import FranchestynRunner


def _resolve_default_reference() -> str:
    """Resolve a local default reference CSV using repository files only."""
    repo_root = Path(__file__).resolve().parents[2]
    candidates = [
        repo_root
        / "franchestyn"
        / "src_csharp"
        / "FraNchEstYN"
        / "FraNchEstYN"
        / "files"
        / "reference"
        / "Indiana.csv",
        repo_root
        / "franchestyn"
        / "src_csharp"
        / "FraNchEstYN"
        / "FraNchEstYN"
        / "bin"
        / "Debug"
        / "net8.0"
        / "files"
        / "reference"
        / "Indiana.csv",
    ]

    for path in candidates:
        if path.exists():
            return str(path)

    # Fall back to a repository-local canonical path if files are not present yet.
    return str(candidates[0])


def _resolve_local_model_file(filename: str) -> str:
    """Resolve a filename relative to this module's directory.

    Args:
        filename: Bare filename to resolve (e.g. ``'fr_crop_parameters.json'``).

    Returns:
        Absolute path string to the file alongside this module.
    """
    return str(Path(__file__).with_name(filename))


@dataclass(frozen=True)
class FranchestynConfig:
    """Configuration for a FraNchEstYN simulation run.

    Attributes:
        param_file: Path to the legacy franchestyn parameters CSV. Leave empty
            when using modular JSON parameter files.
        crop_param_file: Path to ``fr_crop_parameters.json``. Defaults to the
            bundled file.
        disease_param_file: Path to ``fr_disease_parameters.json``. Defaults
            to the bundled file.
        fungicide_param_file: Path to ``fr_fungicide_parameters.json``. Defaults
            to the bundled file.
        reference_path: Path to the reference CSV used for calibration.
        crop_type: Crop type identifier (e.g. ``'wheat'``).
        disease_type: Disease type identifier (e.g. ``'septoria'``).
        fungicide_type: Fungicide type identifier (e.g. ``'protectant'``), or
            ``None`` to disable the fungicide sub-model.
        site: Site identifier matching the sowing and reference CSV.
        variety: Variety identifier matching the sowing and reference CSV.
        disease: Disease column name in the reference CSV.
        is_calibration: If ``True``, runs the Nelder-Mead optimizer before
            the final simulation.
        calibration_variable: ``'crop'``, ``'disease'``, or ``'all'``.
        use_gdd: If ``True``, derives cycle completion from GDD (recommended).
        n_restarts: Number of optimizer multi-start restarts.
        max_iter: Maximum optimizer iterations per restart.
    """
    param_file: str = ""
    crop_param_file: str = field(default_factory=lambda: _resolve_local_model_file("fr_crop_parameters.json"))
    disease_param_file: str = field(default_factory=lambda: _resolve_local_model_file("fr_disease_parameters.json"))
    fungicide_param_file: str = field(default_factory=lambda: _resolve_local_model_file("fr_fungicide_parameters.json"))
    reference_path: str = _resolve_default_reference()
    crop_type: str = "wheat"
    disease_type: str = "septoria"
    fungicide_type: str | None = "protectant"
    site: str = "indiana"
    variety: str = "Generic"
    disease: str = "thisDisease"
    is_calibration: bool = True
    calibration_variable: str = "all"
    use_gdd: bool = True
    n_restarts: int = 1
    max_iter: int = 100


def _outputs_to_records(date_outputs):
    """Convert daily simulation outputs to a list of flat record dicts.

    Args:
        date_outputs: Dict mapping ``datetime`` to ``Outputs`` as returned by
            ``FranchestynRunner.run()``.

    Returns:
        List of dicts with one entry per simulated day, suitable for
        constructing a ``pandas.DataFrame``.
    """
    records = []
    for dt, out in sorted(date_outputs.items()):
        records.append(
            {
                "Date": dt.strftime("%d/%m/%Y"),
                "DaysAfterSowing": out.crop.day_after_sowing,
                "GrowingDegreeDays": out.crop.growing_degree_days,
                "CycleCompletionPercentage": out.crop.cycle_completion_percentage,
                "LightInterception": out.crop.light_interception_attainable,
                "LightIntHealthy": out.crop.light_interception_actual,
                "AGBattainable": out.crop.agb_attainable,
                "AGBactual": out.crop.agb_actual,
                "YieldAttainable": out.crop.yield_attainable,
                "YieldActual": out.crop.yield_actual,
                "HTtimeRinoculum": out.disease.hydro_thermal_time_rate,
                "HTtimeSinoculum": out.disease.hydro_thermal_time_state,
                "Susceptible": out.disease.susceptible_fraction,
                "Latent": out.disease.latent_sum,
                "Sporulating": out.disease.sporulating_sum,
                "Affected": out.disease.affected_sum,
                "Dead": out.disease.dead_sum,
                "DiseaseSeverity": out.disease.disease_severity,
                "FungicideEfficacy": out.fungicide.efficacy,
                "Tmax": out.inputs_daily.tmax if out.inputs_daily else None,
                "Tmin": out.inputs_daily.tmin if out.inputs_daily else None,
                "RHx": out.inputs_daily.rhx if out.inputs_daily else None,
                "RHn": out.inputs_daily.rhn if out.inputs_daily else None,
                "TotalPrec": out.inputs_daily.precipitation if out.inputs_daily else None,
                "TotalRad": out.inputs_daily.rad if out.inputs_daily else None,
                "TotalLW": out.inputs_daily.leaf_wetness if out.inputs_daily else None,
            }
        )
    return records


def run_franchestyn(
    weather_path: str,
    management_path: str,
    start_year: int,
    end_year: int,
    config: FranchestynConfig,
    cropmodel_path: str | None = None,
    crop_param_file: str | None = None,
    disease_param_file: str | None = None,
    fungicide_param_file: str | None = None,
) -> dict:
    """Run the FraNchEstYN model for one or more growing seasons.

    If ``config.is_calibration`` is ``True``, the Nelder-Mead optimizer is
    run before the final simulation. Parameter files can be overridden
    per-call via the optional arguments.

    Args:
        weather_path: Path to the daily weather CSV (or directory containing
            ``daily/{site}.csv``).
        management_path: Path to the sowing management CSV.
        start_year: First simulation year (inclusive).
        end_year: Last simulation year (inclusive).
        config: FraNchEstYN configuration dataclass.
        cropmodel_path: Optional path to the external crop model CSV. Pass
            ``None`` to use the internal crop model.
        crop_param_file: Override for the crop parameters JSON.
        disease_param_file: Override for the disease parameters JSON.
        fungicide_param_file: Override for the fungicide parameters JSON.

    Returns:
        Dictionary with a single key ``'outputs'`` containing:

        - ``'simulation'``: list of per-day record dicts.
        - ``'summary'``: dict with ``'rmse'``, ``'is_calibration'``,
          ``'calibration_variable'``, and ``'best_params'``.
    """
    crop_param_file = crop_param_file or config.crop_param_file
    disease_param_file = disease_param_file or config.disease_param_file
    fungicide_param_file = fungicide_param_file or config.fungicide_param_file

    runner = FranchestynRunner(
        weather_dir=weather_path,
        param_file=config.param_file,
        sowing_file=management_path,
        ref_dir=config.reference_path,
        crop_model_dir=cropmodel_path,
        site=config.site,
        variety=config.variety,
        disease=config.disease,
        start_year=start_year,
        end_year=end_year,
        weather_time_step="daily",
        calibration_variable=config.calibration_variable,
        is_calibration=config.is_calibration,
        use_gdd=config.use_gdd,
        crop_type=config.crop_type,
        crop_param_file=crop_param_file,
        disease_param_file=disease_param_file,
        disease_type=config.disease_type,
        fungicide_param_file=fungicide_param_file,
        fungicide_type=config.fungicide_type,
    )

    best_params = {}
    if config.is_calibration:
        optimizer = FranchestynOptimizer(
            runner=runner,
            calibration_variable=config.calibration_variable,
            n_restarts=config.n_restarts,
            max_iter=config.max_iter,
        )
        best_params = optimizer.calibrate()
        date_outputs = runner.run(param_values=best_params)
    else:
        date_outputs = runner.run()

    include_crop = config.calibration_variable in ("crop", "all")
    include_disease = config.calibration_variable in ("disease", "all")
    rmse = runner.compute_rmse(
        date_outputs,
        include_crop=include_crop,
        include_disease=include_disease,
    )

    records = _outputs_to_records(date_outputs)
    return {
        "outputs": {
            "simulation": records,
            "summary": {
                "rmse": rmse,
                "is_calibration": config.is_calibration,
                "calibration_variable": config.calibration_variable,
                "best_params": best_params,
            },
        }
    }


def save_simulation_results_csv(res_ot_simulation, output_root: Path, filename: str = "franchestyn_simulation_results.csv") -> Path:
    """Write the per-day simulation records to a CSV file.

    Args:
        res_ot_simulation: List of record dicts from ``run_franchestyn``
            (``result['outputs']['simulation']``).
        output_root: Root output directory.
        filename: Output filename; default is
            ``'franchestyn_simulation_results.csv'``.

    Returns:
        Path to the written CSV file.
    """
    output_file = output_root / "SimulationExperimentTemplate" / filename
    df = pd.DataFrame(res_ot_simulation)
    df.to_csv(output_file, index=False)
    return output_file


def build_season_summary(df: pd.DataFrame, site: str, variety: str) -> pd.DataFrame:
    """Aggregate per-day simulation output into one row per growing season.

    Computes AUDPC via the trapezoid rule, peak disease severity, attainable
    and actual yield/AGB, yield loss (raw and percentage), and optional
    seasonal weather averages when those columns are present.

    Args:
        df: Per-day simulation DataFrame (e.g. from ``pd.DataFrame(simulation)``).
        site: Site identifier written to the ``Site`` column.
        variety: Variety identifier written to the ``Variety`` column.

    Returns:
        DataFrame with one row per growing season and columns including
        ``GrowingSeason``, ``Site``, ``Variety``, ``AUDPC``,
        ``DiseaseSeverity``, ``YieldAttainable``, ``YieldActual``,
        ``YieldLossRaw``, ``YieldLossPerc``, ``AGBattainable``, and
        ``AGBactual``. Returns an empty DataFrame when ``df`` is empty.
    """
    if df.empty:
        return pd.DataFrame()

    d = df.copy()
    d["Date"] = pd.to_datetime(d["Date"], dayfirst=True, errors="coerce")
    d = d.dropna(subset=["Date"])

    if "DaysAfterSowing" in d.columns:
        d = d[d["DaysAfterSowing"] > 0]
    if d.empty:
        return pd.DataFrame()

    if "GrowingSeason" not in d.columns:
        d["GrowingSeason"] = d["Date"].dt.year

    rows = []
    for season, g in d.groupby("GrowingSeason"):
        g = g.sort_values("Date")

        y = g["DiseaseSeverity"].fillna(0.0).to_numpy(dtype=float) * 100.0
        x = g["Date"].map(pd.Timestamp.toordinal).to_numpy(dtype=float)
        audpc = float(np.trapezoid(y, x)) if len(g) >= 2 else 0.0

        yield_att = float(g["YieldAttainable"].max()) if "YieldAttainable" in g.columns else 0.0
        yield_act = float(g["YieldActual"].max()) if "YieldActual" in g.columns else 0.0
        agb_att = float(g["AGBattainable"].max()) if "AGBattainable" in g.columns else 0.0
        agb_act = float(g["AGBactual"].max()) if "AGBactual" in g.columns else 0.0
        dis_sev = float(g["DiseaseSeverity"].max()) if "DiseaseSeverity" in g.columns else 0.0

        loss_raw = yield_att - yield_act
        loss_perc = (loss_raw / yield_att * 100.0) if yield_att > 0 else 0.0

        row = {
            "GrowingSeason": int(season),
            "Site": site,
            "Variety": variety,
            "AUDPC": audpc,
            "DiseaseSeverity": dis_sev,
            "YieldAttainable": yield_att,
            "YieldActual": yield_act,
            "YieldLossRaw": loss_raw,
            "YieldLossPerc": loss_perc,
            "AGBattainable": agb_att,
            "AGBactual": agb_act,
        }

        if "Tmax" in g.columns:
            row["AveTx"] = float(g["Tmax"].mean())
        if "Tmin" in g.columns:
            row["AveTn"] = float(g["Tmin"].mean())
        if "RHx" in g.columns:
            row["AveRHx"] = float(g["RHx"].mean())
        if "RHn" in g.columns:
            row["AveRHn"] = float(g["RHn"].mean())
        if "TotalPrec" in g.columns:
            row["TotalPrec"] = float(g["TotalPrec"].sum())
        if "TotalRad" in g.columns:
            row["TotalRad"] = float(g["TotalRad"].sum())
        if "TotalLW" in g.columns:
            row["TotalLW"] = float(g["TotalLW"].sum())

        rows.append(row)

    return pd.DataFrame(rows).sort_values("GrowingSeason").reset_index(drop=True)


def save_season_summary_csv(summary_df: pd.DataFrame, output_root: Path, filename: str = "franchestyn_season_summary.csv") -> Path | None:
    """Write the season summary DataFrame to a CSV file.

    Args:
        summary_df: Season summary DataFrame returned by ``build_season_summary``.
        output_root: Root output directory.
        filename: Output filename; default is
            ``'franchestyn_season_summary.csv'``.

    Returns:
        Path to the written CSV file, or ``None`` if ``summary_df`` is empty.
    """
    if summary_df.empty:
        return None

    output_file = output_root / "SimulationExperimentTemplate" / filename
    summary_df.to_csv(output_file, index=False)
    return output_file


def save_calibrated_parameters_csv(
    best_params: dict,
    output_root: Path,
    site: str,
    variety: str,
    filename: str | None = None,
) -> Path | None:
    """Write calibrated parameter values to a CSV file.

    Writes rows with columns ``model, param, value``, splitting each
    ``'class_ParamName'`` key into its model-class and parameter-name parts.
    The file is placed under
    ``output_root / SimulationExperimentTemplate / calibratedParameters /``.

    Args:
        best_params: Best-fit parameter dict returned by the optimizer
            (``result['outputs']['summary']['best_params']``).
        output_root: Root output directory.
        site: Site identifier used in the default filename.
        variety: Variety identifier used in the default filename.
        filename: Override filename. If ``None``, defaults to
            ``'calibratedParameters_{site}_{variety}.csv'``.

    Returns:
        Path to the written CSV file, or ``None`` if ``best_params`` is empty.
    """
    if not best_params:
        return None

    out_dir = output_root / "SimulationExperimentTemplate" / "calibratedParameters"
    out_dir.mkdir(parents=True, exist_ok=True)
    if filename is None:
        filename = f"calibratedParameters_{site}_{variety}.csv"
    output_file = out_dir / filename

    with output_file.open("w", newline="", encoding="utf-8") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=["model", "param", "value"], delimiter=",")
        writer.writeheader()
        for key, value in sorted(best_params.items()):
            if "_" in key:
                model, param = key.split("_", 1)
            else:
                model, param = "", key
            writer.writerow({"model": model, "param": param, "value": value})

    return output_file
