from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from .fr_optimizer import FranchestynOptimizer
from .fr_runner import FranchestynRunner


def _resolve_default_reference() -> str:
    """
    Resolve a local default reference CSV using repository files only.

    Returns:
        str: Path to the default reference CSV file.
    """
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
    """
    Get the path to a local model file by filename.

    Args:
        filename (str): Name of the file to resolve.

    Returns:
        str: Path to the local model file.
    """
    return str(Path(__file__).with_name(filename))


def deactivate_calibration(params: dict, disable_list) -> None:
    """
    Disable calibration for selected parameter names in a parameter section.

    Args:
        params (dict): Mapping of parameter name to parameter specification dict.
        disable_list (Iterable[str]): Parameter names for which calibration
            should be forced to False.
    """
    for param_name in disable_list:
        if param_name in params:
            params[param_name]["calibration"] = False
    return params


@dataclass(frozen=True)
class FranchestynConfig:
    """
    Configuration for FraNchEstYN model runs.

    Attributes:
        param_file (str): Path to the main parameter file.
        crop_param_file (str): Path to crop parameter file.
        disease_param_file (str): Path to disease parameter file.
        fungicide_param_file (str): Path to fungicide parameter file.
        reference_path (str): Path to reference CSV.
        crop_type (str): Crop type (e.g., 'wheat').
        disease_type (str): Disease type (e.g., 'septoria').
        fungicide_type (str|None): Fungicide type (e.g., 'protectant').
        site (str): Site name.
        variety (str): Variety name.
        disease (str): Disease name.
        is_calibration (bool): Whether to run calibration.
        calibration_variable (str): Calibration variable ('all', 'crop', 'disease').
        use_gdd (bool): Use growing degree days.
        n_restarts (int): Number of calibration restarts.
        max_iter (int): Maximum calibration iterations.
        crop_disabled_params (frozenset[str]): Crop parameter names to hard-exclude
            from calibration.
        disease_disabled_params (frozenset[str]): Disease parameter names to
            hard-exclude from calibration.
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
    crop_disabled_params: frozenset[str] = frozenset()
    disease_disabled_params: frozenset[str] = frozenset()


def _outputs_to_records(date_outputs):
    """
    Convert date_outputs from FraNchEstYN runner to a list of record dicts.

    Args:
        date_outputs (dict): Mapping of date to output objects.

    Returns:
        list of dict: List of output records for each date.
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
    """
    Run the FraNchEstYN model with the given configuration and input files.

    Args:
        weather_path (str): Path to weather input file.
        management_path (str): Path to management input file.
        start_year (int): Start year for simulation.
        end_year (int): End year for simulation.
        config (FranchestynConfig): FraNchEstYN configuration object.
        cropmodel_path (str|None, optional): Path to crop model data file.
        crop_param_file (str|None, optional): Path to crop parameter file.
        disease_param_file (str|None, optional): Path to disease parameter file.
        fungicide_param_file (str|None, optional): Path to fungicide parameter file.

    Returns:
        dict: Dictionary with simulation outputs and summary.
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
        disabled_by_class = {
            "crop": set(config.crop_disabled_params),
            "disease": set(config.disease_disabled_params),
        }
        optimizer = FranchestynOptimizer(
            runner=runner,
            calibration_variable=config.calibration_variable,
            n_restarts=config.n_restarts,
            max_iter=config.max_iter,
            disabled_by_class=disabled_by_class,
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
    """
    Save FraNchEstYN simulation results to a CSV file.

    Args:
        res_ot_simulation (list of dict): Simulation output records.
        output_root (Path): Output root directory.
        filename (str, optional): Output filename. Defaults to 'franchestyn_simulation_results.csv'.

    Returns:
        Path: Path to the saved CSV file.
    """
    output_file = output_root / "SimulationExperimentTemplate" / filename
    df = pd.DataFrame(res_ot_simulation)
    df.to_csv(output_file, index=False)
    return output_file


def build_season_summary(df: pd.DataFrame, site: str, variety: str) -> pd.DataFrame:
    """
    Build a season summary DataFrame from simulation results.

    Args:
        df (pd.DataFrame): Simulation results DataFrame.
        site (str): Site name.
        variety (str): Variety name.

    Returns:
        pd.DataFrame: Season summary DataFrame.
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
    """
    Save season summary DataFrame to a CSV file.

    Args:
        summary_df (pd.DataFrame): Season summary DataFrame.
        output_root (Path): Output root directory.
        filename (str, optional): Output filename. Defaults to 'franchestyn_season_summary.csv'.

    Returns:
        Path|None: Path to the saved CSV file, or None if summary_df is empty.
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
    config: FranchestynConfig | None = None,
    r_like: bool = False,
) -> Path | None:
    """
    Save calibration parameters to a CSV file.

    Args:
        best_params (dict): Best parameter values from calibration.
        output_root (Path): Output root directory.
        site (str): Site name.
        variety (str): Variety name.
        filename (str|None, optional): Output filename. If None, uses a default pattern.
        config (FranchestynConfig|None, optional): Configuration used to load
            parameter metadata when ``r_like=True``.
        r_like (bool, optional): If True, export an R-style diagnostics table
            with defaults, bounds, calibration flags, and calibrated values.

    Returns:
        Path|None: Path to the saved CSV file, or None if best_params is empty.
    """
    out_dir = output_root / "SimulationExperimentTemplate" / "calibratedParameters"
    out_dir.mkdir(parents=True, exist_ok=True)
    if filename is None:
        filename = f"calibratedParameters_{site}_{variety}.csv"
    output_file = out_dir / filename

    if r_like:
        cfg = config or FranchestynConfig(site=site, variety=variety)

        rows = []
        parameter_file = f"parameters_{site}_{variety}.csv"

        def _load_param_defs(json_path: str, model: str, kind: str):
            path = Path(json_path)
            if not path.exists():
                return
            data = json.loads(path.read_text(encoding="utf-8"))
            section = data.get(kind, {})
            for param, spec in section.items():
                default_val = spec.get("value", "NA")
                min_val = spec.get("min", "NA")
                max_val = spec.get("max", "NA")
                unit = spec.get("unit", "unitless")
                calib_enabled = bool(spec.get("calibration", False))

                key = f"{model}_{param}"
                calibrated_val = best_params.get(key, default_val)
                value_col = calibrated_val if calib_enabled else "NA"
                file_col = parameter_file if calib_enabled else "NA"

                rows.append(
                    {
                        "Model": model,
                        "Parameter": param,
                        "unit": unit,
                        "min": min_val,
                        "max": max_val,
                        "default": default_val,
                        "calibration": "TRUE" if calib_enabled else "FALSE",
                        "calibrated": calibrated_val,
                        "value": value_col,
                        "file": file_col,
                        "facet_label": f"{param} ({unit})",
                    }
                )

        _load_param_defs(cfg.crop_param_file, "crop", cfg.crop_type)
        _load_param_defs(cfg.disease_param_file, "disease", cfg.disease_type)

        with output_file.open("w", newline="", encoding="utf-8") as f_out:
            fieldnames = [
                "Model",
                "Parameter",
                "unit",
                "min",
                "max",
                "default",
                "calibration",
                "calibrated",
                "value",
                "file",
                "facet_label",
            ]
            writer = csv.DictWriter(f_out, fieldnames=fieldnames, delimiter=",")
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

        return output_file

    if not best_params:
        return None

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
