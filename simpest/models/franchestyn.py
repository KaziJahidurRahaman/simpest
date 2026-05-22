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
    return str(Path(__file__).with_name(filename))


@dataclass(frozen=True)
class FranchestynConfig:
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
    output_file = output_root / "SimulationExperimentTemplate" / filename
    df = pd.DataFrame(res_ot_simulation)
    df.to_csv(output_file, index=False)
    return output_file


def build_season_summary(df: pd.DataFrame, site: str, variety: str) -> pd.DataFrame:
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
