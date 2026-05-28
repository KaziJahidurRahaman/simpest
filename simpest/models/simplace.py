import csv
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import jpype
import pandas as pd

import simplace
def get_simplace_directories(shell):
    """
    Wrapper for simplace.getSimplaceDirectories to avoid exposing simplace in notebooks.

    Args:
        shell: Simplace shell instance.

    Returns:
        dict: Simplace directories information.
    """
    return simplace.getSimplaceDirectories(shell)


_SIMPLACE_INSTANCE = None


@dataclass(frozen=True)
class SimplaceConfig:
    """
    Configuration for Simplace runs.

    Attributes:
        install_dir (str): Path to Simplace installation directory.
        work_dir (str): Path to Simplace working directory.
        output_dir (str): Path to Simplace output directory.
        solution_path (str): Path to Simplace solution XML file.
        project_path (str): Path to Simplace project XML file.
    """
    install_dir: str = "C:/ParamVC/Research/simplace+/simplace_portable/workspace/"
    work_dir: str = "C:/ParamVC/Research/simplace+/simplace_portable/workspace/simplace_run/simulation/"
    output_dir: str = "C:/ParamVC/Research/simplace+/simplace_out/"
    solution_path: str = "SimulationExperimentTemplate/solution/Lintul5_indiana.sol.xml"
    project_path: str = "SimulationExperimentTemplate/project/Lintul5All_indiana.proj.xml"


def init_simplace(config: SimplaceConfig):
    """
    Initialize and return a Simplace instance using the given configuration.

    Args:
        config (SimplaceConfig): Simplace configuration object.

    Returns:
        Simplace instance.
    """
    global _SIMPLACE_INSTANCE

    if _SIMPLACE_INSTANCE is not None and jpype.isJVMStarted():
        return _SIMPLACE_INSTANCE

    if jpype.isJVMStarted():
        Wrapper = jpype.JClass('net.simplace.sim.wrapper.SimplaceWrapper')
        _SIMPLACE_INSTANCE = Wrapper(config.work_dir, config.output_dir, None, None)
        return _SIMPLACE_INSTANCE

    _SIMPLACE_INSTANCE = simplace.initSimplace(
        installDir=config.install_dir,
        workDir=config.work_dir,
        outputDir=config.output_dir,
    )
    return _SIMPLACE_INSTANCE


def run_simplace(shell, config: SimplaceConfig, project_lines: list[int]):
    """
    Run a Simplace project for the specified project lines.

    Args:
        shell: Simplace shell instance.
        config (SimplaceConfig): Simplace configuration object.
        project_lines (list[int]): List of project line indices to run.
    """
    simplace.openProject(shell, config.solution_path, config.project_path)
    simplace.setProjectLines(shell, project_lines)
    simplace.runProject(shell)


def get_project_row(work_root: Path, selected_line: int) -> dict:
    """
    Get a row from the project CSV file by line index.

    Args:
        work_root (Path): Root directory for Simplace workspace.
        selected_line (int): Line number (1-based) to select from the project CSV.

    Returns:
        dict: Dictionary with project row data.
    """
    project_csv = (
        work_root
        / "SimulationExperimentTemplate"
        / "data"
        / "projectdata"
        / "lintul5all_indiana.csv"
    )
    with project_csv.open(newline="", encoding="utf-8") as f_proj:
        proj_reader = csv.reader(f_proj, delimiter=";")
        _ = next(proj_reader, None)
        project_rows = list(proj_reader)

    row = project_rows[selected_line - 1]
    return {
        "projectid": row[0],
        "simulationid": row[1],
        "startdate": row[2],
        "enddate": row[3],
        "location": row[4],
        "iopt": row[5],
        "idem": int(row[6]),
        "irri": row[7],
    }


def export_crop_model_data(output_root: Path, project_row: dict) -> Path:
    """
    Export crop model data from Simplace daily output to a CSV file for FraNchEstYN.

    Args:
        output_root (Path): Output root directory.
        project_row (dict): Project row dictionary.

    Returns:
        Path: Path to the exported crop model CSV file.
    """
    src = output_root / "SimulationExperimentTemplate" / f"{project_row['location']}{project_row['iopt']}_daily.csv"
    dst = output_root / "SimulationExperimentTemplate" / "cropModel_data.csv"

    with src.open(newline="", encoding="utf-8") as f_in, dst.open("w", newline="", encoding="utf-8") as f_out:
        reader = csv.DictReader(f_in, delimiter=";")
        writer = csv.DictWriter(
            f_out,
            fieldnames=["year", "doy", "agb", "yield", "fint", "lai", "gdd"],
            delimiter=",",
        )
        writer.writeheader()

        for row in reader:
            if all(float(row[name]) == 0.0 for name in ("TAGB", "WSO", "FINT", "LAI")):
                continue

            d = datetime.strptime(row["CURRENT.DATE"], "%d.%m.%Y")
            writer.writerow(
                {
                    "year": d.year,
                    "doy": d.timetuple().tm_yday,
                    "agb": float(row["TAGB"]) * 10,
                    "yield": float(row["WSO"]) * 10,
                    "fint": row["FINT"],
                    "lai": row["LAI"],
                    "gdd": row["TSUM"],
                }
            )
    return dst


def _saturation_vapor_pressure(t_celsius: float) -> float:
    """
    Calculate the saturation vapor pressure for a given temperature.

    Args:
        t_celsius (float): Temperature in Celsius.

    Returns:
        float: Saturation vapor pressure (kPa).
    """
    return 0.6108 * math.exp((17.27 * t_celsius) / (t_celsius + 237.3))


def convert_weather(work_root: Path, output_root: Path, location: str) -> Path:
    """
    Convert Simplace weather file to FraNchEstYN-compatible CSV format.

    Args:
        work_root (Path): Simplace workspace root directory.
        output_root (Path): Output root directory.
        location (str): Location name (used for weather file).

    Returns:
        Path: Path to the converted weather CSV file.
    """
    weather_src = work_root / "SimulationExperimentTemplate" / "data" / "weather" / f"{location}.txt"
    weather_dst = output_root / "SimulationExperimentTemplate" / "weather_franchestyn.csv"

    with weather_src.open(newline="", encoding="utf-8-sig") as f_in, weather_dst.open(
        "w", newline="", encoding="utf-8"
    ) as f_out:
        reader = csv.DictReader(f_in, delimiter="\t")
        reader.fieldnames = [h.lstrip("\ufeff").strip() for h in reader.fieldnames]

        writer = csv.DictWriter(
            f_out,
            fieldnames=["site", "year", "month", "day", "tx", "tn", "p", "rad", "vp", "rhx", "rhn"],
            delimiter=",",
        )
        writer.writeheader()

        for row in reader:
            d = datetime.strptime(row["CURRENTDAY"], "%d.%m.%Y")
            tmax = float(row["Tmax"])
            tmin = float(row["Tmin"])
            vp = float(row["VapourPressure"])

            es_tmax = _saturation_vapor_pressure(tmax)
            es_tmin = _saturation_vapor_pressure(tmin)
            rhx = min(100.0, max(0.0, 100.0 * vp / es_tmin)) if es_tmin > 0 else 0.0
            rhn = min(100.0, max(0.0, 100.0 * vp / es_tmax)) if es_tmax > 0 else 0.0

            writer.writerow(
                {
                    "site": location,
                    "year": d.year,
                    "month": d.month,
                    "day": d.day,
                    "tx": tmax,
                    "tn": tmin,
                    "p": row["Precipitation"],
                    "rad": float(row["Irradiation"]) / 1000.0,
                    "vp": vp,
                    "rhx": rhx,
                    "rhn": rhn,
                }
            )

    return weather_dst


def build_management(output_root: Path, project_row: dict, crop: str = "wheat", variety: str = "generic") -> Path:
    """
    Build and save a management CSV file for FraNchEstYN.

    Args:
        output_root (Path): Output root directory.
        project_row (dict): Project row dictionary.
        crop (str, optional): Crop name. Defaults to "wheat".
        variety (str, optional): Variety name. Defaults to "generic".

    Returns:
        Path: Path to the management CSV file.
    """
    start_date = datetime.strptime(project_row["startdate"], "%d.%m.%Y")
    _ = start_date
    sowing_doy = max(1, project_row["idem"] - 7)

    management_dst = output_root / "SimulationExperimentTemplate" / "management_franchestyn.csv"
    with management_dst.open("w", newline="", encoding="utf-8") as f_out:
        writer = csv.DictWriter(
            f_out,
            fieldnames=["site", "crop", "variety", "year", "sowingDOY"],
            delimiter=",",
        )
        writer.writeheader()
        writer.writerow(
            {
                "site": project_row["location"],
                "crop": crop,
                "variety": variety,
                "year": "All",
                "sowingDOY": sowing_doy,
            }
        )

    return management_dst


def merge_simplace_and_franchestyn(
    output_root: Path,
    project_row: dict,
    franchestyn_df: pd.DataFrame,
    out_name: str = "merged_simulation_data.csv",
) -> Path:
    """
    Merge Simplace and FraNchEstYN daily outputs into a single CSV file.

    Args:
        output_root (Path): Output root directory.
        project_row (dict): Project row dictionary.
        franchestyn_df (pd.DataFrame): FraNchEstYN daily output DataFrame.
        out_name (str, optional): Output filename. Defaults to "merged_simulation_data.csv".

    Returns:
        Path: Path to the merged CSV file.
    """
    simplace_daily_path = output_root / "SimulationExperimentTemplate" / f"{project_row['location']}{project_row['iopt']}_daily.csv"
    simplace_df = pd.read_csv(simplace_daily_path, sep=";")

    simplace_df.columns = simplace_df.columns.str.strip()
    franchestyn_df.columns = franchestyn_df.columns.str.strip()

    simplace_df["date"] = pd.to_datetime(simplace_df["CURRENT.DATE"], format="%d.%m.%Y")
    franchestyn_df["date"] = pd.to_datetime(franchestyn_df["Date"], dayfirst=True)

    sim_df = simplace_df.rename(columns={c: f"{c}_S" for c in simplace_df.columns if c != "date"})
    fran_df = franchestyn_df.rename(columns={c: f"{c}_F" for c in franchestyn_df.columns if c != "date"})

    merged = pd.merge(sim_df, fran_df, on="date", how="left")
    merged = merged.rename(
        columns={
            "WSO_S": "WSO_S_g_m2",
            "YieldAttainable_F": "YieldAttainable_F_kg_ha",
            "YieldActual_F": "YieldActual_F_kg_ha",
        }
    )

    out_path = output_root / "SimulationExperimentTemplate" / out_name
    merged.to_csv(out_path, index=False)
    return out_path
