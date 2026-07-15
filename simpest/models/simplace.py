import csv
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

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
    with project_csv.open(newline="", encoding="utf-8-sig") as f_proj:
        proj_reader = csv.DictReader(f_proj, delimiter=";")
        project_rows = list(proj_reader)

    row = project_rows[selected_line - 1]
    key_map = {k.strip().lower(): k for k in row.keys()}

    def field(*aliases: str, default: str = "") -> str:
        for alias in aliases:
            key = key_map.get(alias.lower())
            if key is not None:
                return str(row.get(key, default)).strip()
        return default

    project_row = {
        "projectid": field("projectid"),
        "simulationid": field("simulationid"),
        "startdate": field("startdate"),
        "enddate": field("enddate"),
        "location": field("location"),
        "iopt": field("iopt"),
        "idem": _to_int(field("idem")) or 0,
        "irri": field("irri"),
    }

    start_dt = _parse_project_date(project_row["startdate"])
    end_dt = _parse_project_date(project_row["enddate"])
    if start_dt and end_dt:
        project_row["yearly_sowing_doy"] = extract_yearly_sowing_doy(
            work_root,
            project_row,
            start_dt.year,
            end_dt.year,
        )

    return project_row


def _parse_project_date(date_value: str) -> Optional[datetime]:
    """Parse a project date in dd.mm.yyyy format."""
    try:
        return datetime.strptime(date_value.strip(), "%d.%m.%Y")
    except (TypeError, ValueError, AttributeError):
        return None


def _to_int(value) -> Optional[int]:
    """Best-effort int coercion for mixed CSV values."""
    if value is None:
        return None
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def extract_yearly_sowing_doy(
    work_root: Path,
    project_row: dict,
    start_year: int,
    end_year: int,
) -> dict[int, int]:
    """
    Extract per-year sowing DOY from projectdata CSV files.

    The function prefers explicit `ISOW` values when present. If no per-year
    values are available, it falls back to a constant sowing DOY derived from
    `IDEM - 7` to preserve current behavior.

    Args:
        work_root (Path): Simplace workspace root.
        project_row (dict): Selected project row metadata.
        start_year (int): Start year (inclusive).
        end_year (int): End year (inclusive).

    Returns:
        dict[int, int]: Mapping year -> sowing DOY.
    """
    projectdata_dir = work_root / "SimulationExperimentTemplate" / "data" / "projectdata"
    if not projectdata_dir.exists():
        return {}

    location = str(project_row.get("location", "")).strip().lower()
    iopt = _to_int(project_row.get("iopt"))
    idem = _to_int(project_row.get("idem"))
    default_sowing_doy = max(1, (idem or 8) - 7)

    rows: list[dict] = []
    for project_csv in sorted(projectdata_dir.glob("*.csv")):
        with project_csv.open(newline="", encoding="utf-8-sig") as f_proj:
            reader = csv.DictReader(f_proj, delimiter=";")
            if not reader.fieldnames:
                continue

            name_to_col = {name.strip().lower(): name for name in reader.fieldnames}
            col_location = name_to_col.get("location")
            col_iopt = name_to_col.get("iopt")
            col_start = name_to_col.get("startdate")
            col_end = name_to_col.get("enddate")
            col_isow = name_to_col.get("isow")
            col_idem = name_to_col.get("idem")

            for raw in reader:
                row_location = str(raw.get(col_location, "")).strip().lower() if col_location else ""
                if location and row_location and row_location != location:
                    continue

                row_iopt = _to_int(raw.get(col_iopt)) if col_iopt else None
                if iopt is not None and row_iopt is not None and row_iopt != iopt:
                    continue

                start_dt = _parse_project_date(raw.get(col_start, "")) if col_start else None
                end_dt = _parse_project_date(raw.get(col_end, "")) if col_end else None
                if start_dt is None and end_dt is None:
                    continue

                year = start_dt.year if start_dt is not None else end_dt.year
                if year < start_year or year > end_year:
                    continue

                sow_doy = _to_int(raw.get(col_isow)) if col_isow else None
                if sow_doy is None:
                    row_idem = _to_int(raw.get(col_idem)) if col_idem else idem
                    sow_doy = max(1, (row_idem or 8) - 7)

                rows.append(
                    {
                        "year": year,
                        "sowing_doy": sow_doy,
                        "span_days": ((end_dt - start_dt).days if start_dt and end_dt else 999999),
                    }
                )

    # Prefer the most specific row per year (shortest start-end span).
    yearly_sowing: dict[int, int] = {}
    for row in sorted(rows, key=lambda x: (x["year"], x["span_days"])):
        yearly_sowing[row["year"]] = row["sowing_doy"]

    if yearly_sowing:
        return yearly_sowing

    # Backward-compatible fallback: constant sowing day across all years.
    return {year: default_sowing_doy for year in range(start_year, end_year + 1)}


# def export_crop_model_data(output_root: Path, project_row: dict) -> Path:
#     """
#     Export crop model data from Simplace daily output to a CSV file for FraNchEstYN.

#     Args:
#         output_root (Path): Output root directory.
#         project_row (dict): Project row dictionary.

#     Returns:
#         Path: Path to the exported crop model CSV file.
#     """
#     src = output_root / "SimulationExperimentTemplate" / f"{project_row['location']}{project_row['iopt']}_daily.csv"
#     dst = output_root / "SimulationExperimentTemplate" / "cropModel_data.csv"

#     with src.open(newline="", encoding="utf-8") as f_in, dst.open("w", newline="", encoding="utf-8") as f_out:
#         reader = csv.DictReader(f_in, delimiter=";")
#         writer = csv.DictWriter(
#             f_out,
#             fieldnames=["year", "doy", "agb", "yield", "fint", "lai", "gdd"],
#             delimiter=",",
#         )
#         writer.writeheader()

#         for row in reader:
#             if all(float(row[name]) == 0.0 for name in ("TAGB", "WSO", "FINT", "LAI")):
#                 continue

#             d = datetime.strptime(row["CURRENT.DATE"], "%d.%m.%Y")
#             writer.writerow(
#                 {
#                     "year": d.year,
#                     "doy": d.timetuple().tm_yday,
#                     "agb": float(row["TAGB"]) * 10,
#                     "yield": float(row["WSO"]) * 10,
#                     "fint": row["FINT"],
#                     "lai": row["LAI"],
#                     "gdd": row["TSUM"],
#                 }
#             )
#     return dst


def export_crop_model_data(output_root: Path, project_row: dict) -> Path:
    """Exports daily SIMPLACE outputs to FraNchEstYN crop model format.

    The exported CSV contains the variables required by FraNchEstYN for
    coupling with an external crop model.

    Expected output units:
        - agb: g/m²
        - yield: g/m²
        - fint: fraction (0–1)
        - lai: m²/m²
        - gdd: °C·day

    Notes:
        SIMPLACE (LINTUL5) already exports ``TAGB`` and ``WSO`` in g/m².
        Therefore, no unit conversion is applied.

    Args:
        output_root: Root directory containing the
            ``SimulationExperimentTemplate`` folder.
        project_row: Dictionary containing project information. Must include
            the keys ``location`` and ``iopt``.

    Returns:
        Path to the exported ``cropModel_data.csv`` file.

    Raises:
        FileNotFoundError: If the SIMPLACE daily output file does not exist.
        KeyError: If required columns are missing from the input CSV.
        ValueError: If numeric conversion of required variables fails.
    """
    sim_dir = output_root / "SimulationExperimentTemplate"

    src = sim_dir / f"{project_row['location']}{project_row['iopt']}_daily.csv"
    dst = sim_dir / "cropModel_data.csv"

    if not src.exists():
        raise FileNotFoundError(f"SIMPLACE daily output not found: {src}")

    with src.open("r", newline="", encoding="utf-8") as f_in, \
         dst.open("w", newline="", encoding="utf-8") as f_out:

        reader = csv.DictReader(f_in, delimiter=";")

        writer = csv.DictWriter(f_out, fieldnames=["year", "doy", "agb", "yield", "fint", "lai", "gdd"],)
        writer.writeheader()

        for row in reader:

            agb = float(row["TAGB"])
            yield_ = float(row["WSO"])
            fint = float(row["FINT"])
            lai = float(row["LAI"])
            gdd = float(row["TSUM"])

            # Skip rows before crop emergence.
            if agb == 0.0 and yield_ == 0.0 and fint == 0.0 and lai == 0.0:
                continue

            date = datetime.strptime(row["CURRENT.DATE"], "%d.%m.%Y")

            writer.writerow(
                {
                    "year": date.year,
                    "doy": date.timetuple().tm_yday,
                    "agb": agb,
                    "yield": yield_,
                    "fint": fint,
                    "lai": lai,
                    "gdd": gdd,
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


def build_management(
    output_root: Path,
    project_row: dict,
    crop: str = "wheat",
    variety: str = "generic",
    yearly_sowing_doy: Optional[dict[int, int]] = None,
) -> Path:
    """
    Build and save a management CSV file for FraNchEstYN.

    Args:
        output_root (Path): Output root directory.
        project_row (dict): Project row dictionary.
        crop (str, optional): Crop name. Defaults to "wheat".
        variety (str, optional): Variety name. Defaults to "generic".
        yearly_sowing_doy (dict[int, int] | None, optional): Mapping of
            simulation year to sowing DOY. If omitted, a single "All" row
            is written to preserve backward compatibility.

    Returns:
        Path: Path to the management CSV file.
    """
    sowing_doy = max(1, project_row["idem"] - 7)
    if yearly_sowing_doy is None:
        candidate = project_row.get("yearly_sowing_doy") if isinstance(project_row, dict) else None
        if isinstance(candidate, dict) and candidate:
            yearly_sowing_doy = {
                int(year): int(doy)
                for year, doy in candidate.items()
            }

    management_dst = output_root / "SimulationExperimentTemplate" / "management_franchestyn.csv"
    with management_dst.open("w", newline="", encoding="utf-8") as f_out:
        writer = csv.DictWriter(
            f_out,
            fieldnames=["site", "crop", "variety", "year", "sowingDOY"],
            delimiter=",",
        )
        writer.writeheader()
        if yearly_sowing_doy:
            for year in sorted(yearly_sowing_doy):
                writer.writerow(
                    {
                        "site": project_row["location"],
                        "crop": crop,
                        "variety": variety,
                        "year": year,
                        "sowingDOY": max(1, int(yearly_sowing_doy[year])),
                    }
                )
        else:
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
