"""Main module."""
from pathlib import Path

import pandas as pd
import simplace

import simpest.models.simplace as simplace_mod
import simpest.models.franchestyn as franchestyn_mod


def run_pipeline(
    implace_config: simplace_mod.SimplaceConfig,
    franchestyn_config: franchestyn_mod.FranchestynConfig,
    project_lines: list[int],
    shutdown_simplace: bool = False,
) -> dict:


    """
    Run the full SIMPLACE and FraNchEstYN pipeline.

    This function initializes and runs a SIMPLACE simulation, prepares the required input files,
    and then runs the FraNchEstYN model using the outputs from SIMPLACE. It saves all intermediate
    and final results to disk and returns paths and results for further analysis.

    Args:
        simplace_config (simplace_mod.SimplaceConfig):
            Configuration object for SIMPLACE (install, work, output, solution, project paths).
        franchestyn_config (franchestyn_mod.FranchestynConfig):
            Configuration object for FraNchEstYN (reference, crop, disease, site, etc.).
        project_lines (list of int):
            List of project line indices to run in SIMPLACE (usually [1]).
        shutdown_simplace (bool, optional):
            If True, shuts down the SIMPLACE JVM after running (not recommended in Jupyter). Default is False.

    Returns:
        dict: Dictionary with keys:
            - 'work_root': Path to SIMPLACE work directory used
            - 'output_root': Path to SIMPLACE output directory used
            - 'project_row': Project row dictionary used for the run
            - 'merged_csv': Path to merged output CSV (SIMPLACE + FraNchEstYN)
            - 'result': Full FraNchEstYN result dictionary
            - 'simplace_shutdown': Whether SIMPLACE was shut down

    Raises:
        Exception: Any exception raised by SIMPLACE or FraNchEstYN will propagate unless handled internally.
    """

    shell = simplace_mod.init_simplace(simplace_config)
    try:
        simplace_mod.run_simplace(shell, simplace_config, project_lines=project_lines)


        dirs = simplace.getSimplaceDirectories(shell)
        work_root = Path(dirs["_WORKDIR_"])
        output_root = Path(dirs["_OUTPUTDIR_"])

        selected_line = project_lines[0]
        project_row = simplace_mod.get_project_row(work_root, selected_line=selected_line)

        crop_model_path = simplace_mod.export_crop_model_data(output_root, project_row)
        weather_path = simplace_mod.convert_weather(work_root, output_root, project_row["location"])
        management_path = simplace_mod.build_management(output_root, project_row)

        start_year = int(project_row["startdate"].split(".")[-1])
        end_year = int(project_row["enddate"].split(".")[-1])

        result = franchestyn_mod.run_franchestyn(
            weather_path=str(weather_path),
            management_path=str(management_path),
            cropmodel_path=str(crop_model_path),
            start_year=start_year,
            end_year=end_year,
            config=franchestyn_config,
        )

        summary = result.get("outputs", {}).get("summary", {})
        simulation = result.get("outputs", {}).get("simulation", [])

        franchestyn_mod.save_simulation_results_csv(simulation, output_root)

        simulation_df = pd.DataFrame(simulation)
        season_summary = franchestyn_mod.build_season_summary(
            simulation_df,
            site=franchestyn_config.site,
            variety=franchestyn_config.variety,
        )
        franchestyn_mod.save_season_summary_csv(season_summary, output_root)

        best_params = summary.get("best_params", {}) if isinstance(summary, dict) else {}
        franchestyn_mod.save_calibrated_parameters_csv(
            best_params,
            output_root,
            site=franchestyn_config.site,
            variety=franchestyn_config.variety,
        )

        merged_path = simplace_mod.merge_simplace_and_franchestyn(output_root, project_row, simulation_df)

        return {
            "work_root": str(work_root),
            "output_root": str(output_root),
            "project_row": project_row,
            "merged_csv": str(merged_path),
            "result": result,
            "simplace_shutdown": bool(shutdown_simplace),
        }
    finally:
        if shutdown_simplace:
            try:
                simplace.shutDown(shell)
            except Exception as exc:
                print(f"Warning: SIMPLACE shutdown failed: {exc}")
