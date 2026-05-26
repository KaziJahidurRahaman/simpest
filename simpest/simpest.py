"""Main module."""
from pathlib import Path

import pandas as pd
import simplace

from simpest.models.franchestyn import (
    FranchestynConfig,
    build_season_summary,
    run_franchestyn,
    save_calibrated_parameters_csv,
    save_season_summary_csv,
    save_simulation_results_csv,
)
from simpest.models.simplace import (
    SimplaceConfig,
    build_management,
    convert_weather,
    export_crop_model_data,
    get_project_row,
    init_simplace,
    merge_simplace_and_franchestyn,
    run_simplace,
)


def run_pipeline(
    simplace_config: SimplaceConfig,
    franchestyn_config: FranchestynConfig,
    project_lines: list[int],
    shutdown_simplace: bool = False,
) -> dict:
    """Run the full SIMPLACE -> FraNchEstYN pipeline.

    Args:
        simplace_config: Configuration used to initialize and run SIMPLACE.
        franchestyn_config: Configuration used to run FraNchEstYN.
        project_lines: SIMPLACE project line numbers to run.
        shutdown_simplace: If True, calls ``simplace.shutDown(shell)`` in
            ``finally``. Keep this False in Jupyter to avoid possible kernel
            restarts when the underlying JVM is terminated.

    Returns:
        A dictionary containing paths, selected project metadata, merged CSV
        location, FraNchEstYN result payload, and shutdown state.
    """
    shell = init_simplace(simplace_config)
    try:
        run_simplace(shell, simplace_config, project_lines=project_lines)

        dirs = simplace.getSimplaceDirectories(shell)
        work_root = Path(dirs["_WORKDIR_"])
        output_root = Path(dirs["_OUTPUTDIR_"])

        selected_line = project_lines[0]
        project_row = get_project_row(work_root, selected_line=selected_line)

        crop_model_path = export_crop_model_data(output_root, project_row)
        weather_path = convert_weather(work_root, output_root, project_row["location"])
        management_path = build_management(output_root, project_row)

        start_year = int(project_row["startdate"].split(".")[-1])
        end_year = int(project_row["enddate"].split(".")[-1])

        result = run_franchestyn(
            weather_path=str(weather_path),
            management_path=str(management_path),
            cropmodel_path=str(crop_model_path),
            start_year=start_year,
            end_year=end_year,
            config=franchestyn_config,
        )

        summary = result.get("outputs", {}).get("summary", {})
        simulation = result.get("outputs", {}).get("simulation", [])

        save_simulation_results_csv(simulation, output_root)

        simulation_df = pd.DataFrame(simulation)
        season_summary = build_season_summary(
            simulation_df,
            site=franchestyn_config.site,
            variety=franchestyn_config.variety,
        )
        save_season_summary_csv(season_summary, output_root)

        best_params = summary.get("best_params", {}) if isinstance(summary, dict) else {}
        save_calibrated_parameters_csv(
            best_params,
            output_root,
            site=franchestyn_config.site,
            variety=franchestyn_config.variety,
        )

        merged_path = merge_simplace_and_franchestyn(output_root, project_row, simulation_df)

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
