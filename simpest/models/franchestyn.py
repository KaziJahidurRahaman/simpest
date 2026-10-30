from __future__ import annotations

import csv
import json
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from .fr_optimizer_base import build_optimizer
from .fr_param_reader import load_crop_specs, load_disease_specs, load_fungicide_specs
from .fr_runner import FranchestynRunner


def _resolve_local_model_file(filename: str) -> str:
    """
    Get the path to a local model file by filename.

    Args:
        filename (str): Name of the file to resolve.

    Returns:
        str: Path to the local model file.
    """
    return str(Path(__file__).with_name(filename))


def deactivate_calibration(params: dict, disable_list) -> dict:
    """
    Disable calibration for selected parameter names in a parameter section.

    Each parameter carries a boolean ``calibration`` flag, and deactivating a
    parameter simply forces that flag to ``False``. The calibrator then skips
    any parameter whose flag is off.

    The ``params`` mapping is mutated in place *and* returned, so both
    ``deactivate_calibration(specs, names)`` and
    ``specs = deactivate_calibration(specs, names)`` are valid.

    Args:
        params (dict): Mapping of parameter name to parameter specification dict
            (e.g. ``config.crop_parameters``).
        disable_list (Iterable[str]): Parameter names for which calibration
            should be forced to ``False``.

    Returns:
        dict: The same ``params`` mapping, with the selected flags set to False.
    """
    for param_name in disable_list:
        if param_name in params:
            params[param_name]["calibration"] = False
    return params


def apply_bound_overrides(params: dict, overrides: dict) -> dict:
    """
    Override parameter bounds (or any other spec field) in place.

    Use this to widen a bound for one analysis without editing the canonical
    ``fr_*_parameters.json`` files.

    Args:
        params (dict): Mapping of parameter name to parameter specification dict.
        overrides (dict): Mapping of parameter name to the fields to set, e.g.
            ``{"PathogenSpread": {"max": 0.6}}``.

    Returns:
        dict: The same ``params`` mapping.

    Raises:
        KeyError: If an override names a parameter that ``params`` lacks.
    """
    for name, fields in overrides.items():
        if name not in params:
            raise KeyError(f"no parameter {name!r} to override")
        params[name].update(fields)
    return params


def apply_best_params(config: "FranchestynConfig", best_params: dict) -> None:
    """
    Write calibrated values back into a config's parameter specs.

    Args:
        config (FranchestynConfig): Config whose ``crop_parameters``,
            ``disease_parameters`` and ``fungicide_parameters`` are updated in
            place.
        best_params (dict): Calibrated values keyed ``"<class>_<Name>"``, as
            returned in ``summary["best_params"]`` by :func:`run_franchestyn`.

    Raises:
        KeyError: If a key has an unknown class or names a missing parameter.
    """
    sections = {
        "crop": config.crop_parameters,
        "disease": config.disease_parameters,
        "fungicide": config.fungicide_parameters,
    }
    for key, value in best_params.items():
        cls, name = key.split("_", 1)
        sections[cls][name]["value"] = float(value)


def inject_reference_params(params: dict, model_name: str, reference_df: pd.DataFrame) -> tuple[list, list]:
    """
    Copy parameter values from a FraNchEstYN calibrated-parameters table.

    Each matching parameter gets the table's ``calibrated`` value and has its
    ``calibration`` flag turned off, so a run reproduces the reference
    engine's parameters instead of searching for its own. The ``calibrated``
    column holds the value the reference engine used: the fitted value where
    it calibrated the parameter, the default otherwise.

    Args:
        params (dict): Mapping of parameter name to parameter specification
            dict, updated in place.
        model_name (str): Which rows to use, by the table's ``Model`` column
            (case-insensitive), e.g. ``"crop"`` or ``"disease"``.
        reference_df (pd.DataFrame): Table with ``Model``, ``Parameter`` and
            ``calibrated`` columns.

    Returns:
        tuple[list, list]: ``(matched, unmatched)`` parameter names, so naming
        gaps between the table and ``params`` are visible. Rows whose value is
        not numeric are skipped.
    """
    matched, unmatched = [], []
    rows = reference_df[reference_df["Model"].astype(str).str.lower() == model_name.lower()]
    for _, r in rows.iterrows():
        name = str(r["Parameter"])
        try:
            val = float(r["calibrated"])
        except (TypeError, ValueError):
            continue
        if name in params:
            params[name]["value"] = val
            params[name]["calibration"] = False
            matched.append(name)
        else:
            unmatched.append(name)
    return matched, unmatched


@dataclass
class FranchestynConfig:
    """
    Configuration for a simpest crop-disease-fungicide simulation run.

    Attributes:
        param_file (str): Path to the main parameter file.
        crop_param_file (str): Path to crop parameter file.
        disease_param_file (str): Path to disease parameter file.
        fungicide_param_file (str): Path to fungicide parameter file.
        crop_type (str): Crop type (e.g., 'wheat').
        disease_type (str): Disease type (e.g., 'septoria'). Also the disease
            on/off switch — ``None`` skips the disease model entirely, giving a
            crop-only run.
        fungicide_type (str|None): Fungicide type (e.g., 'protectant'). Also the
            fungicide on/off switch — ``None`` skips the fungicide model. Set it
            whenever treatments are scheduled, or they will be ignored.
        site (str): Site name.
        variety (str): Variety name.
        disease (str): Disease name.
        is_calibration (bool): Whether to run calibration.
        calibration_variable (str): Calibration variable ('all', 'crop', 'disease').
        optimizer (str): Calibration optimizer to use. ``"nelder-mead"``
            (default) runs a multi-start Nelder–Mead simplex search via
            :func:`scipy.optimize.minimize`; ``"optuna"`` runs an Optuna
            (TPE / Bayesian) search; ``"multistart-simplex"`` runs the
            port of the C# FraNchEstYN multi-start simplex
            (:mod:`~simpest.models.fr_optimizer_simplex`).
        n_trials (int): Optuna trial budget (number of parameter sets to
            evaluate). Only used when ``optimizer="optuna"``.
        optuna_timeout (float|None): Optional wall-clock cap in seconds for the
            Optuna search. ``None`` (default) means no time limit. Only used
            when ``optimizer="optuna"``.
        seed (int|None): RNG seed for the optimizer, making the search
            reproducible. ``None`` (default) is non-deterministic. Applies to
            all optimizers.
        ftol (float): Nelder–Mead convergence tolerance on the objective spread
            across the simplex vertices. Used by ``optimizer="nelder-mead"``
            and ``"multistart-simplex"``.
        use_gdd (bool): Method for crop cycle completion. ``False`` (default)
            uses calendar-day interpolation of the external crop-model series;
            ``True`` uses the thermal-time (GDD) based cycle percentage.
        use_prev_day_alignment (bool): Day alignment used by the calibration
            objective. ``True`` (default) compares the previous simulated day to
            the current reference observation (sim[d-1] vs ref[d]); ``False``
            uses same-day alignment (sim[d] vs ref[d]).
        all_row_includes_end_year (bool): Whether an ``"All"`` sowing row covers
            the final year. ``False`` (default) omits ``end_year``; ``True``
            includes it. Has no effect for per-year sowing CSVs.
        weather_time_step (str): Time step of ``weather_df``. ``"daily"``
            (default) synthesizes hourly temperature, RH and leaf wetness from
            daily values; ``"hourly"`` reads measured hourly records (columns
            ``date, hour, temp, prec, rh, rad`` with ``rad`` in MJ m-2 h-1),
            so relative humidity and leaf wetness come from the data.
        latitude (float): Site latitude in decimal degrees. Only needed when
            radiation must be estimated because ``weather_df`` has none.
        rain_limited_primary_inoculum (bool): Let a rainless spell starve a
            splash-borne epidemic of primary inoculum, by scaling external
            infection with the rain-detachment efficiency that already scales
            secondary spread. ``False`` (default) matches the C# model, where
            primary inoculum does not depend on rain.
        htt_state_dry_decay (float): Fraction of the accumulated hydro-thermal
            time state that survives a day whose dry spell exceeds
            ``DryCriticalInterruption``. ``1.0`` (default) matches the C# model,
            where the state only accumulates and the onset gate is therefore
            crossed in every season sooner or later. Below 1, a long dry spell
            can keep an epidemic from ever starting. Needs
            ``carry_dry_counter=True`` to have any effect.
        carry_dry_counter (bool): Carry the dry-hour counter across midnight.
            ``False`` (default) matches the C# model, where the counter is reset
            with each daily output and therefore never exceeds 23 — so a
            ``DryCriticalInterruption`` above 23 hours can never trigger and the
            parameter is inert.
        upper_leaf_pool (bool): Track a second tissue pool for the upper
            leaves, which emerge shortly before heading and are what field
            severity assessments usually score. The pool is infected by the
            same rule as the canopy and reported as ``UpperLeafSeverity``;
            with the switch on, the calibration objective compares the
            reference severity with it instead of ``DiseaseSeverity``. The
            canopy epidemic and the yield damage are not changed. ``False``
            (default) matches the C# model, which has a single pool.
        upper_leaf_cycle_start (float): Crop cycle percentage at which the
            upper leaves start to emerge. The default (36) and
            ``upper_leaf_cycle_full`` (52) are for the upper four leaves of
            soft red winter wheat cv. Monon at West Lafayette, Indiana: in
            1990 leaf F-2 was fully expanded 27 days, F-1 21 days and the flag
            leaf 14 days before spike emergence (Shaner and Buechley 1995,
            Plant Disease 79:928-938), which is 43, 47 and 52 % of the
            simulated crop cycle; the start is extrapolated to the emergence
            of F-3. Set both for another crop, cultivar or leaf layer.
        upper_leaf_cycle_full (float): Crop cycle percentage at which the
            upper leaves are fully emerged.
        score_harvest_after_maturity (bool): Score a yield observation dated
            after the simulated maturity of its season against the season's
            final simulated yield. Grain yield is measured at harvest, which
            can fall after the day the simulated crop matures, and the run has
            no output after maturity. ``False`` (default) matches the C# model,
            where such an observation is compared with zero output and gets
            the ×1000 zero-yield penalty.
        n_restarts (int): Number of calibration restarts (simplexes). Used by
            ``optimizer="nelder-mead"`` and ``"multistart-simplex"``.
        max_iter (int): Maximum calibration iterations per simplex. Used by
            ``optimizer="nelder-mead"`` and ``"multistart-simplex"``.
        n_jobs (int): Number of independent model evaluations to run
            concurrently via ``joblib``. ``1`` (default) runs sequentially
            in-process, matching prior behavior exactly. Used by
            ``optimizer="nelder-mead"``'s restarts, by ``optimizer="optuna"``
            (trials evaluated in reproducible batches of ``n_jobs``) and by
            :func:`~simpest.models.fr_sensitivity.run_morris_sensitivity`'s
            trajectory evaluations; each independent unit of work gets its own
            worker process, so results are unaffected beyond wall-clock time.
        crop_parameters (dict): Raw crop parameter spec dict for ``crop_type``,
            loaded from ``crop_param_file`` at construction. Editable in place;
            pass it through :func:`deactivate_calibration` to turn off
            calibration for selected parameters before running.
        disease_parameters (dict): Raw disease parameter spec dict for
            ``disease_type``, loaded from ``disease_param_file``.
        fungicide_parameters (dict): Raw fungicide parameter spec dict for
            ``fungicide_type`` (empty when ``fungicide_type`` is None).
        crop_disabled_params (frozenset[str]): Crop parameter names to hard-exclude
            from calibration. Legacy alternative to editing ``crop_parameters``;
            both are honored.
        disease_disabled_params (frozenset[str]): Disease parameter names to
            hard-exclude from calibration. Legacy alternative to editing
            ``disease_parameters``; both are honored.
    """
    param_file: str = ""
    crop_param_file: str = field(default_factory=lambda: _resolve_local_model_file("fr_crop_parameters.json"))
    disease_param_file: str = field(default_factory=lambda: _resolve_local_model_file("fr_disease_parameters.json"))
    fungicide_param_file: str = field(default_factory=lambda: _resolve_local_model_file("fr_fungicide_parameters.json"))
    crop_type: str = "wheat"
    disease_type: str = "septoria"
    fungicide_type: str | None = "protectant"
    site: str = "indiana"
    variety: str = "Generic"
    disease: str = "thisDisease"
    is_calibration: bool = True
    calibration_variable: str = "all"
    optimizer: str = "nelder-mead"
    n_trials: int = 100
    optuna_timeout: float | None = None
    seed: int | None = None
    ftol: float = 1e-12
    use_gdd: bool = False
    use_prev_day_alignment: bool = True
    all_row_includes_end_year: bool = False
    weather_time_step: str = "daily"
    latitude: float = 0.0
    rain_limited_primary_inoculum: bool = False
    htt_state_dry_decay: float = 1.0
    carry_dry_counter: bool = False
    upper_leaf_pool: bool = False
    upper_leaf_cycle_start: float = 36.0
    upper_leaf_cycle_full: float = 52.0
    score_harvest_after_maturity: bool = False
    n_restarts: int = 1
    max_iter: int = 100
    n_jobs: int = 1
    crop_disabled_params: frozenset[str] = frozenset()
    disease_disabled_params: frozenset[str] = frozenset()

    # Editable parameter spec dicts, populated from the JSON files in
    # __post_init__ (unless the caller supplied their own).
    crop_parameters: dict = field(default_factory=dict)
    disease_parameters: dict = field(default_factory=dict)
    fungicide_parameters: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Load the editable parameter spec dicts from the JSON files.

        Only fills a dict that the caller left empty, so an explicitly passed
        ``crop_parameters`` / ``disease_parameters`` / ``fungicide_parameters``
        is preserved.
        """
        if not self.crop_parameters:
            self.crop_parameters = load_crop_specs(self.crop_param_file, self.crop_type)
        if not self.disease_parameters and self.disease_type:
            self.disease_parameters = load_disease_specs(self.disease_param_file, self.disease_type)
        if not self.fungicide_parameters and self.fungicide_type:
            self.fungicide_parameters = load_fungicide_specs(self.fungicide_param_file, self.fungicide_type)


def _outputs_to_records(date_outputs):
    """
    Convert date_outputs from the simulation runner to a list of record dicts.

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
                "GrowingSeason": out.crop.growing_season,
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
                "UpperLeafEmerged": out.disease.upper_leaf_emerged,
                "UpperLeafSeverity": out.disease.upper_leaf_severity,
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


@contextmanager
def _build_franchestyn_runner(
    start_year: int,
    end_year: int,
    config: FranchestynConfig,
    weather_df: pd.DataFrame,
    management_df: pd.DataFrame,
    crop_model_df: pd.DataFrame | None,
    ref_df: pd.DataFrame,
    crop_param_file: str | None = None,
    disease_param_file: str | None = None,
    fungicide_param_file: str | None = None,
):
    """
    Yield a configured :class:`FranchestynRunner` backed by temporary CSVs of the
    in-memory input frames.

    The runner is valid only inside the ``with`` block; the temporary directory
    holding its input CSVs is removed on exit. Shared by :func:`run_franchestyn`
    and :func:`simpest.models.fr_sensitivity.run_morris_sensitivity` so both use
    one runner-construction path.

    Args:
        start_year (int): Start year for simulation.
        end_year (int): End year for simulation.
        config (FranchestynConfig): Simulation configuration object.
        weather_df (pd.DataFrame): In-memory weather input.
        management_df (pd.DataFrame): In-memory management input.
        crop_model_df (pd.DataFrame|None): In-memory external crop-model
            series, or ``None`` to grow the crop with the internal crop model.
        ref_df (pd.DataFrame): In-memory reference input.
        crop_param_file (str|None, optional): Path to crop parameter file.
        disease_param_file (str|None, optional): Path to disease parameter file.
        fungicide_param_file (str|None, optional): Path to fungicide parameter file.

    Yields:
        FranchestynRunner: Fully configured runner.
    """
    crop_param_file = crop_param_file or config.crop_param_file
    disease_param_file = disease_param_file or config.disease_param_file
    fungicide_param_file = fungicide_param_file or config.fungicide_param_file

    if weather_df is None or weather_df.empty:
        raise ValueError("weather_df must be a non-empty DataFrame.")
    if management_df is None or management_df.empty:
        raise ValueError("management_df must be a non-empty DataFrame.")
    if crop_model_df is not None and crop_model_df.empty:
        raise ValueError("crop_model_df must be None (internal crop model) or a non-empty DataFrame.")
    if ref_df is None or ref_df.empty:
        raise ValueError("ref_df must be a non-empty DataFrame.")

    def _write_temp_csv(df: pd.DataFrame, tmp_dir: str, name: str) -> str:
        path = Path(tmp_dir) / name
        df.to_csv(path, index=False)
        return str(path)

    with tempfile.TemporaryDirectory(prefix="simpest_franchestyn_") as tmp_dir:
        weather_input = _write_temp_csv(weather_df, tmp_dir, "weather.csv")
        management_input = _write_temp_csv(management_df, tmp_dir, "management.csv")
        crop_model_input = (
            None if crop_model_df is None else _write_temp_csv(crop_model_df, tmp_dir, "crop_model.csv")
        )
        reference_input = _write_temp_csv(ref_df, tmp_dir, "reference.csv")

        runner = FranchestynRunner(
            weather_dir=weather_input,
            param_file=config.param_file,
            sowing_file=management_input,
            ref_dir=reference_input,
            crop_model_dir=crop_model_input,
            site=config.site,
            variety=config.variety,
            disease=config.disease,
            start_year=start_year,
            end_year=end_year,
            weather_time_step=config.weather_time_step,
            latitude=config.latitude,
            rain_limited_primary_inoculum=config.rain_limited_primary_inoculum,
            htt_state_dry_decay=config.htt_state_dry_decay,
            carry_dry_counter=config.carry_dry_counter,
            upper_leaf_pool=config.upper_leaf_pool,
            upper_leaf_cycle_start=config.upper_leaf_cycle_start,
            upper_leaf_cycle_full=config.upper_leaf_cycle_full,
            score_harvest_after_maturity=config.score_harvest_after_maturity,
            calibration_variable=config.calibration_variable,
            is_calibration=config.is_calibration,
            use_gdd=config.use_gdd,
            use_prev_day_alignment=config.use_prev_day_alignment,
            all_row_includes_end_year=config.all_row_includes_end_year,
            crop_type=config.crop_type,
            crop_param_file=crop_param_file,
            disease_param_file=disease_param_file,
            disease_type=config.disease_type,
            fungicide_param_file=fungicide_param_file,
            fungicide_type=config.fungicide_type,
            crop_param_specs=config.crop_parameters or None,
            disease_param_specs=config.disease_parameters or None,
            fungicide_param_specs=config.fungicide_parameters or None,
        )
        yield runner


def build_runner(
    start_year: int,
    end_year: int,
    config: FranchestynConfig,
    weather_df: pd.DataFrame,
    management_df: pd.DataFrame,
    crop_model_df: pd.DataFrame | None,
    ref_df: pd.DataFrame,
    crop_param_file: str | None = None,
    disease_param_file: str | None = None,
    fungicide_param_file: str | None = None,
) -> FranchestynRunner:
    """
    Build a standalone :class:`FranchestynRunner` the caller owns.

    Unlike :func:`_build_franchestyn_runner`, the returned runner outlives the
    temporary input CSVs: its weather series is parsed once (and cached, see
    :meth:`FranchestynRunner._load_weather`) while the temp directory is still
    alive, then the runner is handed back with no further dependency on disk.

    Reuse the same runner across an entire calibration session — every stage
    of a staged sweep, every forward-model check, every calibration restart —
    via ``run_franchestyn(..., runner=this_runner)`` instead of rebuilding a
    fresh runner (and re-parsing weather/sowing/reference data) on every call.

    Args mirror :func:`run_franchestyn`.

    Returns:
        FranchestynRunner: A runner with its weather series already cached.
    """
    with _build_franchestyn_runner(
        start_year, end_year, config, weather_df, management_df, crop_model_df,
        ref_df, crop_param_file, disease_param_file, fungicide_param_file,
    ) as runner:
        runner._load_weather()  # populate the weather cache before the temp CSVs vanish
        return runner


def run_franchestyn(
    start_year: int,
    end_year: int,
    config: FranchestynConfig,
    weather_df: pd.DataFrame,
    management_df: pd.DataFrame,
    crop_model_df: pd.DataFrame | None,
    ref_df: pd.DataFrame,
    crop_param_file: str | None = None,
    disease_param_file: str | None = None,
    fungicide_param_file: str | None = None,
    runner: FranchestynRunner | None = None,
) -> dict:
    """
    Run the crop-disease-fungicide simulation with in-memory DataFrame inputs.

    Args:
        start_year (int): Start year for simulation.
        end_year (int): End year for simulation.
        config (FranchestynConfig): Simulation configuration object.
        weather_df (pd.DataFrame): In-memory weather input.
        management_df (pd.DataFrame): In-memory management input.
        crop_model_df (pd.DataFrame|None): In-memory external crop-model
            series, or ``None`` to grow the crop with the internal crop model.
        ref_df (pd.DataFrame): In-memory reference input.
        crop_param_file (str|None, optional): Path to crop parameter file.
        disease_param_file (str|None, optional): Path to disease parameter file.
        fungicide_param_file (str|None, optional): Path to fungicide parameter file.
        runner (FranchestynRunner|None, optional): A pre-built runner (from
            :func:`build_runner`) to reuse instead of constructing a new one.
            When given, ``weather_df``/``management_df``/``crop_model_df``/
            ``ref_df``/``*_param_file`` are ignored for construction purposes
            (the runner already has its own data) — only ``config``'s current
            parameter specs are applied, via
            :meth:`FranchestynRunner.update_param_specs`, before running.
            This skips the temp-CSV write and full runner rebuild that
            constructing a fresh runner requires, so repeated calls in a
            calibration loop don't re-parse weather/sowing/reference data
            every time.

    Returns:
        dict: Dictionary with simulation outputs and summary.
    """
    if runner is not None:
        runner.update_param_specs(
            config.crop_parameters, config.disease_parameters, config.fungicide_parameters,
        )
        best_params, date_outputs = _calibrate_or_run(runner, config)
        include_crop = config.calibration_variable in ("crop", "all")
        include_disease = config.calibration_variable in ("disease", "all")
        rmse = runner.compute_rmse(
            date_outputs, include_crop=include_crop, include_disease=include_disease,
        )
        records = _outputs_to_records(date_outputs)
    else:
        with _build_franchestyn_runner(
            start_year, end_year, config, weather_df, management_df, crop_model_df,
            ref_df, crop_param_file, disease_param_file, fungicide_param_file,
        ) as built_runner:
            best_params, date_outputs = _calibrate_or_run(built_runner, config)
            include_crop = config.calibration_variable in ("crop", "all")
            include_disease = config.calibration_variable in ("disease", "all")
            rmse = built_runner.compute_rmse(
                date_outputs, include_crop=include_crop, include_disease=include_disease,
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


def _calibrate_or_run(runner: FranchestynRunner, config: FranchestynConfig):
    """Run ``runner`` per ``config``, calibrating first if requested.

    Returns:
        tuple[dict, dict]: ``(best_params, date_outputs)``, where
        ``best_params`` is empty when ``config.is_calibration`` is ``False``.
    """
    best_params = {}
    if config.is_calibration:
        disabled_by_class = {
            "crop": set(config.crop_disabled_params),
            "disease": set(config.disease_disabled_params),
        }
        optimizer = build_optimizer(config.optimizer, runner, config, disabled_by_class)
        best_params = optimizer.calibrate()
        date_outputs = runner.run(param_values=best_params)
    else:
        date_outputs = runner.run()
    return best_params, date_outputs


def save_simulation_results_csv(res_ot_simulation, output_root: Path, filename: str = "franchestyn_simulation_results.csv") -> Path:
    """
    Save simulation results to a CSV file.

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
    """Aggregate daily simulation results into a per-season summary.

    Rows are grouped by growing season (sowing year) and, for each season, the
    function reports the epidemic and yield outcomes:

    - **AUDPC** (area under the disease progress curve) is the time integral of
      disease severity (expressed as a percentage) over the season, computed by
      the trapezoidal rule against the calendar date.
    - **Yield loss** is the gap between attainable and actual yield, reported
      both in absolute terms and as a percentage,
      ``(attainable − actual) / attainable × 100``.
    - Peak attainable and actual yield and above-ground biomass, peak disease
      severity, and season-level weather aggregates (mean temperatures and
      humidity, total precipitation, radiation, and leaf wetness) are also
      included where the corresponding columns are present.

    Args:
        df (pd.DataFrame): Daily simulation results.
        site (str): Site name written to each summary row.
        variety (str): Variety name written to each summary row.

    Returns:
        pd.DataFrame: One row per growing season; empty if the input has no
        valid post-sowing days.
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

    # Group by the crop's growing season (= sowing year). Fall back to the
    # calendar year only where the season label is missing or unset (0), so
    # seasons that cross 1 January are not split across two summary rows.
    if "GrowingSeason" not in d.columns:
        d["GrowingSeason"] = d["Date"].dt.year
    else:
        season = pd.to_numeric(d["GrowingSeason"], errors="coerce")
        d["GrowingSeason"] = season.where(season > 0, d["Date"].dt.year)

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
        r_like (bool, optional): If True, export an extended diagnostics table
            that includes each parameter's default, bounds, unit, calibration
            flag, and calibrated value, in addition to the calibrated values.

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
