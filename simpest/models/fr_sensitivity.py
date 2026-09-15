"""Morris elementary-effects screening of the calibration parameters.

Before calibrating, it is worth knowing which parameters actually move the
objective. This module runs a Morris (elementary effects) screen: it perturbs
one parameter at a time along random trajectories through the bounded parameter
space and, for each parameter, reports

* ``mu_star`` – the mean absolute elementary effect on the run's RMSE against
  the reference data. High ``mu_star`` means the parameter is influential.
* ``sigma`` – the spread of the elementary effects. High ``sigma`` relative to
  ``mu_star`` means the effect is non-linear or interacts with other
  parameters.

Parameters with ``mu_star`` indistinguishable from zero can be frozen at their
defaults and dropped from calibration. The screen costs roughly
``n_trajectories * (k + 1)`` model runs for ``k`` parameters.

The parameter selection and the RMSE objective are the same as the optimizers'
(:class:`~simpest.models.fr_optimizer_base.BaseFranchestynOptimizer`); this
class only swaps the search for a screen, so use :meth:`MorrisSensitivity.analyze`
— it does not calibrate.
"""

from __future__ import annotations

import sys
from typing import Dict, Optional, Set

import numpy as np
import pandas as pd
from SALib.analyze import morris as _morris_analyze
from SALib.sample import morris as _morris_sample

from .fr_optimizer_base import BaseFranchestynOptimizer
from .fr_runner import FranchestynRunner

_TABLE_COLUMNS = ["parameter", "mu", "mu_star", "sigma", "mu_star_conf"]


class MorrisSensitivity(BaseFranchestynOptimizer):
    """Morris (elementary effects) screening for a simpest calibration run.

    Reuses :class:`~simpest.models.fr_optimizer_base.BaseFranchestynOptimizer`
    for parameter selection and the shared RMSE objective, but *screens* rather
    than optimises: it ranks each calibratable parameter by how strongly it
    moves the run's RMSE against the reference data. Call :meth:`analyze`; the
    inherited :meth:`calibrate` is not implemented for this class.

    Args:
        runner (FranchestynRunner): Fully configured runner instance.
        calibration_variable (str): Calibration target scope: ``"crop"``,
            ``"disease"``, or ``"all"``.
        disabled_by_class (Optional[Dict[str, Set[str]]]): Parameter names to
            exclude, keyed by parameter class.
        seed (Optional[int]): Seed for the trajectory sampling, giving a
            reproducible screen. ``None`` (default) is non-deterministic.
        n_trajectories (int): Number of Morris trajectories (SALib ``N``). The
            screen runs ``n_trajectories * (k + 1)`` model evaluations.
        num_levels (int): Number of grid levels per parameter (SALib
            ``num_levels``); 4 is the usual choice.
    """

    def __init__(
        self,
        runner: FranchestynRunner,
        calibration_variable: str = "all",
        disabled_by_class: Optional[Dict[str, Set[str]]] = None,
        seed: Optional[int] = None,
        n_trajectories: int = 20,
        num_levels: int = 4,
    ) -> None:
        super().__init__(
            runner=runner,
            calibration_variable=calibration_variable,
            disabled_by_class=disabled_by_class,
            seed=seed,
        )
        self.n_trajectories = n_trajectories
        self.num_levels = num_levels

    @classmethod
    def from_config(cls, runner, config, disabled_by_class=None) -> "MorrisSensitivity":
        """Build from a :class:`FranchestynConfig`-like object (reads
        ``sa_n_trajectories`` / ``sa_num_levels`` / ``seed`` if present)."""
        return cls(
            runner=runner,
            calibration_variable=config.calibration_variable,
            disabled_by_class=disabled_by_class,
            seed=getattr(config, "seed", None),
            n_trajectories=getattr(config, "sa_n_trajectories", 20),
            num_levels=getattr(config, "sa_num_levels", 4),
        )

    def analyze(self) -> pd.DataFrame:
        """
        Run the Morris screen and return a ranked sensitivity table.

        Returns:
            pd.DataFrame: One row per calibratable parameter, columns
            ``parameter``, ``mu``, ``mu_star``, ``sigma``, ``mu_star_conf``,
            sorted by ``mu_star`` descending. Empty when nothing is flagged for
            calibration.
        """
        if not self.calib_keys:
            print("No calibration parameters found — nothing to screen.")
            return pd.DataFrame(columns=_TABLE_COLUMNS)

        degenerate = [k for k, (lo, hi) in zip(self.calib_keys, self.bounds) if lo >= hi]
        if degenerate:
            raise ValueError(
                "Morris screening needs a non-empty range for every parameter; "
                f"these have min >= max: {degenerate}. Widen their bounds or "
                "disable them from calibration."
            )

        problem = {
            "num_vars": len(self.calib_keys),
            "names": list(self.calib_keys),
            "bounds": [[float(lo), float(hi)] for lo, hi in self.bounds],
        }

        x_samples = _morris_sample.sample(
            problem, N=self.n_trajectories, num_levels=self.num_levels, seed=self.seed
        )
        n_runs = len(x_samples)
        print(
            f"- Morris screening {len(self.calib_keys)} parameters: "
            f"{self.n_trajectories} trajectories -> {n_runs} model runs.\n"
            f"- Parameters:\n{self.calib_keys}"
        )

        y = np.empty(n_runs, dtype=float)
        best = np.inf
        for i, x in enumerate(x_samples):
            y[i] = self._evaluate_rmse(x)
            best = min(best, y[i])
            sys.stdout.write(
                f"\rRun {i + 1}/{n_runs} CURR RMSE={self._last_rmse:.4f} BEST RMSE={best:.4f}"
            )
            sys.stdout.flush()
        print()

        n_failed = int(np.count_nonzero(y >= 1e299))
        if n_failed:
            print(
                f"WARNING: {n_failed}/{n_runs} runs hit the failure penalty "
                f"(1e300); elementary effects for the parameters involved are "
                f"inflated."
            )

        si = _morris_analyze.analyze(
            problem,
            x_samples,
            y,
            num_levels=self.num_levels,
            seed=self.seed,
            print_to_console=False,
        )

        table = (
            pd.DataFrame(
                {
                    "parameter": si["names"],
                    "mu": si["mu"],
                    "mu_star": si["mu_star"],
                    "sigma": si["sigma"],
                    "mu_star_conf": si["mu_star_conf"],
                }
            )
            .sort_values("mu_star", ascending=False)
            .reset_index(drop=True)
        )

        print(table.to_string(index=False))
        return table


def run_morris_sensitivity(
    start_year: int,
    end_year: int,
    config,
    weather_df: "pd.DataFrame",
    management_df: "pd.DataFrame",
    crop_model_df: "pd.DataFrame",
    ref_df: "pd.DataFrame",
    *,
    n_trajectories: int = 20,
    num_levels: int = 4,
    seed: int = 0,
    crop_param_file: Optional[str] = None,
    disease_param_file: Optional[str] = None,
    fungicide_param_file: Optional[str] = None,
) -> dict:
    """
    Morris-screen the calibratable parameters of a simpest run.

    Builds the runner once (via
    :func:`simpest.models.franchestyn._build_franchestyn_runner`) and evaluates
    it along Morris trajectories. Which parameters are screened, and their
    ranges, come from ``config`` exactly as for calibration — parameters flagged
    ``calibration`` in the crop/disease spec dicts, within
    ``config.calibration_variable`` scope, minus the disabled ones.

    Args:
        start_year (int): Start year for simulation.
        end_year (int): End year for simulation.
        config (FranchestynConfig): Simulation configuration object.
        weather_df, management_df, crop_model_df, ref_df (pd.DataFrame):
            In-memory inputs, as for :func:`run_franchestyn`.
        n_trajectories (int): Morris trajectories (~``n_trajectories*(k+1)`` runs).
        num_levels (int): Morris grid levels per parameter.
        seed (int): Seed for reproducible trajectory sampling.
        crop_param_file, disease_param_file, fungicide_param_file (str|None):
            Optional parameter-file overrides.

    Returns:
        dict: ``{"table": <DataFrame>, "n_runs": int, "settings": {...}}``. The
        table is the ranked output of :meth:`MorrisSensitivity.analyze`.
    """
    from .franchestyn import _build_franchestyn_runner

    disabled_by_class = {
        "crop": set(config.crop_disabled_params),
        "disease": set(config.disease_disabled_params),
    }

    with _build_franchestyn_runner(
        start_year,
        end_year,
        config,
        weather_df,
        management_df,
        crop_model_df,
        ref_df,
        crop_param_file,
        disease_param_file,
        fungicide_param_file,
    ) as runner:
        screen = MorrisSensitivity(
            runner=runner,
            calibration_variable=config.calibration_variable,
            disabled_by_class=disabled_by_class,
            seed=seed,
            n_trajectories=n_trajectories,
            num_levels=num_levels,
        )
        table = screen.analyze()

    return {
        "table": table,
        "n_runs": int(n_trajectories * (len(table) + 1)) if len(table) else 0,
        "settings": {
            "n_trajectories": n_trajectories,
            "num_levels": num_levels,
            "seed": seed,
        },
    }
