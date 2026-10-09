"""A FraNchEstYN session: one set of inputs, one runner, many runs.

Calibration, sensitivity screens and forward checks all run the model many
times on the same inputs. `FranchestynSession` keeps those inputs together,
builds the `FranchestynRunner` once (see `franchestyn.build_runner`) and
offers the runs a notebook needs: a plain run, a forward run with the current
parameter values, and a calibration repeated over several seeds.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import pandas as pd

from . import franchestyn
from .fr_runner import FranchestynRunner


@dataclass
class SeedCalibration:
    """Result of `FranchestynSession.calibrate_seeds`.

    Attributes:
        best: The winning run: ``{"seed", "rmse", "best_params"}``.
        runs: Every run, in seed order, in the same form.
        start_params: The parameter specs every seed started from, as
            ``{"crop": ..., "disease": ..., "fungicide": ...}``.
    """

    best: dict
    runs: list
    start_params: dict

    def table(self) -> pd.DataFrame:
        """RMSE and fitted values of every seed, one column per seed."""
        return pd.DataFrame({
            r["seed"]: {"RMSE": r["rmse"], **{k.split("_", 1)[1]: v for k, v in r["best_params"].items()}}
            for r in self.runs
        })


class FranchestynSession:
    """Runs FraNchEstYN repeatedly on one set of inputs.

    The runner is built on first use and reused for every later run, so the
    input CSVs are written and the weather is parsed only once. Every run
    applies the config's *current* parameter specs, so edit
    ``session.config`` between runs to change what is calibrated or fixed.

    Args:
        config: Run and calibration settings. Held by reference, not copied.
        start_year: First simulated year.
        end_year: Last simulated year.
        weather_df: Weather input.
        management_df: Management (sowing) input.
        crop_model_df: External crop-model series.
        ref_df: Reference observations.
    """

    def __init__(
        self,
        config: franchestyn.FranchestynConfig,
        start_year: int,
        end_year: int,
        weather_df: pd.DataFrame,
        management_df: pd.DataFrame,
        crop_model_df: pd.DataFrame,
        ref_df: pd.DataFrame,
    ):
        self.config = config
        self.inputs = dict(
            start_year=start_year, end_year=end_year, weather_df=weather_df,
            management_df=management_df, crop_model_df=crop_model_df, ref_df=ref_df,
        )
        self._runner: FranchestynRunner | None = None

    @property
    def runner(self) -> FranchestynRunner:
        """The session's runner, built on first access."""
        if self._runner is None:
            self._runner = franchestyn.build_runner(config=self.config, **self.inputs)
        return self._runner

    def run(self) -> dict:
        """Run (and calibrate, if ``config.is_calibration``) with the current specs.

        Returns:
            dict: The result of `franchestyn.run_franchestyn`.
        """
        return franchestyn.run_franchestyn(config=self.config, runner=self.runner, **self.inputs)

    def run_forward(self) -> dict:
        """Run once with every parameter at its current value, without calibrating.

        ``config.is_calibration`` is restored afterwards.

        Returns:
            dict: The result of `franchestyn.run_franchestyn`.
        """
        was_calibration = self.config.is_calibration
        self.config.is_calibration = False
        try:
            return self.run()
        finally:
            self.config.is_calibration = was_calibration

    def calibrate_seeds(self, seeds, n_trials: int) -> SeedCalibration:
        """Calibrate from the same start once per seed and keep the best run.

        Every seed starts from the parameter specs the config holds when this
        is called. Afterwards the config holds those start specs with the
        best seed's values written in, ready for `run_forward`.

        Args:
            seeds: Optimizer seeds to try.
            n_trials: Trial budget per seed (``config.n_trials``).

        Returns:
            SeedCalibration: The best run, every run and the start specs.
        """
        cfg = self.config
        start_params = {
            "crop": copy.deepcopy(cfg.crop_parameters),
            "disease": copy.deepcopy(cfg.disease_parameters),
            "fungicide": copy.deepcopy(cfg.fungicide_parameters),
        }

        runs = []
        for seed in seeds:
            self._restore(start_params)
            cfg.is_calibration, cfg.n_trials, cfg.seed = True, n_trials, seed
            summary = self.run()["outputs"]["summary"]
            runs.append({"seed": seed, "rmse": summary["rmse"], "best_params": summary["best_params"]})
            print(f"seed {seed}: RMSE={summary['rmse']:.3f}")

        best = min(runs, key=lambda r: r["rmse"])
        self._restore(start_params)
        franchestyn.apply_best_params(cfg, best["best_params"])
        return SeedCalibration(best=best, runs=runs, start_params=start_params)

    def _restore(self, params: dict) -> None:
        self.config.crop_parameters = copy.deepcopy(params["crop"])
        self.config.disease_parameters = copy.deepcopy(params["disease"])
        self.config.fungicide_parameters = copy.deepcopy(params["fungicide"])
