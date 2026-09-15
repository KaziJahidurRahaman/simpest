"""Optuna-based calibration for the simulation model.

An alternative to the multi-start Nelder–Mead search in
:mod:`simpest.models.fr_optimizer`. Instead of refining local simplexes, this
optimizer uses Optuna's Tree-structured Parzen Estimator (TPE) sampler: a
sequential model-based ("Bayesian") search that builds a probabilistic picture
of which regions of the parameter space produce low RMSE and concentrates
sampling there. It copes well with the noisy, non-smooth objective surface the
crop/disease model produces and keeps every candidate inside the parameter
bounds by construction.

The search is controlled by ``n_trials`` (how many parameter sets to evaluate)
and an optional ``timeout`` (wall-clock cap in seconds). Supplying a ``seed``
makes the TPE sampling reproducible.
"""

from __future__ import annotations

import sys
from typing import Dict, Optional, Set

import numpy as np
import optuna

from .fr_optimizer_base import BaseFranchestynOptimizer
from .fr_runner import FranchestynRunner


class OptunaOptimizer(BaseFranchestynOptimizer):
    """Optuna (TPE) calibration for a simpest simulation run.

    Wraps a configured runner and searches for the parameter set that minimises
    the run's RMSE against reference data. Only parameters flagged for
    calibration (and not explicitly disabled) within the requested scope are
    optimised; all others are held at their default values.

    Args:
        runner (FranchestynRunner): Fully configured runner instance.
        calibration_variable (str): Calibration target scope: ``"crop"``,
            ``"disease"``, or ``"all"``.
        disabled_by_class (Optional[Dict[str, Set[str]]]): Parameter names to
            exclude from calibration, keyed by parameter class.
        seed (Optional[int]): Seed for the TPE sampler, giving a reproducible
            search. ``None`` (default) is non-deterministic.
        n_trials (int): Number of parameter sets to evaluate.
        timeout (Optional[float]): Optional wall-clock limit in seconds; the
            search stops once either ``n_trials`` or ``timeout`` is reached.
    """

    def __init__(
        self,
        runner: FranchestynRunner,
        calibration_variable: str = "all",
        disabled_by_class: Optional[Dict[str, Set[str]]] = None,
        seed: Optional[int] = None,
        n_trials: int = 100,
        timeout: Optional[float] = None,
    ) -> None:
        super().__init__(
            runner=runner,
            calibration_variable=calibration_variable,
            disabled_by_class=disabled_by_class,
            seed=seed,
        )
        self.n_trials = n_trials
        self.timeout = timeout

    @classmethod
    def from_config(cls, runner, config, disabled_by_class=None) -> "OptunaOptimizer":
        """Build from a :class:`FranchestynConfig` (reads ``n_trials`` /
        ``optuna_timeout`` / ``seed``)."""
        return cls(
            runner=runner,
            calibration_variable=config.calibration_variable,
            disabled_by_class=disabled_by_class,
            seed=getattr(config, "seed", None),
            n_trials=getattr(config, "n_trials", 100),
            timeout=getattr(config, "optuna_timeout", None),
        )

    def calibrate(self) -> Dict[str, float]:
        """
        Run the Optuna study and return the best parameter set.

        Returns:
            Dict[str, float]: Best-fit parameter values keyed by
            ``class_ParamName``.
        """

        if not self.calib_keys:
            print("No calibration parameters found — returning defaults.")
            return {}

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        print(
            f"- Calibrating {len(self.calib_keys)} using Optuna (TPE), "
            f"n_trials={self.n_trials}.\n- Parameters:\n{self.calib_keys}"
        )

        def objective(trial: "optuna.Trial") -> float:
            x = np.array(
                [
                    trial.suggest_float(key, lo, hi)
                    for key, (lo, hi) in zip(self.calib_keys, self.bounds)
                ],
                dtype=float,
            )
            return self._evaluate_rmse(x)

        def _progress(study: "optuna.Study", trial: "optuna.trial.FrozenTrial") -> None:
            """Print a one-line trial / best-so-far progress update."""
            done = trial.number + 1
            curr = trial.value if trial.value is not None else float("nan")
            sys.stdout.write(
                f"\rTrial {done}/{self.n_trials} CURR RMSE={curr:.4f} BEST RMSE={study.best_value:.4f}"
            )
            sys.stdout.flush()

        sampler = optuna.samplers.TPESampler(seed=self.seed)
        study = optuna.create_study(direction="minimize", sampler=sampler)
        study.optimize(
            objective,
            n_trials=self.n_trials,
            timeout=self.timeout,
            show_progress_bar=False,
            callbacks=[_progress],
        )

        best_params = {key: float(study.best_params[key]) for key in self.calib_keys}

        print(f"\nBest RMSE: {study.best_value:.4f}")
        print(best_params)

        return best_params
