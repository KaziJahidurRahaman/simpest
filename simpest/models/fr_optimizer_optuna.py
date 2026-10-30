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

``n_jobs`` fans trials out across worker *processes* rather than the threads
Optuna's own ``study.optimize(n_jobs=...)`` would use -- threads don't help a
CPU-bound, pure-Python objective like this one (GIL). The main process keeps
the only study and runs a batched ask/tell loop: it asks the sampler for
``n_jobs`` trials, evaluates them concurrently in a persistent process pool,
and tells the results back in ask order. The sampler uses Optuna's
"constant liar" so trials within one batch spread out instead of piling onto
the same point. Because the order of every ask and tell is fixed, a run is
reproducible for a given ``seed`` and ``n_jobs``. Different ``n_jobs`` values
give different (but individually reproducible) trial sequences, because the
batch size changes what history each ask sees.
"""

from __future__ import annotations

import sys
from typing import Dict, Optional, Set

import numpy as np
import optuna

from .fr_optimizer_base import BaseFranchestynOptimizer
from .fr_runner import FranchestynRunner


# Each pool worker holds one optimizer, sent once through the pool
# initializer, so a batch only has to ship parameter vectors, not the runner
# and its cached weather series.
_WORKER_OPTIMIZER: Optional["OptunaOptimizer"] = None


def _init_worker(optimizer: "OptunaOptimizer") -> None:
    global _WORKER_OPTIMIZER
    _WORKER_OPTIMIZER = optimizer


def _evaluate_in_worker(x: np.ndarray) -> float:
    return _WORKER_OPTIMIZER._evaluate_rmse(x)


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
        n_jobs (int): Number of trials to evaluate concurrently in worker
            processes. ``1`` (default) runs sequentially in-process. Larger
            values evaluate trials in batches of ``n_jobs`` (see the module
            docstring); results are reproducible for a fixed ``seed`` and
            ``n_jobs``.
    """

    def __init__(
        self,
        runner: FranchestynRunner,
        calibration_variable: str = "all",
        disabled_by_class: Optional[Dict[str, Set[str]]] = None,
        seed: Optional[int] = None,
        n_trials: int = 100,
        timeout: Optional[float] = None,
        n_jobs: int = 1,
    ) -> None:
        super().__init__(
            runner=runner,
            calibration_variable=calibration_variable,
            disabled_by_class=disabled_by_class,
            seed=seed,
        )
        self.n_trials = n_trials
        self.timeout = timeout
        self.n_jobs = n_jobs

    @classmethod
    def from_config(cls, runner, config, disabled_by_class=None) -> "OptunaOptimizer":
        """Build from a :class:`FranchestynConfig` (reads ``n_trials`` /
        ``optuna_timeout`` / ``seed`` / ``n_jobs``)."""
        return cls(
            runner=runner,
            calibration_variable=config.calibration_variable,
            disabled_by_class=disabled_by_class,
            seed=getattr(config, "seed", None),
            n_trials=getattr(config, "n_trials", 100),
            timeout=getattr(config, "optuna_timeout", None),
            n_jobs=getattr(config, "n_jobs", 1),
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

        if self.n_jobs == 1:
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
            best_value = study.best_value
        else:
            import time
            from concurrent.futures import ProcessPoolExecutor

            n_workers = min(self.n_jobs, self.n_trials)
            print(
                f"- Evaluating trials in batches of {n_workers} across "
                f"{n_workers} worker processes."
            )
            sampler = optuna.samplers.TPESampler(seed=self.seed, constant_liar=True)
            study = optuna.create_study(direction="minimize", sampler=sampler)
            started = time.monotonic()

            with ProcessPoolExecutor(
                max_workers=n_workers, initializer=_init_worker, initargs=(self,)
            ) as pool:
                while len(study.trials) < self.n_trials:
                    if self.timeout is not None and time.monotonic() - started > self.timeout:
                        break
                    batch = min(n_workers, self.n_trials - len(study.trials))
                    trials = [study.ask() for _ in range(batch)]
                    xs = [
                        np.array(
                            [
                                t.suggest_float(key, lo, hi)
                                for key, (lo, hi) in zip(self.calib_keys, self.bounds)
                            ],
                            dtype=float,
                        )
                        for t in trials
                    ]
                    for trial, value in zip(trials, pool.map(_evaluate_in_worker, xs)):
                        study.tell(trial, value)
                        _progress(study, study.trials[trial.number])

            best_params = {key: float(study.best_params[key]) for key in self.calib_keys}
            best_value = study.best_value

        print(f"\nBest RMSE: {best_value:.4f}")
        print(best_params)

        return best_params
