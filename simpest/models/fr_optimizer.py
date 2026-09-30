"""Nelder–Mead calibration for the simulation model.

A downhill-simplex search driven by :func:`scipy.optimize.minimize`
(``method="Nelder-Mead"``), relying on SciPy's native ``bounds`` handling
rather than a large-penalty term.

This runs a *multi-start* search: SciPy's Nelder–Mead is a single local
minimisation, so it is launched from ``n_restarts`` random starting points
drawn inside the parameter bounds and the best result is kept. ``max_iter``
caps the iterations per start and ``ftol`` is the convergence tolerance on the
objective spread. Supplying a ``seed`` makes the random starts reproducible.
"""

from __future__ import annotations

import math
import sys
from typing import Dict, Optional, Set, Tuple

import numpy as np
import scipy.optimize

from .fr_optimizer_base import BaseFranchestynOptimizer
from .fr_runner import FranchestynRunner


def _run_restart(
    optimizer: "NelderMeadOptimizer", rng: np.random.Generator
) -> Tuple[float, np.ndarray]:
    """Run one restart of ``optimizer`` and return its ``(best_rmse, best_x)``.
    Module-level so ``joblib`` can pickle the call cleanly when dispatching
    restarts to worker processes."""
    optimizer._iter_in_restart = 0
    return optimizer._run_one_restart(rng)


class NelderMeadOptimizer(BaseFranchestynOptimizer):
    """Multi-start Nelder–Mead calibration for a simpest simulation run.

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
        seed (Optional[int]): Seed for the random-start generator, giving
            reproducible restarts. ``None`` (default) is non-deterministic.
        n_restarts (int): Number of independent minimisations (random restarts).
        max_iter (int): Maximum iterations per restart (SciPy ``maxiter``).
        ftol (float): Convergence tolerance on the objective spread across the
            simplex vertices (SciPy ``fatol``).
        n_jobs (int): Number of restarts to run concurrently via ``joblib``.
            ``1`` (default) runs restarts sequentially in-process, with the
            live per-iteration progress line. Any other value runs restarts
            in separate worker processes — each restart is fully independent,
            so this is embarrassingly parallel — but per-iteration progress
            isn't shown (child-process stdout doesn't interleave sensibly);
            joblib reports per-restart completion instead.
    """

    def __init__(
        self,
        runner: FranchestynRunner,
        calibration_variable: str = "all",
        disabled_by_class: Optional[Dict[str, Set[str]]] = None,
        seed: Optional[int] = None,
        n_restarts: int = 5,
        max_iter: int = 1000,
        ftol: float = 1e-12,
        n_jobs: int = 1,
    ) -> None:
        super().__init__(
            runner=runner,
            calibration_variable=calibration_variable,
            disabled_by_class=disabled_by_class,
            seed=seed,
        )
        self.n_restarts = n_restarts
        self.max_iter = max_iter
        self.ftol = ftol
        self.n_jobs = n_jobs

        self._current_restart = 0
        self._iter_in_restart = 0
        # Running best-so-far across restarts, for the live progress line
        # only (sequential mode). The value actually returned by calibrate()
        # is reduced from each restart's own local best (see
        # _run_one_restart), not from this.
        self._best_rmse = math.inf

    @classmethod
    def from_config(cls, runner, config, disabled_by_class=None) -> "NelderMeadOptimizer":
        """Build from a :class:`FranchestynConfig` (reads ``n_restarts`` /
        ``max_iter`` / ``ftol`` / ``seed`` / ``n_jobs``)."""
        return cls(
            runner=runner,
            calibration_variable=config.calibration_variable,
            disabled_by_class=disabled_by_class,
            seed=getattr(config, "seed", None),
            n_restarts=config.n_restarts,
            max_iter=config.max_iter,
            ftol=getattr(config, "ftol", 1e-12),
            n_jobs=getattr(config, "n_jobs", 1),
        )

    def calibrate(self) -> Dict[str, float]:
        """
        Run the multi-start Nelder–Mead search and return the best set.

        Returns:
            Dict[str, float]: Best-fit parameter values keyed by
            ``class_ParamName``.
        """

        if not self.calib_keys:
            print("No calibration parameters found — returning defaults.")
            return {}

        print(
            f"- Calibrating {len(self.calib_keys)} using SciPy Nelder–Mead "
            f"(multi-start).\n- Parameters:\n{self.calib_keys}"
        )

        # One independent RNG stream per restart, so results (and their
        # reproducibility under a fixed seed) don't depend on whether
        # restarts run sequentially or concurrently.
        child_rngs = np.random.default_rng(self.seed).spawn(self.n_restarts)

        if self.n_jobs == 1:
            self._best_rmse = math.inf
            results = []
            for restart, child_rng in enumerate(child_rngs):
                self._current_restart = restart + 1
                self._iter_in_restart = 0
                results.append(self._run_one_restart(child_rng))
        else:
            from joblib import Parallel, delayed

            print(
                f"- Running {self.n_restarts} restarts across up to {self.n_jobs} "
                f"worker processes (per-iteration progress is not shown while "
                f"parallel; per-restart completion is reported below)."
            )
            results = Parallel(n_jobs=self.n_jobs, verbose=10)(
                delayed(_run_restart)(self, child_rng) for child_rng in child_rngs
            )

        best_rmse, best_x = min(results, key=lambda r: r[0])
        best_params = {key: float(v) for key, v in zip(self.calib_keys, best_x)}

        print(f"\nBest RMSE: {best_rmse:.4f}")
        print(best_params)

        return best_params

    def _run_one_restart(self, rng: np.random.Generator) -> Tuple[float, np.ndarray]:
        """Run a single restart from a random start point in ``rng``.

        Returns the best RMSE/point seen across every evaluation of *this*
        restart, tracked via a wrapped objective since SciPy's bounded
        Nelder-Mead can end on a slightly worse vertex than its best visited
        point.
        """
        best_rmse = math.inf
        best_x: Optional[np.ndarray] = None

        def objective(x: np.ndarray) -> float:
            nonlocal best_rmse, best_x
            val = self._evaluate_rmse(x)
            if val < best_rmse:
                best_rmse = val
                best_x = np.array(x, dtype=float)
            if val < self._best_rmse:
                self._best_rmse = val
            return val

        x0 = np.array([rng.uniform(lo, hi) for lo, hi in self.bounds], dtype=float)
        scipy.optimize.minimize(
            objective,
            x0,
            method="Nelder-Mead",
            bounds=self.bounds,
            options={"maxiter": self.max_iter, "fatol": self.ftol, "xatol": 1e-9},
            callback=self._on_iteration,
        )
        return best_rmse, best_x

    def _on_iteration(self, _xk: np.ndarray) -> None:
        """Progress callback for each simplex iteration.

        A no-op when restarts run in parallel (``n_jobs != 1``): the callback
        would fire in a worker process, whose stdout doesn't interleave
        sensibly with the main process's — joblib's own per-restart
        completion reporting covers progress in that case instead.
        """
        self._iter_in_restart += 1
        if self.n_jobs != 1:
            return
        sys.stdout.write(
            f"\rRun {self._current_restart}/{self.n_restarts} "
            f"Iteration {self._iter_in_restart}/{self.max_iter} "
            f"CURR RMSE={self._last_rmse:.4f} BEST RMSE={self._best_rmse:.4f}"
        )
        sys.stdout.flush()
