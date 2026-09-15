"""SciPy Nelder–Mead calibration for the simulation model.

The same downhill-simplex search as :mod:`simpest.models.fr_optimizer`, but
driven by :func:`scipy.optimize.minimize` (``method="Nelder-Mead"``) instead of
the bespoke implementation. Useful as a maintained reference to cross-check the
hand-written optimizer against, and it relies on SciPy's native ``bounds``
handling rather than a large-penalty term.

Like the built-in optimizer this runs a *multi-start* search: SciPy's
Nelder–Mead is a single local minimisation, so it is launched from
``n_restarts`` random starting points drawn inside the parameter bounds and the
best result is kept. ``max_iter`` caps the iterations per start and ``ftol`` is
the convergence tolerance on the objective spread. Supplying a ``seed`` makes
the random starts reproducible.
"""

from __future__ import annotations

import math
import sys
from typing import Dict, Optional, Set

import numpy as np
import scipy.optimize

from .fr_optimizer_base import BaseFranchestynOptimizer
from .fr_runner import FranchestynRunner


class ScipyNelderMeadOptimizer(BaseFranchestynOptimizer):
    """Multi-start SciPy Nelder–Mead calibration for a simpest simulation run.

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

        self._current_restart = 0
        self._iter_in_restart = 0
        self._best_rmse = math.inf
        self._best_x: Optional[np.ndarray] = None

    @classmethod
    def from_config(cls, runner, config, disabled_by_class=None) -> "ScipyNelderMeadOptimizer":
        """Build from a :class:`FranchestynConfig` (reads ``n_restarts`` /
        ``max_iter`` / ``ftol`` / ``seed``)."""
        return cls(
            runner=runner,
            calibration_variable=config.calibration_variable,
            disabled_by_class=disabled_by_class,
            seed=getattr(config, "seed", None),
            n_restarts=config.n_restarts,
            max_iter=config.max_iter,
            ftol=getattr(config, "ftol", 1e-12),
        )

    def calibrate(self) -> Dict[str, float]:
        """
        Run the multi-start SciPy Nelder–Mead search and return the best set.

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

        rng = np.random.default_rng(self.seed)

        # Best-so-far across every evaluation of every restart. Tracked in the
        # wrapped objective so it needs no extra model runs, and so the returned
        # point is the true global best rather than a restart's final vertex
        # (SciPy's bounded Nelder-Mead can end on a slightly worse vertex).
        self._best_rmse = math.inf
        self._best_x = None

        def objective(x: np.ndarray) -> float:
            val = self._evaluate_rmse(x)
            if val < self._best_rmse:
                self._best_rmse = val
                self._best_x = np.array(x, dtype=float)
            return val

        for restart in range(self.n_restarts):
            self._current_restart = restart + 1
            self._iter_in_restart = 0

            x0 = np.array([rng.uniform(lo, hi) for lo, hi in self.bounds], dtype=float)
            scipy.optimize.minimize(
                objective,
                x0,
                method="Nelder-Mead",
                bounds=self.bounds,
                options={"maxiter": self.max_iter, "fatol": self.ftol, "xatol": 1e-9},
                callback=self._on_iteration,
            )

        best_params = {key: float(v) for key, v in zip(self.calib_keys, self._best_x)}

        print(f"\nBest RMSE: {self._best_rmse:.4f}")
        print(best_params)

        return best_params

    def _on_iteration(self, _xk: np.ndarray) -> None:
        """Progress callback for each simplex iteration."""
        self._iter_in_restart += 1
        sys.stdout.write(
            f"\rRun {self._current_restart}/{self.n_restarts} "
            f"Iteration {self._iter_in_restart}/{self.max_iter} "
            f"CURR RMSE={self._last_rmse:.4f} BEST RMSE={self._best_rmse:.4f}"
        )
        sys.stdout.flush()
