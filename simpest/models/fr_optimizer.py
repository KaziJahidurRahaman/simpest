"""
optimizer.py – Nelder-Mead calibration wrapper for FraNchEstYN.

Wraps FranchestynRunner.compute_rmse() as a scipy.optimize.minimize objective.
Mirrors the multi-start simplex logic from optimizer.cs:ObjfuncVal().
"""

from __future__ import annotations

import math
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize, OptimizeResult

from .fr_runner import FranchestynRunner
from .fr_data import Parameter


class FranchestynOptimizer:
    """Nelder-Mead calibration for FraNchEstYN.

    Args:
        runner: Fully-configured runner instance (reads data from disk once).
        calibration_variable: ``'crop'``, ``'disease'``, or ``'all'``.
            Selects which parameters to calibrate.
        n_restarts: Number of independent Nelder-Mead restarts (multi-start).
        max_iter: Maximum number of iterations per restart.
    """

    def __init__(
        self,
        runner: FranchestynRunner,
        calibration_variable: str = "all",
        n_restarts: int = 5,
        max_iter: int = 1000,
    ) -> None:
        self.runner = runner
        self.calibration_variable = calibration_variable.lower()
        self.n_restarts = n_restarts
        self.max_iter = max_iter

        # Select calibration parameters and record their bounds
        self.calib_keys, self.bounds = self._select_calib_params()

        self._n_eval = 0
        self._current_restart = 0
        self._iter_in_restart = 0
        self._last_rmse = math.inf

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def calibrate(self) -> Dict[str, float]:
        """Run multi-start Nelder-Mead and return the best parameter set.

        Returns:
            Best-fit parameter values keyed by ``'ClassName_ParamName'``.
        """
        
        if not self.calib_keys:
            print("No calibration parameters found — returning defaults.")
            return {}

        best_rmse = math.inf
        best_params: Dict[str, float] = {}
        rng = np.random.default_rng(seed=42)

        print(f'- Calibrating {len(self.calib_keys)} using Nelder-Mead method. \n-Parameters: \n{self.calib_keys}')

        for restart in range(self.n_restarts):
            self._current_restart = restart + 1
            self._iter_in_restart = 0

            # Random starting point within parameter bounds
            x0 = np.array([
                rng.uniform(lo, hi)
                for lo, hi in self.bounds
            ])

            result: OptimizeResult = minimize(
                self._objective,
                x0,
                method="Nelder-Mead",
                options={
                    "maxiter": self.max_iter,
                    "xatol": 1e-4,
                    "fatol": 1e-4,
                    "disp": False,
                },
                callback=self._on_iteration,
                bounds=self.bounds,  # used only to clamp in objective
            )

            rmse = result.fun
            # print(
            #     # f"\nRestart {restart + 1}/{self.n_restarts} complete  "
            #     f"\nIterations={result.nit}  evals={result.nfev}  RMSE={rmse:.4f}",
            #     flush=True,
            # )

            if rmse < best_rmse:
                best_rmse = rmse
                best_params = dict(zip(self.calib_keys, result.x))

        print(f"\nBest RMSE: {best_rmse:.4f}")
        return best_params

    # -----------------------------------------------------------------------
    # Private helpers
    # -----------------------------------------------------------------------

    def _objective(self, x: np.ndarray) -> float:
        """RMSE objective function for ``scipy.optimize.minimize``.

        Args:
            x: Candidate parameter vector aligned with ``self.calib_keys``.

        Returns:
            RMSE value, or ``1e300`` when ``x`` is out of bounds or a
            simulation error occurs.
        """
        self._n_eval += 1

        # Penalise out-of-bounds (mirrors C# behaviour)
        for val, (lo, hi) in zip(x, self.bounds):
            if val <= lo or val > hi:
                return 1e300

        param_values = dict(zip(self.calib_keys, x))
        try:
            date_outputs = self.runner.run(param_values)
        except Exception:
            return 1e300

        include_crop    = self.calibration_variable in ("crop", "all")
        include_disease = self.calibration_variable in ("disease", "all")

        rmse = self.runner.compute_rmse(
            date_outputs,
            include_crop=include_crop,
            include_disease=include_disease,
        )
        self._last_rmse = rmse
        return rmse

    def _on_iteration(self, _xk: np.ndarray) -> None:
        """Scipy callback invoked after each optimizer iteration.

        Args:
            _xk: Current parameter vector (unused; progress is tracked via
                instance counters).
        """
        self._iter_in_restart += 1
        sys.stdout.write(
            f"\rRun {self._current_restart}/{self.n_restarts} Iteration {self._iter_in_restart}/{self.max_iter} CURR RMSE={self._last_rmse:.4f}"
        )
        sys.stdout.flush()

    def _select_calib_params(self) -> Tuple[List[str], List[Tuple[float, float]]]:
        """Select parameters flagged for calibration.

        Returns:
            Tuple of ``(keys, bounds)`` where ``keys`` is a list of
            ``'ClassName_ParamName'`` strings and ``bounds`` is the
            corresponding list of ``(minimum, maximum)`` tuples.
        """
        calib_keys: List[str] = []
        bounds: List[Tuple[float, float]] = []

        for key, p in self.runner.name_param.items():
            # Skip non-calibrated params
            if not p.calibration.strip():
                continue

            # Restrict to the requested calibration variable
            param_class = key.split("_", 1)[0].lower()
            if self.calibration_variable not in ("all", param_class):
                continue

            # Skip boolean parameters
            if p.is_boolean:
                continue

            calib_keys.append(key)
            bounds.append((p.minimum, p.maximum))

        return calib_keys, bounds
