"""optimizer.py - C#-style MultiStartSimplex calibration for FraNchEstYN.

This module implements a pure-Python multi-start Nelder-Mead routine to
mirror the C# optimizer flow (UNIMI MultiStartSimplex usage):
- `n_restarts` as number of random simplexes,
- `max_iter` as simplex iteration budget,
- `ftol` as objective convergence criterion.
"""

from __future__ import annotations

import math
import sys
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from .fr_runner import FranchestynRunner
class FranchestynOptimizer:
    """
    C#-style multi-start simplex calibration wrapper for FraNchEstYN.

    Args:
        runner (FranchestynRunner): Fully configured runner instance.
        calibration_variable (str): Calibration target scope: "crop",
            "disease", or "all".
        n_restarts (int): Number of simplexes (C# ``NofSimplexes`` analogue).
        max_iter (int): Maximum iterations per simplex (C# ``Itmax`` analogue).
        ftol (float): Objective convergence tolerance (C# ``Ftol`` analogue).
    """

    def __init__(
        self,
        runner: FranchestynRunner,
        calibration_variable: str = "all",
        n_restarts: int = 5,
        max_iter: int = 1000,
        ftol: float = 1e-12,
        disabled_by_class: Optional[Dict[str, Set[str]]] = None,
    ) -> None:
        self.runner = runner
        self.calibration_variable = calibration_variable.lower()
        self.n_restarts = n_restarts
        self.max_iter = max_iter
        self.ftol = ftol
        self.disabled_by_class = disabled_by_class or {}

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
        """
        Run multi-start simplex and return the best parameter set.

        Returns:
            Dict[str, float]: Best-fit parameter values keyed by
            ``class_ParamName``.
        """
        
        if not self.calib_keys:
            print("No calibration parameters found — returning defaults.")
            return {}

        best_rmse = math.inf
        best_params: Dict[str, float] = {}
        rng = np.random.default_rng()

        print(f"- Calibrating {len(self.calib_keys)} using multi-start simplex method.\n"f"- Parameters:\n{self.calib_keys}")

        for restart in range(self.n_restarts):
            self._current_restart = restart + 1
            self._iter_in_restart = 0

            simplex, fvals, _nit = self._nelder_mead_single_restart(rng)
            best_idx = int(np.argmin(fvals))
            rmse = float(fvals[best_idx])

            if rmse < best_rmse:
                best_rmse = rmse
                best_params = dict(zip(self.calib_keys, simplex[best_idx]))

        print(f"\nBest RMSE: {best_rmse:.4f}")
        return best_params

    # -----------------------------------------------------------------------
    # Private helpers
    # -----------------------------------------------------------------------

    def _objective(self, x: np.ndarray) -> float:
        """RMSE objective function for scipy.optimize.minimize."""
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
        """Progress callback for each simplex iteration."""
        self._iter_in_restart += 1
        sys.stdout.write(
            f"\rRun {self._current_restart}/{self.n_restarts} Iteration {self._iter_in_restart}/{self.max_iter} CURR RMSE={self._last_rmse:.4f}"
        )
        sys.stdout.flush()

    def _select_calib_params(self) -> Tuple[List[str], List[Tuple[float, float]]]:
        """Return (keys, bounds) for parameters flagged for calibration."""
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

            # Explicitly exclude user-deactivated parameters by name.
            param_name = key.split("_", 1)[1] if "_" in key else key
            if param_name in self.disabled_by_class.get(param_class, set()):
                continue

            # Skip boolean parameters
            if p.is_boolean:
                continue

            calib_keys.append(key)
            bounds.append((p.minimum, p.maximum))

        return calib_keys, bounds

    def _random_simplex(self, rng: np.random.Generator) -> np.ndarray:
        """Create a random initial simplex fully contained in parameter bounds."""
        dim = len(self.bounds)
        simplex = np.empty((dim + 1, dim), dtype=float)

        # Anchor vertex
        simplex[0] = np.array([rng.uniform(lo, hi) for lo, hi in self.bounds], dtype=float)

        # One perturbed vertex per dimension
        for i in range(dim):
            v = simplex[0].copy()
            lo, hi = self.bounds[i]
            span = hi - lo
            delta = rng.uniform(0.05 * span, 0.25 * span)
            direction = -1.0 if rng.random() < 0.5 else 1.0
            v[i] = np.clip(v[i] + direction * delta, lo, hi)
            simplex[i + 1] = v

        return simplex

    def _nelder_mead_single_restart(
        self, rng: np.random.Generator
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        """Run one Nelder-Mead restart using standard coefficients."""
        # Standard Nelder-Mead coefficients
        alpha = 1.0  # reflection
        gamma = 2.0  # expansion
        rho = 0.5    # contraction
        sigma = 0.5  # shrink

        simplex = self._random_simplex(rng)
        fvals = np.array([self._objective(v) for v in simplex], dtype=float)

        nit = 0
        while nit < self.max_iter:
            order = np.argsort(fvals)
            simplex = simplex[order]
            fvals = fvals[order]

            # Convergence check driven by objective spread (C#-style Ftol usage)
            if np.max(np.abs(fvals - fvals[0])) <= self.ftol:
                break

            centroid = np.mean(simplex[:-1], axis=0)
            worst = simplex[-1]

            # Reflection
            xr = centroid + alpha * (centroid - worst)
            fr = self._objective(xr)

            if fvals[0] <= fr < fvals[-2]:
                simplex[-1] = xr
                fvals[-1] = fr
            elif fr < fvals[0]:
                # Expansion
                xe = centroid + gamma * (xr - centroid)
                fe = self._objective(xe)
                if fe < fr:
                    simplex[-1] = xe
                    fvals[-1] = fe
                else:
                    simplex[-1] = xr
                    fvals[-1] = fr
            else:
                # Contraction
                if fr < fvals[-1]:
                    # Outside contraction
                    xc = centroid + rho * (xr - centroid)
                    fc = self._objective(xc)
                    if fc <= fr:
                        simplex[-1] = xc
                        fvals[-1] = fc
                    else:
                        # Shrink
                        best = simplex[0].copy()
                        for i in range(1, len(simplex)):
                            simplex[i] = best + sigma * (simplex[i] - best)
                            fvals[i] = self._objective(simplex[i])
                else:
                    # Inside contraction
                    xc = centroid - rho * (centroid - worst)
                    fc = self._objective(xc)
                    if fc < fvals[-1]:
                        simplex[-1] = xc
                        fvals[-1] = fc
                    else:
                        # Shrink
                        best = simplex[0].copy()
                        for i in range(1, len(simplex)):
                            simplex[i] = best + sigma * (simplex[i] - best)
                            fvals[i] = self._objective(simplex[i])

            nit += 1
            self._on_iteration(simplex[0])

        return simplex, fvals, nit
