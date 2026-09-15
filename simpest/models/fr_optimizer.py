"""Multi-start Nelder–Mead calibration for the simulation model.

This module estimates model parameters by minimising the run's root-mean-square
error against reference observations. It uses a self-contained, pure-Python
multi-start Nelder–Mead (downhill simplex) search: several simplexes are each
optimised from a random starting configuration and the best result across all
restarts is returned. Multiple restarts reduce the chance of settling in a poor
local minimum of the objective surface.

The objective is the runner's :meth:`~simpest.models.fr_runner.FranchestynRunner.compute_rmse`,
and candidate parameter sets that fall outside their bounds receive a large
penalty so the search remains in the feasible region. The search is controlled
by three knobs: ``n_restarts`` (number of independent simplexes), ``max_iter``
(iterations per simplex), and ``ftol`` (objective-spread convergence tolerance).

Because the initial simplex is drawn at random, results vary from run to run
unless a fixed ``seed`` is supplied. For deterministic, fixed-parameter
validation, run the model directly rather than through calibration.

Parameter selection and the model-run objective are shared with the other
optimizers via :class:`~simpest.models.fr_optimizer_base.BaseFranchestynOptimizer`.
"""

from __future__ import annotations

import math
import sys
from typing import Dict, Optional, Set, Tuple

import numpy as np

from .fr_optimizer_base import BaseFranchestynOptimizer
from .fr_runner import FranchestynRunner


class FranchestynOptimizer(BaseFranchestynOptimizer):
    """Multi-start Nelder–Mead calibration for a simpest simulation run.

    Wraps a configured runner and searches for the parameter set that minimises
    the run's RMSE against reference data. Only parameters flagged for
    calibration (and not explicitly disabled) within the requested scope are
    optimised; all others are held at their default values.

    Args:
        runner (FranchestynRunner): Fully configured runner instance.
        calibration_variable (str): Calibration target scope: ``"crop"``,
            ``"disease"``, or ``"all"``.
        n_restarts (int): Number of independent simplexes (random restarts).
        max_iter (int): Maximum iterations per simplex.
        ftol (float): Convergence tolerance on the objective spread across the
            simplex vertices.
        disabled_by_class (Optional[Dict[str, Set[str]]]): Parameter names to
            exclude from calibration, keyed by parameter class.
        seed (Optional[int]): Seed for the random number generator, giving
            reproducible restarts. ``None`` (default) is non-deterministic.
    """

    def __init__(
        self,
        runner: FranchestynRunner,
        calibration_variable: str = "all",
        n_restarts: int = 5,
        max_iter: int = 1000,
        ftol: float = 1e-12,
        disabled_by_class: Optional[Dict[str, Set[str]]] = None,
        seed: Optional[int] = None,
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

    @classmethod
    def from_config(cls, runner, config, disabled_by_class=None) -> "FranchestynOptimizer":
        """Build from a :class:`FranchestynConfig` (reads ``n_restarts`` /
        ``max_iter`` / ``ftol`` / ``seed``)."""
        return cls(
            runner=runner,
            calibration_variable=config.calibration_variable,
            n_restarts=config.n_restarts,
            max_iter=config.max_iter,
            ftol=getattr(config, "ftol", 1e-12),
            disabled_by_class=disabled_by_class,
            seed=getattr(config, "seed", None),
        )

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
        rng = np.random.default_rng(self.seed)

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

        print(best_params)

        return best_params

    # -----------------------------------------------------------------------
    # Private helpers
    # -----------------------------------------------------------------------

    def _objective(self, x: np.ndarray) -> float:
        """Evaluate the calibration objective at parameter vector ``x``.

        Runs the model with the candidate parameters and returns the resulting
        RMSE. Candidates that violate the parameter bounds, or that raise during
        the run, return a large penalty (``1e300``) so the search avoids the
        infeasible region.
        """
        # Penalise out-of-bounds candidates with a large finite objective value
        for val, (lo, hi) in zip(x, self.bounds):
            if val <= lo or val > hi:
                return 1e300

        return self._evaluate_rmse(x)

    def _on_iteration(self, _xk: np.ndarray) -> None:
        """Progress callback for each simplex iteration."""
        self._iter_in_restart += 1
        sys.stdout.write(
            f"\rRun {self._current_restart}/{self.n_restarts} Iteration {self._iter_in_restart}/{self.max_iter} CURR RMSE={self._last_rmse:.4f}"
        )
        sys.stdout.flush()

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

            # Convergence check driven by the objective spread across vertices
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
