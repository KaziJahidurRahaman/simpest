"""Shared foundation for simpest calibration optimizers.

Every optimizer solves the same problem: find the parameter set that minimises
the run's RMSE against reference observations, searching only the parameters
flagged for calibration and holding the rest at their defaults. This module
factors out that common machinery — parameter selection, bounds, and the
model-run objective — so a concrete optimizer only has to implement the search
strategy in :meth:`BaseFranchestynOptimizer.calibrate`.

It also provides a small name-keyed registry (:data:`OPTIMIZERS`) and
:func:`build_optimizer`, which :func:`simpest.models.franchestyn.run_franchestyn`
uses to pick an optimizer from a :class:`~simpest.models.franchestyn.FranchestynConfig`.
"""

from __future__ import annotations

import importlib
import math
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from .fr_runner import FranchestynRunner


class BaseFranchestynOptimizer:
    """Common base for simpest calibration optimizers.

    Wraps a configured runner, selects the calibratable parameters within the
    requested scope, and exposes the calibration objective. Subclasses implement
    :meth:`calibrate` with a concrete search strategy.

    Args:
        runner (FranchestynRunner): Fully configured runner instance.
        calibration_variable (str): Calibration target scope: ``"crop"``,
            ``"disease"``, or ``"all"``.
        disabled_by_class (Optional[Dict[str, Set[str]]]): Parameter names to
            exclude from calibration, keyed by parameter class.
        seed (Optional[int]): Seed for the search's random number generator,
            giving reproducible runs. ``None`` (default) is non-deterministic.
    """

    def __init__(
        self,
        runner: FranchestynRunner,
        calibration_variable: str = "all",
        disabled_by_class: Optional[Dict[str, Set[str]]] = None,
        seed: Optional[int] = None,
    ) -> None:
        self.runner = runner
        self.calibration_variable = calibration_variable.lower()
        self.disabled_by_class = disabled_by_class or {}
        self.seed = seed

        # Select calibration parameters and record their bounds.
        self.calib_keys, self.bounds = self._select_calib_params()

        self._n_eval = 0
        self._last_rmse = math.inf

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def calibrate(self) -> Dict[str, float]:
        """Run the search and return the best parameter set.

        Returns:
            Dict[str, float]: Best-fit parameter values keyed by
            ``class_ParamName``. Empty when nothing is flagged for calibration.
        """
        raise NotImplementedError

    @classmethod
    def from_config(
        cls,
        runner: FranchestynRunner,
        config,
        disabled_by_class: Optional[Dict[str, Set[str]]] = None,
    ) -> "BaseFranchestynOptimizer":
        """Build an optimizer from a :class:`FranchestynConfig`.

        Subclasses override this to read their own knobs from ``config``. The
        base implementation passes only the shared arguments.
        """
        return cls(
            runner=runner,
            calibration_variable=config.calibration_variable,
            disabled_by_class=disabled_by_class,
            seed=getattr(config, "seed", None),
        )

    # -----------------------------------------------------------------------
    # Shared helpers
    # -----------------------------------------------------------------------

    def _evaluate_rmse(self, x: np.ndarray) -> float:
        """Run the model with candidate parameters ``x`` and return its RMSE.

        Runs that raise return a large penalty (``1e300``) so a search treats
        the region as infeasible. Bounds are *not* enforced here — an optimizer
        that can propose out-of-bounds candidates must guard against that
        itself (see :class:`~simpest.models.fr_optimizer.FranchestynOptimizer`).
        """
        self._n_eval += 1

        param_values = dict(zip(self.calib_keys, x))
        try:
            date_outputs = self.runner.run(param_values)
        except Exception:
            return 1e300

        include_crop = self.calibration_variable in ("crop", "all")
        include_disease = self.calibration_variable in ("disease", "all")

        rmse = self.runner.compute_rmse(
            date_outputs,
            include_crop=include_crop,
            include_disease=include_disease,
        )
        self._last_rmse = rmse
        return rmse

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


# ---------------------------------------------------------------------------
# Optimizer registry
# ---------------------------------------------------------------------------

# Name -> "module:ClassName". Resolved lazily by build_optimizer so this module
# does not import the concrete optimizer modules (and their dependencies).
OPTIMIZERS: Dict[str, str] = {
    "nelder-mead": "simpest.models.fr_optimizer:FranchestynOptimizer",
    "scipy-nelder-mead": "simpest.models.fr_optimizer_scipy:ScipyNelderMeadOptimizer",
    "optuna": "simpest.models.fr_optimizer_optuna:OptunaOptimizer",
}


def build_optimizer(
    name: str,
    runner: FranchestynRunner,
    config,
    disabled_by_class: Optional[Dict[str, Set[str]]] = None,
) -> BaseFranchestynOptimizer:
    """Instantiate the optimizer registered under ``name``.

    Args:
        name (str): Optimizer key, case-insensitive (e.g. ``"nelder-mead"``,
            ``"optuna"``).
        runner (FranchestynRunner): Fully configured runner instance.
        config (FranchestynConfig): Configuration supplying the optimizer knobs.
        disabled_by_class (Optional[Dict[str, Set[str]]]): Parameter names to
            exclude from calibration, keyed by parameter class.

    Returns:
        BaseFranchestynOptimizer: A ready-to-run optimizer.

    Raises:
        ValueError: If ``name`` is not a registered optimizer.
    """
    key = (name or "").strip().lower()
    if key not in OPTIMIZERS:
        valid = ", ".join(sorted(OPTIMIZERS))
        raise ValueError(f"Unknown optimizer {name!r}. Valid options: {valid}.")

    module_path, class_name = OPTIMIZERS[key].split(":")
    cls = getattr(importlib.import_module(module_path), class_name)
    return cls.from_config(runner, config, disabled_by_class)
