#!/usr/bin/env python

"""Tests for the calibration optimizers and their registry."""

import unittest
from dataclasses import dataclass
from typing import Optional

from simpest.models.fr_optimizer import FranchestynOptimizer
from simpest.models.fr_optimizer_base import build_optimizer
from simpest.models.fr_optimizer_optuna import OptunaOptimizer
from simpest.models.fr_optimizer_scipy import ScipyNelderMeadOptimizer


@dataclass
class _FakeParam:
    """Minimal stand-in for :class:`simpest.models.fr_param_reader.Parameter`."""

    minimum: float
    maximum: float
    calibration: str = "x"
    is_boolean: bool = False


class _FakeRunner:
    """Runner stub with a convex RMSE surface, minimised at ``targets``.

    ``run`` just echoes the candidate overrides; ``compute_rmse`` turns them into
    a simple bowl so the optimizers have a well-posed problem with no simulation.
    """

    def __init__(self, name_param, targets):
        self.name_param = name_param
        self._targets = targets

    def run(self, param_values=None):
        return dict(param_values or {})

    def compute_rmse(self, date_outputs, include_crop=True, include_disease=True):
        return sum((date_outputs[k] - t) ** 2 for k, t in self._targets.items())


@dataclass
class _FakeConfig:
    calibration_variable: str = "all"
    n_restarts: int = 2
    max_iter: int = 60
    ftol: float = 1e-12
    n_trials: int = 80
    optuna_timeout: Optional[float] = None
    seed: Optional[int] = 0


def _two_param_setup():
    name_param = {
        "crop_A": _FakeParam(minimum=0.0, maximum=10.0),
        "crop_B": _FakeParam(minimum=-5.0, maximum=5.0),
    }
    targets = {"crop_A": 4.0, "crop_B": 1.0}
    return _FakeRunner(name_param, targets), targets


class TestOptimizerRegistry(unittest.TestCase):
    def test_builds_registered_optimizers(self):
        runner, _ = _two_param_setup()
        cfg = _FakeConfig()
        self.assertIsInstance(
            build_optimizer("nelder-mead", runner, cfg, None), FranchestynOptimizer
        )
        self.assertIsInstance(
            build_optimizer("scipy-nelder-mead", runner, cfg, None),
            ScipyNelderMeadOptimizer,
        )
        self.assertIsInstance(
            build_optimizer("optuna", runner, cfg, None), OptunaOptimizer
        )

    def test_name_is_case_insensitive(self):
        runner, _ = _two_param_setup()
        opt = build_optimizer("  OptUna ", runner, _FakeConfig(), None)
        self.assertIsInstance(opt, OptunaOptimizer)
        opt = build_optimizer(" SciPy-Nelder-Mead ", runner, _FakeConfig(), None)
        self.assertIsInstance(opt, ScipyNelderMeadOptimizer)

    def test_unknown_name_raises_value_error(self):
        runner, _ = _two_param_setup()
        with self.assertRaises(ValueError):
            build_optimizer("simulated-annealing", runner, _FakeConfig(), None)


class TestOptunaOptimizer(unittest.TestCase):
    def test_finds_minimum_within_bounds(self):
        runner, targets = _two_param_setup()
        opt = OptunaOptimizer(runner=runner, n_trials=150, seed=0)
        best = opt.calibrate()

        self.assertEqual(set(best), {"crop_A", "crop_B"})
        for key, (lo, hi) in zip(opt.calib_keys, opt.bounds):
            self.assertGreaterEqual(best[key], lo)
            self.assertLessEqual(best[key], hi)

        # Comfortably better than the domain-centre parameter set.
        mid_rmse = sum(
            ((lo + hi) / 2 - targets[k]) ** 2
            for k, (lo, hi) in zip(opt.calib_keys, opt.bounds)
        )
        self.assertLess(runner.compute_rmse(best), mid_rmse)

    def test_is_reproducible_with_seed(self):
        runner, _ = _two_param_setup()
        a = OptunaOptimizer(runner=runner, n_trials=40, seed=7).calibrate()
        b = OptunaOptimizer(runner=runner, n_trials=40, seed=7).calibrate()
        self.assertEqual(a, b)

    def test_no_calibratable_params_returns_empty(self):
        name_param = {"crop_A": _FakeParam(0.0, 1.0, calibration="")}
        runner = _FakeRunner(name_param, {})
        self.assertEqual(OptunaOptimizer(runner=runner, n_trials=10).calibrate(), {})


class TestScipyNelderMeadOptimizer(unittest.TestCase):
    def test_finds_minimum_within_bounds(self):
        runner, targets = _two_param_setup()
        opt = ScipyNelderMeadOptimizer(
            runner=runner, n_restarts=4, max_iter=200, seed=0
        )
        best = opt.calibrate()

        self.assertEqual(set(best), {"crop_A", "crop_B"})
        for key, (lo, hi) in zip(opt.calib_keys, opt.bounds):
            self.assertGreaterEqual(best[key], lo)
            self.assertLessEqual(best[key], hi)

        mid_rmse = sum(
            ((lo + hi) / 2 - targets[k]) ** 2
            for k, (lo, hi) in zip(opt.calib_keys, opt.bounds)
        )
        self.assertLess(runner.compute_rmse(best), mid_rmse)

    def test_is_reproducible_with_seed(self):
        runner, _ = _two_param_setup()
        a = ScipyNelderMeadOptimizer(runner=runner, n_restarts=3, seed=7).calibrate()
        b = ScipyNelderMeadOptimizer(runner=runner, n_restarts=3, seed=7).calibrate()
        self.assertEqual(a, b)

    def test_no_calibratable_params_returns_empty(self):
        name_param = {"crop_A": _FakeParam(0.0, 1.0, calibration="")}
        runner = _FakeRunner(name_param, {})
        self.assertEqual(ScipyNelderMeadOptimizer(runner=runner).calibrate(), {})


if __name__ == "__main__":
    unittest.main()
