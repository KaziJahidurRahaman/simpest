#!/usr/bin/env python

"""Tests for the Morris sensitivity screen."""

import unittest

from simpest.models.fr_sensitivity import MorrisSensitivity

from tests.test_optimizer import _FakeParam, _FakeRunner


def _dominant_b_setup():
    """Runner whose RMSE is driven almost entirely by ``crop_A``."""
    name_param = {
        "crop_A": _FakeParam(minimum=0.0, maximum=10.0),
        "crop_B": _FakeParam(minimum=0.0, maximum=10.0),
    }
    runner = _FakeRunner(name_param, {"crop_A": 5.0, "crop_B": 5.0})
    # crop_B contributes 10^-6 as much as crop_A to the objective.
    runner.compute_rmse = lambda out, **kw: (
        (out["crop_A"] - 5.0) ** 2 + 1e-6 * (out["crop_B"] - 5.0) ** 2
    )
    return runner


class TestMorrisSensitivity(unittest.TestCase):
    def test_ranks_influential_parameter_first(self):
        screen = MorrisSensitivity(_dominant_b_setup(), seed=0, n_trajectories=10)
        table = screen.analyze()

        self.assertEqual(list(table["parameter"]), ["crop_A", "crop_B"])
        by_param = table.set_index("parameter")
        self.assertGreater(by_param.loc["crop_A", "mu_star"], by_param.loc["crop_B", "mu_star"])
        self.assertTrue(table[["mu", "mu_star", "sigma", "mu_star_conf"]].to_numpy().tolist())
        self.assertFalse(table[["mu_star", "sigma"]].isna().any().any())

    def test_deterministic_with_seed(self):
        a = MorrisSensitivity(_dominant_b_setup(), seed=3, n_trajectories=6).analyze()
        b = MorrisSensitivity(_dominant_b_setup(), seed=3, n_trajectories=6).analyze()
        self.assertTrue(a.equals(b))

    def test_no_calibratable_params_returns_empty(self):
        name_param = {"crop_A": _FakeParam(0.0, 1.0, calibration="")}
        runner = _FakeRunner(name_param, {})
        table = MorrisSensitivity(runner, n_trajectories=4).analyze()
        self.assertTrue(table.empty)

    def test_degenerate_bounds_raise(self):
        name_param = {"crop_A": _FakeParam(minimum=2.0, maximum=2.0)}
        runner = _FakeRunner(name_param, {"crop_A": 2.0})
        with self.assertRaises(ValueError):
            MorrisSensitivity(runner, n_trajectories=4).analyze()


if __name__ == "__main__":
    unittest.main()
