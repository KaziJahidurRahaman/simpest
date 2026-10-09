"""Tests for FranchestynSession and the parameter-spec helpers it relies on."""

import copy
import unittest
from unittest import mock

import pandas as pd

from simpest.models import fr_session, franchestyn


def _config():
    config = franchestyn.FranchestynConfig(fungicide_type=None)
    config.is_calibration = False
    return config


class TestParamHelpers(unittest.TestCase):
    def test_apply_bound_overrides(self):
        params = {"A": {"min": 0.0, "max": 0.3, "value": 0.1}}
        franchestyn.apply_bound_overrides(params, {"A": {"max": 0.6}})
        self.assertEqual(params["A"]["max"], 0.6)
        with self.assertRaises(KeyError):
            franchestyn.apply_bound_overrides(params, {"B": {"max": 1.0}})

    def test_apply_best_params(self):
        config = _config()
        name = next(iter(config.disease_parameters))
        franchestyn.apply_best_params(config, {f"disease_{name}": 0.123})
        self.assertEqual(config.disease_parameters[name]["value"], 0.123)

    def test_inject_reference_params(self):
        params = {"A": {"value": 1.0, "calibration": True}}
        table = pd.DataFrame({"Model": ["Disease", "disease", "crop", "disease"],
                              "Parameter": ["A", "Missing", "A", "Bad"],
                              "calibrated": [2.5, 1.0, 9.0, "n/a"]})
        matched, unmatched = franchestyn.inject_reference_params(params, "disease", table)
        self.assertEqual((matched, unmatched), (["A"], ["Missing"]))
        self.assertEqual(params["A"], {"value": 2.5, "calibration": False})


class TestFranchestynSession(unittest.TestCase):
    def setUp(self):
        self.config = _config()
        frame = pd.DataFrame()
        self.session = fr_session.FranchestynSession(
            self.config, 1980, 1981, weather_df=frame, management_df=frame, crop_model_df=frame, ref_df=frame)

    def test_runner_is_built_once(self):
        with mock.patch.object(franchestyn, "build_runner", return_value="runner") as build:
            self.assertEqual(self.session.runner, "runner")
            self.assertEqual(self.session.runner, "runner")
        build.assert_called_once()

    def test_run_forward_restores_calibration_flag(self):
        self.config.is_calibration = True
        seen = []

        def fake_run(config, runner, **_):
            seen.append(config.is_calibration)
            return {"outputs": {}}

        with mock.patch.object(franchestyn, "build_runner"), mock.patch.object(franchestyn, "run_franchestyn", fake_run):
            self.session.run_forward()
        self.assertEqual(seen, [False])
        self.assertTrue(self.config.is_calibration)

    def test_calibrate_seeds_keeps_best_and_restarts_each_seed(self):
        name = next(iter(self.config.disease_parameters))
        start_value = self.config.disease_parameters[name]["value"]
        rmse_by_seed = {1: 5.0, 2: 3.0, 3: 4.0}
        start_values_seen = []

        def fake_run(config, runner, **_):
            start_values_seen.append(config.disease_parameters[name]["value"])
            config.disease_parameters[name]["value"] = -1.0   # a run may leave the specs edited
            return {"outputs": {"summary": {"rmse": rmse_by_seed[config.seed],
                                            "best_params": {f"disease_{name}": float(config.seed)}}}}

        with mock.patch.object(franchestyn, "build_runner"), mock.patch.object(franchestyn, "run_franchestyn", fake_run), \
                mock.patch("builtins.print"):
            result = self.session.calibrate_seeds([1, 2, 3], n_trials=10)

        self.assertEqual(start_values_seen, [start_value] * 3)
        self.assertEqual(result.best["seed"], 2)
        self.assertEqual(self.config.disease_parameters[name]["value"], 2.0)
        self.assertEqual(self.config.n_trials, 10)
        self.assertEqual(result.table().loc["RMSE"].tolist(), [5.0, 3.0, 4.0])
        self.assertEqual(result.start_params["disease"][name]["value"], start_value)
        # Other specs are back at their start values.
        other = copy.deepcopy(result.start_params["disease"])
        other[name]["value"] = 2.0
        self.assertEqual(self.config.disease_parameters, other)


if __name__ == "__main__":
    unittest.main()
