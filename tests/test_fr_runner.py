#!/usr/bin/env python

"""Tests for FranchestynRunner's weather caching and parameter-spec updates.

These target the two changes that let a single runner be reused across a
whole calibration session instead of being rebuilt (and re-parsing weather
from disk) on every call: `_load_weather()` caching and `update_param_specs()`
refreshing only the cheap, per-stage parameter state.

Runner instances are built via `FranchestynRunner.__new__` with only the
attributes each method under test needs, bypassing `__init__`'s sowing/
reference-file I/O, which is unrelated to what's being tested here.
"""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

from simpest.models import franchestyn
from simpest.models.fr_runner import FranchestynRunner


def _write_daily_weather_csv(path: Path, n_days: int = 5) -> None:
    lines = ["Date,Tmax,Tmin,Radiation"]
    for d in range(1, n_days + 1):
        lines.append(f"2000-01-{d:02d},20.0,5.0,15.0")
    path.write_text("\n".join(lines) + "\n")


def _bare_weather_runner(weather_file, start_year=2000, end_year=2000, latitude=40.0):
    runner = FranchestynRunner.__new__(FranchestynRunner)
    runner.weather_dir = str(weather_file)
    runner.weather_time_step = "daily"
    runner.site = "test"
    runner.start_year = start_year
    runner.end_year = end_year
    runner.latitude = latitude
    runner._weather_cache = None
    runner._weather_hours_cache = None
    return runner


def _bare_param_runner():
    runner = FranchestynRunner.__new__(FranchestynRunner)
    runner.disease_type = "septoria"
    runner.fungicide_type = None
    runner.name_param = {}
    return runner


class TestWeatherCaching(unittest.TestCase):
    def test_load_weather_is_cached(self):
        with tempfile.TemporaryDirectory() as tmp:
            weather_file = Path(tmp) / "weather.csv"
            _write_daily_weather_csv(weather_file)
            runner = _bare_weather_runner(weather_file)

            first = runner._load_weather()
            second = runner._load_weather()
            self.assertIs(first, second)

    def test_weather_hours_are_cached_and_sorted(self):
        with tempfile.TemporaryDirectory() as tmp:
            weather_file = Path(tmp) / "weather.csv"
            _write_daily_weather_csv(weather_file)
            runner = _bare_weather_runner(weather_file)

            hours = runner._weather_hours()
            self.assertEqual(hours, sorted(hours))
            self.assertIs(runner._weather_hours(), hours)

    def test_cached_weather_survives_source_file_removal(self):
        """Once cached, the runner no longer touches the source file -- this
        is what lets a runner outlive the temp CSVs written for construction
        (see franchestyn.build_runner)."""
        with tempfile.TemporaryDirectory() as tmp:
            weather_file = Path(tmp) / "weather.csv"
            _write_daily_weather_csv(weather_file)
            runner = _bare_weather_runner(weather_file)

            first = runner._load_weather()
            weather_file.unlink()

            second = runner._load_weather()
            self.assertIs(first, second)


class TestHourlyWeather(unittest.TestCase):
    def test_hourly_file_keeps_measured_humidity(self):
        """An hourly file's RH is used as-is (not re-synthesised from
        temperature), so leaf wetness follows the measured humidity."""
        with tempfile.TemporaryDirectory() as tmp:
            weather_file = Path(tmp) / "weather.csv"
            lines = ["date,hour,temp,rh,prec,rad"]
            for hr in range(24):
                rh = 95 if hr < 6 else 60
                lines.append(f"2000-01-01,{hr},10.0,{rh},0.0,0.5")
            weather_file.write_text("\n".join(lines) + "\n")
            runner = _bare_weather_runner(weather_file)
            runner.weather_time_step = "hourly"

            weather = runner._load_weather()

            self.assertEqual(len(weather), 24)
            wet_hours = sum(rec.leaf_wetness for rec in weather.values())
            self.assertEqual(wet_hours, 6)
            self.assertEqual(weather[min(weather)].relative_humidity, 95.0)


class TestConfigPassThrough(unittest.TestCase):
    def test_weather_time_step_and_latitude_reach_runner(self):
        config = franchestyn.FranchestynConfig(
            fungicide_type=None, weather_time_step="hourly", latitude=40.5,
        )
        empty = pd.DataFrame({"x": [0]})
        with mock.patch.object(franchestyn, "FranchestynRunner") as fake_runner:
            with franchestyn._build_franchestyn_runner(
                2000, 2000, config, empty, empty, empty, empty,
            ):
                pass
        kwargs = fake_runner.call_args.kwargs
        self.assertEqual(kwargs["weather_time_step"], "hourly")
        self.assertEqual(kwargs["latitude"], 40.5)

    def test_no_crop_model_df_selects_internal_crop_model(self):
        config = franchestyn.FranchestynConfig(fungicide_type=None)
        frame = pd.DataFrame({"x": [0]})
        with mock.patch.object(franchestyn, "FranchestynRunner") as fake_runner:
            with franchestyn._build_franchestyn_runner(
                2000, 2000, config, frame, frame, None, frame,
            ):
                pass
        self.assertIsNone(fake_runner.call_args.kwargs["crop_model_dir"])

    def test_empty_crop_model_df_is_rejected(self):
        config = franchestyn.FranchestynConfig(fungicide_type=None)
        frame = pd.DataFrame({"x": [0]})
        with self.assertRaises(ValueError):
            with franchestyn._build_franchestyn_runner(
                2000, 2000, config, frame, frame, pd.DataFrame(), frame,
            ):
                pass

    def test_defaults_keep_daily_weather(self):
        config = franchestyn.FranchestynConfig(fungicide_type=None)
        self.assertEqual(config.weather_time_step, "daily")
        self.assertEqual(config.latitude, 0.0)


class TestUpdateParamSpecs(unittest.TestCase):
    def test_updates_crop_and_disease_values_in_place(self):
        runner = _bare_param_runner()

        crop_specs = {"TbaseCrop": {"value": 1.0, "min": -1, "max": 5, "calibration": False}}
        disease_specs = {"Tmin": {"value": 2.0, "min": 0, "max": 10, "calibration": True}}
        runner.update_param_specs(crop_param_specs=crop_specs, disease_param_specs=disease_specs)

        self.assertEqual(runner.name_param["crop_TbaseCrop"].value, 1.0)
        self.assertEqual(runner.name_param["crop_TbaseCrop"].calibration, "")
        self.assertEqual(runner.name_param["disease_Tmin"].value, 2.0)
        self.assertEqual(runner.name_param["disease_Tmin"].calibration, "x")

        # A second call with new values overwrites in place -- this is the
        # mechanism a reused runner relies on between calibration stages.
        runner.update_param_specs(crop_param_specs={"TbaseCrop": {"value": 3.0, "min": -1, "max": 5, "calibration": True}})
        self.assertEqual(runner.name_param["crop_TbaseCrop"].value, 3.0)
        self.assertEqual(runner.name_param["crop_TbaseCrop"].calibration, "x")
        # Disease params, not passed this time, are untouched.
        self.assertEqual(runner.name_param["disease_Tmin"].value, 2.0)

    def test_skips_disease_specs_when_disease_type_is_none(self):
        runner = _bare_param_runner()
        runner.disease_type = None

        runner.update_param_specs(disease_param_specs={"Tmin": {"value": 2.0, "min": 0, "max": 10, "calibration": True}})
        self.assertNotIn("disease_Tmin", runner.name_param)

    def test_does_not_touch_weather_cache(self):
        runner = _bare_param_runner()
        sentinel = object()
        runner._weather_cache = sentinel
        runner._weather_hours_cache = ["unchanged"]

        runner.update_param_specs(
            crop_param_specs={"TbaseCrop": {"value": 1.0, "min": -1, "max": 5, "calibration": False}}
        )

        self.assertIs(runner._weather_cache, sentinel)
        self.assertEqual(runner._weather_hours_cache, ["unchanged"])


if __name__ == "__main__":
    unittest.main()
