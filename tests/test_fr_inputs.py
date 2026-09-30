"""Tests for the FraNchEstYN input builders."""

import unittest

import numpy as np
import pandas as pd

from simpest.models.fr_inputs import build_crop_model_df, build_management_df, build_weather_df


def _season_arrays():
    """Two seasons of 400 days; the first matures on day 300, the second on 310."""
    days = np.arange(400, dtype=float)
    dvs = np.vstack([np.clip(days / 150.0, 0, 2), np.clip(days / 155.0, 0, 2)])
    ramp = np.vstack([days, days * 2])
    return dict(agb=ramp, yield_=ramp / 10, fint=ramp / 1000, gdd=ramp, dvs=dvs)


class TestBuildCropModelDf(unittest.TestCase):
    def test_runs_from_sowing_to_maturity(self):
        df = build_crop_model_df(**_season_arrays(), sow_years=[1980, 1981], sowing_doys=[280, 290])
        first = df[df["season"] == 1980]
        self.assertEqual(len(first), 300 - 279 + 1)   # sowing index 279 .. maturity index 300
        self.assertEqual((first["year"].iloc[0], first["doy"].iloc[0]), (1980, 280))
        self.assertEqual(first["agb"].iloc[0], 279.0)
        self.assertEqual(first["agb"].iloc[-1], 300.0)
        self.assertNotIn("dvs", df.columns)

    def test_seasons_are_sorted_by_date(self):
        df = build_crop_model_df(**_season_arrays(), sow_years=[1981, 1980], sowing_doys=[290, 280])
        self.assertEqual(list(df["season"].unique()), [1980, 1981])

    def test_optional_dvs_column(self):
        df = build_crop_model_df(**_season_arrays(), sow_years=[1980, 1981], sowing_doys=[280, 290],
                                 include_dvs=True)
        self.assertEqual(df.columns[-2:].tolist(), ["dvs", "season"])

    def test_immature_season_raises(self):
        arrays = _season_arrays()
        arrays["dvs"] = arrays["dvs"] * 0.5
        with self.assertRaises(ValueError):
            build_crop_model_df(**arrays, sow_years=[1980, 1981], sowing_doys=[280, 290])


class TestBuildWeatherDf(unittest.TestCase):
    def setUp(self):
        dates = pd.date_range("1980-01-01", periods=3, freq="D")
        self.raw = pd.DataFrame({"Date": dates, "TempMax": [10.0, 11, 12], "TempMin": [0.0, 1, 2],
                                 "Precipitation": [0.0, 5, 0], "Radiation": [10000.0, 12000, 8000]})
        self.hourly = pd.DataFrame({"date": np.repeat(dates.strftime("%Y-%m-%d"), 2),
                                    "rh": [60.0, 90, 50, 95, 70, 80]})

    def test_daily_converts_radiation(self):
        df = build_weather_df(self.raw, site="x")
        self.assertEqual(df["rad"].tolist(), [10.0, 12.0, 8.0])
        self.assertNotIn("rhx", df.columns)

    def test_daily_rh_adds_extremes(self):
        df = build_weather_df(self.raw, site="x", mode="daily_rh", weather_hourly=self.hourly)
        self.assertEqual(df["rhx"].tolist(), [90.0, 95.0, 80.0])
        self.assertEqual(df["rhn"].tolist(), [60.0, 50.0, 70.0])

    def test_daily_rh_prefers_daily_columns(self):
        raw = self.raw.assign(RHmax=[91.0, 92, 93], RHmin=[41.0, 42, 43])
        df = build_weather_df(raw, site="x", mode="daily_rh", weather_hourly=self.hourly)
        self.assertEqual(df["rhx"].tolist(), [91.0, 92.0, 93.0])
        self.assertEqual(df["rhn"].tolist(), [41.0, 42.0, 43.0])
        # The daily columns are enough on their own.
        df = build_weather_df(raw, site="x", mode="daily_rh")
        self.assertEqual(df["rhx"].tolist(), [91.0, 92.0, 93.0])

    def test_hourly_returns_hourly_records(self):
        self.assertIs(build_weather_df(self.raw, "x", mode="hourly", weather_hourly=self.hourly), self.hourly)

    def test_bad_inputs_raise(self):
        with self.assertRaises(ValueError):
            build_weather_df(self.raw, "x", mode="weekly")
        with self.assertRaises(ValueError):
            build_weather_df(self.raw, "x", mode="daily_rh")


class TestBuildManagementDf(unittest.TestCase):
    def test_from_indexed_site_table(self):
        site = pd.DataFrame({"LATITUDE": [40.0, 40.0], "IDPL": [280, 290]},
                            index=pd.Index([1980, 1981], name="location"))
        df = build_management_df(site, site="s", crop="wheat", variety="v")
        self.assertEqual(df.columns.tolist(), ["site", "crop", "variety", "year", "sowingDOY"])
        self.assertEqual(df["sowingDOY"].tolist(), [280, 290])
        self.assertEqual(df["year"].tolist(), [1980, 1981])

    def test_from_plain_columns(self):
        sowing = pd.DataFrame({"year": [1980], "sowingDOY": [280]})
        df = build_management_df(sowing, "s", "wheat", "v", year_col="year", sowing_doy_col="sowingDOY")
        self.assertEqual(df.iloc[0].tolist(), ["s", "wheat", "v", 1980, 280])


if __name__ == "__main__":
    unittest.main()
