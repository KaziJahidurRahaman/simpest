"""Tests for FraNchEstYN run diagnostics."""

import unittest

import numpy as np
import pandas as pd

from simpest.models.fr_diagnostics import (
    days_between,
    error_stats,
    first_date,
    match_ref_to_sim,
    season_diagnostics,
    season_rmse,
    simulation_frame,
)


def _sim_df():
    """Season sown in 1980, simulated 1981-04-01 .. 1981-06-30, severity rising 1 %/day."""
    dates = pd.date_range("1981-04-01", "1981-06-30", freq="D")
    return pd.DataFrame({"Date": dates, "GrowingSeason": 1980, "DiseaseSeverity": np.arange(len(dates), dtype=float),
                         "TotalLW": 24.0})


def _ref_df():
    # DOY 100 = 1981-04-10 (sim 9), DOY 120 = 1981-04-30 (sim 29), DOY 200 is after the season.
    return pd.DataFrame({"year": [1981, 1981, 1981], "DOY": [100, 120, 200], "sev": [10.0, 25.0, 80.0]})


class TestSimulationFrame(unittest.TestCase):
    def test_parses_dates_and_scales_severity(self):
        result = {"outputs": {"simulation": [{"Date": "02/03/1981", "DiseaseSeverity": 0.25}]}}
        sim = simulation_frame(result)
        self.assertEqual(sim["Date"].iloc[0], pd.Timestamp("1981-03-02"))
        self.assertEqual(sim["DiseaseSeverity"].iloc[0], 25.0)
        self.assertEqual(simulation_frame(result, severity_percent=False)["DiseaseSeverity"].iloc[0], 0.25)


class TestMatchRefToSim(unittest.TestCase):
    def test_matches_same_date_and_skips_out_of_season(self):
        err = match_ref_to_sim(_sim_df(), _ref_df(), seasons=[1980, 1990], ref_col="sev", sim_col="DiseaseSeverity")
        self.assertEqual(err["simulated"].tolist(), [9.0, 29.0])
        self.assertEqual(err["error"].tolist(), [-1.0, 4.0])
        stats = error_stats(err)
        self.assertEqual(stats["n"], 2)
        self.assertAlmostEqual(stats["MAE"], 2.5)
        self.assertAlmostEqual(season_rmse(err).loc[1980], np.sqrt((1 + 16) / 2))

    def test_no_matches_gives_empty_table(self):
        err = match_ref_to_sim(_sim_df(), _ref_df(), seasons=[1990], ref_col="sev", sim_col="DiseaseSeverity")
        self.assertTrue(err.empty)
        self.assertIn("error", err.columns)


class TestThresholdDates(unittest.TestCase):
    def test_first_date_and_days_between(self):
        sim = _sim_df()
        self.assertEqual(first_date(sim["Date"], sim["DiseaseSeverity"], 20), "04-21")
        self.assertEqual(days_between(sim["Date"], sim["DiseaseSeverity"], 10, 60), 50)
        self.assertEqual(first_date(sim["Date"], sim["DiseaseSeverity"], 1000), "-")
        self.assertTrue(np.isnan(days_between(sim["Date"], sim["DiseaseSeverity"], 10, 1000)))


class TestSeasonDiagnostics(unittest.TestCase):
    def test_counts_observations_after_season(self):
        sim, ref = _sim_df(), _ref_df()
        err = match_ref_to_sim(sim, ref, [1980], "sev", "DiseaseSeverity")
        diag = season_diagnostics(sim, ref, err, [1980], "sev")
        row = diag.loc["1980/81"]
        self.assertEqual(row["n_obs"], 3)
        self.assertEqual(row["n_obs_after_season"], 1)
        self.assertEqual(row["peak_obs"], 80.0)
        self.assertEqual(row["fully_wet_days_AMJ"], len(sim))


if __name__ == "__main__":
    unittest.main()
