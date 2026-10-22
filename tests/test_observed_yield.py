"""Tests for observed-yield reading, trend fitting and comparison."""

import os
import tempfile
import unittest

import numpy as np
import pandas as pd

from simpest.models.observed_yield import (
    BU_ACRE_TO_KG_HA,
    GRAIN_DRY_FRACTION,
    LinearTrend,
    TwoSegmentTrend,
    compare_with_potential,
    fit_linear_trend,
    fit_stats,
    fit_two_segment_trend,
    measured_disease_loss,
    read_nass_yield,
)


class TestReadNassYield(unittest.TestCase):
    def test_converts_and_sorts(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "nass.csv")
            pd.DataFrame({"Program": ["SURVEY"] * 2, "Year": [1991, 1990], "Value": [50.0, 40.0]}).to_csv(path)
            df = read_nass_yield(path)
        self.assertEqual(df["harvest_year"].tolist(), [1990, 1991])
        self.assertAlmostEqual(df["yield_kg_ha"].iloc[0], 40.0 * BU_ACRE_TO_KG_HA)
        self.assertAlmostEqual(BU_ACRE_TO_KG_HA, 67.25, places=2)


class TestTwoSegmentTrend(unittest.TestCase):
    def test_recovers_a_known_hinge(self):
        years = np.arange(1930, 2021)
        truth = TwoSegmentTrend(breakpoint=1950, slope_before=5.0, slope_after=50.0,
                                level_at_breakpoint=1500.0, rmse=0.0, rmse_single_line=0.0)
        trend = fit_two_segment_trend(years, truth(years))
        self.assertEqual(trend.breakpoint, 1950)
        self.assertAlmostEqual(trend.slope_before, 5.0, places=6)
        self.assertAlmostEqual(trend.slope_after, 50.0, places=6)
        self.assertAlmostEqual(trend.rmse, 0.0, places=6)
        self.assertGreater(trend.rmse_single_line, 100.0)

    def test_breakpoint_respects_min_segment(self):
        years = np.arange(1930, 1960)
        values = np.where(years < 1933, 0.0, 10.0 * (years - 1933))
        trend = fit_two_segment_trend(years, values, min_segment_years=10)
        self.assertGreaterEqual(trend.breakpoint, 1940)
        self.assertLessEqual(trend.breakpoint, 1949)

    def test_too_short_record_raises(self):
        with self.assertRaises(ValueError):
            fit_two_segment_trend(np.arange(1990, 2005), np.arange(15.0), min_segment_years=10)


class TestLinearTrend(unittest.TestCase):
    def test_recovers_a_known_line(self):
        years = np.arange(1972, 1991)
        trend = fit_linear_trend(years, 3000.0 + 50.0 * (years - 1972))
        self.assertAlmostEqual(trend.slope, 50.0, places=6)
        self.assertAlmostEqual(trend.rmse, 0.0, places=6)
        self.assertTrue(np.allclose(trend([1972, 1990]), [3000.0, 3900.0]))

    def test_short_record_is_fine(self):
        trend = fit_linear_trend([1990, 1991, 1992], [1.0, 2.0, 4.0])
        self.assertAlmostEqual(trend.slope, 1.5)


class TestCompareWithPotential(unittest.TestCase):
    def test_accepts_linear_trend(self):
        trend = LinearTrend(slope=100.0, intercept=-190000.0, rmse=0.0)
        obs = pd.DataFrame({"harvest_year": [1981], "yield_kg_ha": [8100.0]})
        cmp_df = compare_with_potential(obs, sow_years=[1980], potential_kg_ha=[7000], trend=trend,
                                        ref_harvest_year=1983)
        self.assertAlmostEqual(cmp_df["observed_detrended"].iloc[0], 8300.0 * GRAIN_DRY_FRACTION)

    def test_detrends_to_reference_year(self):
        trend = TwoSegmentTrend(breakpoint=1900, slope_before=0.0, slope_after=100.0,
                                level_at_breakpoint=0.0, rmse=0.0, rmse_single_line=0.0)
        obs = pd.DataFrame({"harvest_year": [1981, 1982], "yield_kg_ha": [8100.0, 8200.0]})
        cmp_df = compare_with_potential(obs, sow_years=[1980, 1981, 1982], potential_kg_ha=[7000, 7000, 7000],
                                        trend=trend)
        # Observations sit exactly on the trend, so detrending moves both to 1983's level.
        expected = 8300.0 * GRAIN_DRY_FRACTION
        self.assertTrue(np.allclose(cmp_df["observed_detrended"].iloc[:2], expected))
        self.assertTrue(np.isnan(cmp_df["observed"].iloc[2]))   # 1983 harvest not observed
        self.assertAlmostEqual(cmp_df["yield_gap_ratio"].iloc[0], expected / 7000)


class TestFitStats(unittest.TestCase):
    def test_bias_and_rmse(self):
        stats = fit_stats([2.0, 4.0, 6.0], [1.0, 3.0, 5.0])
        self.assertAlmostEqual(stats["bias"], 1.0)
        self.assertAlmostEqual(stats["RMSE"], 1.0)
        self.assertAlmostEqual(stats["r"], 1.0)


if __name__ == "__main__":
    unittest.main()


class TestMeasuredDiseaseLoss(unittest.TestCase):
    def test_loss_per_group(self):
        plots = pd.DataFrame({
            "n_rate_kg_ha": [0, 0, 0, 0, 60, 60, 120],
            "trt": ["fung", "fung", "none", "none", "fung", "none", "fung"],
            "yield_kg_ha": [4000.0, 4200.0, 3000.0, 3200.0, 5000.0, 4500.0, 6000.0],
        })
        loss = measured_disease_loss(plots, "trt", treated="fung", untreated="none")
        # 120 kg N has no untreated plot, so it is left out.
        self.assertEqual(loss["n_rate_kg_ha"].tolist(), [0, 60])
        self.assertAlmostEqual(loss["loss_pct"].iloc[0], 100 * (1 - 3100 / 4100))
        self.assertAlmostEqual(loss["loss_pct"].iloc[1], 10.0)
        self.assertEqual(loss["n_treated"].tolist(), [2, 1])

    def test_ignores_other_treatments(self):
        plots = pd.DataFrame({
            "n_rate_kg_ha": [0, 0, 0],
            "trt": ["fung", "none", "resistant"],
            "yield_kg_ha": [4000.0, 3000.0, 9999.0],
        })
        loss = measured_disease_loss(plots, "trt", treated="fung", untreated="none")
        self.assertAlmostEqual(loss["loss_pct"].iloc[0], 25.0)
