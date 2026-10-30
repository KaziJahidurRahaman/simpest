"""Tests for how the objective scores observations outside the simulated season.

``compute_rmse`` follows the C# ``Optimizer.ObjfuncVal``: every observation in
the simulated years is scored against the state at hour 0 of its day, which is
zero output before sowing and after maturity. ``score_harvest_after_maturity``
(simpest only) scores a post-maturity yield against the season's final yield.
"""

import math
import unittest
from datetime import date, datetime
from types import SimpleNamespace
from unittest import mock

import pandas as pd

from simpest.models.fr_data import Outputs
from simpest.models import franchestyn
from simpest.models.franchestyn import FranchestynConfig
from simpest.models.fr_runner import FranchestynRunner


def _day(cycle_pct, yield_att, yield_act, fint=0.5):
    out = Outputs()
    out.crop.cycle_completion_percentage = cycle_pct
    out.crop.day_after_sowing = 10
    out.crop.yield_attainable = yield_att
    out.crop.yield_actual = yield_act
    out.crop.light_interception_attainable = fint
    return out


def _runner(switch, yield_attainable=None, yield_actual=None, fint=None, sowing_doy=None):
    runner = FranchestynRunner.__new__(FranchestynRunner)
    runner.disease = "sev"
    runner.upper_leaf_pool = False
    runner.use_prev_day_alignment = True
    runner.score_harvest_after_maturity = switch
    runner.start_year, runner.end_year = 2013, 2014
    runner.sim_unit = SimpleNamespace(
        year_sowing_doy=sowing_doy or {},
        reference_data=SimpleNamespace(
            date_agb={}, date_fint=fint or {}, disease_date_disease_sev={},
            date_yield_attainable=yield_attainable or {}, date_yield_actual=yield_actual or {},
        ))
    return runner


# A season that matures on 10 May; the yield is measured on 15 May.
OUTPUTS = {
    datetime(2014, 5, 9, 23): _day(99.0, 6000.0, 3000.0),
    datetime(2014, 5, 10, 23): _day(100.0, 7000.0, 3500.0),
}
# Zero yield after maturity: (7600 / 100)^2 x 1000.
ZERO_YIELD_PENALTY = round(math.sqrt(76.0 ** 2 * 1000.0), 3)


class TestPostMaturityObservations(unittest.TestCase):
    def test_default_matches_csharp(self):
        self.assertFalse(FranchestynConfig(fungicide_type=None).score_harvest_after_maturity)

    def test_yield_after_maturity_is_a_penalised_zero(self):
        runner = _runner(False, yield_attainable={date(2014, 5, 15): 7600.0})
        self.assertEqual(runner.compute_rmse(OUTPUTS, include_crop=True, include_disease=False),
                         ZERO_YIELD_PENALTY)

    def test_day_after_maturity_already_sees_zero(self):
        # C# resets the outputs at hour 0 of the day after maturity.
        runner = _runner(False, yield_attainable={date(2014, 5, 11): 7600.0})
        self.assertEqual(runner.compute_rmse(OUTPUTS, include_crop=True, include_disease=False),
                         ZERO_YIELD_PENALTY)

    def test_growing_crop_is_compared_with_the_previous_day(self):
        runner = _runner(False, yield_attainable={date(2014, 5, 10): 6600.0})
        self.assertEqual(runner.compute_rmse(OUTPUTS, include_crop=True, include_disease=False), 6.0)

    def test_before_sowing_zero_without_penalty(self):
        runner = _runner(False, yield_attainable={date(2014, 5, 1): 7600.0})
        self.assertEqual(runner.compute_rmse(OUTPUTS, include_crop=True, include_disease=False), 76.0)

    def test_fint_on_sowing_day_counts_as_planted(self):
        # Sowing on 9 May (DOY 129): zero interception while planted is x1000.
        runner = _runner(False, fint={date(2014, 5, 9): 0.5}, sowing_doy={2014: 129})
        self.assertEqual(runner.compute_rmse(OUTPUTS, include_crop=True, include_disease=False),
                         round(math.sqrt(50.0 ** 2 * 1000.0), 3))

    def test_observations_outside_the_simulated_years_are_ignored(self):
        runner = _runner(False, yield_attainable={date(2016, 5, 15): 7600.0})
        self.assertEqual(runner.compute_rmse(OUTPUTS, include_crop=True, include_disease=False), 0.0)

    def test_switch_scores_against_the_final_simulated_yield(self):
        runner = _runner(True, yield_attainable={date(2014, 5, 15): 7600.0},
                         yield_actual={date(2014, 5, 15): 3400.0})
        # Attainable only: (7600 - 7000) / 100 = 6.
        self.assertAlmostEqual(runner.compute_rmse(OUTPUTS, include_crop=True, include_disease=False), 6.0)
        # Both on the same date sum into one error: sqrt(36 + 1).
        self.assertAlmostEqual(runner.compute_rmse(OUTPUTS, include_crop=True, include_disease=True),
                               round(37 ** 0.5, 3))

    def test_switch_reaches_the_runner(self):
        config = FranchestynConfig(fungicide_type=None, score_harvest_after_maturity=True)
        frame = pd.DataFrame({"x": [0]})
        with mock.patch.object(franchestyn, "FranchestynRunner") as fake_runner:
            with franchestyn._build_franchestyn_runner(2000, 2000, config, frame, frame, None, frame):
                pass
        self.assertTrue(fake_runner.call_args.kwargs["score_harvest_after_maturity"])
