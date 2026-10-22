"""Tests for scoring yield observations dated after the simulated maturity."""

import unittest
from datetime import date, datetime
from types import SimpleNamespace
from unittest import mock

import pandas as pd

from simpest.models.fr_data import Outputs
from simpest.models import franchestyn
from simpest.models.franchestyn import FranchestynConfig
from simpest.models.fr_runner import FranchestynRunner


def _day(cycle_pct, yield_att, yield_act):
    out = Outputs()
    out.crop.cycle_completion_percentage = cycle_pct
    out.crop.day_after_sowing = 10
    out.crop.yield_attainable = yield_att
    out.crop.yield_actual = yield_act
    return out


def _runner(switch, yield_attainable=None, yield_actual=None):
    runner = FranchestynRunner.__new__(FranchestynRunner)
    runner.disease = "sev"
    runner.upper_leaf_pool = False
    runner.use_prev_day_alignment = True
    runner.score_harvest_after_maturity = switch
    runner.sim_unit = SimpleNamespace(reference_data=SimpleNamespace(
        date_agb={}, date_fint={}, disease_date_disease_sev={},
        date_yield_attainable=yield_attainable or {}, date_yield_actual=yield_actual or {},
    ))
    return runner


# A season that matures on 10 May; the yield is measured on 15 May.
OUTPUTS = {
    datetime(2014, 5, 9, 23): _day(99.0, 6000.0, 3000.0),
    datetime(2014, 5, 10, 23): _day(100.0, 7000.0, 3500.0),
}


class TestScoreHarvestAfterMaturity(unittest.TestCase):
    def test_default_matches_csharp(self):
        self.assertFalse(FranchestynConfig(fungicide_type=None).score_harvest_after_maturity)

    def test_off_leaves_post_maturity_yield_out(self):
        runner = _runner(False, yield_attainable={date(2014, 5, 15): 7600.0})
        self.assertEqual(runner.compute_rmse(OUTPUTS, include_crop=True, include_disease=False), 0.0)

    def test_on_scores_against_the_final_simulated_yield(self):
        runner = _runner(True, yield_attainable={date(2014, 5, 15): 7600.0},
                         yield_actual={date(2014, 5, 15): 3400.0})
        # Attainable only: (7600 - 7000) / 100 = 6.
        self.assertAlmostEqual(runner.compute_rmse(OUTPUTS, include_crop=True, include_disease=False), 6.0)
        # Both on the same date sum into one error: sqrt(36 + 1).
        self.assertAlmostEqual(runner.compute_rmse(OUTPUTS, include_crop=True, include_disease=True),
                               round(37 ** 0.5, 3))

    def test_observation_before_maturity_is_not_matched_to_a_later_day(self):
        # An unsimulated date whose last earlier day has not matured is skipped.
        outputs = {datetime(2014, 5, 9, 23): _day(99.0, 6000.0, 3000.0)}
        runner = _runner(True, yield_attainable={date(2014, 5, 15): 7600.0})
        self.assertEqual(runner.compute_rmse(outputs, include_crop=True, include_disease=False), 0.0)

    def test_simulated_day_is_left_to_the_daily_loop(self):
        # 10 May is simulated, so the 11 May observation is scored once, by the daily loop.
        runner = _runner(True, yield_attainable={date(2014, 5, 11): 7600.0})
        self.assertAlmostEqual(runner.compute_rmse(OUTPUTS, include_crop=True, include_disease=False), 6.0)

    def test_switch_reaches_the_runner(self):
        config = FranchestynConfig(fungicide_type=None, score_harvest_after_maturity=True)
        frame = pd.DataFrame({"x": [0]})
        with mock.patch.object(franchestyn, "FranchestynRunner") as fake_runner:
            with franchestyn._build_franchestyn_runner(2000, 2000, config, frame, frame, None, frame):
                pass
        self.assertTrue(fake_runner.call_args.kwargs["score_harvest_after_maturity"])


if __name__ == "__main__":
    unittest.main()
