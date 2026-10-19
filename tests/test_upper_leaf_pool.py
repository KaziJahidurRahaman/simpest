"""Tests for the opt-in upper-leaf tissue pool of the disease model."""

import unittest
from datetime import date, datetime
from types import SimpleNamespace

from simpest.models.fr_data import InputsDaily, Outputs, Parameters
from simpest.models.fr_disease_model import DiseaseModel, _upper_leaf_emerged
from simpest.models.franchestyn import FranchestynConfig, _outputs_to_records
from simpest.models.fr_runner import FranchestynRunner


def _infection_day(cycle_pct, rainfall=0.0, latency=100.0, **switches):
    """A day that is ready to infect, at a given crop cycle percentage.

    Primary infection pressure is 1.0 (efficiency) x 0.5 (constant inoculum).
    """
    parameters = Parameters()
    pd_ = parameters.par_disease
    pd_.is_splash_borne = 1.0
    pd_.outer_inoculum_shape_release = 0.0
    pd_.outer_inoculum_max = 0.5
    pd_.rain50_detachment = 5.0
    pd_.pathogen_spread = 0.2
    pd_.hydro_thermal_time_onset = 1.0
    pd_.cycle_percentage_onset = 0.0
    pd_.latency_duration = latency
    pd_.sporulation_duration = 100.0
    pd_.tmin, pd_.topt, pd_.tmax = 5.0, 20.0, 30.0
    for name, value in switches.items():
        setattr(parameters, name, value)

    output = Outputs()
    output.disease.hydro_thermal_time_state = 10.0
    output.disease.hydro_thermal_time_infection = 1.0
    output.disease.sporulation_efficiency = 1.0
    output.disease.susceptible_fraction = 1.0
    output.crop.cycle_completion_percentage = cycle_pct
    output.crop.light_interception_attainable = 1.0

    output1 = Outputs()
    output1.crop.light_interception_attainable = 1.0

    input_ = InputsDaily()
    input_.tmin, input_.tmax = 20.0, 20.0   # at Topt: one day of progress per day
    input_.precipitation = rainfall
    return parameters, input_, output, output1


def _run(cycle_pct, upper_sporulating=0.0, **kwargs):
    parameters, input_, output, output1 = _infection_day(cycle_pct, **kwargs)
    output.disease.upper_leaf_sporulating = upper_sporulating
    output.disease.upper_leaf_affected = upper_sporulating
    DiseaseModel().run_daily(input_, parameters, output, output1)
    return output1.disease


class TestUpperLeafEmerged(unittest.TestCase):
    def test_linear_between_start_and_full(self):
        self.assertEqual(_upper_leaf_emerged(30.0, 36.0, 52.0), 0.0)
        self.assertEqual(_upper_leaf_emerged(36.0, 36.0, 52.0), 0.0)
        self.assertAlmostEqual(_upper_leaf_emerged(44.0, 36.0, 52.0), 0.5)
        self.assertEqual(_upper_leaf_emerged(52.0, 36.0, 52.0), 1.0)
        self.assertEqual(_upper_leaf_emerged(80.0, 36.0, 52.0), 1.0)

    def test_step_when_full_is_not_above_start(self):
        self.assertEqual(_upper_leaf_emerged(39.9, 40.0, 40.0), 0.0)
        self.assertEqual(_upper_leaf_emerged(40.0, 40.0, 40.0), 1.0)


class TestUpperLeafPool(unittest.TestCase):
    def test_off_by_default_and_leaves_no_trace(self):
        disease = _run(44.0)
        self.assertEqual(disease.upper_leaf_emerged, 0.0)
        self.assertEqual(disease.upper_leaf_latent, 0.0)
        self.assertEqual(disease.upper_leaf_severity, 0.0)

    def test_canopy_epidemic_is_unchanged(self):
        off = _run(44.0, rainfall=10.0)
        on = _run(44.0, rainfall=10.0, upper_leaf_pool=True)
        for name in ("latent_sum", "sporulating_sum", "dead_sum", "affected_sum",
                     "disease_severity", "susceptible_fraction"):
            self.assertEqual(getattr(on, name), getattr(off, name), name)

    def test_no_infection_before_the_leaves_emerge(self):
        disease = _run(30.0, upper_leaf_pool=True)
        self.assertEqual(disease.upper_leaf_emerged, 0.0)
        self.assertEqual(disease.upper_leaf_latent, 0.0)

    def test_infection_is_limited_to_the_emerged_fraction(self):
        half = _run(44.0, upper_leaf_pool=True)
        full = _run(60.0, upper_leaf_pool=True)
        self.assertAlmostEqual(half.upper_leaf_emerged, 0.5)
        self.assertAlmostEqual(half.upper_leaf_latent, 0.5 * 0.5)   # pressure x emerged
        self.assertAlmostEqual(full.upper_leaf_latent, 0.5 * 1.0)
        self.assertAlmostEqual(half.upper_leaf_affected, 0.25)
        self.assertEqual(half.upper_leaf_severity, 0.0)             # still latent

    def test_window_is_configurable(self):
        disease = _run(44.0, upper_leaf_pool=True, upper_leaf_cycle_start=40.0,
                       upper_leaf_cycle_full=60.0)
        self.assertAlmostEqual(disease.upper_leaf_emerged, 0.2)

    def test_own_sporulating_tissue_adds_secondary_infection_when_it_rains(self):
        dry = _run(60.0, upper_sporulating=0.2, upper_leaf_pool=True)
        wet = _run(60.0, upper_sporulating=0.2, rainfall=5.0, upper_leaf_pool=True)
        # Dry: primary only, on the unaffected 0.8. Rain of Rain50 x fint
        # detaches half: + 0.2 (sporulating) x 0.2 (PathogenSpread) x 0.5.
        self.assertAlmostEqual(dry.upper_leaf_latent, 0.5 * 0.8)
        self.assertAlmostEqual(wet.upper_leaf_latent, (0.5 + 0.2 * 0.2 * 0.5) * 0.8)

    def test_severity_is_relative_to_the_emerged_area(self):
        # Latency of one day at Topt: the new cohort sporulates the same day.
        disease = _run(44.0, latency=1.0, upper_leaf_pool=True)
        self.assertAlmostEqual(disease.upper_leaf_sporulating, 0.25)
        self.assertAlmostEqual(disease.upper_leaf_severity, 0.5)

    def test_pool_fills_up_over_days(self):
        parameters, input_, output, output1 = _infection_day(60.0, upper_leaf_pool=True)
        model = DiseaseModel()
        for _ in range(60):
            model.run_daily(input_, parameters, output, output1)
            output.disease.upper_leaf_affected = output1.disease.upper_leaf_affected
            output.disease.upper_leaf_sporulating = output1.disease.upper_leaf_sporulating
        self.assertLessEqual(output1.disease.upper_leaf_affected, 1.0)
        self.assertGreater(output1.disease.upper_leaf_affected, 0.99)


class TestUpperLeafPoolWiring(unittest.TestCase):
    def _bare_runner(self, upper_leaf_pool):
        runner = FranchestynRunner.__new__(FranchestynRunner)
        runner.name_param = {}
        runner.param_out_calibration = {}
        runner.rain_limited_primary_inoculum = False
        runner.htt_state_dry_decay = 1.0
        runner.carry_dry_counter = False
        runner.upper_leaf_pool = upper_leaf_pool
        runner.upper_leaf_cycle_start = 40.0
        runner.upper_leaf_cycle_full = 55.0
        return runner

    def test_defaults_match_csharp(self):
        config = FranchestynConfig(fungicide_type=None)
        self.assertFalse(config.upper_leaf_pool)
        self.assertEqual((config.upper_leaf_cycle_start, config.upper_leaf_cycle_full), (36.0, 52.0))
        self.assertFalse(Parameters().upper_leaf_pool)

    def test_switch_reaches_the_parameters_object(self):
        parameters = self._bare_runner(True)._build_parameters({})
        self.assertTrue(parameters.upper_leaf_pool)
        self.assertEqual(parameters.upper_leaf_cycle_start, 40.0)
        self.assertEqual(parameters.upper_leaf_cycle_full, 55.0)

    def test_objective_uses_upper_leaf_severity_when_on(self):
        out = Outputs()
        out.disease.disease_severity = 0.10
        out.disease.upper_leaf_severity = 0.40
        outputs = {datetime(2000, 5, 1, 23): out}
        reference = SimpleNamespace(
            disease_date_disease_sev={"sev": {date(2000, 5, 1): 40.0}}, date_yield_actual={},
        )
        for switch, expected in ((False, 30.0), (True, 0.0)):
            runner = self._bare_runner(switch)
            runner.disease = "sev"
            runner.use_prev_day_alignment = False
            runner.sim_unit = SimpleNamespace(reference_data=reference)
            rmse = runner.compute_rmse(outputs, include_crop=False, include_disease=True)
            self.assertAlmostEqual(rmse, expected)

    def test_records_carry_the_upper_leaf_columns(self):
        out = Outputs()
        out.disease.upper_leaf_emerged = 0.5
        out.disease.upper_leaf_severity = 0.25
        record = _outputs_to_records({datetime(2000, 5, 1, 23): out})[0]
        self.assertEqual(record["UpperLeafEmerged"], 0.5)
        self.assertEqual(record["UpperLeafSeverity"], 0.25)


if __name__ == "__main__":
    unittest.main()
