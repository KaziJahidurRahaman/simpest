"""Tests for the opt-in switches that let an epidemic fail to establish."""

import unittest

from simpest.models.fr_data import InputsDaily, Outputs, Parameters
from simpest.models.fr_disease_model import DiseaseModel
from simpest.models.franchestyn import FranchestynConfig
from simpest.models.fr_runner import FranchestynRunner


def _infection_setup(rainfall):
    """A day that is ready to infect: onset passed, full inoculum, wet enough."""
    parameters = Parameters()
    pd_ = parameters.par_disease
    pd_.is_splash_borne = 1.0
    pd_.outer_inoculum_shape_release = 0.0   # constant release
    pd_.outer_inoculum_max = 0.5
    pd_.rain50_detachment = 5.0
    pd_.hydro_thermal_time_onset = 1.0
    pd_.cycle_percentage_onset = 0.0
    pd_.latency_duration = 100.0
    pd_.sporulation_duration = 100.0
    pd_.tmin, pd_.topt, pd_.tmax = 5.0, 20.0, 30.0

    output = Outputs()
    output.disease.hydro_thermal_time_state = 10.0
    output.disease.hydro_thermal_time_infection = 1.0
    output.disease.susceptible_fraction = 1.0
    output.crop.cycle_completion_percentage = 50.0
    output.crop.light_interception_attainable = 1.0

    output1 = Outputs()
    output1.crop.light_interception_attainable = 1.0

    input_ = InputsDaily()
    input_.tmin, input_.tmax = 15.0, 25.0
    input_.precipitation = rainfall
    return parameters, input_, output, output1


def _latent_after_one_day(rainfall, **switches):
    parameters, input_, output, output1 = _infection_setup(rainfall)
    for name, value in switches.items():
        setattr(parameters, name, value)
    DiseaseModel().run_daily(input_, parameters, output, output1)
    return output1.disease.latent_sum


class TestRainLimitedPrimaryInoculum(unittest.TestCase):
    def test_default_ignores_rain(self):
        """C# behaviour: primary infection is the same wet or dry."""
        self.assertAlmostEqual(_latent_after_one_day(0.0), _latent_after_one_day(20.0))
        self.assertGreater(_latent_after_one_day(0.0), 0.0)

    def test_switch_starves_a_dry_day(self):
        dry = _latent_after_one_day(0.0, rain_limited_primary_inoculum=True)
        wet = _latent_after_one_day(20.0, rain_limited_primary_inoculum=True)
        self.assertEqual(dry, 0.0)
        self.assertGreater(wet, 0.0)

    def test_switch_is_ignored_for_non_splash_borne(self):
        parameters, input_, output, output1 = _infection_setup(0.0)
        parameters.par_disease.is_splash_borne = 0.0
        parameters.rain_limited_primary_inoculum = True
        DiseaseModel().run_daily(input_, parameters, output, output1)
        self.assertGreater(output1.disease.latent_sum, 0.0)


class TestDryCounterAndDecay(unittest.TestCase):
    def _bare_runner(self, **switches):
        runner = FranchestynRunner.__new__(FranchestynRunner)
        runner.name_param = {}
        runner.param_out_calibration = {}
        runner.rain_limited_primary_inoculum = switches.get("rain_limited_primary_inoculum", False)
        runner.htt_state_dry_decay = switches.get("htt_state_dry_decay", 1.0)
        runner.carry_dry_counter = switches.get("carry_dry_counter", False)
        return runner

    def test_switches_reach_the_parameters_object(self):
        parameters = self._bare_runner(
            rain_limited_primary_inoculum=True, htt_state_dry_decay=0.5
        )._build_parameters({})
        self.assertTrue(parameters.rain_limited_primary_inoculum)
        self.assertEqual(parameters.htt_state_dry_decay, 0.5)

    def test_defaults_match_csharp(self):
        parameters = self._bare_runner()._build_parameters({})
        self.assertFalse(parameters.rain_limited_primary_inoculum)
        self.assertEqual(parameters.htt_state_dry_decay, 1.0)
        config = FranchestynConfig(fungicide_type=None)
        self.assertFalse(config.rain_limited_primary_inoculum)
        self.assertEqual(config.htt_state_dry_decay, 1.0)
        self.assertFalse(config.carry_dry_counter)

    def test_dry_decay_shrinks_carried_state(self):
        """A dry-interrupted day keeps only the surviving fraction."""
        parameters = Parameters()
        parameters.htt_state_dry_decay = 0.5
        parameters.par_disease.dry_critical_interruption = 10.0
        parameters.par_disease.wetness_duration_optimum = 10.0
        parameters.par_disease.wetness_duration_minimum = 1.0

        output = Outputs()
        output.disease.hydro_thermal_time_state = 8.0
        output1 = Outputs()
        output1.disease.counter_dry = 20.0          # longer than the interruption
        output1.disease.hydro_thermal_time_rate = 0.0

        model = DiseaseModel()
        from datetime import datetime
        from simpest.models.fr_data import InputsHourly
        model.run_hourly(InputsHourly(date=datetime(2000, 5, 1, 23)), parameters, output, output1)

        self.assertEqual(output1.disease.hydro_thermal_time_state, 4.0)
        self.assertEqual(output1.disease.hydro_thermal_time_infection, 0.0)


if __name__ == "__main__":
    unittest.main()
