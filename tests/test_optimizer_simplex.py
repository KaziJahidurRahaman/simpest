#!/usr/bin/env python

"""Tests for the port of the C# multi-start simplex (fr_optimizer_simplex).

The reference numbers were produced by the original ``optimizerSimplex.dll``
of FraNchEstYN (called through pythonnet on 2026-10-22), so these tests pin
the port to the C# library without needing .NET.
"""

import unittest

import numpy as np

from simpest.models.fr_optimizer_base import build_optimizer
from simpest.models.fr_optimizer_simplex import (
    MAX_SEED,
    MultiStartSimplexOptimizer,
    RndUnif,
    amoeba,
    draw_vertices,
    sort_rows,
)
from tests.test_optimizer import _FakeConfig, _two_param_setup

ROSEN_BOUNDS = [(-2.0, 2.0), (-1.0, 3.0)]


def _rosenbrock(x):
    """The library's ``ObjFuncRosenbrok``, including its bounds check."""
    for j, (low, high) in enumerate(ROSEN_BOUNDS):
        if x[j] <= low or x[j] > high:
            return 1e300
    return 100.0 * (x[1] - x[0] * x[0]) ** 2 + (1.0 - x[0]) * (1.0 - x[0])


class TestRndUnif(unittest.TestCase):
    def test_matches_csharp_draws(self):
        rnd = RndUnif()
        draws = [rnd.uniform(-12345)] + [rnd.uniform(7) for _ in range(4)]
        self.assertEqual(draws, [0.5594846911194373, 0.4795962319446297, 0.9985495857792038,
                                 0.8524612036487474, 0.6107650482440231])

    def test_seed_that_overflows_csharp_int_is_rejected(self):
        RndUnif().uniform(-MAX_SEED)
        with self.assertRaises(ValueError):
            RndUnif().uniform(-(MAX_SEED + 1))


class TestMultiStartSimplex(unittest.TestCase):
    def test_vertices_match_csharp(self):
        v = draw_vertices(42, ROSEN_BOUNDS, 3)
        self.assertEqual(v.shape, (9, 2))
        self.assertEqual(v[0, 0], 0.286855579450612)
        self.assertEqual(v[8, 1], 2.1711349118535135)

    def test_search_and_sort_match_csharp(self):
        # MultiStartSimplex.Multistart with Itmax 25 on ObjFuncRosenbrok.
        v = draw_vertices(42, ROSEN_BOUNDS, 3)
        results = np.zeros((9, 4))
        for k in range(3):
            p = v[3 * k:3 * k + 3].copy()
            y = np.array([_rosenbrock(row) for row in p])
            amoeba(_rosenbrock, p, y, 1e-12, 25)
            results[3 * k:3 * k + 3, :2], results[3 * k:3 * k + 3, 2], results[3 * k:3 * k + 3, 3] = p, y, k + 1
        sort_rows(results, 2)
        np.testing.assert_array_equal(
            results[0], [0.9991733917629975, 0.9985816065697812, 6.165424020929838e-06, 2.0])
        np.testing.assert_array_equal(
            results[8], [0.4594238913689296, 0.21710995571490505, 0.29587025889063867, 3.0])

    def test_sort_is_ascending(self):
        arr = np.array([[5.0, 0], [1.0, 1], [3.0, 2], [2.0, 3], [4.0, 4]])
        sort_rows(arr, 0)
        self.assertEqual(arr[:, 0].tolist(), [1.0, 2.0, 3.0, 4.0, 5.0])


class TestMultiStartSimplexOptimizer(unittest.TestCase):
    def _optimizer(self, **overrides):
        runner, targets = _two_param_setup()
        cfg = _FakeConfig(**{"n_restarts": 3, "max_iter": 200, "seed": 7, **overrides})
        return build_optimizer("multistart-simplex", runner, cfg, None), targets

    def test_registered_and_built_from_config(self):
        opt, _ = self._optimizer()
        self.assertIsInstance(opt, MultiStartSimplexOptimizer)
        self.assertEqual((opt.n_simplexes, opt.max_iter, opt.seed), (3, 200, 7))

    def test_finds_minimum_and_is_reproducible(self):
        opt, targets = self._optimizer()
        best = opt.calibrate()
        for key, target in targets.items():
            self.assertAlmostEqual(best[key], target, places=4)
        again, _ = self._optimizer()
        self.assertEqual(again.calibrate(), best)

    def test_parallel_gives_the_same_result(self):
        serial, _ = self._optimizer()
        parallel, _ = self._optimizer(n_jobs=2)
        self.assertEqual(parallel.calibrate(), serial.calibrate())

    def test_bounds_follow_csharp(self):
        # (min, max]: the lower bound itself is out, the upper bound is in.
        opt, _ = self._optimizer()
        self.assertEqual(opt._objfunc(np.array([0.0 + 1e-9, 5.0])), (4.0 - 1e-9) ** 2 + 16.0)
        self.assertEqual(opt._objfunc(np.array([10.0, -4.0])), 36.0 + 25.0)
        self.assertEqual(opt._objfunc(np.array([10.1, 1.0])), 1e300)
        self.assertEqual(opt._objfunc(np.array([1.0, -5.0])), 1e300)

    def test_random_seed_is_recorded(self):
        opt, _ = self._optimizer(seed=None)
        opt.calibrate()
        self.assertTrue(0 <= opt.seed_used <= MAX_SEED)


if __name__ == "__main__":
    unittest.main()
