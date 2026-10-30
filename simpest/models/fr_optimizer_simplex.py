"""Multi-start downhill simplex, ported from the C# FraNchEstYN calibration.

The C# model calibrates with ``UNIMI.optimizer.MultiStartSimplex`` from
``optimizerSimplex.dll``, which ships with FraNchEstYN as a binary only. This
module is a line-by-line port of that library, so a simpest calibration can
follow the same search as the C# (and R) FraNchEstYN:

- **Starting points.** All ``(n + 1) * n_simplexes`` vertices are drawn up
  front, uniformly inside the bounds, from the library's own random generator
  (``RndUnif``, a three-congruential shuffle generator in the style of
  Numerical Recipes ``ran1``), vertex by vertex and parameter by parameter.
- **Search.** Each block of ``n + 1`` vertices is one simplex, minimised by the
  library's ``amoeba``: reflection (1), expansion (2), contraction (0.5) and
  shrink towards the best vertex (0.5). It stops when
  ``2 |y_hi - y_lo| / (|y_hi| + |y_lo| + 1e-20) < ftol`` or after ``max_iter``
  iterations (shrink evaluations are not counted as iterations).
- **Bounds.** A candidate outside ``(min, max]`` gets the objective ``1e300``,
  as in the C# ``Optimizer.ObjfuncVal``. That check stops at the first
  coefficient that is exactly 0 (it guards the trailing 0 the library appends
  to the starting vertices), so later coefficients are then not checked; this
  quirk is kept.
- **Result.** The final vertices of all simplexes are sorted by objective with
  the library's Shell sort, and the first row is the calibrated set.

The C# model seeds the generator from ``new Random().Next()``, so a C# run is
not reproducible. Here ``seed`` replaces that draw; with ``seed=None`` it is
drawn at random, as in C#. Given the same seed, vertices and objective values,
the search is the same as the C# one. The objective itself is
:meth:`~simpest.models.fr_runner.FranchestynRunner.compute_rmse`, the port of
the C# objective (rounded to three decimals, as in C#).

Simplexes are independent once the vertices are drawn, so ``n_jobs > 1`` runs
them in parallel through ``joblib`` without changing the result.
"""

from __future__ import annotations

import math
import random
from typing import Callable, Dict, List, Optional, Set, Tuple

import numpy as np

from .fr_optimizer_base import BaseFranchestynOptimizer
from .fr_runner import FranchestynRunner

# Largest seed whose generator initialisation stays inside a C# int
# (54773 + seed must not overflow).
MAX_SEED = 2**31 - 1 - 54773

_PENALTY = 1e300


class RndUnif:
    """Port of ``UNIMI.optimizer.RndUnif``, the library's uniform generator.

    In C# the generator state is static. Here each instance holds its own
    state, which matches a fresh C# process.
    """

    _M1, _IA1, _IC1 = 259200, 7141, 54773
    _M2, _IA2, _IC2 = 134456, 8121, 28411
    _M3, _IA3, _IC3 = 243000, 4561, 51349
    _RM1, _RM2 = 3.8580247e-06, 7.4373773e-06

    def __init__(self) -> None:
        self._iff = 0
        self._r = [0.0] * 97
        self._ix1 = self._ix2 = self._ix3 = 0

    def uniform(self, idum: int) -> float:
        """Next number in (0, 1). A negative ``idum`` (re)initialises the
        generator with seed ``-idum``, as does the first call."""
        if idum < 0 or self._iff == 0:
            if 54773 - idum > 2**31 - 1:
                raise ValueError(f"Seed {-idum} overflows the C# generator (max {MAX_SEED}).")
            self._iff = 1
            ix1 = (self._IC1 - idum) % self._M1
            ix1 = (self._IA1 * ix1 + self._IC1) % self._M1
            self._ix2 = ix1 % self._M2
            ix1 = (self._IA1 * ix1 + self._IC1) % self._M1
            self._ix3 = ix1 % self._M3
            for i in range(97):
                ix1 = (self._IA1 * ix1 + self._IC1) % self._M1
                self._ix2 = (self._IA2 * self._ix2 + self._IC2) % self._M2
                self._r[i] = (float(ix1) + float(self._ix2) * self._RM2) * self._RM1
            self._ix1 = ix1
        self._ix1 = (self._IA1 * self._ix1 + self._IC1) % self._M1
        self._ix2 = (self._IA2 * self._ix2 + self._IC2) % self._M2
        self._ix3 = (self._IA3 * self._ix3 + self._IC3) % self._M3
        j = 96 * self._ix3 // self._M3
        self._r[j] = (float(self._ix1) + float(self._ix2) * self._RM2) * self._RM1
        return self._r[j]

    def uniform_between(self, idum: int, low: float, high: float) -> float:
        """Next number in (``low``, ``high``)."""
        return low + (high - low) * self.uniform(idum)


def draw_vertices(seed: int, bounds: List[Tuple[float, float]], n_simplexes: int) -> np.ndarray:
    """Starting vertices of all simplexes, as ``MultiStartSimplex.Multistart``
    draws them.

    Args:
        seed (int): Generator seed, the C# ``new Random().Next()`` draw.
        bounds (List[Tuple[float, float]]): ``(min, max)`` per parameter.
        n_simplexes (int): Number of simplexes (C# caps it at 10000).

    Returns:
        np.ndarray: ``((n + 1) * n_simplexes, n)`` vertices; rows
        ``k (n + 1) ... (k + 1)(n + 1) - 1`` form simplex ``k``.
    """
    n_simplexes = min(n_simplexes, 10000)
    rnd = RndUnif()
    first = rnd.uniform(-seed)
    idum = int(first * 1000.0)
    n = len(bounds)
    vertices = np.empty(((n + 1) * n_simplexes, n))
    for i in range(vertices.shape[0]):
        for j, (low, high) in enumerate(bounds):
            vertices[i, j] = rnd.uniform_between(idum, low, high)
    return vertices


def amoeba(
    func: Callable[[np.ndarray], float],
    p: np.ndarray,
    y: np.ndarray,
    ftol: float,
    itmax: int,
) -> int:
    """Port of ``Simplex.amoeba``: minimise ``func`` from simplex ``p``.

    ``p`` (``(n + 1, n)``) and ``y`` (its ``n + 1`` objective values) are
    updated in place. The arithmetic follows the C# code term by term so the
    vertices come out bit for bit the same.

    Returns:
        int: Number of iterations done.
    """
    alpha, beta, gamma, tiny = 1.0, 0.5, 2.0, 1e-20
    ndim = p.shape[1]
    mpts = ndim + 1
    iterations = 0
    while True:
        ilo = 0
        if y[0] > y[1]:
            ihi, inhi = 0, 1
        else:
            ihi, inhi = 1, 0
        for i in range(mpts):
            if y[i] < y[ilo]:
                ilo = i
            if y[i] > y[ihi]:
                inhi = ihi
                ihi = i
            elif y[i] > y[inhi] and i != ihi:
                inhi = i
        rtol = 2.0 * abs(y[ihi] - y[ilo]) / (abs(y[ihi]) + abs(y[ilo]) + tiny)
        if rtol < ftol or iterations == itmax:
            return iterations
        iterations += 1

        pbar = np.zeros(ndim)
        for k in range(mpts):
            if k != ihi:
                pbar += p[k]
        pbar /= ndim
        pr = (1.0 + alpha) * pbar - alpha * p[ihi]
        ypr = func(pr)
        if ypr <= y[ilo]:
            prr = gamma * pr + (1.0 - gamma) * pbar
            yprr = func(prr)
            if yprr < y[ilo]:
                p[ihi], y[ihi] = prr, yprr
            else:
                p[ihi], y[ihi] = pr, ypr
        elif ypr >= y[inhi]:
            if ypr < y[ihi]:
                p[ihi], y[ihi] = pr, ypr
            prr = beta * p[ihi] + (1.0 - beta) * pbar
            yprr = func(prr)
            if yprr < y[ihi]:
                p[ihi], y[ihi] = prr, yprr
                continue
            for i in range(mpts):
                if i != ilo:
                    p[i] = 0.5 * (p[i] + p[ilo])
                    y[i] = func(p[i].copy())
        else:
            p[ihi], y[ihi] = pr, ypr


def sort_rows(arr: np.ndarray, col: int) -> None:
    """Port of ``Sort2dArray.Sort2d`` (ascending): Shell sort of the rows of
    ``arr`` by column ``col``, in place. Ties keep the order this sort gives
    them, which decides which of several equally good vertices is returned."""
    first, last = 0, arr.shape[0] - 1
    gap = 0
    while True:
        gap = gap * 3 + 1
        if gap > last - first + 1:
            break
    while True:
        gap = int(round(gap / 3.0))   # C# Math.Round: half to even, as Python
        for i in range(gap + first, last + 1):
            value = arr[i, col]
            row = arr[i].copy()
            j = i
            while arr[j - gap, col] > value:
                arr[j] = arr[j - gap]
                j -= gap
                if j - gap < first:
                    break
            arr[j] = row
        if gap == 1:
            break


def _run_simplex(optimizer: "MultiStartSimplexOptimizer", p: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int]:
    """Minimise one simplex. Module-level so ``joblib`` can pickle the call."""
    y = np.array([optimizer._objfunc(p[i].copy()) for i in range(p.shape[0])])
    iterations = amoeba(optimizer._objfunc, p, y, optimizer.ftol, optimizer.max_iter)
    return p, y, iterations


class MultiStartSimplexOptimizer(BaseFranchestynOptimizer):
    """Calibration with the C# FraNchEstYN multi-start simplex.

    Args:
        runner (FranchestynRunner): Fully configured runner instance.
        calibration_variable (str): Calibration target scope: ``"crop"``,
            ``"disease"``, or ``"all"``.
        disabled_by_class (Optional[Dict[str, Set[str]]]): Parameter names to
            exclude from calibration, keyed by parameter class.
        seed (Optional[int]): Seed of the vertex generator, ``0`` to
            :data:`MAX_SEED`. ``None`` (default) draws one at random, as C#
            does; the seed used is printed and kept in :attr:`seed_used`.
        n_simplexes (int): Number of simplexes (C# ``simplexes``; the R
            package uses 3).
        max_iter (int): Maximum iterations per simplex (C# ``iterations``).
        ftol (float): Relative tolerance on the objective spread of a simplex
            (C# uses ``1e-12``).
        n_jobs (int): Simplexes to run concurrently through ``joblib``. The
            result does not depend on it.
    """

    def __init__(
        self,
        runner: FranchestynRunner,
        calibration_variable: str = "all",
        disabled_by_class: Optional[Dict[str, Set[str]]] = None,
        seed: Optional[int] = None,
        n_simplexes: int = 3,
        max_iter: int = 100,
        ftol: float = 1e-12,
        n_jobs: int = 1,
    ) -> None:
        super().__init__(
            runner=runner,
            calibration_variable=calibration_variable,
            disabled_by_class=disabled_by_class,
            seed=seed,
        )
        self.n_simplexes = n_simplexes
        self.max_iter = max_iter
        self.ftol = ftol
        self.n_jobs = n_jobs
        self.seed_used: Optional[int] = None

    @classmethod
    def from_config(cls, runner, config, disabled_by_class=None) -> "MultiStartSimplexOptimizer":
        """Build from a :class:`FranchestynConfig` (reads ``seed``,
        ``n_restarts`` as the number of simplexes, ``max_iter``, ``ftol`` and
        ``n_jobs``)."""
        return cls(
            runner=runner,
            calibration_variable=config.calibration_variable,
            disabled_by_class=disabled_by_class,
            seed=getattr(config, "seed", None),
            n_simplexes=config.n_restarts,
            max_iter=config.max_iter,
            ftol=getattr(config, "ftol", 1e-12),
            n_jobs=getattr(config, "n_jobs", 1),
        )

    def calibrate(self) -> Dict[str, float]:
        """Run the multi-start simplex and return the best parameter set.

        Returns:
            Dict[str, float]: Best-fit parameter values keyed by
            ``class_ParamName``. Empty when nothing is flagged for calibration.
        """
        if not self.calib_keys:
            print("No calibration parameters found — returning defaults.")
            return {}

        seed = self.seed
        if seed is None:
            seed = random.SystemRandom().randint(0, MAX_SEED)
        if not 0 <= seed <= MAX_SEED:
            raise ValueError(f"seed must be in 0..{MAX_SEED}, got {seed}.")
        self.seed_used = seed

        n = len(self.calib_keys)
        print(
            f"- Calibrating {n} parameters with the C# multi-start simplex "
            f"({self.n_simplexes} simplexes, {self.max_iter} iterations, seed {seed}).\n"
            f"- Parameters:\n{self.calib_keys}"
        )
        vertices = draw_vertices(seed, self.bounds, self.n_simplexes)
        blocks = [vertices[k:k + n + 1].copy() for k in range(0, vertices.shape[0], n + 1)]

        if self.n_jobs == 1:
            done = []
            for k, block in enumerate(blocks):
                done.append(_run_simplex(self, block))
                print(f"  simplex {k + 1}/{len(blocks)}: best {done[-1][1].min():.3f}, "
                      f"{done[-1][2]} iterations")
        else:
            from joblib import Parallel, delayed

            done = Parallel(n_jobs=self.n_jobs, verbose=10)(
                delayed(_run_simplex)(self, block) for block in blocks
            )

        # MultiStartSimplex.Multistart: every final vertex with its objective
        # and simplex number, sorted by objective; the first row wins.
        results = np.zeros((vertices.shape[0], n + 2))
        for k, (p, y, _) in enumerate(done):
            rows = slice(k * (n + 1), (k + 1) * (n + 1))
            results[rows, :n] = p
            results[rows, n] = y
            results[rows, n + 1] = k + 1
        sort_rows(results, n)

        best_params = {key: float(v) for key, v in zip(self.calib_keys, results[0, :n])}
        print(f"\nBest RMSE: {results[0, n]:.4f}")
        print(best_params)
        return best_params

    def _objfunc(self, x: np.ndarray) -> float:
        """The C# ``Optimizer.ObjfuncVal``: bounds check, then the model RMSE."""
        for j, (low, high) in enumerate(self.bounds):
            if x[j] == 0:
                break
            if x[j] <= low or x[j] > high:
                return _PENALTY
        return self._evaluate_rmse(x)
