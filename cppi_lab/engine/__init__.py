"""Simulation engine: signal pre-computation, CPPI kernel and Monte Carlo runner."""

from numba import config as _numba_config

# Prefer OpenMP for ``parallel=True`` kernels.  It is thread safe (the ensemble
# runs on a worker thread) and avoids the warning Numba prints when it probes
# an outdated system TBB before falling back.
_numba_config.THREADING_LAYER_PRIORITY = ["omp", "tbb", "workqueue"]
