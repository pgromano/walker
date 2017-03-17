import numpy as np
from . import simulators, stats
from multiprocessing import Pool
from functools import partial

class Walker(object):
	def __init__(self, method="Gaussian", **kwargs):
		if method == 'Gaussian':
			stats.set_attributes.gaussian(self, **kwargs)

	def log_liklihood(self, X):
		return np.log(-(stats.functions._gaussian.distribution(self, X)))+np.log(-(stats.functions._gaussian.distribution(self, X))).min()

	def potential(self, X):
		return stats.functions._gaussian.distribution(self, X)

	def gradient(self, X):
		return stats.functions._gaussian.gradient(self, X)

	def simulate(self, steps, n_walkers=1, n_jobs=2, **kwargs):
		if n_walkers == 1:
			return self._simulate(steps, **kwargs)
		else:
			return self._mpi_simulate(steps, n_walkers=n_walkers, n_jobs=n_jobs, **kwargs)

	def string(self, a, b, **kwargs):
		return simulators.zero_temp_string.run(self, a, b, **kwargs)

	def _simulate(self, steps, **kwargs):
		return simulators.diffusion.run(self, steps, **kwargs)

	def _mpi_simulate(self, steps, n_walkers=2, n_jobs=1, **kwargs):
		n_walker_iter = np.empty(n_walkers)
		n_walker_iter.fill(steps)
		p = Pool(n_jobs)
		return p.map(partial(self.simulate, **kwargs), n_walker_iter.astype(int))

	def accelerated_simulate(self, steps, n_walkers=1, n_jobs=2, **kwargs):
		if n_walkers == 1:
			return self._acc_simulate(steps, **kwargs)
		else:
			return self._mpi_acc_simulate(steps, n_walkers=n_walkers, n_jobs=n_jobs, **kwargs)

	def _acc_simulate(self, steps, **kwargs):
		return simulators.temp_acc.run(self, steps, **kwargs)

	def _mpi_acc_simulate(self, steps, n_walkers=2, n_jobs=1, **kwargs):
		n_walker_iter = np.empty(n_walkers)
		n_walker_iter.fill(steps)
		p = Pool(n_jobs)
		return p.map(partial(self.simulate, **kwargs), n_walker_iter.astype(int))

	def metasimulate(self, steps, h=0.1, w=0.1, **kwargs):
		pass
