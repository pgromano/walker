import numpy as np
from . import stats

class Walker(object):
	def __init__(self, method="Gaussian", **kwargs):
		if method == 'Gaussian':
			stats.set_attributes.gaussian(self, **kwargs)

	def potential(self, X):
		return stats.functions._gaussian.distribution(self, X)

	def gradient(self, X):
		return stats.functions._gaussian.gradient(self, X)

	def simulate(self, steps, **kwargs):
		x0 = kwargs.get('x0', self.mu[np.random.choice(self.n_peaks)])
		kT = kwargs.get('kT', 1.0)
		gamma = kwargs.get('gamma', 10.0)
		dt = kwargs.get('dt', 0.001)

		# Build simulation array and fill with starting point
		X = np.zeros((steps+1, self.n_features))
		X[0] = x0

		# Generate random forces
		F = np.random.normal(scale=np.sqrt((2.0 * kT * dt)/gamma),
			size=(steps, self.n_features))

		# Simulate!
		for i in range(steps):
			X[i+1] = X[i] - (dt/gamma)*self.gradient(X[i]) + F[i]
		return X
