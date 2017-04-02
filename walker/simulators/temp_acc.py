import numpy as np
from .src import _temp_acc

def run(self, steps, **kwargs):
    x0 = kwargs.get('x0', self.mu[np.random.choice(self.n_peaks)])
    T0 = kwargs.get('T0', 1.0)
    Tf = kwargs.get('Tf', self._surface_kT*0.25)
    gamma = kwargs.get('gamma', 10.0)
    dt = kwargs.get('dt', 0.001)

    # Build simulation array and fill with starting point
    X = np.zeros((steps, self.n_features)).astype(float, order='F')
    X[0] = x0

    mu = np.copy(self.mu).astype(float, order='F')
    icov = np.array([np.linalg.pinv(self.covariance[i]) for i in range(self.n_peaks)]).astype(float, order='F')
    A = np.array([-self._intensity[i]*self._surface_kT for i in range(self.n_peaks)]).astype(float, order='F')

    _temp_acc.run(X, mu, icov, A, dt, gamma, T0, Tf, self.n_features, steps, self.n_peaks)
    return X
