import numpy as np
#from .src import _evolve

def run(self, steps, **kwargs):
    x0 = kwargs.get('x0', self.mu[np.random.choice(self.n_peaks)])
    T0 = kwargs.get('T0', 1.0)
    Tf = kwargs.get('Tf', self._surface_kT)
    gamma = kwargs.get('gamma', 10.0)
    dt = kwargs.get('dt', 0.001)

    # Build simulation array and fill with starting point
    X = np.zeros((steps+1, self.n_features))
    X[0] = x0

    # Generate random forces
    kT = np.linspace(T0, Tf, steps)
    F = np.squeeze([np.random.normal(scale=np.sqrt((2.0 * kTi * dt)/gamma),
        size=(self.n_features)) for kTi in kT])

    # Simulate!
    for i in range(steps):
        X[i+1] = X[i] - (dt/gamma)*self.gradient(X[i]) + F[i]
    return X
