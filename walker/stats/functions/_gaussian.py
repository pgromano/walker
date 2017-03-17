import numpy as np

def distribution(self, X):
    if self.covariance is None:
        N = self.n_peaks+1
        self.covariance = []
        compute_cov = True
    else:
        N = self.n_peaks
        compute_cov = False

    for i in range(N):
        # Define peak amplitude
        A = -self._intensity[i]*self._surface_kT

        # Displacement of all X values to mean center
        diff = X-self._mu[i]

        if compute_cov is True:
            # Build eigenvalue matrix from variances
            cov = np.eye(self.n_features)*self._sigma[i]

            # Solve covariance matrix from principal components rotated by theta to orthonormal frame
            if self.n_features > 1:
                cov = _rotate_covariance(cov, self._theta[i], axis=self._axis)
            self.covariance.append(cov)
        else:
            cov = self.covariance[i]

        # Invert covariance matrix
        icov = np.linalg.pinv(cov)
        if i == 0:
            P = A*np.exp(-np.inner(diff, np.inner(icov, diff.T))/2.0)
        else:
            P += A*np.exp(-np.inner(diff, np.inner(icov, diff.T))/2.0)
    return P

def gradient(self, X):
    if self.covariance is None:
        N = self.n_peaks+1
        self.covariance = []
        compute_cov = True
    else:
        N = self.n_peaks
        compute_cov = False

    for i in range(N):
        # Define peak amplitude
        A = -self._intensity[i]*self._surface_kT

        # Displacement of all X values to mean center
        diff = X-self._mu[i]

        if compute_cov is True:
            # Build eigenvalue matrix from variances
            cov = np.eye(self.n_features)*self._sigma[i]

            # Solve covariance matrix from principal components rotated by theta to orthonormal frame
            if self.n_features > 1:
                cov = _rotate_covariance(cov, self._theta[i], axis=self._axis)
            self.covariance.append(cov)
        else:
            cov = self.covariance[i]

        # Invert covariance matrix
        icov = np.linalg.pinv(cov)
        if i == 0:
            P = A*np.exp(-np.inner(diff, np.inner(icov, diff.T))/2.0)
            dP = -P*(np.inner(icov, diff.T))
        else:
            P = A*np.exp(-np.inner(diff, np.inner(icov, diff.T))/2.0)
            dP += -P*(np.inner(icov, diff.T))
    return dP


def _rotate_covariance(cov, theta, axis=None):
    def _R(theta, n, axis):
        R = np.eye(n)
        R[:2,:2] = np.array([[np.cos(theta), -np.sin(theta)],
                             [np.cos(theta), np.sin(theta)]])
        return np.roll(R, axis, axis=[0,1])

    # Number of dimensions
    ndim = cov.shape[0]

    if type(theta) == float:
        theta = [theta]

    if ndim == 2:
        axis = [2 for i in range(len(theta))]
    elif ndim > 2 and axis is None:
        axis = [0 for i in range(len(theta))]

    for i,t in enumerate(theta):
        # Get rotation operator for n dimensions
        if theta[i] == 0:
            pass
        else:
            R = _R(theta[i], ndim, axis=axis[i])
            cov = R.dot(cov.dot(np.linalg.pinv(R)))
    return cov
