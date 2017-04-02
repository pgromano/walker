import numpy as np

def gaussian(self, **kwargs):
    '''
    Attributes:
    -----------

    mu: (N,K) ndarray
        Gaussian centers, positions for N minima for K dimensions.

    sigma: (N,K) ndarray
        Gaussian width

    theta: (N,l) ndarray
        Array of l rotation angles for all N Gaussian peaks. This value is only value for
        when K=2 dimensions. Must include values to axis to specify which axis
        to rotate.

    axis: (N,l) ndarray
        Axis around which to rotate by angle theta. Arbitrary number of
        rotations are allowed. Axis defaults to None for N=1, 2 (z-axis)
        N=2, and 0 (x-axis) for N>3. For N>=3, order of rotations is exceedingly
        important as rotational operations *do not commute*!

    intensity: (N,) ndarray
        Relative intensity term. This sets the ratio between peak height for all
        peaks.

    References
    ----------
    [1]
    '''

    # Surface energy scale
    self._surface_kT = kwargs.get('kT', 20)

    # Gaussian minima
    self.mu = _check_array_type(kwargs.get('mu', np.zeros((1, 2))))
    self._mu = np.row_stack([self.mu, self.mu.mean(0)])

    # Get number of peaks and dimensions (features)
    self.n_peaks = self.mu.shape[0]
    self.n_features = self.mu.shape[1]

    self.covariance = _check_array_type(kwargs.get('cov', None))
    if self.covariance is None:
        self.covariance = []
        # Setup gaussian widths
        self.sigma = _check_array_type(kwargs.get('sigma', np.ones((self.n_peaks, self.n_features))*0.1))
        self._sigma = np.row_stack([self.sigma, self.sigma.sum(0)*4])

        # Setup gaussian skews
        if self.n_features > 1:
            self.theta = _check_array_type(kwargs.get('theta', np.zeros((self.n_peaks, 1))))
            self._theta = np.row_stack([self.theta, np.zeros(self.theta.shape[1])])
            self.axis = _check_array_type(kwargs.get('axis', 2*np.ones((self.n_peaks, 1)))).astype(int)
            self._axis = np.row_stack([self.axis, 2*np.ones(self.axis.shape[1])]).astype(int)

        for i in range(self.n_peaks):
            # Build eigenvalue matrix from variances
            cov = np.eye(self.n_features)*self._sigma[i]

            # Solve covariance matrix from principal components rotated by theta to orthonormal frame
            if self.n_features > 1:
                cov = _rotate_covariance(cov, self._theta[i], axis=self._axis)
            self.covariance.append(cov)

    # Setup gaussian intensitys
    self.intensity = _check_array_type(kwargs.get('intensity', np.ones((self.n_peaks, 1))))
    self._intensity = np.row_stack([self.intensity, self.intensity.max()*0.25])

def _check_array_type(attr):
    if type(attr) == list:
        attr = np.array(attr)
    return attr

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
