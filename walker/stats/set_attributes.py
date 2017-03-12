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

    # Setup gaussian widths
    self.sigma = _check_array_type(kwargs.get('sigma', np.ones((self.n_peaks, self.n_features))*0.1))
    self._sigma = np.row_stack([self.sigma, self.sigma.sum(0)*4])

    # Setup gaussian skews
    if self.n_features > 1:
        self.theta = _check_array_type(kwargs.get('theta', np.zeros((self.n_peaks, 1))))
        self._theta = np.row_stack([self.theta, np.zeros(self.theta.shape[1])])
        self.axis = _check_array_type(kwargs.get('axis', 2*np.ones((self.n_peaks, 1)))).astype(int)
        self._axis = np.row_stack([self.axis, 2*np.ones(self.axis.shape[1])]).astype(int)

    # Setup gaussian intensitys
    self.intensity = _check_array_type(kwargs.get('intensity', np.ones((self.n_peaks, 1))))
    self._intensity = np.row_stack([self.intensity, self.intensity.max()*0.25])

def _check_array_type(attr):
    if type(attr) == list:
        attr = np.array(attr)
    return attr
