import numpy as np
from scipy.interpolate import interp1d
from sklearn.linear_model import LinearRegression

def run(self, a, b, **kwargs):
    if self.n_peaks < 2:
        raise AttributeError('''
        There must be at least two minima along surface.''')

    tol = kwargs.get('tol', 1e-6)
    tol_check = 1e6
    max_steps = int(kwargs.get('max_steps', 1e3))
    n_sites = kwargs.get('n_sites', 25)
    dt = kwargs.get('dt', 1e-2)

    # Create linespace
    axis = np.linspace(0,1,n_sites)
    line = np.row_stack([a, b])
    X = line[:,:-1]
    y = line[:,-1]
    lr = LinearRegression().fit(X,y)

    X = np.column_stack([np.linspace(X[:,i].min(), X[:,i].max(), n_sites) for i in range(X.shape[1])])
    y = np.linspace(y.min(), y.max(), n_sites)

    dX = X-np.roll(X, -1, axis=1)
    dX[0,:] = 0
    dy = y - np.roll(y, -1)
    dy[0] = 0

    lxy = np.cumsum(np.sqrt(dX.sum(1)**2 + dy**2))
    lxy /= lxy[-1]

    interp_X = [interp1d(lxy, X[:,i]) for i in range(X.shape[1])]
    interp_y = interp1d(lxy, y)

    X = np.column_stack([interp_X[i](axis) for i in range(X.shape[1])])
    y = interp_y(axis)

    step = 0
    for step in range(max_steps):
        X0 = np.copy(X)
        y0 = np.copy(y)
        dV = np.array([self.gradient(ri) for ri in np.column_stack([X, y[:,None]])])

        # 1. Evolve
        X = X - dt*dV[:,:-1]
        y = y - dt*dV[:,-1]

        # 2. Reparametrize
        dX = X-np.roll(X, -1, axis=1)
        dX[0,:] = 0
        dy = y - np.roll(y, -1)
        dy[0] = 0

        lxy = np.cumsum(np.sqrt(dX.sum(1)**2 + dy**2))
        lxy /= lxy[-1]

        interp_X = [interp1d(lxy, X[:,i]) for i in range(X.shape[1])]
        interp_y = interp1d(lxy, y)

        X = np.column_stack([interp_X[i](axis) for i in range(X.shape[1])])
        y = interp_y(axis)

        tol_check = (np.linalg.norm(X-X0, axis=0).sum()+ np.linalg.norm(y-y0))/n_sites
        if tol_check <= tol:
            print('Converged!')
            print(str(tol_check)+' < '+str(tol))
            break

    return np.column_stack([X,y])
