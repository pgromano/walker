import numpy as np
import theano
from walker.util import attributes

class walker(object):
    def __init__(self, minima, width=None, skew=None, depth=None, T=298):
        '''
        minima: float
            The x,y positions for all minima that define the potential. The
            length of `minima` defines the number of gaussians to be summed to
            create the full potential energy surface.

            minima =   [[x0, y0],
                        [x1, y1],
                        [x2, y2],
                        ...
                        [xN, yN]]

        width: float (default: 1)
            Gaussian width along the x,y axes. Length must be equivalent to the
            number of total minima!

            width =    [[width_x0, width_y0],
                        [width_x1, width_y1],
                        [width_x2, width_y2],
                        ...
                        [width_xN, width_yN]]

        References
        ----------
        [1] https://en.wikipedia.org/wiki/Gaussian_function#Two-dimensional_Gaussian_function
        [2] Code adapted from https://cims.nyu.edu/~eve2/ztsMueller.m
    	[3] Code re-adapted from https://github.com/rmcgibbo/mullermsm
        '''
        # Setup thermal scale
        self._T = T
        self._kbT = 1.38064852E-23*T

        # Setup gaussian minima
        if type(minima) == list:
            minima = np.squeeze(minima)
        self._minima = minima
        self._N = len(minima)

        # Setup gaussian widths
        if width is None:
            width = np.ones((len(minima), 2))
        self._width = width

        # Setup gaussian skews
        if skew is None:
            skew = np.zeros(len(minima))
        self._skew = width

        # Setup gaussian depths
        if depth is None:
            depth = np.ones(len(minima))
        self._depth = depth
        '''TODO: Setup an automatic depth calculator to ensure proper
        barrier heights and no particle escape.'''

        self.attr = attributes.attributes(self)

    def potential(self,position):
        def shape(sx,sy,skew):
    		aa = np.zeros(len(sx))
    		bb = np.copy(aa)
    		cc = np.copy(aa)
    		for n in range(len(sx)):
    			aa[n] = np.cos(skew[n])**2/(2*sx[n]**2)+np.sin(skew[n])**2/(2*sy[n]**2)
    			bb[n] = -np.sin(2*skew[n])/(4*sx[n]**2)+np.sin(2*skew[n])/(4*sy[n]**2)
    			cc[n] = np.sin(skew[n])**2/(2*sx[n]**2)+np.cos(skew[n])**2/(2*sy[n]**2)
    		return aa,bb,cc

        # Add wide minima to reduce liklihood of particle escape
        XX = np.insert(self.attr.minima[:,0], 0, self.attr.minima[:,0].median())
    	YY = np.insert(self.attr.minima[:,1], 0, self.attr.minima[:,1].median())
        sx = np.insert(self.attr.width[:,0], 0, self.attr.width[:,0].max()*10)
        sy = np.insert(self.attr.width[:,1], 0, self.attr.width[:,1].max()*10)

    	aa, bb, cc = shape(sx, sy, np.insert(self.attr.skew, 0, 0))
    	AA = -np.insert(self.attr.depth, 0, 1)*self.attr.T

    	# use symbolic algebra if you supply symbolic quantities
    	exp = theano.tensor.exp if isinstance(x, theano.tensor.TensorVariable) else np.exp

    	value = 0
    	for j in range(0, len(AA)):
    		value += AA[j]*exp(-(aa[j]*(x-XX[j])**2 -
                     2*bb[j]*(x-XX[j])*(y-YY[j]) +
                     cc[j]*(y-YY[j])**2))
    	return value

    def simulate(self, steps, dt=0.1, mGamma=1000.0, init=None):
        F_random = np.random.normal(scale=nq.sqrt((2.0*self.attr.kbT*dt)/mGamma),
                    size=(steps-1,2))
        position = np.zeros((steps, 2))
        if init is not None:
            position[0,:] = init

        for t in range(steps-1):
            position[t+1,:] = position[t,:] +
                            np.multiply((dt/mGamma), _Force(position[t,:])) +
                            F_random[t,:]
        self.trajectory = position

    def _Force(position):
        """Compile a theano function to compute the negative grad
    	 of the muller potential"""
    	sym_x, sym_y = tensor.scalar(), tensor.scalar()
    	sym_V = potential(sym_x, sym_y,file=file)
    	sym_F =  tensor.grad(-sym_V, [sym_x, sym_y])

    	# F takes two arguments, x,y and returns a 2 element python list
    	F = theano.function([sym_x, sym_y], sym_F)
    	return np.array(F(*position))
