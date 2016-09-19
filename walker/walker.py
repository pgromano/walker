import numpy as np
from walker import util

class walker(object):
	def __init__(self, minima, width=None, skew=None, intensity=None, extent=None, kbT=10):
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
		self._surface_kbT = kbT

		# Setup gaussian minima
		if type(minima) == list:
			minima = np.squeeze(minima)
		self._minima = minima
		self._N = len(minima)

		# Setup gaussian widths
		if width is None:
			width = np.ones((len(minima), 2))*0.2
		if type(width) == list:
			width = np.squeeze(width)
		self._width = width

		# Setup gaussian skews
		if skew is None:
			skew = np.zeros(len(minima))
		if type(skew) == list:
			skew = np.squeeze(skew)
		self._skew = skew

		# Setup gaussian intensitys
		if intensity is None:
			intensity = np.ones(len(minima))
		if type(intensity) == list:
			intensity = np.squeeze(intensity)
		self._intensity = intensity
		'''TODO: Setup an automatic intensity calculator to ensure proper
		barrier heights and no particle escape.'''

		# Setup coordinate range
		if extent is None:
			extent = [-1,1,-1,1]
		if type(extent) == list:
			extent = np.squeeze(extent)
		self._extent = extent

		self.attr = util.attributes.attributes(self)

	def potential(self, x, y):
		def shape(sx, sy, skew):
			aa = np.zeros(len(sx))
			bb = np.copy(aa)
			cc = np.copy(aa)
			for n in range(len(sx)):
				aa[n] =  np.cos(skew[n])**2/(2*sx[n]**2)+np.sin(skew[n])**2/(2*sy[n]**2)
				bb[n] = -np.sin(2*skew[n])/(4*sx[n]**2)+np.sin(2*skew[n])/(4*sy[n]**2)
				cc[n] =  np.sin(skew[n])**2/(2*sx[n]**2)+np.cos(skew[n])**2/(2*sy[n]**2)
			return aa,bb,cc

		# Add wide minima to reduce liklihood of particle escape
		XX = np.insert(self.attr.minima[:,0], 0, self.attr.minima[:,0].mean())
		YY = np.insert(self.attr.minima[:,1], 0, self.attr.minima[:,1].mean())
		sx = np.insert(self.attr.width[:,0], 0, self.attr.width[:,0].sum()*3.75)
		sy = np.insert(self.attr.width[:,1], 0, self.attr.width[:,1].sum()*3.75)
		AA = -np.insert(self.attr.intensity, 0, self.attr.intensity.max()*0.25)*self.attr.kbT

		aa, bb, cc = shape(sx, sy, np.insert(self.attr.skew, 0, 0))

		value = 0
		for j in range(0, len(AA)):
			value += AA[j]*np.exp(-(aa[j]*(x-XX[j])**2 -
					 2*bb[j]*(x-XX[j])*(y-YY[j]) +
					 cc[j]*(y-YY[j])**2))
		return value

	def calculate_surface(self, extent=None, bins=100):
		if extent is None:
			x = np.linspace(-1,1,bins)
			y = np.linspace(-1,1,bins)
		else:
			x = np.linspace(extent[0], extent[1], bins)
			y = np.linspace(extent[2], extent[3], bins)
		XX, YY = np.meshgrid(x,y)
		self.surface = self.potential(XX, YY)

	def simulate(self, steps, kbT=1, dt=0.001, mGamma=1000.0, init=None):
		# Setup thermal scale
		if init is None:
			x = 0.0
			y = 0.0
		else:
			x = init[0]
			y = init[1]

		XX = np.insert(self.attr.minima[:,0], 0, self.attr.minima[:,0].mean())
		YY = np.insert(self.attr.minima[:,1], 0, self.attr.minima[:,1].mean())
		sx = np.insert(self.attr.width[:,0], 0, self.attr.width[:,0].sum()*3.75)
		sy = np.insert(self.attr.width[:,1], 0, self.attr.width[:,1].sum()*3.75)
		AA = -np.insert(self.attr.intensity, 0, self.attr.intensity.max()*0.25)*self.attr.kbT
		sk = np.insert(self.attr.skew, 0, 0)

		util.simulate(x, y, steps, dt, mGamma, kbT,
					XX.astype(float, order='F'),
					YY.astype(float, order='F'),
					AA.astype(float, order='F'),
					sx.astype(float, order='F'),
					sy.astype(float, order='F'),
					sk.astype(float, order='F'),
					len(XX))
