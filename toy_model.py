#!/usr/bin/python
import numpy as np
import matplotlib.pyplot as plt
import pinky
import theano
import theano.tensor as tensor
import time
from tqdm import *
import os.path

def set_params(x0,y0,sx,sy,skew=None,height=None,file=None):
	""" __Define parameters for Gaussian formed Potential Surface__

	Parameters
	----------
	x0: Center of gaussian potential along the x-axis. The array is of length N,
		where N is the number of gaussians to be summed in the total potential.
		Length of x0 must be equivalent to all other inputs!

	y0: Center of gaussian potential along the y-axis. The array is of length N,
		where N is the number of gaussians to be summed in the total potential.
		Length of y0 must be equivalent to all other inputs!

	sx: Gaussian width along the x-axis. Length of sx must be equivalent to all
		other inputs!

	sy: Gaussian width along the y-axis. Length of sy must be equivalent to all
		other inputs!

	skew: By default, skew is set to None, which will produce 0 rotation to the
		gaussian shape. Skew is the theta angle of rotation in the standard
		2D elliptical gaussian function. Negative values will rotate counter-
		clockwise, and positive values will rotate clockwise.

	height: By default, height is set to None, which will produce a well size of
		-1 for all gaussians. Height is the prefactor that determines the magnitude
		for each gaussians. Negative values will produce wells, and positive
		values will produce barriers.

	References
	----------
	..[1] https://en.wikipedia.org/wiki/Gaussian_function#Two-dimensional_Gaussian_function
	"""
	parameters = np.zeros((6,len(x0)))
	parameters[0,:] = x0
	parameters[1,:] = y0
	parameters[2,:] = sx
	parameters[3,:] = sy
	if skew is not None:
		parameters[4,:] = skew
	if height is None:
		parameters[5,:] = -np.ones(len(x0))
	else:
		parameters[5,:] = height
	if file is None:
		np.save('.parameters.npy',parameters)
	else:
		np.save(file+'.npy',parameters)


def potential(x,y,file=None):
	"""__Multi-Gaussian Potential__

	Parameters
	----------
	x : {float, np.ndarray}
		X coordinate. Can be either a single number or an array. If you supply
		an array, x and y need to be the same shape.
	y : {float, np.ndarray}
		Y coordinate. Can be either a single number or an array. If you supply
		an array, x and y need to be the same shape.
	Returns
	-------
	potential : {float, np.ndarray}
		Potential energy. Will be the same shape as the inputs, x and y.

	References
	---------
	..[2] Code adapted from https://cims.nyu.edu/~eve2/ztsMueller.m
	..[3] Code re-adapted from https://github.com/rmcgibbo/mullermsm
	"""
	if file is None:
		if not os.path.exists('.parameters.npy'):
			print('ERROR: Must set parameters into file.')
		else:
			prms = np.load('.parameters.npy')
	else:
		prms = np.load(file+'.npy')

	def shape(sx,sy,skew):
		aa = np.zeros(len(sx))
		bb = np.copy(aa)
		cc = np.copy(aa)
		for n in range(len(sx)):
			aa[n] = np.cos(skew[n])**2/(2*sx[n]**2)+np.sin(skew[n])**2/(2*sy[n]**2)
			bb[n] = -np.sin(2*skew[n])/(4*sx[n]**2)+np.sin(2*skew[n])/(4*sy[n]**2)
			cc[n] = np.sin(skew[n])**2/(2*sx[n]**2)+np.cos(skew[n])**2/(2*sy[n]**2)
		return aa,bb,cc

	XX = prms[0,:]
	YY = prms[1,:]
	aa,bb,cc = shape(prms[2,:],prms[3,:],prms[4,:])
	AA = prms[5,:]

	# use symbolic algebra if you supply symbolic quantities
	exp = tensor.exp if isinstance(x, tensor.TensorVariable) else np.exp

	value = 0
	for j in range(0, len(AA)):
		value += AA[j]*exp(-(aa[j]*(x-XX[j])**2-2*bb[j]*(x-XX[j])*(y-YY[j])+cc[j]*(y-YY[j])**2))
	return value

def force(position,file=None):
	"""Compile a theano function to compute the negative grad
	 of the muller potential"""
	sym_x, sym_y = tensor.scalar(), tensor.scalar()
	sym_V = potential(sym_x, sym_y,file=file)
	sym_F =  tensor.grad(-sym_V, [sym_x, sym_y])

	# F takes two arguments, x,y and returns a 2
	# element python list
	F = theano.function([sym_x, sym_y], sym_F)

	return np.array(F(*position))

def FES(V):
	F = -np.ma.log(V)
	return F

def sample(V,time,qrange=None,est_err=False,plot=False):
	if len(V[0,:]) != len(V[:,0]):
		print('ERROR: Potential must be square matrix!')
	else:
		bins = len(V[0,:])

	if qrange is None:
		x_axes=np.linspace(-1,1,bins)
		y_axes=np.linspace(-1,1,bins)
		X,Y=np.meshgrid(x_axes,y_axes)
	else:
		x_axes=np.linspace(qrange[0][0],qrange[0][1],bins)
		y_axes=np.linspace(qrange[1][0],qrange[1][1],bins)
		X,Y=np.meshgrid(x_axes,y_axes)

	samplings = np.empty((time,2))
	for t in tqdm(range(time)):
		samplings[t,:]=pinky.pinky(x_axes,y_axes,np.copy(V))

	if est_err is True:
		z,x,y = np.histogram2d(samplings[:,0],samplings[:,1],bins=bins)
		rmse = np.sqrt(((z.T/np.amax(z)-V)**2).mean(axis=None))

		if plot is True:
			plt.figure(1,figsize=(10,10))
			plt.contour(X,Y,V,15,linewidths=3)
			plt.contourf(X,Y,z.T,15,cmap="CMRmap",alpha=0.5)
			plt.title('Root mean squared error: '+str(rmse)+'%')
			plt.xlim(np.amin(x_axes),np.amax(x_axes))
			plt.ylim(np.amin(y_axes),np.amax(y_axes))
			plt.show()
		else:
			print('Root mean squared error: '+str(rmse)+'%')
	else:
		if plot is True:
			z,x,y = np.histogram2d(samplings[:,0],samplings[:,1],bins=bins)
			plt.figure(1,figsize=(10,10))
			plt.contour(X,Y,V,15,linewidths=3)
			plt.contourf(X,Y,z.T,15,cmap="CMRmap",alpha=0.5)
			plt.xlim(np.amin(x_axes),np.amax(x_axes))
			plt.ylim(np.amin(y_axes),np.amax(y_axes))
			plt.show()
	return samplings

def simulate(steps=1000,dt=0.1,mGamma=1000.0,kT=1.0,p0=None,file=None):
	F_random = np.random.normal(scale=np.sqrt((2.0*kT*dt)/mGamma),size=(steps-1,2))
	position = np.zeros((steps,2))
	if p0 is not None:
		position[0,:] = p0
	for t in tqdm(range(steps-1)):
		time.sleep(0.001)
		position[t+1] = position[t]+np.multiply((dt/mGamma),force(position[t],file=file))+F_random[t]
	return position

def T(centroids,V):
	c = np.copy(centroids)
	bins = len(V[:,0])
	c0 = np.digitize(c[0],np.linspace(-1,1,bins))
	c1 = np.digitize(c[1],np.linspace(-1,1,bins))

	Tmat = np.empty((len(c[0]),len(c[1])))
	for i in range(len(c[0])):
		for j in range(len(c[0])):
			Tmat[i,j] = np.amin([1,np.exp(-(V[c0[j],c1[j]]-V[c0[i],c1[i]]))])
		Tmat[i,:] /= np.sum(Tmat[i,:])
	return Tmat

def load(file):
	if not os.path.exists(file):
		print('ERROR: Simulation file does not exist.')
		trajectory = None
	elif(file.endswith('.npy')):
		trajectory = np.load(file)
	else:
		n_sim = np.shape(file)[0]
		n_coor = np.shape(file)[1]
		trajectory = [pd.concat([pd.read_table(file[j,i],
			index_col=False, header=None, names=[i])
		for i in range(n_coor)],axis=1) for j in range(n_sim)]
	return trajectory
