# walker
Python API for random walk on Gaussian potential

This code is a modification of the work of https://github.com/rmcgibbo/mullermsm, and instead is designed to generate brownian simulations along any desired potential. 

Simulations of the random particle walker are executed in fortran code using f2py. Compilation is simple can be completed by

  `f2py -f ./walker/util/_simulate.f95 --fcompiler='gfortran' -f90flags='-Wno-tabs' -m ./walker/util/_simulate`
  

  
