import numpy as np
from scipy.special import legendre
import sys
import time
import numba_progress

from progressbar import  Bar,Percentage,ProgressBar

from .generate_tabflux_tabintflux import *
from .generate_flux_intflux import *
from .compute_coag import *
from .L2_proj import *
from .limiter import *
from .utils_polynomials import *


def precomputing_coala_coag_k0(verbose,kernel,K0,nbins,kpol,Q,massgrid):

   """
   Function to precompute tensor_coag_tabflux needed for coagulation solver

   Function for ballistic kernel with differential velocities dv

   DG scheme k=0, piecewise constant approximation

   Parameters
   ----------
   verbose : bolean
      to display details on calculations
   kernel : scalar, type -> integer 
      select the collisional kernel function
   K0 : scalar, type -> float
      constant value of the kernel function (used to adapt to code unit)
   nbins : scalar, type -> integer
      number of dust bins
   kpol : scalar, type -> integer
      degree of polynomials for approximation
   Q : scalar, type -> integer
      number of points for Gauss-Legendre quadrature
   massgrid : 1D array (dim = nbins+1), type -> float
      grid of masses given borders value of mass bins

      
   Returns
   -------
   tensor_tabflux_coag : 3D array (dim = (nbins,nbins,nbins)) , type -> float
      precomputed array for coagulation solver k=0

   """


   vecnodes,vecweights = np.polynomial.legendre.leggauss(Q)


   # Legendre polynomial coefficients
   mat_coeffs_leg = np.zeros((kpol+1,kpol+1))
   mat_coeffs_leg = legendre_coeffs(kpol)

   # print("mat_coeffs_leg=",mat_coeffs_leg)
   
   start = time.time()
   if (kernel == 3):

      tensor_tabflux_coag = np.zeros((nbins,nbins,nbins))

      #numba version
      with numba_progress.ProgressBar(total=nbins, desc="Precomputing coagtabflux k=%d with numba"%(kpol)) as progress:
         compute_coagtabflux_k0_numba(kernel,K0,Q,vecnodes,vecweights,nbins,massgrid,mat_coeffs_leg,tensor_tabflux_coag,progress)

   else:

      print("Need to choose a kernel = 3 for ballistic kernel with dv array.")
      sys.exit()


   finish = time.time()
   if (verbose):
      print("Tensor tabflux generated in %.5f s"%(finish-start))

   return tensor_tabflux_coag


def coala_coag_k0(verbose,kernel,K0,nbins,kpol,dthydro,coeff_CFL,Q,eps,massgrid,tensor_tabflux_coag,dv,gij_init):
   """
   Function to compute coagulation solver in hydro code for 1 hydro time-step 

   Function for ballistic kernel with differential velocities dv

   DG scheme k=0, piecewise constant approximation

   Parameters
   ----------
   kernel : scalar, type -> integer 
      select the collisional kernel function
   K0 : scalar, type -> float
      constant value of the kernel function (used to adapt to code unit)
   nbins : scalar, type -> integer
      number of dust bins
   kpol : scalar, type -> integer
      degree of polynomials for approximation
   dthydro : scalar, type -> float
      hydro timestep, used as timestep to reach for coagulation process
   coeff_CFL : scalar, type -> float
      timestep coefficient for stability of the SSPRK order 3 scheme
   Q : scalar, type -> integer
      number of points for Gauss-Legendre quadrature
   eps : scalar, type -> float
      minimum value for mass distribution approximation gij
   massgrid : 1D array (dim = nbins+1), type -> float
      grid of masses given borders value of mass bins
   tensor_tabflux_coag : 3D array (dim = (nbins,nbins,nbins)) , type -> float
      precomputed array for coagulation solver k=0
   dv : 2D array (dim = (nbins,nbins)), type -> float
      array of the differential velocity between grains
   gij_init : 1D array (dim = nbins), type -> float
      initial components of g on the polynomial basis
   

   Returns
   -------
   gij : 1D array (dim = nbins) or 2D array (dim = (nbins.kpol+1)), type -> float
      evolved components of g on the polynomial basis

   """

   if (kernel != 3):
      print("Need to choose a kernel = 3 for ballistic kernel with dv array.")
      sys.exit()
   
   #read initial condition
   gij = np.copy(gij_init)
   
   if (verbose):
      #total mass density
      M1_t0 = np.sum((massgrid[1:]-massgrid[0:nbins])*gij)
      print("gij t0 =",gij)
      print("M1 t0 = ",M1_t0)


   tot_nsub = 0
   tot_ndt = 0


   if (verbose):
      print("Time solver in progress")
   #time solver DG
   start = time.time()

   gij,nsub,ndt = compute_coag_k0_kdv(eps,coeff_CFL,nbins,massgrid,gij,tensor_tabflux_coag,dv,dthydro)
   
   tot_nsub += nsub
   tot_ndt += ndt

   finish = time.time()

   
   if (verbose):
      print("")
      print("total nsub =",tot_nsub)
      print("total ndt =",tot_ndt)
      print("total number time-steps =",tot_ndt+tot_nsub)

      print("")
      print("gij tend =",gij)
   

      M1_tend = np.sum((massgrid[1:]-massgrid[0:nbins])*gij)

      print("total dust mass density t0 = ",M1_t0)
      print("total dust mass density tend = ",M1_tend)
      # print("diff M1 = ",M1_tend-M1_t0)
      print("abs err M1 = ",abs(M1_tend-M1_t0)/M1_t0)


      print("Time solver in %.5f s"%(finish-start))

   return gij











