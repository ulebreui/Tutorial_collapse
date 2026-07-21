import numpy as np

def init_grid_log(nbins,massmax,massmin):
   """
   Generate massgrid and massbins in logscale

   Parameters
   ----------
      nbins : scalar, type -> integer
         number of dust bins
      massmax : scalar, type -> float
         maximium mass value of dust mass range
      massmin : scalar, type -> float
         minimum mass value of dust mass range

   Returns
   ----------
      massgrid : 1D array (dim = nbins+1), type -> float
         grid of masses given borders value of mass bins
      massbins : 1D array (dim = nbins), type -> float
         arithmetic mean value of massgrid for each mass bins

   """
   r = (massmax/massmin)**(1./float(nbins))
   massgrid = np.zeros(nbins+1)
   massbins = np.zeros(nbins)
   massgrid[0] = massmin

   for j in range(nbins):
      massgrid[j+1] = r*massgrid[j]
      massbins[j] = 0.5*(massgrid[j]+massgrid[j+1])

   return massgrid,massbins


def init_grid_log_phy(nbins,smax,smin,rhograin,unit_m):
   """
   Generate massgrid and massbins in logscale from smin and smax

   Parameters
   ----------
      nbins : scalar, type -> integer
         number of dust bins
      smax : scalar, type -> float
         maximium mass value of dust size range
      smin : scalar, type -> float
         minimum mass value of dust size range
      rhograin : scalar, type -> float
         intrincisc dust density
      unit_m : scalar, type -> float
         unit mass

   Returns
   ----------
      massgrid : 1D array (dim = nbins+1), type -> float
         grid of masses given borders value of mass bins
      massbins : 1D array (dim = nbins), type -> float
         arithmetic mean value of massgrid for each mass bins

   """
   
   massmax = 4.*np.pi*rhograin*smax**3/3. / unit_m
   massmin = 4.*np.pi*rhograin*smin**3/3. / unit_m
   r = (massmax/massmin)**(1./float(nbins))
   massgrid = np.zeros(nbins+1)
   massbins = np.zeros(nbins)
   massgrid[0] = massmin

   for j in range(nbins):
      massgrid[j+1] = r*massgrid[j]
      massbins[j] = 0.5*(massgrid[j]+massgrid[j+1])

   return massgrid,massbins