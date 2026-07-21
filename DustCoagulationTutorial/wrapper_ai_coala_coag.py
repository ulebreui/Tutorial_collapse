import sys
import os
# print(sys.path)
# Remove current dir from path (the bad one)
current_bad_dir = os.getcwd()
# print("current_bad_dir =",current_bad_dir)
if current_bad_dir in sys.path:
    sys.path.remove(current_bad_dir)

# print(sys.path)
from coala_py.src import *
import numpy as np

from matplotlib import pyplot as plt


def init_power_law_rho_dust(eps_rho_dust,nbins,massgrid,massbins,bin_scut,coeff_pl,dtg,rho_gas):
    """
    Function to compute the initial rho_dust for power law distribution with physical quantity

    DG scheme k=0, piecewise constant approximation

    Parameters
    ----------
    eps_rho_dust : scalar, type -> float
        minimum value for rho_dust mass distribution
    nbins : scalar, type -> integer
        number of dust bins
    massgrid : 1D array (dim = nbins+1), type -> float
        grid of masses given borders value of mass bins
    massbins : 1D array (dim = nbins), type -> float
        arithmetic mean value of massgrid for each mass bins
    bins_cut : scalar, type -> integer
        bin including the cut in size of the power law
    coeff_pl : scalar, type -> float
        coefficient for the power law distribution
    dtg : scalar, type -> float
        mass dust to gas ratio
    rho_gas : scalar, type -> float
        gas density



    Returns
    -------
    rho_dust : 1D array (dim = nbins), type -> float
        initial rho_dust distribution with mass

    """

    Q=5
    vecnodes,vecweights = np.polynomial.legendre.leggauss(Q)

    #Cin value from Paruta et al., 2016 paper
    mcut = massgrid[bin_scut+1]
    Cin = dtg*rho_gas * (4.+coeff_pl)/3. /(mcut**((4. + coeff_pl)/3.) - massgrid[0]**((4. + coeff_pl)/3.))
    rho_dust = np.zeros(nbins)
    for j in range(nbins):
        if (j <= bin_scut):
            hj = massgrid[j+1]-massgrid[j]
            xj = massbins[j]

            term_sum = 0.
            for alpha in range(Q):
                xjalpha = xj + 0.5*hj*vecnodes[alpha]

                term_sum += vecweights[alpha]*xjalpha**((1.+coeff_pl)/3.)


            rho_dust[j] = np.fmax(Cin*term_sum*hj/2.,eps_rho_dust)

            # rho_dust[j] = np.fmax(Cin*xj**((1.-coeff_pl)/3.)*hj,eps)

        else:
            rho_dust[j] = eps_rho_dust

    return rho_dust

def compute_dv_ormel(rho_gas,Temp,rhograin,grainsize,tff,alpha_turb):
    """
    Function to compute 2D array for grain-grain differential velocity from Ormel's model

    Parameters
    ----------
    rho_gas : scalar, type -> float
        gas mass density
    Temp : scalar, type -> float
        gas temperature
    rhograin : scalar, type -> float
        intrinsic grain density
    grainsize : 1D array (dim = nbins), type -> float
        grainsizes from geometric mean of size grid
    tff : scalar, type -> float
        free fall time

    Returns
    -------
    dv : 2D array (dim = (nbins,nbins)), type -> float
        grain-grain differential velocity from Ormel's model in cgs

    """

    #physical constant in cgs
    kB      = 1.38e-16
    u       = 1660538921e-33 #hydrogen atomic mass
    mu_gas  = 2.3            #mean molecular weight
    mh      = 100749e-5*u    #mass hydrogen


    w_th = np.sqrt(8.*kB*Temp/(np.pi*mu_gas*mh))
    ts = rhograin*grainsize/(rho_gas*w_th)
    St = ts/tff
    cs = np.sqrt(kB*Temp/(mu_gas*mh))

    nbins = len(grainsize)
    dv_ormel = np.zeros((nbins,nbins))
    for i in range(nbins):
        for j in range(nbins):
            dv_ormel[i,j] = np.sqrt(alpha_turb)*cs*np.fmax(np.sqrt(St[i]),np.sqrt(St[j]))


    return dv_ormel

def wrapper_coala_coag_k0(verbose,eps_rho_dust,rho_gas,rhograin,massgrid,massbins,list_dt,dv,rho_dust_in):

    """
    Function to compute the evolved rho_dust from grain-grain differential velocity dv

    DG scheme k=0, piecewise constant approximation

    Parameters
    ----------
    eps_rho_dust : scalar, type -> float
        minimum value for rho_dust mass distribution
    rho_gas : scalar, type -> float
        gas mass density
    rhograin : scalar, type -> float
        intrinsic grain density
    massgrid : 1D array (dim = nbins+1), type -> float
        grid of masses given borders value of mass bins
    massbins : 1D array (dim = nbins), type -> float
        arithmetic mean value of massgrid for each mass bins
    list_dt : 1D array, type -> float
        list of time steps in cgs
    dv : 2D array (dim = (nbins,nbins)), type -> float
        grain-grain differential velocity in cgs
    rho_dust_in : 1D array (dim = nbins), type -> float
        initial rho_dust distribution with mass



    Returns
    -------
    rho_dust_out : 1D array (dim = nbins), type -> float
        evolved rho_dust distribution with mass

    """

    nbins = len(rho_dust_in)

    # print("nbins=",nbins)
    # print("shape dv =",np.shape(dv))
    #coala parameters
    kernel    = 3
    kpol      = 0
    Q         = 5
    eps       = 1e-30
    coeff_CFL = 0.3


    #precomputing part
    # to adapt collision kernel function in cgs
    K0 = np.pi*(4./3.*np.pi*rhograin)**(-2./3.)
    tensor_tabflux_coag = precomputing_coala_coag_k0(verbose,kernel,K0,nbins,kpol,Q,massgrid)


    #init distribution
    eps_gij = 1e-50
    gij_init = np.zeros(nbins)
    for j in range(nbins):
        if (rho_dust_in[j] > eps_rho_dust):
            gij_init[j] = rho_dust_in[j]/(massgrid[j+1]-massgrid[j])
        else:
            gij_init[j] = eps_gij



    #Run coala to solve coagulation equation for a list of dt

    #check if list_dt is sorted in ascending order
    i_sorted = all(list_dt[i] <= list_dt[i + 1] for i in range(len(list_dt) - 1))


    # define incremental dt in order to perform a continuous simulations in time and keep onle data at list_dt values
    arr_rho_dust_out = np.zeros((len(list_dt),nbins))

    if (i_sorted):
        for i in range(len(list_dt)):

            if (i==0):
                dt_increment = list_dt[i]
            else:
                dt_increment = list_dt[i]-list_dt[i-1]

            gij_new = coala_coag_k0(verbose,kernel,K0,nbins,kpol,dt_increment,coeff_CFL,Q,eps_gij,massgrid,tensor_tabflux_coag,dv,gij_init)

            #evolved distribution
            rho_dust_out = (massgrid[1:]-massgrid[0:nbins])*gij_new
            rho_dust_out[rho_dust_out < eps_rho_dust] = eps_rho_dust
            arr_rho_dust_out[i,:] = rho_dust_out

            #gij <- gij_init
            gij_init = gij_new

    else:
        for i in range(len(list_dt)):

            gij_new = coala_coag_k0(verbose,kernel,K0,nbins,kpol,list_dt[i],coeff_CFL,Q,eps_gij,massgrid,tensor_tabflux_coag,dv,gij_init)

            #evolved distribution
            rho_dust_out = (massgrid[1:]-massgrid[0:nbins])*gij_new
            rho_dust_out[rho_dust_out < eps_rho_dust] = eps_rho_dust
            arr_rho_dust_out[i,:] = rho_dust_out


    return arr_rho_dust_out


if __name__ == '__main__':
    ##################
    # example for using the wrap function
    ##################

    nbins = 50


    rho_gas = 1e-11  # gas mass density in cgs
    Temp = 10        #K
    Grav = 6.67e-8   #cgs
    tff = np.sqrt(3.*np.pi/(32.*Grav*rho_gas)) #free fall time in cgs
    dtg = 1e-2       # mass dust to gas ratio

    #MRN initial condition
    rhograin = 2.3  #cgs
    smax = 1e-1     #cgs
    smin = 5e-7     #cgs
    scut = 250e-7   #cgs
    eps_rho_dust = 1e-30

    alpha_turb = 1.5

    coeff_pl = -3.5 #MRN distribution

    size_grid = np.logspace(np.log10(smin),np.log10(smax),nbins+1)
    # print("size_grid =",size_grid)

    for j in range(nbins):
        if (size_grid[j] < scut <= size_grid[j+1]):
            bin_scut = j

    massgrid,massbins = init_grid_log_phy(nbins,smax,smin,rhograin,1.)
    # print("massgrid=",massgrid)


    rho_dust_in = init_power_law_rho_dust(eps_rho_dust,nbins,massgrid,massbins,bin_scut,coeff_pl,dtg,rho_gas)

    # print("rho_dust_in =",rho_dust_in)
    # print("M1 in= ",np.sum(rho_dust_in))
    # print("check dtg =",np.sum(rho_dust_in)/rho_gas)


    #compute dv ormel
    grainsize = np.sqrt(size_grid[1:]*size_grid[0:nbins])
    dv_ormel = compute_dv_ormel(rho_gas,Temp,rhograin,grainsize,tff,alpha_turb)

    #time solver
    #sorted list_dt
    # list_dt = np.array(np.linspace(0.,2,101))[1:]*tff

    #random list_dt then sorted
    list_dt = np.zeros(100)
    for i in range(100):
        list_dt[i] = np.random.uniform(0,2)*tff

    # print("list_dt random =",list_dt)
    # list_dt = np.sort(list_dt)
    # print("sorted random list =",np.sort(list_dt))
    # sys.exit()


    verbose_coala = False
    arr_rho_dust_out = wrapper_coala_coag_k0(verbose_coala,eps_rho_dust,rho_gas,rhograin,massgrid,massbins,list_dt,dv_ormel,rho_dust_in)


    # print("arr_rho_dust_out =",arr_rho_dust_out)
    # print("M1 out = ",np.sum(rho_dust_out))
    # print("check dtg =",np.sum(rho_dust_out)/rho_gas)



    #quick plot to check
    plt.figure(1)

    cm_to_mu = 1e4

    plt.loglog(grainsize*cm_to_mu,rho_dust_in,c='black',alpha=0.5)
    for i in range(len(list_dt)):
        plt.loglog(grainsize*cm_to_mu,arr_rho_dust_out[i,:],label="t=%.1f tff"%(list_dt[i]/tff))
    plt.ylabel(r'dust mass density')




    #mass frac plots
    # mass_frac_t0 = rho_dust_in/np.sum(rho_dust_in)
    # plt.loglog(grainsize*cm_to_mu,mass_frac_t0,c='black',alpha=0.5)

    # mass_frac_tend = rho_dust_out/np.sum(rho_dust_out)
    # plt.loglog(grainsize*cm_to_mu,mass_frac_tend,c='black')
    # plt.ylabel(r'mass fraction')

    # plt.ylim(1.e-4, 1.)
    plt.xlim(smin*cm_to_mu,smax*cm_to_mu)

    plt.title("dv Ormel, nbins=%d"%(nbins))
    plt.xlabel(r'size [µm] ')

    plt.legend(loc='lower left',ncol=1)
    plt.tight_layout()
    plt.show()

