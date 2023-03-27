"""
Fit GSMF for all redshifts and all simulations
"""

import sys
import numpy as np
from swiftsimio import load as simload
from velociraptor import load

import scipy

from emcee.autocorr import integrated_time

import fitDF.fitDF as fitDF
import fitDF.models as models
import fitDF.analyse as analyse

from schwimmbad import MultiPool
from functools import partial

def mass_bins():
    massBinLimits = np.linspace(7.95, 13.35, 28)
    massBins = np.logspace(8.05, 13.25, 27)
    return massBins, massBinLimits
# print(np.log10(massBins))
# print(massBinLimits)

def fitdf(N_up, N, V, mstar_temp, cprior, name):

    obs = [{'bin_edges': massBinLimits, 'N': N_up, 'volume': V, 'sigma': N_up / np.sqrt(N)}] 
    # print("upscaled N", N_up,
    #       "\nN", N,
    #       "\nsigma", obs[0]['sigma'])

    model = models.DoubleSchechter()

    priors = {}

    # scale = 2.0
    # priors['log10phi*_1'] = scipy.stats.norm(loc=cprior['phi1'], scale=scale)
    # priors['log10phi*_2'] = scipy.stats.norm(loc=cprior['phi2'], scale=scale)
    priors['log10phi*_1'] = scipy.stats.uniform(loc=-8, scale=6.0)
    priors['log10phi*_2'] = scipy.stats.uniform(loc=-8, scale=6.0)
    
    # scale = 2.0
    # priors['alpha_1'] = scipy.stats.norm(loc=cprior['a1'], scale=scale)
    # priors['alpha_2'] = scipy.stats.norm(loc=cprior['a2'], scale=scale)
    priors['alpha_1'] = scipy.stats.uniform(loc=-4.5, scale=3.0)
    priors['alpha_2'] = scipy.stats.uniform(loc=-1.001, scale=0.002)
    
    priors['D*'] = scipy.stats.uniform(loc = 8., scale = 4.0)
 
    fitter = fitDF.fitter(obs, model=model, priors=priors, output_directory='samples')
    fitter.lnlikelihood = fitter.gaussian_lnlikelihood
    samples = fitter.fit(nsamples=int(1e4), burn=1000, sample_save_ID=name, 
                             use_autocorr=True, verbose=True)

    # from methods import switch_samples
    # _samples = switch_samples(samples)
 
    observations = [{'volume':V, 'sample': np.log10(mstar_temp), 'bin_edges': massBinLimits, 'N': N_up}]
    a = analyse.analyse(ID='samples', model=model, sample_save_ID=name, observations=observations)#, samples=_samples)

    for ip, p in enumerate(a.parameters):
        acorr = integrated_time(a.samples[p], quiet=True)
        print("Autocorrelation time %s:"%p,acorr)

    #fig = a.triangle(hist2d = True, ccolor='0.5')
    #plt.savefig('images/%s_posteriors.png'%name)
 
    #fig = a.LF(observations=True,xlabel='$\mathrm{log_{10}}(M_{*} \,/\, M_{\odot})$')
    #plt.savefig('images/%s_fit_MF.png'%name)
 


def fit(tag, prev_z): 
    
    snap = tag

    # Load swiftsimio dataset to get volume and redshift
    sim_data = simload("../EAGLE_50/snapshots/fb1p0/cowshed50_%s.hdf5" % snap)
    z = sim_data.metadata.redshift
    boxsize = sim_data.metadata.boxsize

    print(snap, z)

    if prev_z != None:
        if prev_z - z < 0.5:
            return

    prev_z = z

    # Load halos
    try:
        halo_data = load("../EAGLE_50/galaxies/cowshed50_%s.properties" % snap)
    except OSError as e:
        print(e)
        return

    # Extract masses
    halo_data.masses.mass_star.convert_to_units("msun")
    stellar_mass = halo_data.masses.mass_star
    mstar_temp = stellar_mass[stellar_mass > 0]

    massBins, massBinLimits = mass_bins()

    if mstar_temp.size == 0:
        return

    V = np.product(boxsize.value)

    hist_all, _ = np.histogram(np.log10(mstar_temp), bins=massBinLimits)
    phi_all = (hist_all / V) / (massBinLimits[1] - massBinLimits[0])

    if np.sum(hist_all) < 50:
        print("Less than 10 counts")
        return
    
    phi_sigma = (np.sqrt(hist_all) / V) / (massBinLimits[1] - massBinLimits[0])
        
    ## ---- Get fit
    sample_ID = 'cowshed50_gsmf_%s' % snap

    # Mask out 0s
    okinds = hist_all > 0
    binlim_okinds = np.zeros(hist_all.size + 1, dtype=bool)
    binlim_okinds[0] = 1
    binlim_okinds[1:][okinds] = 1
    hist_all = hist_all[okinds]
    phi_all = phi_all[okinds]
    massBinLimits = massBinLimits[binlim_okinds]
    massBins = massBins[okinds]
    
    V = 50 ** 3
    N = models.phi_to_N(phi_all, V, massBinLimits)

    print(hist_all)

    fitdf(N, hist_all, V, mstar_temp, cprior=custom_priors[tag], name=sample_ID)
    print(sample_ID, "fit done?")
   
    # ## Ref ##
    # # if len(mstar_ref[rtag]) > 0:
    # Phi, phi_sigma, hist = fl.calc_df(mstar_ref[rtag], rtag, 100**3, massBinLimits)
    # fitdf(hist, 100**3, mstar_ref[rtag] * 1e10, cprior=custom_priors[tag], name='ref_gsmf_%s'%rtag)

    ## AGN ##
    # # if len(mstar_agn[rtag]) > 0:
    # Phi, phi_sigma, hist = fl.calc_df(mstar_agn[rtag], rtag, 50**3, massBinLimits)
    # fitdf(hist, 50**3, mstar_agn[rtag] * 1e10, cprior=custom_priors[tag], name='agn_gsmf_%s'%rtag)
    
    return prev_z



# Set up snapshot list
custom_priors = {}
str_snap_int = sys.argv[1]
snap = str_snap_int.zfill(4)
custom_priors[snap] = {'phi1':float(sys.argv[2]),'phi2':float(sys.argv[3]),
                       'a1': float(sys.argv[4]),'a2': float(sys.argv[5])}

prev_z = fit(snap, prev_z=None)

