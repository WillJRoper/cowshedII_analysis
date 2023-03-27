import sys
import numpy as np
from swiftsimio import load as simload
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from astropy.cosmology import Planck18 as cosmo
from astropy.cosmology import z_at_value
import astropy.units as u
from matplotlib.lines import Line2D
import eagle_IO.eagle_IO as eagle_io
from velociraptor import load
from scipy.optimize import curve_fit
from swiftascmaps import lover


import fitDF.fitDF as fitDF
import fitDF.models as models
import fitDF.analyse as analyse


def mass_bins():
    massBinLimits = np.linspace(7.95, 12.05, 20)
    massBins = 10 ** ((massBinLimits[1:] + massBinLimits[:-1]) / 2)
    return massBins, massBinLimits

def plot_df(ax, phi, phi_sigma, hist, massBins,
            label, color, hist_lim=10, lw=3, alpha=0.7, lines=True):

    kwargs = {}
    kwargs_lo = {}

    if lines:
        kwargs['lw']=lw
        
        kwargs_lo['lw']=lw
        kwargs_lo['linestyle']='dotted'
    else:
        kwargs['ls']=''
        kwargs['marker']='o'
        
        kwargs_lo['ls']=''
        kwargs_lo['marker']='o'
        kwargs_lo['markerfacecolor']='white'
        kwargs_lo['markeredgecolor']=color
        
        
    def yerr(phi,phi_sigma):

        p = phi
        ps = phi_sigma
        
        mask = (ps == p)
        
        err_up = np.abs(np.log10(p) - np.log10(p + ps))
        err_lo = np.abs(np.log10(p) - np.log10(p - ps))
        
        err_lo[mask] = 100
        
        return err_up, err_lo, mask


    err_up, err_lo, mask = yerr(phi,phi_sigma)

    # err_lo = np.log10(phi) - np.log10(phi - phi_sigma[0])
    # err_up = np.log10(phi) - np.log10(phi + phi_sigma[1])
        
    ax.errorbar(np.log10(massBins[phi > 0.]),
                np.log10(phi[phi > 0.]),
                yerr=[err_lo[phi > 0.],
                      err_up[phi > 0.]],
                #uplims=(mask[phi > 0.]),
                label=label, c=color, alpha=alpha, **kwargs)

# Set up snapshot list
custom_priors = {}
snap_ints = list(range(0, 22))
snaps = []
for s in snap_ints:
    str_snap_int = "%s" % s
    snaps.append(str_snap_int.zfill(4))
    custom_priors[snaps[-1]] = {'phi1':-5.0,'phi2':-5.0,'a1':-2.0,'a2':-1.0}

# Define the normalisation and colormap
norm = Normalize(vmin=2, vmax=12)
cmap = lover

# Set up plot
fig = plt.figure()
ax = fig.add_subplot(111)
ax.grid(True)
ax.loglog()

# Define mass bins
massBins, massBinLimits = mass_bins() 

model = models.Schechter()

def yerr(phi,phi_sigma):

    p = phi
    ps = phi_sigma
    
    mask = (ps == p)
        
    err_up = np.abs(np.log10(p) - np.log10(p + ps))
    err_lo = np.abs(np.log10(p) - np.log10(p - ps))
        
    err_lo[mask] = 100
    
    return err_up, err_lo, mask


# Loop over snapshots
prev_z = None
for snap in snaps:

    # Load swiftsimio dataset to get volume and redshift
    sim_data = simload("../EAGLE_50/snapshots/fb1p0/cowshed50_%s.hdf5" % snap)
    z = sim_data.metadata.redshift
    boxsize = sim_data.metadata.boxsize

    if prev_z != None:
        if prev_z - z < 0.5:
            continue

    prev_z = z

    # Load halos
    try:
        halo_data = load("../EAGLE_50/galaxies/cowshed50_%s.properties" % snap)
    except OSError:
        continue

    # Extract masses
    halo_data.masses.mass_star.convert_to_units("msun")
    stellar_mass = halo_data.masses.mass_star
    mstar_temp = stellar_mass[stellar_mass > 0]

    if mstar_temp.size == 0:
        continue

    V = np.product(boxsize.value)

    hist_all, _ = np.histogram(np.log10(mstar_temp), bins=massBinLimits)
    hist = np.float64(hist_all)
    phi_all = (hist / V) / (massBinLimits[1] - massBinLimits[0])

    if np.sum(hist_all) < 10:
        print("Less than 10 counts")
        continue

    print("Plotting:", z)

    okinds = phi_all > 0
    ax.plot(massBins[okinds], phi_all[okinds], color=cmap(norm(z)))

cbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax)
cbar.set_label("$z$")

ax.set_xlabel("$M_\star / \mathrm{M}_\odot$")
ax.set_ylabel("$\phi / [\mathrm{cMpc}^{-3} \mathrm{dex}^{-1}]$")

# ax.legend()

fig.savefig("../plots/gsmf.png", bbox_inches="tight", dpi=100)

plt.close(fig)
