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
from velociraptor.tools import create_mass_function
from unyt import Msun, Mpc


import fitDF.fitDF as fitDF
import fitDF.models as models
import fitDF.analyse as analyse


def mass_bins():
    massBinLimits = np.logspace(7.95, 12.05, 25)
    massBins = (massBinLimits[1:] + massBinLimits[:-1]) / 2
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

    
# Define EAGLE snapshots
pre_snaps = ['000_z020p000', '002_z009p993', '003_z008p988', '006_z005p971',
             '009_z004p485',
             '012_z003p017', '015_z002p012', '018_z001p259', '021_z000p736',
             '024_z000p366', '027_z000p101', '001_z015p132', '004_z008p075',
             '007_z005p487', '010_z003p984', '013_z002p478', '016_z001p737',
             '019_z001p004', '022_z000p615', '025_z000p271', '028_z000p000',
             '002_z009p993', '005_z007p050', '008_z005p037', '011_z003p528',
             '014_z002p237', '017_z001p487', '020_z000p865', '023_z000p503',
             '026_z000p183']

# Sort EAGLE snapshots
snaps = np.zeros(len(pre_snaps), dtype=object)
for s in pre_snaps:
    ind = int(s.split('_')[0])
    snaps[ind] = s

snaps = snaps[snaps != 0]

# Sort EAGLE snapshots
eagle_snaps = []
prev_z = 100
for s in snaps:
    z = float(s.split('_')[1][1:].replace("p", "."))
    if z < 2:
        continue
    if z > 12:
        continue
    if prev_z - z < 1.0:
        continue
    prev_z = z
    eagle_snaps.append(s)

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

# Get eagle data
ref_path = "/cosma7/data/Eagle/ScienceRuns/Planck1/L0100N1504/PE/REFERENCE/data"

# Set up plot
fig = plt.figure()
ax = fig.add_subplot(111)
ax.grid(True)
ax.loglog()

# Define mass bins
massBins, massBinLimits = mass_bins()
bin_width = massBinLimits[1:] - massBinLimits[:-1]

def yerr(phi,phi_sigma):

    p = phi
    ps = phi_sigma
    
    mask = (ps == p)
        
    err_up = np.abs(np.log10(p) - np.log10(p + ps))
    err_lo = np.abs(np.log10(p) - np.log10(p - ps))
        
    err_lo[mask] = 100
    
    return err_up, err_lo, mask

for snap in eagle_snaps:

    z = float(snap.split('_')[1][1:].replace("p", "."))
    boxsize = np.array([100, 100, 100])
    mass = eagle_io.read_array("SUBFIND", ref_path,
                               snap,
                               "Subhalo/ApertureMeasurements/Mass/030kpc",
                               noH=True, physicalUnits=True,
                               numThreads=8)[:, 4] * 10 ** 10

    mstar_temp = mass[mass > 0]

    if mstar_temp.size == 0:
        continue

    V = np.product(boxsize) * Mpc**3

    maxxBins, phi_all, _ = create_mass_function(mstar_temp * Msun, 10**7.95,
                                                10**12.05, box_volume=V,
                                                n_bins=25)

    if np.sum(hist_all) < 10:
        print("Less than 10 counts")
        continue

    print("Plotting:", snap, z)

    ax.plot(massBins, phi_all, color=cmap(norm(z)),
            linestyle="dotted")

# Loop over snapshots
prev_z = None
for snap in snaps:

    # Load swiftsimio dataset to get volume and redshift
    sim_data = simload("../EAGLE_50/snapshots/fb1p0/cowshed50_%s.hdf5" % snap)
    z = sim_data.metadata.redshift
    boxsize = np.array([50, 50, 50])

    if prev_z != None:
        if prev_z - z < 1.0:
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

    V = np.product(boxsize) * Mpc**3

    maxxBins, phi_all, _ = create_mass_function(mstar_temp * Msun, 10**7.95,
                                                10**12.05, box_volume=V,
                                                n_bins=25)

    if np.sum(hist_all) < 10:
        print("Less than 10 counts")
        continue

    print("Plotting:", snap, z)

    okinds = phi_all > 0
    ax.plot(massBins[okinds], phi_all[okinds], color=cmap(norm(z)))

legend_elements1 = [Line2D([0], [0], color='k',
                           label="COWSHED 50 Mpc",
                           linestyle="-"),
                    Line2D([0], [0], color='k',
                           label="EAGLE 100 Mpc",
                           linestyle="dotted"),
                    ]

cbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax)
cbar.set_label("$z$")

ax.set_xlabel("$M_\star / \mathrm{M}_\odot$")
ax.set_ylabel("$\phi / [\mathrm{cMpc}^{-3} \mathrm{dex}^{-1}]$")

ax.legend(handles=legend_elements1, loc="upper right")

fig.savefig("../plots/gsmf.pdf", bbox_inches="tight", dpi=100)

plt.close(fig)
