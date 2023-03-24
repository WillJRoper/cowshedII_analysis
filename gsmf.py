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


def log10phi(m, Mstar, phi_star, alpha):

    phi = phi_star * np.exp(- m / Mstar) * (m / Mstar) ** alpha
    
    return phi

# Set up snapshot list
snap_ints = list(range(0, 22))
snaps = []
for s in snap_ints:
    str_snap_int = "%s" % s
    snaps.append(str_snap_int.zfill(4))

# Define the normalisation and colormap
norm = Normalize(vmin=0, vmax=16)
cmap = plt.cm.plasma

# Set up plot
fig = plt.figure()
ax = fig.add_subplot(111)
ax.grid(True)

# Define mass bins
mass_bins = np.logspace(8, 12, 20)
bin_cents = (mass_bins[1:] + mass_bins[:-1]) / 2
bin_widths = mass_bins[1:] - mass_bins[:-1]


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
        halo_data = load("../EAGLE_50/galaxies/cowshed50_%s.properties.0" % snap)
    except OSError:
        continue

    # Extract masses
    halo_data.masses.mass_star.convert_to_units("msun")
    stellar_mass = halo_data.masses.mass_star
    stellar_mass = stellar_mass[stellar_mass > 0]

    if stellar_mass.size == 0:
        continue

    print(z, boxsize, np.log10(np.min(stellar_mass)), np.log10(np.max(stellar_mass)))

    # Histogram these masses
    H, _ = np.histogram(stellar_mass, bins=mass_bins)

    if np.sum(H) < 10:
        continue

    # Convert histogram to mass function
    gsmf = H / np.product(boxsize) / np.log10(bin_widths)

    # # Fit the data
    okinds = gsmf > 0
    # popt, pcov = curve_fit(log10phi, bin_cents[okinds], gsmf[okinds], p0=[10*4, 10**10, -1])

    # # Plot this line
    # xs = np.linspace(mass_bins.min(), mass_bins.max(), 1000)
    ax.errorbar(np.log10(bin_cents[okinds]), np.log10(gsmf[okinds]),
                yerr=1 / np.sqrt(H[okinds]), linestyle="none",
                marker="o", color=cmap(norm(z)))



fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax)

ax.set_xlabel("$\log_{10}(M_\star / \mathrm{M}_\odot)$")
ax.set_ylabel("$\log_{10}(\phi / [\mathrm{cMpc}^{-3} \mathrm{dex}^{-1}])$")

# ax.legend()

fig.savefig("../plots/gsmf.png", bbox_inches="tight", dpi=100)

plt.close(fig)
