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



def plot_median(x, y, num_bins, ax, redshift):
    
    # Define bin edges based on the input x values and number of bins
    bin_edges = np.linspace(min(x), max(x), num_bins+1)
    bin_cents = (bin_edges[1:] + bin_edges[:-1]) / 2
    
    # Calculate the median of y values in each bin
    bin_medians = []
    for i in range(num_bins):
        bin_y = y[(x >= bin_edges[i]) & (x <= bin_edges[i+1])]
        bin_median = np.median(bin_y)
        bin_medians.append(bin_median)

    ax.plot(bin_cents, bin_medians, color=cmap(norm(redshift)))

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
ax.loglog()
ax.grid(True)

# Define mass bins
mass_bins = np.logspace(8, 12, 30)
bin_cents = (mass_bins[1:] + mass_bins[:-1]) / 2
bin_widths = mass_bins[1:] - mass_bins[:-1]


# Loop over snapshots
for snap in snaps:

    # Load swiftsimio dataset to get volume and redshift
    sim_data = simload("../EAGLE_50/snapshots/fb1p0/cowshed50_%s.hdf5" % snap)
    z = sim_data.metadata.redshift
    boxsize = sim_data.metadata.boxsize

    # Load halos
    try:
        halo_data = load("../EAGLE_50/galaxies/cowshed50_%s.properties.0" % snap)
    except OSError:
        continue

    # Extract masses
    halo_data.masses.mass_star.convert_to_units("msun")
    halo_data.radii.r_halfmass_star.convert_to_units("kpc")
    stellar_mass = halo_data.masses.mass_star
    okinds = stellar_mass > 0
    stellar_mass = stellar_mass[okinds]
    radii = halo_data.radii.r_halfmass_star
    radii = radii[okinds]

    if stellar_mass.size == 0:
        continue

    plot_median(stellar_mass, radii, 30, ax, z)

fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax)

ax.set_xlabel("$M_\star / \mathrm{M}_\odot$")
ax.set_ylabel("$R_\star / [\mathrm{pkpc}]$")

# ax.legend()

fig.savefig("../plots/size_mass.png", bbox_inches="tight", dpi=100)

plt.close(fig)
