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
from velociraptor.swift.swift import to_swiftsimio_dataset
from velociraptor.particles import load_groups
from swiftgalaxy import SWIFTGalaxy, Velociraptor
from schwimmbad import MultiPool


def log10phi(D, D_star, log10phi_star, alpha):

    y = D - D_star
    phi = np.log(10) * 10 ** log10phi_star * np.exp(-10**y) * 10 ** (y * (alpha + 1))
    
    return np.log10(phi)

# Set up snapshot list
snap_ints = [4, 5, 6, 8]
snaps = []
for s in snap_ints:
    str_snap_int = "%s" % s
    snaps.append(str_snap_int.zfill(4))

# Define the normalisation and colormap
norm = Normalize(vmin=2, vmax=16)
cmap = lover

# Set up plot
fig = plt.figure()
ax = fig.add_subplot(111)
ax.grid(True)
ax.loglog()

# Define mass bins
sfr_bins = np.logspace(-3, 2, 30)
bin_cents = (sfr_bins[1:] + sfr_bins[:-1]) / 2
bin_widths = sfr_bins[1:] - sfr_bins[:-1]

# Define the normalisation and colormap
norm = Normalize(vmin=2, vmax=12)
cmap = lover


# Loop over snapshots
for snap in zip(snaps):

    # Load swiftsimio dataset to get volume and redshift
    sim_data = simload("../EAGLE_50/snapshots/fb1p0/cowshed50_%s.hdf5" % snap)
    z = sim_data.metadata.redshift
    boxsize = sim_data.metadata.boxsize

    # Load halos
    try:
        halo_data = load("../EAGLE_50/galaxies/cowshed50_%s.properties" % snap)
        groups = load_groups(
            "../EAGLE_50/galaxies/cowshed50_%s.catalog_groups" % snap,
            catalogue=halo_data
        )
    except OSError:
        continue

    # Extract sfrs
    halo_data.star_formation_rate.sfr_gas.convert_to_units("Msun/yr")
    ngal = halo_data.star_formation_rate.sfr_gas.size

    # Extract masses
    halo_data.masses.mass_star.convert_to_units("msun")
    stellar_mass = halo_data.masses.mass_star

    # Define the redshift limit for the SFR bin_cents
    z_high = z_at_value(cosmo.age, cosmo.age(z) - 100 * u.Myr,
                        zmin=-1, zmax=127)

    print(z, z_high, cosmo.age(z), cosmo.age(z_high))

    # Define galaxy ids.
    gal_ids = np.array(list(range(ngal)))
    gal_ids = gal_ids[stellar_mass > 10**7.5]

    print("There are %d galaxies above mass threshold (out of %d" %
          (gal_ids.size, ngal))

    def calc_sfr(i):

        sg = SWIFTGalaxy(
            "../EAGLE_50/snapshots/fb1p0/cowshed50_%s.hdf5" % snap,
            Velociraptor(
                "../EAGLE_50/galaxies/cowshed50_%s" % snap,
                halo_index=i
            )
        )

        # Get redshift of stellar birth
        zs = (1 / sg.stars.birth_scale_factors.value) - 1
        ms = sg.stars.masses.value * 10 ** 10

        # Get only particles formed in bin_cents
        okinds = zs > z_high
        return np.sum(ms[okinds]) / (100 * 10 ** 6)
    
    with MultiPool(int(sys.argv[1])) as pool:
        sfrs = list(pool.map(calc_sfr, gal_ids))

    sfr = np.array(sfrs)

    print(z, boxsize, np.log10(np.min(sfr)), np.log10(np.max(sfr)))

    # Histogram these masses
    H, _ = np.histogram(sfr, bins=sfr_bins)

    # Convert histogram to mass function
    sfrf = H / np.product(boxsize.value) / bin_widths

    # # Fit the data
    okinds = H > 0
    # popt, pcov = curve_fit(log10phi, bin_cents[okinds], gsmf[okinds], po=[10, 10, 1])

    # # Plot this line
    # xs = np.linspace(sfr_bins.min(), sfr_bins.max(), 1000)
    ax.plot(bin_cents[okinds], sfrf[okinds], color=cmap(norm(z)))

    # ax.text(0.95, 0.05, f'$z={z:.1f}$',
    #         bbox=dict(boxstyle="round,pad=0.3", fc='w',
    #                   ec="k", lw=1, alpha=0.8),
    #         transform=ax.transAxes, horizontalalignment='right',
    #         fontsize=8)

ax.set_xlabel("$\mathrm{SFR}_{100}/\mathrm{M}_\odot \mathrm{yr}^{-1}$")
ax.set_ylabel("$\phi / [\mathrm{cMpc}^{-3} \mathrm{dex}^{-1}]$")

# ax.legend()

fig.savefig("../plots/sfrf.png", bbox_inches="tight", dpi=100)

plt.close(fig)
