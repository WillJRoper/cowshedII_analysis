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


def log10phi(D, D_star, log10phi_star, alpha):

    y = D - D_star
    phi = np.log(10) * 10 ** log10phi_star * np.exp(-10**y) * 10 ** (y * (alpha + 1))
    
    return np.log10(phi)

# Set up snapshot list
snap_ints = [4, 8, 15, 18]
snaps = []
for s in snap_ints:
    str_snap_int = "%s" % s
    snaps.append(str_snap_int.zfill(4))

# Define the normalisation and colormap
norm = Normalize(vmin=2, vmax=16)
cmap = lover

# Set up plot
fig = plt.figure()
ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222)
ax3 = fig.add_subplot(223)
ax4 = fig.add_subplot(224)
for ax in [ax1, ax2, ax3, ax4]:
    ax.grid(True)
    ax.loglog()

# Define mass bins
sfr_bins = np.logspace(-3, 7, 20)
bin_cents = (sfr_bins[1:] + sfr_bins[:-1]) / 2
bin_widths = sfr_bins[1:] - sfr_bins[:-1]


# Loop over snapshots
for ax, snap in zip([ax1, ax2, ax3, ax4], snaps):

    # Load swiftsimio dataset to get volume and redshift
    sim_data = simload("../EAGLE_50/snapshots/fb1p0/cowshed50_%s.hdf5" % snap)
    z = sim_data.metadata.redshift
    boxsize = sim_data.metadata.boxsize

    # Load halos
    try:
        halo_data = load("../EAGLE_50/galaxies/cowshed50_%s.properties.0" % snap)
        groups = load_groups(
            "../EAGLE_50/galaxies/cowshed50_%s.catalog_groups.0" % snap,
            catalogue=halo_data
        )
    except OSError:
        continue

    # Extract sfrs
    halo_data.star_formation_rate.sfr_gas.convert_to_units("Msun/yr")
    ngal = halo_data.star_formation_rate.sfr_gas.size

    # Define the redshift limit for the SFR bin_cents
    z_high = z_at_value(cosmo.age, cosmo.age(z) - 100 * u.Myr,
                        zmin=-1, zmax=127)

    print(z, z_high, cosmo.age(z), cosmo.age(z_high))

    sfr = np.zeros(ngal)
    for i in range(ngal):

        print(i, "of", ngal, end="\r")

        particles, unbound_particles = groups.extract_halo(halo_index=i)

        data, mask = to_swiftsimio_dataset(
            particles,
            "../EAGLE_50/snapshots/fb1p0/cowshed50_%s.hdf5" % snap,
            generate_extra_mask=True
        )

        # Get redshift of stellar birth
        zs = (1 / data.stars.birth_scale_factors[mask.stars].value) - 1
        ms = data.stars.masses[mask.stars].value * 10 ** 10

        # Get only particles formed in bin_cents
        okinds = zs > z_high
        sfr[i] = np.sum(ms[okinds]) / (100 * 10 ** 6)

    np.array(sfr)

    print(z, boxsize, np.log10(np.min(sfr)), np.log10(np.max(sfr)))

    # Histogram these masses
    H, _ = np.histogram(sfr, bins=sfr_bins)

    # Convert histogram to mass function
    sfrf = H / np.product(boxsize.value) / np.log10(bin_widths)
    sigma = np.sqrt(H / np.product(boxsize.value) / np.log10(bin_widths))

    # # Fit the data
    okinds = H > 0
    # popt, pcov = curve_fit(log10phi, bin_cents[okinds], gsmf[okinds], po=[10, 10, 1])

    # # Plot this line
    # xs = np.linspace(sfr_bins.min(), sfr_bins.max(), 1000)
    ax.scatter(bin_cents[okinds], sfrf[okinds],
               marker="o", linestyle="none")

    ax.text(0.95, 0.05, f'$z={z:.1f}$',
            bbox=dict(boxstyle="round,pad=0.3", fc='w',
                      ec="k", lw=1, alpha=0.8),
            transform=ax.transAxes, horizontalalignment='right',
            fontsize=8)

ax.set_xlabel("$\mathrm{SFR}_{100}/\mathrm{M}_\odot \mathrm{yr}^{-1}$")
ax.set_ylabel("$\phi / [\mathrm{cMpc}^{-3} \mathrm{dex}^{-1}]$")

# ax.legend()

fig.savefig("../plots/sfrf.png", bbox_inches="tight", dpi=100)

plt.close(fig)
