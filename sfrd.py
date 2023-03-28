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
from swiftascmaps import nineteen_eighty_nine
from velociraptor.swift.swift import to_swiftsimio_dataset
from velociraptor.particles import load_groups
from swiftgalaxy import SWIFTGalaxy, Velociraptor
from schwimmbad import MultiPool
from unyt import c, h, nJy, erg, s, Hz, pc, angstrom, eV, Msun, yr


tenpc = 10*pc  # ten parsecs
# the surface area (in cm) at 10 pc. I HATE the magnitude system
geo = 4*np.pi*(tenpc.to('cm').value)**2

def M_to_Lnu(M):
    """ Convert absolute magnitude (M) to L_nu """
    return 10**(-0.4 * (M + 48.6)) * geo * erg / s / Hz

def lnu_to_sfr(lnu):
   """
   L1500 = SFR x (7.1e-29 Msun^-1 yr^-1 erg^-1 s Hz)
   This if from a paper by Pratika Dayal (Dayal+2022, REBELs paper) 
   """
   return lnu * (7.1 * 10 ** -29 * Msun * yr**-1 * erg**-1 * s * Hz)
   
# Define observational data
# (format -> {z: [M deltaM phi phi_err_low phi_err_upp]})
obs = {}
obs["Donnan+2022"] = {8: [[-22.17, 1.0, 0.63E-6, 0.32E-6, 0.32E-6],
                          [-21.42, 0.5, 3.92E-6, 1.63E-6, 1.63E-6]],
                      9: [[-22.30, 1.0, 0.17E-6, 0.17E-6, 0.17E-6],
                          [-21.30, 1.0, 3.02E-6, 2.74E-6, 2.74E-6],
                          [-18.50, 1.0, 1200E-6, 537E-6, 537E-6]],
                      10.5: [[-22.57, 1.0, 0.18E-6, 0.18E-6, 0.18E-6],
                             [-20.10, 1.0, 16.2E-6, 11.5E-6, 11.5E-6],
                             [-19.35, 0.5, 136.0E-6, 48.2E-6, 48.2E-6],
                             [-18.85, 0.5, 234.9E-6, 100.5E-6, 100.5E-6],
                             [-18.23, 0.75, 630.8E-6, 448.2E-6, 448.2E-6]]}
obs["Bouwens+2022"] = {8.5: [[-20.02, 1.25, 0.000164, 0.000162, 0.000162],
                             [-18.77, 1.25, 0.000378, 0.000306, 0.000306]],
                       10.5: [[-18.65, 1.0, 0.000290, 0.000238, 0.000238]],
                       12.5: [[-20.31, 1.0, 0.000116, 0.000094, 0.000094],
                              [-19.31, 1.0, 0.000190, 0.000152, 0.000152]]}

markers = {"Donnan+2022": "s", "Bouwens+2022": "^"}

# Convert the magnitudes
for study in obs:
    for key in obs[study]:
        for lst in obs[study][key]:
            mag = lst[0]
            mag_err = lst[1]
            mag_high = mag - mag_err
            mag_low = mag + mag_err
            
            lum = M_to_Lnu(mag)
            lum_low = M_to_Lnu(mag_low)
            lum_high = M_to_Lnu(mag_high)

            print(lum)
            
            sfr = lnu_to_sfr(lum)
            sfr.convert_to_units("Msun/yr")
            sfr_low = lnu_to_sfr(lum_low)
            sfr_low.convert_to_units("Msun/yr")
            sfr_high = lnu_to_sfr(lum_high)
            sfr_high.convert_to_units("Msun/yr")
            sfr_err_low = sfr - sfr_low
            sfr_err_high = sfr_high - sfr

            lst.append(sfr.value)
            lst.append(sfr_err_low.value)
            lst.append(sfr_err_high.value)
        
            print(sfr, sfr_err_low, sfr_err_high)

# Set up snapshot list
snap_ints = [4, 5, 6, 8]
snaps = []
for s in snap_ints:
    str_snap_int = "%s" % s
    snaps.append(str_snap_int.zfill(4))

# Define the normalisation and colormap
norm = Normalize(vmin=6, vmax=13)
cmap = nineteen_eighty_nine

# Set up plot
fig = plt.figure()
ax = fig.add_subplot(111)
ax.grid(True)
ax.loglog()

# Define mass bins
sfr_bins = np.logspace(-3, 2, 50)
bin_cents = (sfr_bins[1:] + sfr_bins[:-1]) / 2
bin_widths = sfr_bins[1:] - sfr_bins[:-1]


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

legend_elements1 = [Line2D([0], [0], color='k',
                           label="COWSHED 50 Mpc",
                           linestyle="-"),
                    ]
    
# Plot the observations
for study in obs:

    legend_elements1.append(Line2D([0], [0], color='k',
                                   label=study, linestyle="none",
                                   marker=markers[study]))
    
    for z in obs[study]:
        for lst in obs[study][z]:

            ax.errorbar(lst[-3], lst[2], xerr=np.array([[lst[-2]], [lst[-1]]]),
                        yerr=np.array([[lst[3]], [lst[4]]]), linestyle="none",
                        marker=markers[study], color=cmap(norm(z)))

ax.set_xlabel("$\mathrm{SFR}_{100}/\mathrm{M}_\odot \mathrm{yr}^{-1}$")
ax.set_ylabel("$\phi / [\mathrm{cMpc}^{-3} \mathrm{dex}^{-1}]$")

ax.legend(handles=legend_elements1, loc="upper right")

cbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax)
cbar.set_label("$z$")

# ax.legend()

fig.savefig("../plots/sfrf.png", bbox_inches="tight", dpi=100)

plt.close(fig)
