"""
run with:
python cosmic_sfr_density.py 0036 <directories...>
"""
import sys
import numpy as np
from swiftsimio import load
import matplotlib.pyplot as plt
from astropy.cosmology import Planck18 as cosmo
from astropy.cosmology import z_at_value
import astropy.units as u
from matplotlib.lines import Line2D
import eagle_IO.eagle_IO as eagle_io


# Get command line args
snap12 = sys.argv[1]
snap50 = sys.argv[2]
paths = sys.argv[3:]

# Define the bin resolution in Gyr
bin_width = 0.05

# Set up plot
fig = plt.figure()
ax = fig.add_subplot(111)
ax.semilogy()
ax.grid(True)

# Loop over files
for path in paths:

    print(path, path.split("/"))

    # Load data
    try:
        data = load(path + "/cowshed12_<snap>.hdf5".replace("<snap>", snap12))
        vol = "12.5 cMpc"
        ls = "--"
    except OSError:
        data = load(path + "/cowshed50_<snap>.hdf5".replace("<snap>", snap50))
        vol = "50 cMpc"
        ls = "-"

    # Get the label for this run
    f_bov = float(path.split("/")[-2][2:].replace("p", "."))

    # Get redshift of stellar birth
    zs = (1 / data.stars.birth_scale_factors.value) - 1
    # Create age bins
    age_bins = np.arange(cosmo.age(100).to(u.Gyr).value, 14, bin_width) * u.Gyr
    bin_edges = z_at_value(cosmo.age, age_bins, zmin=-1, zmax=127)[::-1]
    bin_cents = (bin_edges[1:] + bin_edges[:-1]) / 2
    mass_formed = np.zeros(bin_cents.size)

    # Bin the stars
    H, _ = np.histogram(zs, bins=bin_edges,
                        weights=data.stars.masses.value * 10 ** 10)

    # Convert the mass sum in H to SFR in M_sun / Myr
    sfr = H / (bin_width * u.Gyr).to(u.yr).value

    # Convert to cSFRD in M_sun / Myr / Mpc^3
    if "12" in path:
        csfrd = sfr / (12.5 ** 3)
        c = "mediumpurple"
    else:
        csfrd = sfr / (50 ** 3)
        c = "darkorange"

    # Get the right color
    if "1p0" in path:
        c = "mediumpurple"
    elif "0p5" in path:
        c = "darkorange"
    else:
        c = "yellowgreen"

    # Plot curve
    okinds = csfrd > 0
    ax.plot(bin_cents[okinds], csfrd[okinds], color=c, linestyle=ls)

# Get eagle data
ref_path = "/cosma7/data/Eagle/ScienceRuns/Planck1/L0100N1504/PE/REFERENCE/data"

aborn = eagle_io.read_array('PARTDATA', ref_path, '027_z000p101',
                            'PartType4/StellarFormationTime',
                            noH=True,
                            physicalUnits=True,
                            numThreads=8)
eagle_ms = eagle_io.read_array('PARTDATA', ref_path, '027_z000p101',
                               'PartType4/InitialMass',
                               noH=True,
                               physicalUnits=True,
                               numThreads=8) * 10 ** 10
zs = 1 / aborn - 1

# Create age bins
age_bins = np.arange(cosmo.age(100).to(u.Gyr).value, 14, bin_width) * u.Gyr
bin_edges = z_at_value(cosmo.age, age_bins, zmin=-1, zmax=127)[::-1]
bin_cents = (bin_edges[1:] + bin_edges[:-1]) / 2

# Bin the stars
H, _ = np.histogram(zs, bins=bin_edges,
                    weights=eagle_ms)

# Convert the mass sum in H to SFR in M_sun / yr
sfr = H / (bin_width * u.Gyr).to(u.yr).value
csfrd = sfr / (100 ** 3)

# Plot curve
okinds = csfrd > 0
ax.plot(bin_cents[okinds], csfrd[okinds], color="yellowgreen",
        linestyle="dotted")


def fit(z):
    # Define the fit
    return 0.015 * (1 + z) ** 2.7 / (1 + ((1 + z) / 2.9) ** 5.6)

legend_elements1 = [Line2D([0], [0], color='k',
                           label="COWSHED 50 Mpc",
                           linestyle="-"),
                    Line2D([0], [0], color='k',
                           label="COWSHED 12.5 Mpc",
                           linestyle="--"),
                    Line2D([0], [0], color='k',
                           label="EAGLE 100 Mpc",
                           linestyle="dotted"),
                    Line2D([0], [0], color='grey',
                           label="Madau & Dickinson (2014)",
                           linestyle="dashdot"),
                    ]
legend_elements2 = [Line2D([0], [0], color='mediumpurple',
                           label="$f_\mathrm{bov}= 1.0$",
                           linestyle="-"),
                    Line2D([0], [0], color='darkorange',
                           label="$f_\mathrm{bov}= 0.5$",
                           linestyle="-"),
                    Line2D([0], [0], color='yellowgreen',
                           label="$f_\mathrm{bov}= 0.0$",
                           linestyle="-"),
                    ]


# Plot the fit
okinds = bin_cents <= 8
ax.plot(bin_cents[okinds], fit(bin_cents[okinds]), color="grey",
        linestyle="dashdot")

# Label axes
ax.set_xlabel("$z$")
ax.set_ylabel("CSFRD / [M$_\odot$ / yr / Mpc$^{3}$]")

ax.set_xlim(0, 25)

first_legend = ax.legend(handles=legend_elements2, loc="upper right")
ax.legend(handles=legend_elements1,
          loc='lower left')
ax.add_artist(first_legend)

fig.savefig("../plots/csfrd.png", bbox_inches="tight", dpi=100)

plt.close(fig)
