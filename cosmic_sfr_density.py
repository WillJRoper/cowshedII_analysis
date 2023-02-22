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


# Get command line args
snap = sys.argv[1]
paths = sys.argv[2:]

# Set up plot
fig = plt.figure()
ax = fig.add_subplot(111)
ax.semilogy()
ax.grid(True)

for path in paths:

    print(path, path.split("/"))

    # Load data
    data = load(path + "/cowshed12_<snap>.hdf5".replace("<snap>", snap))

    # Get the label for this run
    f_bov = float(path.split("/")[-2][2:].replace("p", "."))
    lab = "$f_\mathrm{bovine}=%.2f$" % f_bov

    # Get redshift of stellar birth
    zs = (1 / data.stars.birth_scale_factors.value) - 1
    print(zs.min(), zs.max())
    # Create age bins
    age_bins = np.arange(cosmo.age(30).to(u.Gyr).value, 14, 0.1) * u.Gyr
    print(age_bins)
    bin_edges = z_at_value(cosmo.age, age_bins, zmin=-1, zmax=50)[::-1]
    bin_cents = (bin_edges[1:] + bin_edges[:-1]) / 2
    mass_formed = np.zeros(bin_cents.size)

    # Bin the stars
    H, _ = np.histogram(zs, bins=bin_edges, weights=data.stars.masses)

    # Convert the mass sum in H to SFR in M_sun / Myr
    sfr = H / 100

    # Convert to cSFRD in M_sun / Myr / Mpc^3
    csfrd = sfr / (12.5 ** 3)

    # Plot curve
    ax.plot(bin_cents, csfrd, label=lab)

# Label axes
ax.set_xlabel("$z$")
ax.set_ylabel("CSFRD / [M$_\odot$ / Myr / Mpc$^{3}$]")

ax.legend()

fig.savefig("../plots/csfrd.png", bbox_inches="tight", dpi=100)

plt.close(fig)
