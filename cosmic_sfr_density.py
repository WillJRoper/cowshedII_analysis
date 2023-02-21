import sys
import numpy as np
from swiftsimio import load
import matplotlib.pyplot as plt
from astropy.cosmology import Planck18 as cosmo
import astropy.units as u


# Load data
data = load(sys.argv[1])

# Get redshift of stellar birth
zs = (1 / data.stars.birth_scale_factors) - 1

# Get the stars age at z=0
ages = cosmo.age(zs).to(u.Myr).value

# Create age bins
bin_edges = np.arange(0, 13.8 * 1000, 100)
bin_cents = (bin_edges[1:] + bin_edges[:-1]) / 2
mass_formed = np.zero(bin_cents.size)

# Bin the stars
H, _ = np.histogram(ages, bins=bin_edges, weights=data.stars.masses)

# Convert the mass sum in H to SFR in M_sun / Myr
sfr = H / 100

# Convert to cSFRD in M_sun / Myr / Mpc^3
csfrd = sfr / (12.5 ** 3)

# Set up plot
fig = plt.figure()
ax = fig.add_subplot(111)

# Plot curve
ax.plot(bin_cents, csfrd)

# Label axes
ax.set_xlabel("Age / [Myr]")
ax.set_ylabel("CSFRD / [M$_\odot$ / Myr / Mpc$^{3}$]")

fig.savefig("../plots/csfrd.png", bbox_inches="tight", dpi=100)

plt.close(fig)
