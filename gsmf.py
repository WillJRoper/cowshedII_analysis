import sys
import numpy as np
from swiftsimio import load as simload
import matplotlib.pyplot as plt
from astropy.cosmology import Planck18 as cosmo
from astropy.cosmology import z_at_value
import astropy.units as u
from matplotlib.lines import Line2D
import eagle_IO.eagle_IO as eagle_io
from velociraptor import load

# Set up snapshot list
snap_ints = list(range(0, 22))
snaps = []
for s in snap_ints:
    str_snap_int = "%s" % s
    snaps.append(str_snap_int.zfill(4))

# Set up plot
fig = plt.figure()
ax = fig.add_subplot(111)
ax.loglog()
ax.grid(True)

# Loop over snapshots
for snap in snaps:

    # Load swiftsimio dataset to get volume and redshift
    sim_data = simload("../EAGLE_50/snapshots/fb1p0/cowshed50_%s.hdf5" % snap)
    z = data.metadata.redshift
    boxsize = data.metadata.boxsize

    # Load halos
    halo_data = load("galaxies/cowshed50_%s.properties.0" % snap)

    # Extract masses
    data.masses.mass_star_30kpc.convert_to_units("msun")
    stellar_mass = data.masses.mass_star_30kpc

    if stellar_mass.size == 0:
        continue

    print(z, boxsize, np.log10(np.min(stellar_mass)), np.log10(np.max(stellar_mass)))

    # Histogram these masses

    
