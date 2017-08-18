"""
Example of how to make a silver sphere  and plot material data,
scattering, absorption, and scattering per multipole
"""

import numpy as np
import matplotlib.pyplot as plt
from miepy.materials import Ag, plot_material
from miepy import single_mie_sphere

#create a silver material (wavelengths 300-1100nm)
silver = Ag()     #material object

#calculate scattering coefficients
rad = 200       # 200 nm radius
Nmax = 10       # Use up to 10 multipoles
m = single_mie_sphere(Nmax, silver, rad).scattering() #scattering object

# Figure 1: Ag eps & mu data
plt.figure(1)
plot_material(silver)


# Figure 2: Scattering and Absorption
plt.figure(2)
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
C,A = m.scattering()     # Returns scat,absorp arrays
plt.plot(m.energy,C,label="Scattering", linewidth=2)
plt.plot(m.energy,A,label="Absorption", linewidth=2)
plt.legend()
plt.xlabel("Photon energy (eV)")
plt.ylabel("Scattering Intensity")

# Figure 3: Scattering per multipole
plt.figure(3)
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.plot(m.energy,C,label="Total", linewidth=2)  #plot total scattering
m.plot_scattering_modes(4)    #plots all modes up n=4
plt.legend()
plt.xlabel("Photon energy (eV)")
plt.ylabel("Scattering Intensity")
plt.show()

