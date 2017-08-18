"""
Example of how to make a dielectric material and plot scattering,
absorption, and scattering per multipole
"""

import numpy as np
import matplotlib.pyplot as plt
from miepy.materials import material
from miepy import single_mie_sphere

#wavelength from 400nm to 1000nm
wav = np.linspace(400,1000,1000)

#create a material with n = 3.7 (eps = n^2) at all wavelengths
eps = 3.7**2*np.ones(1000)
mu = 1*np.ones(1000)
dielectric = material(wav,eps,mu)     #material object

#calculate scattering coefficients
rad = 100       # 100 nm radius
Nmax = 10       # Use up to 10 multipoles
m = single_mie_sphere(Nmax, dielectric, rad).scattering() #scattering object

# Figure 1: Scattering and Absorption
plt.figure(1)
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
C,A = m.scattering()     # Returns scat,absorp arrays
plt.plot(m.energy,C,label="Scattering", linewidth=2)
plt.plot(m.energy,A,label="Absorption", linewidth=2)
plt.legend()
plt.xlabel("Photon energy (eV)")
plt.ylabel("Scattering Intensity")

# Figure 2: Scattering per multipole
plt.figure(2)
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.plot(m.energy,C,label="Total", linewidth=2)  #plot total scattering
m.plot_scattering_modes(2)    #plots all modes up n=2 (dipole,quadrupole)
plt.legend()
plt.xlabel("Photon energy (eV)")
plt.ylabel("Scattering Intensity")
plt.show()

