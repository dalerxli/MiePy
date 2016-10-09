"""
Example of how to make a silver sphere  and plot material data,
scattering, absorption, and scattering per multipole
"""

import numpy as np
import matplotlib.pyplot as plt
from miepy.materials import material, Ag, plot_material,Au
from miepy import sphere, core_shell

#create a silver material (wavelengths 300-1100nm)
wav = np.linspace(400,1000,1000)
silver = Ag()     #material object
eps = 1.46**2*np.ones(1000)
mu = 1*np.ones(1000)
dielectric = material(wav,eps,mu)     #material object

#calculate scattering coefficients
rad = 125       # 200 nm radius
Nmax = 10       # Use up to 10 multipoles
rad = 135
m = core_shell(Nmax, dielectric, silver,rad,rad+30, eps_b=1.5**2) #scattering object



# Figure 2: Scattering and Absorption
plt.figure(2)
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
C,A = m.scattering()     # Returns scat,absorp arrays
plt.plot(m.energy,C,label="Scattering", linewidth=2)
plt.plot(m.energy,A,label="Absorption", linewidth=2)
plt.legend()
plt.xlabel("Wavelength (nm)")
plt.ylabel("Scattering Intensity")

# Figure 3: Scattering per multipole
plt.figure(3)
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.plot(m.energy,C,label="Total", linewidth=2)  #plot total scattering
m.plot_scattering_modes(3)    #plots all modes up n=4
plt.legend()
plt.xlabel("Wavelength (nm)")
plt.ylabel("Scattering Intensity")
plt.show()

