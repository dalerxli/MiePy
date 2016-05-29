import numpy as np
import matplotlib.pyplot as plt
from miepy.materials import material, load_material, Ag
from miepy import sphere

#create a silver material miepy.materials (wavelengths 300-1100nm)
eps = 3.7**2*np.ones(1000)
mu = 1*np.ones(1000)
silver = Ag()     #material object

#calculate scattering coefficients
rad = 200       # 200 nm radius
Nmax = 10       # Use up to 10 multipoles
m = sphere(Nmax, silver, rad) #scattering object

# Figure 1: Scattering and Absorption
plt.figure(1)
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
C,A = m.scattering()     # Returns scat,absorp arrays
plt.plot(m.wav,C,label="Scattering", linewidth=2)
plt.plot(m.wav,A,label="Absorption", linewidth=2)
plt.legend()
plt.xlabel("Wavelength (nm)")
plt.ylabel("Scattering Intensity")

# Figure 2: Scattering per multipole
plt.figure(2)
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.plot(m.wav,C,label="Total", linewidth=2)  #plot total scattering
m.plot_scattering_modes(4)    #plots all modes up n=4
plt.legend()
plt.xlabel("Wavelength (nm)")
plt.ylabel("Scattering Intensity")
plt.show()

