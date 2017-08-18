"""
Comparison of single Mie theory and GMT
"""

import numpy as np
import matplotlib.pyplot as plt
from miepy.materials import material
from miepy import sphere
from miepy.particles import particle_system
from miepy.sources import xpol
from tqdm import tqdm

#wavelength from 400nm to 1000nm
wav = np.linspace(400,1000,1000)

#create a material with n = 3.7 (eps = n^2) at all wavelengths
eps = 3.7**2*np.ones(1000)
mu = 1*np.ones(1000)
dielectric = material(wav,eps,mu)     #material object

#calculate scattering coefficients
rad = 100       # 100 nm radius

# Single Mie Theory
Nmax = 10       # Use up to 10 multipoles
m = sphere(Nmax, dielectric, rad).scattering() #scattering object
C,A = m.scattering()     # Returns scat,absorp arrays
plt.plot(m.energy,C,label="Single Mie theory", linewidth=2)

fluxes = []
idx = np.arange(0,1000,15)
for i in tqdm(idx):
    mat = material([wav[i]], [eps[i]], [mu[i]])
    system = particle_system([dict(Nmax=2, material=mat, position=[0,0,0], radius=rad)], xpol())
    flux = system.particle_flux(0)
    fluxes.append(flux)

plt.plot(m.energy[idx],  fluxes, label='GMT')

plt.xlabel("Photon energy (eV)")
plt.ylabel("Scattering Intensity")
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.legend()
plt.show()
