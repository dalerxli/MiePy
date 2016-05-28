import numpy as np
import matplotlib.pyplot as plt
from miepy.materials import material, load_material
from miepy.mie_sphere import sphere

wav = np.linspace(400,1000,1000)
eps = 3.7**2*np.ones(1000)
mu = 1*np.ones(1000)
rad = 100
Nmax = 10
diel = material(wav,eps,mu)
# diel = load_material("miepy/materials/ag.dat")
m = sphere(Nmax, diel, rad)

plt.figure(1)
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
C,A = m.scattering()
plt.plot(m.wav,C,label="Scattering", linewidth=2)
plt.plot(m.wav,A,label="Absorption", linewidth=2)
plt.legend()
plt.xlabel("Wavelength (nm)")
plt.ylabel("Scattering Intensity")

plt.figure(2)
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.plot(m.wav,C,label="Total", linewidth=2)
m.plot_scattering_modes(2)
plt.legend()
plt.xlabel("Wavelength (nm)")
plt.ylabel("Scattering Intensity")
plt.show()

