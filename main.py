import numpy as np
import matplotlib.pyplot as plt
from miepy.materials import material
from miepy.mie_sphere import sphere

wav = np.linspace(400,1000,1000)
eps = 3.7**2*np.ones(1000)
mu = 1*np.ones(1000)
diel = material(wav,eps,mu)
m = sphere(10, diel, 100)

plt.figure(1)
C,A = m.scattering()
CeD = m.mode_scattering('e',1)
CeQ = m.mode_scattering('e',2)
CeO = m.mode_scattering('e',3)
CmD = m.mode_scattering('m',1)
CmQ = m.mode_scattering('m',2)
CmO = m.mode_scattering('m',3)
plt.plot(wav,C,label="Total", linewidth=2)
plt.plot(wav,CeD,label="eD", linewidth=2)
plt.plot(wav,CeQ,label="eQ", linewidth=2)
plt.plot(wav,CmD,label="mD", linewidth=2)
plt.plot(wav,CmQ,label="mQ", linewidth=2)
plt.plot(wav,CeO,label="eO", linewidth=2)
plt.plot(wav,CmO,label="mO", linewidth=2)
plt.legend()

plt.figure(2)
plt.plot(wav,C,label="Scattering", linewidth=2)
plt.plot(wav,A,label="Absorption", linewidth=2)
plt.legend()
plt.show()

