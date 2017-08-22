"""
GMT for dimer, varying the separation distance between the dimer pair, with and without interactions
"""

import numpy as np
import matplotlib.pyplot as plt
import miepy
from tqdm import tqdm

Ag = miepy.materials.predefined.Ag()
radius = 75e-9
source = miepy.sources.rhc_polarized_plane_wave(amplitude=1)
separations = np.linspace(2*radius+150e-9,2*radius+1500e-9, 50)

force1 = []
force2 = []

torque1 = []
torque2 = []

for separation in tqdm(separations):
    spheres = miepy.spheres([[separation/2,0,0], [-separation/2,0,0]], radius, Ag)

    sol = miepy.gmt(spheres, source, 600e-9, 1, interactions=False)
    F,T = map(np.squeeze, sol.force_on_particle(0))
    force1.append(F[0])
    torque1.append(T[2])

    sol = miepy.gmt(spheres, source, 600e-9, 1, interactions=True)
    F,T = map(np.squeeze, sol.force_on_particle(0))
    force2.append(F[0])
    torque2.append(T[2])

plt.figure(1)
plt.plot(separations*1e9, force1, label="Fx, no interactions")
plt.plot(separations*1e9, force2, label="Fx, interactions")
plt.axhline(y=0, color='black')
plt.legend()

plt.figure(2)
plt.plot(separations*1e9, torque1, label="Tz, no interactions")
plt.plot(separations*1e9, torque2, label="Tz, interactions")
plt.legend()

plt.show()

