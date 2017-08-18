"""
GMT for dimer, varying the separation distance between the dimer pair, with and without interactions
"""
import numpy as np
import matplotlib.pyplot as plt
from miepy import particle,particle_system,materials,sources
from my_pytools.my_matplotlib.colors import cmap
from tqdm import tqdm

Ag = materials.Ag()
mat = materials.material([Ag.wav[375]], [Ag.eps[375]])  # 600 nm
radius = 75

source = sources.rhc(amplitude=1)
separations= np.linspace(2*radius+150,2*radius+1500, 50)

force1 = []
force2 = []

torque1 = []
torque2 = []


for separation in tqdm(separations):
    particles = [
                  dict(position=[separation/2,0,0], radius=radius, material=mat, Nmax=1),
                  dict(position=[-separation/2,0,0], radius=radius, material=mat, Nmax=1),
                ]

    sol = particle_system(particles, source, interactions=False)
    F,T = sol.particle_force(0)
    force1.append(F[0])
    torque1.append(T[2])

    sol = particle_system(particles, source, interactions=True)
    F,T = sol.particle_force(0)
    force2.append(F[0])
    torque2.append(T[2])

plt.figure(1)
plt.plot(separations, force1, label="Fx, no interactions")
plt.plot(separations, force2, label="Fx, interactions")
plt.axhline(y=0, color='black')
plt.legend()

plt.figure(2)
plt.plot(separations, torque1, label="Tz, no interactions")
plt.plot(separations, torque2, label="Tz, interactions")
plt.legend()

plt.show()

