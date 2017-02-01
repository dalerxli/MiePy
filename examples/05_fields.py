"""
Displaying the fields in an xy cross section of the sphere (x polarized light, z-propagating)
"""

import numpy as np
import matplotlib.pyplot as plt
from miepy.materials import material
from miepy import sphere
import my_pytools.my_matplotlib.style as style
import my_pytools.my_matplotlib.plots as plots
import my_pytools.my_matplotlib.colors as colors

style.screen()

#wavelength from 400nm to 1000nm
wav = np.linspace(400,1000,1000)

#create a material with n = 3.7 (eps = n^2) at all wavelengths
eps = 2.5*np.ones(1000) + 0.1j
mu = 1*np.ones(1000)
dielectric = material(wav,eps,mu)     #material object

#calculate scattering coefficients
rad = 800       # 100 nm radius
Nmax = 5       # Use up to 10 multipoles
m = sphere(Nmax, dielectric, rad)

E_func = m.E_field(999)
x = np.linspace(-2*rad,2*rad,200)
y = np.linspace(-2*rad,2*rad,200)
z = np.array([rad*0.0])

X,Y,Z = np.meshgrid(x,y,z, indexing='xy')
R = (X**2 + Y**2 + Z**2)**0.5
THETA = np.arccos(Z/R)
PHI = np.arctan2(Y,X)
PHI[PHI<0] += 2*np.pi
# R, THETA, PHI = np.meshgrid(r,theta,phi, indexing='ij')

E = E_func(R,THETA,PHI)
E = np.squeeze(E)
I = np.sum(np.abs(E)**2, axis=0)
# I /= np.max(I)

fig = plt.figure()
plt.pcolormesh(np.squeeze(X),np.squeeze(Y), I, shading="gouraud")
plt.colorbar()

# plt.streamplot(np.squeeze(X), np.squeeze(Y), np.real(E[0]), np.real(E[1]), color='white')

THETA = np.squeeze(THETA)
PHI = np.squeeze(PHI)
Ex = E[0]*np.sin(THETA)*np.cos(PHI) + E[1]*np.cos(THETA)*np.cos(PHI) - E[2]*np.sin(PHI)
Ey = E[0]*np.sin(THETA)*np.sin(PHI) + E[1]*np.cos(THETA)*np.sin(PHI) + E[2]*np.cos(PHI)

# Ex[np.squeeze(R) > rad] = 0

step=10
# plt.quiver(np.squeeze(X)[::step,::step], np.squeeze(Y)[::step,::step], np.real(Ex)[::step,::step], np.real(Ey)[::step,::step], color='white')


plt.streamplot(np.squeeze(X), np.squeeze(Y), np.real(Ex), np.real(Ey), color='white', linewidth=2)
# plt.streamplot(np.squeeze(X), np.squeeze(Y), np.real(Ey.T), np.real(Ex.T), color='white', linewidth=2)
plt.xlim([-2*rad,2*rad])
plt.ylim([-2*rad,2*rad])
plt.xlabel("X (nm)")
plt.ylabel("Y (nm)")
plots.set_num_ticks(8,8)

plt.show()
print(I.shape)

