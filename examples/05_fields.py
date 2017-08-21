"""
Displaying the fields in an xy cross section of the sphere (x polarized light, z-propagating)
"""

import numpy as np
import matplotlib.pyplot as plt
import miepy
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm

#wavelength from 400nm to 1000nm
wavelength = np.linspace(400e-9,1000e-9,1000)

Ag = miepy.materials.predefined.Ag()
Ag = miepy.material_functions.constant_material(1.1)

#calculate scattering coefficients
radius = 20e-9       # 100 nm radius
Lmax = 1       # Use up to 10 multipoles
sphere = miepy.single_mie_sphere(radius, Ag, wavelength, Lmax)

E_func = sphere.E_field(index=999)
x = np.linspace(-2*radius,2*radius,100)
y = np.linspace(-2*radius,2*radius,100)
z = np.array([radius*0.0])

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
plt.pcolormesh(np.squeeze(X),np.squeeze(Y), I, shading="gouraud", cmap=cm.viridis)
plt.colorbar(label='electric field intensity')

# plt.streamplot(np.squeeze(X), np.squeeze(Y), np.real(E[0]), np.real(E[1]), color='white')

THETA = np.squeeze(THETA)
PHI = np.squeeze(PHI)
Ex = E[0]*np.sin(THETA)*np.cos(PHI) + E[1]*np.cos(THETA)*np.cos(PHI) - E[2]*np.sin(PHI)
Ey = E[0]*np.sin(THETA)*np.sin(PHI) + E[1]*np.cos(THETA)*np.sin(PHI) + E[2]*np.cos(PHI)

# Ex[np.squeeze(R) > radius] = 0

step=10
# plt.quiver(np.squeeze(X)[::step,::step], np.squeeze(Y)[::step,::step], np.real(Ex)[::step,::step], np.real(Ey)[::step,::step], color='white')


plt.streamplot(np.squeeze(X), np.squeeze(Y), np.real(Ex), np.real(Ey), color='black', linewidth=1.5)
# plt.streamplot(np.squeeze(X), np.squeeze(Y), np.real(Ey.T), np.real(Ex.T), color='white', linewidth=2)
plt.xlim([-2*radius,2*radius])
plt.ylim([-2*radius,2*radius])
plt.xlabel("X (nm)")
plt.ylabel("Y (nm)")
plt.title(r'$\varepsilon = 4 + 0.1i$, $\lambda = 1 \mu m$' + '\n')

theta = np.linspace(0,np.pi,50)
phi = np.linspace(0,2*np.pi,50)
r = np.array([10000])

R,THETA,PHI = np.meshgrid(r,theta,phi)
X = R*np.sin(THETA)*np.cos(PHI)
Y = R*np.sin(THETA)*np.sin(PHI)
Z = R*np.cos(THETA)

X = X.squeeze()
Y = Y.squeeze()
Z = Z.squeeze()

E = E_func(R,THETA,PHI)
I = np.sum(np.abs(E)**2, axis=0)
I = np.squeeze(I)
I -= np.min(I)
I /= np.max(I)


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

shape = X.shape
C = np.zeros((shape[0], shape[1], 4))
cmap_3d = cm.viridis
for i in range(shape[0]):
    for j in range(shape[1]):
        # C[i,j,:] = matplotlib.cm.rainbow(R[i,j])
        C[i,j,:] = cmap_3d(I[i,j])
surf = ax.plot_surface(X*1e9, Y*1e9, Z*1e9, rstride=1, cstride=1,shade=False, facecolors=C,linewidth=.0, edgecolors='#000000', antialiased=False)
m = cm.ScalarMappable(cmap=cmap_3d)
m.set_array(I)
plt.colorbar(m)
surf.set_edgecolor('k')
ax.set_xlabel('X')

plt.show()
