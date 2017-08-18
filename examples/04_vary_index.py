"""
Scattering intensity of a dielectric sphere for variable wavelength and dielectric constant
"""

import numpy as np
import matplotlib.pyplot as plt
from miepy.materials import material
from miepy import single_mie_sphere
import my_pytools.my_matplotlib.style as style
import my_pytools.my_matplotlib.plots as plots
import my_pytools.my_matplotlib.colors as colors
import matplotlib.cm as cm
from tqdm import tqdm

style.screen()

#wavelength from 400nm to 1000nm

#create a material with n = 3.7 (eps = n^2) at all wavelengths
N1 = 250
N2 = 250
data = np.zeros(shape=(N2,N1))
indices = np.linspace(1,4,N2)
wav = np.linspace(400,1000,N1)
cmap = colors.cmap['parula']
for i,index in enumerate(tqdm(indices)):
    eps = index**2*np.ones(N1)
    mu = 1*np.ones(N1)
    dielectric = material(wav,eps,mu)     #material object

    #calculate scattering coefficients
    rad = 165       # 100 nm radius
    Nmax = 20       # Use up to 10 multipoles
    m = single_mie_sphere(Nmax, dielectric, rad, eps_b=1.0**2).scattering() #scattering object

    # Figure 1: Scattering and Absorption
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    C,A = m.scattering()     # Returns scat,absorp arrays
    # plt.plot(m.energy,C, color=cm.viridis((index-1)/3),linewidth=1)
    # plt.plot(m.energy,A,label="Absorption", linewidth=2)
    # plt.legend()
    # plt.xlabel("Photon energy (eV)")
    # plt.ylabel("Scattering Intensity")
    data[i] = C

    # Figure 2: Scattering per multipole
    # plt.figure(2)
    # plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    # plt.plot(m.energy,C,label="Total", linewidth=2)  #plot total scattering
    # m.plot_scattering_modes(2)    #plots all modes up n=2 (dipole,quadrupole)
    # plt.legend()
    # plt.xlabel("Photon energy (eV)")
    # plt.ylabel("Scattering Intensity")

plt.figure(1)
data /= np.max(data)


X,Y = np.meshgrid(indices, wav, indexing='ij')
# plt.pcolormesh(X,Y,data, shading="gouraud", cmap="gnuplot2")
plt.pcolormesh(X,Y,data, cmap=cmap)
plots.pcolor_z_info(data.T,indices,wav)
plt.xlabel("index of refraction")
plt.ylabel("wavelength (nm)")
plt.colorbar(label="scattering intensity")


plt.show()

