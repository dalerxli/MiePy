"""
materials defines the material class, containing eps and mu data as a function of wavelength

Additional functions are supplied to create different kinds of materials: from files, by 
a drude-lorentz model, or some of the MiePy built-in materials
"""

import matplotlib.pyplot as plt
import numpy as np
import miepy
from scipy import constants

class material:
    """Contains eps and mu (both complex) as function of wavelength"""

    def __init__(self,wav,eps,mu = None):
        """ Create a new material
                wav[N]            wavelengths (in nm)
                eps[N][complex]   complex permitivitty
                mu[N][complex]    complex permeability (default = 1)   """
        
        wav = np.asarray(wav)
        eps = np.asarray(eps)
        if mu is None:
            mu = np.ones(shape = eps.shape)
        mu = np.asarray(mu)

        self.Nfreq = len(wav)
        self.wav = wav
        self.eps = eps
        self.mu = mu
        self.k = 2*np.pi/wav
        self.energy = constants.h*constants.c*1e9/wav/constants.e  # in eV 

def plot_material(mat):
    """Plot material properties of 'mat' """
    plt.plot(mat.wav, mat.eps.real, 'b', linewidth=2, label="eps real")
    plt.plot(mat.wav, mat.eps.imag, 'b--', linewidth=2, label="eps imag")
    plt.plot(mat.wav, mat.mu.real, 'r', linewidth=2, label="mu real")
    plt.plot(mat.wav, mat.mu.imag, 'r--', linewidth=2, label="mu imag")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("eps/mu")
    plt.legend()
    

def load_material(filename):
    """Given a filename containing the following columns (separated by any whitespace, and the last two columns optional):

                 wavelength, eps_real, eps_imag, (mu_real, mu_imag)

       or as a .npy file of shape (Nfreq, 3) or (Nfreq, 5)

       return a material object with the contents"""

    data = miepy.array_io.load(filename) 
    Nc = data.shape[1]     #number of columns
    Nfreq = data.shape[0]  #number of frequencies

    if Nc != 3 and Nc != 5:
        raise ValueError("Number of columns must be 3 or 5")

    wav = data[:,0]
    eps = data[:,1] + 1j*data[:,2]

    if Nc == 5:
        mu = data[:,3] + 1j*data[:,4]
    else:
        mu = np.ones(Nfreq)

    return material(wav,eps,mu)


def save_material(filename, mat):
    """Output a material 'mat' to file 'filename' """
    out = np.array([mat.wav, mat.eps.real, mat.eps.imag,
                    mat.mu.real, mat.mu.imag]).T
    header = "Wavelength\teps_real\teps_imag\tmu_real\tmu_imag"
    miepy.array_io.save(filename, out, header=header)



def drude_lorentz(wp, sig, f, gam, wav, magnetic_only=False, eps_inf=1):
    """Create a material using a Drude-Lorentz function.
       All arguments must be in eV units, except wav (in nm).

       Arguments can be either 1D or 2D arrays: if 2D, eps/mu
       are both specified. If 1D, eps or mu is specified,
       depending on the variable magnetic_only.

            wp    =  plasma frequency      (eV), [2] array
            sig   =  strength factors      (eV), [2, #poles] array
            f     =  resonant frequencies  (eV), [2, #poles] array
            gam   =  damping factors       (eV), [2, #poles] array
            wav   =  wavelengths           (nm), [Nfreq] array
            magnetic_only   =  True if 1D array is to specify mu"""

    #convert iterable input to numpy arrays if necessary
    sig = np.asarray(sig)
    f = np.asarray(f)
    gam = np.asarray(gam)
    wav = np.asarray(wav)

    if len(sig.shape) == 1:
        size = len(sig)
        c = np.array([0,1]) if magnetic_only else np.array([1,0])
        wp = np.array([wp,wp])*c
        sig = np.array([sig,sig])
        f = np.array([f,f])
        gam = np.array([gam,gam])

    Nfreq = len(wav)
    size = len(sig[0])
    omega = 2*np.pi*constants.c*constants.hbar/(constants.e*wav*1e-9)

    eps = eps_inf*np.ones(Nfreq, dtype=np.complex)
    mu = np.ones(Nfreq, dtype=np.complex)
    for i in range(size):
        eps_add = sig[0,i]*wp[0]**2/(f[0,i]**2 - omega**2 -1j*omega*gam[0,i])
        mu_add = sig[1,i]*wp[1]**2/(f[1,i]**2 - omega**2 -1j*omega*gam[1,i])
        eps += eps_add
        mu += mu_add

    return material(wav,eps,mu)

def Ag():
    """Return silver material from MiePy Ag data"""
    # return load_material(miepy.__path__[0] + "/materials/ag.npy")

    wp = 9.01
    sig = [1.01889808, 0.62834151]
    f = [0,5.05635462]
    gam  = [0.01241231, 0.54965831]
    wav = np.linspace(300,1100,1000)
    return drude_lorentz(wp,sig,f,gam,wav)

def Au():
    """Return gold material from MiePy Ag data"""
    # return load_material(miepy.__path__[0] + "/materials/au.dat")

    wp = 9.01
    sig = [0.91765493, 1.47444698]
    gam  = [0.03145002, 3.27922159]
    f = [0, 4.61436431]
    wav = np.linspace(300,1100,1000)
    return drude_lorentz(wp,sig,f,gam,wav)

