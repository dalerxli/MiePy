import matplotlib.pyplot as plt
import numpy as np
import miepy
from scipy import constants



class material:
    """Contains eps and mu (both complex) as function of wavelength"""
    def __init__(self,wav,eps,mu):
        self.Nfreq = len(wav)
        self.wav = wav
        self.eps = eps
        self.mu = mu
        self.k = 2*np.pi/wav

def plot_material(mat):
    """Plot material properties of 'mat' """
    plt.plot(mat.wav, mat.eps.real, linewidth=2, label="eps real")
    plt.plot(mat.wav, mat.eps.imag, linewidth=2, label="eps imag")
    plt.plot(mat.wav, mat.mu.real, linewidth=2, label="mu real")
    plt.plot(mat.wav, mat.mu.imag, linewidth=2, label="mu imag")
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
    out = np.arrat([mat.wav, mat.eps.real, mat.eps.imag,
                    mat.mu.real, mat.mu.imag]).T
    miepy.array_io.save(filename, out)



def drude_lorentz(wp, sig, om, gam, wav):
    """Create a material using a Drude-Lorentz function
            wp    =  plasma frequency
            sig   =  strength factors
            om    =  resonant frequencies
            gam   =  damping factors
            wav   =  wavelengths"""

    Nfreq = len(wav)
    size = len(sig)
    omega = 2*np.pi*constants.c*constants.hbar/(constants.e*wav*1e-9)

    eps = np.ones(Nfreq, dtype=np.complex)
    mu = np.ones(Nfreq, dtype=np.complex)
    for i in range(size):
        eps_add = sig[i]*wp**2/(om[i]**2 - omega**2 -1j*omega*gam[i])
        eps += eps_add

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
