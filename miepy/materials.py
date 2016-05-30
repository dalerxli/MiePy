import numpy as np
import miepy

class material:
    """Contains eps and mu as function of wavelength"""
    def __init__(self,wav,eps,mu):
        self.Nfreq = len(wav)
        self.wav = wav
        self.eps = eps
        self.mu = mu
        self.k = 2*np.pi/wav


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

def Ag():
    """Return silver material from MiePy Ag data"""
    return load_material(miepy.__path__[0] + "/materials/ag.npy")
