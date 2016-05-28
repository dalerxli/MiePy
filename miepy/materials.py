import numpy as np

class material:
    """Contains eps and mu as function of wavelength"""
    def __init__(self,wav,eps,mu):
        self.Nfreq = len(wav)
        self.wav = wav
        self.eps = eps
        self.mu = mu
        self.k = 2*np.pi/wav
