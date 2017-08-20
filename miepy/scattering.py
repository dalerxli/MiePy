"""
Scattering defines all functions that make use of the scattering coefficients an, bn
Calculations include scattering, absorbption, and electric and magnetic field computations
Mie sphere and Mie core shell both contain an, bn as part of their solution
"""

import numpy as np
import matplotlib.pyplot as plt
import miepy
import scipy.constants as constants

def scattering_per_multipole(an, bn, k):
    """Scattering cross-section per multipole. Returns scat[Nfreq,2,Lmax].
            an[N]    an scattering coefficients
            bn[N]    bn scattering coefficients
            k[N]     wavenumbers
    """
    Nfreq, Lmax = an.shape
    flux = np.zeros([Nfreq,2,Lmax])
    nvals = np.arange(1, Lmax+1)
    flux[:,0,:] = 2*np.pi*(2*nvals+1)*np.abs(an)**2/k[:,np.newaxis]**2
    flux[:,1,:] = 2*np.pi*(2*nvals+1)*np.abs(bn)**2/k[:,np.newaxis]**2

    return flux

def extinction_per_multipole(an, bn, k):
    """Extinction cross-section per multipole. Returns extinct[Nfreq,2,Lmax].
            an[N]    an scattering coefficients
            bn[N]    bn scattering coefficients
            k[N]     wavenumbers
    """
    Nfreq, Lmax = an.shape
    flux = np.zeros([Nfreq,2,Lmax])
    nvals = np.arange(1, Lmax+1)
    flux[:,0,:] = 2*np.pi*(2*nvals+1)*np.real(an)/k[:,np.newaxis]**2
    flux[:,1,:] = 2*np.pi*(2*nvals+1)*np.real(bn)/k[:,np.newaxis]**2

    return flux

def absorbption_per_multipole(an, bn, k):
    """Absorbption cross-section per multipole. Returns absorb[Nfreq,2,Lmax].
            an[N]    an scattering coefficients
            bn[N]    bn scattering coefficients
            k[N]     wavenumbers
    """
    return extinction_per_multipole(an, bn, k) - scattering_per_multipole(an, bn, k)

def cross_sections(an, bn, k):
    """Return the 3 cross-sections, (Scattering, Absorbption, Extinction)
            an[N]    an scattering coefficients
            bn[N]    bn scattering coefficients
            k[N]     wavenumbers
    """
    scat_flux = scattering_per_multipole(an, bn, k)
    extinct_flux = extinction_per_multipole(an, bn, k)
    abs_flux = extinct_flux - scat_flux

    return map(lambda arr: np.sum(arr, axis=(1,2)), [scat_flux, abs_flux, extinct_flux])

def multipole_label(T,L):
    """Get multipole label.
            T = 0 (electric), 1(magnetic)
            L = 0,1,2... (order)
    """
    first = ['e', 'm'][T]
    if L <= 3:
        last = ['D', 'Q', 'O', 'H'][L]
    else:
        last = f" (L = {L})" 
    return first + last