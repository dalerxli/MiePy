"""
Pre-defined sources that can be used with particle_system
"""

import numpy as np

class plane_wave:
    def __init__(self, polarization, amplitude=1):
        polarization = np.asarray(polarization, dtype=np.complex)
        self.polarization = polarization
        self.polarization /= np.linalg.norm(polarization)
        self.amplitude = amplitude
    
    def E(self, r, k):
        amp = self.amplitude*np.exp(1j*k*r[2])
        pol = np.array([*self.polarization, 0])
        return np.einsum('i...,...->i...', pol, amp)

    def H(self, r, k):
        amp = self.amplitude*np.exp(1j*k*r[2])
        H0_x, H0_y = -self.polarization[1], self.polarization[0]
        pol = np.array([H0_x, H0_y, 0])
        return np.einsum('i...,...->i...', pol, amp)

xpol = lambda amplitude=1: plane_wave(polarization=[1,0],   amplitude=amplitude)
ypol = lambda amplitude=1: plane_wave(polarization=[0,1],   amplitude=amplitude)
rhc  = lambda amplitude=1: plane_wave(polarization=[1,1j],  amplitude=amplitude)
lhc  = lambda amplitude=1: plane_wave(polarization=[1,-1j], amplitude=amplitude)

class azimuthal:
    def E(self, r, k):
        rho = (r[0]**2 + r[1]**2)**0.5
        theta = np.arctan2(r[1], r[0])
        amp = np.exp(1j*k*r[2])*rho*np.exp(-(rho/100)**2)
        pol = np.array([-np.sin(theta), np.cos(theta), np.zeros_like(theta)])
        return pol*amp 

    def H(self, r, k):
        rho = (r[0]**2 + r[1]**2)**0.5
        theta = np.arctan2(r[1], r[0])
        amp = np.exp(1j*k*r[2])*rho*np.exp(-(rho/100)**2)
        pol = np.array([np.cos(theta), np.sin(theta), np.zeros_like(theta)])
        return pol*amp 


class radial:
    def E(self, r, k):
        rho = (r[0]**2 + r[1]**2)**0.5
        theta = np.arctan2(r[1], r[0])
        amp = np.exp(1j*k*r[2])*rho*np.exp(-(rho/100)**2)
        pol = np.array([np.cos(theta), np.sin(theta), np.zeros_like(theta)])
        return pol*amp 

    def H(self, r, k):
        rho = (r[0]**2 + r[1]**2)**0.5
        theta = np.arctan2(r[1], r[0])
        amp = np.exp(1j*k*r[2])*rho*np.exp(-(rho/100)**2)
        pol = np.array([-np.sin(theta), np.cos(theta), np.zeros_like(theta)])
        return pol*amp 