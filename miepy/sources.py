"""
Pre-defined sources that can be used with particle_system
"""

import numpy as np

class xpol:
    def E(r, k):
        amp = np.exp(1j*k*r[2])
        pol = np.array([1, 0, 0])
        return np.einsum('i...,...->i...', pol, amp)

    def H(r, k):
        amp = np.exp(1j*k*r[2])
        pol = np.array([0, 1, 0])
        return np.einsum('i...,...->i...', pol, amp)

class azimuthal:
    def E(r, k):
        rho = (r[0]**2 + r[1]**2)**0.5
        theta = np.arctan2(r[1], r[0])
        amp = np.exp(1j*k*r[2])*rho*np.exp(-(rho/100)**2)
        pol = np.array([-np.sin(theta), np.cos(theta), np.zeros_like(theta)])
        return pol*amp 

    def H(r, k):
        rho = (r[0]**2 + r[1]**2)**0.5
        theta = np.arctan2(r[1], r[0])
        amp = np.exp(1j*k*r[2])*rho*np.exp(-(rho/100)**2)
        pol = np.array([np.cos(theta), np.sin(theta), np.zeros_like(theta)])
        return pol*amp 


class radial:
    def E(r, k):
        rho = (r[0]**2 + r[1]**2)**0.5
        theta = np.arctan2(r[1], r[0])
        amp = np.exp(1j*k*r[2])*rho*np.exp(-(rho/100)**2)
        pol = np.array([np.cos(theta), np.sin(theta), np.zeros_like(theta)])
        return pol*amp 

    def H(r, k):
        rho = (r[0]**2 + r[1]**2)**0.5
        theta = np.arctan2(r[1], r[0])
        amp = np.exp(1j*k*r[2])*rho*np.exp(-(rho/100)**2)
        pol = np.array([-np.sin(theta), np.cos(theta), np.zeros_like(theta)])
        return pol*amp 