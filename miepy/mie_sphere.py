"""
mie_sphere calculates the scattering coefficients of a sphere using Mie theory
"""
import numpy as np
from miepy.scattering import multipoles
from miepy.special_functions import riccati_1,riccati_2,vector_spherical_harmonics

class sphere
    def __init__(self, nmax, mat, r, eps_b=1, mu_b=1):
    """Determine an, bn coefficients for a sphere

           nmax  = maximum number of orders to use
           mat   = material object
           r     = radius
           eps_b = background permittivity
           mu_b  = background permeability

       Returns multipoles object"""

        self.n_b = (eps_b)**.5
        self.mat = mat
        self.r = r
        self.eps_b = eps_b
        self.mu_b = mu_b
        self.nmax = nmax

        self.an = np.zeros((nmax,mat.Nfreq), dtype=np.complex)
        self.bn = np.zeros((nmax,mat.Nfreq), dtype=np.complex)
        self.cn = np.zeros((nmax,mat.Nfreq), dtype=np.complex)
        self.dn = np.zeros((nmax,mat.Nfreq), dtype=np.complex)

        self.exterior_computed = False
        self.interior_computed = False
    
    def scattering(self):
        xvals = self.mat.k*self.r*self.n_b
        for i,x in enumerate(xvals):
            m = (self.mat.eps[i]/self.eps_b)**.5
            mt = m*self.mu_b/self.mat.mu[i]

            jn,jn_p = riccati_1(self.nmax,x)
            jnm,jnm_p = riccati_1(self.nmax,m*x)
            yn,yn_p = riccati_2(self.nmax,x)
            a = (mt*jnm*jn_p - jn*jnm_p)/(mt*jnm*yn_p - yn*jnm_p)
            b = (jnm*jn_p - mt*jn*jnm_p)/(jnm*yn_p - mt*yn*jnm_p)
            self.an[:,i] = a[1:]
            self.bn[:,i] = b[1:]
        
        self.an = np.nan_to_num(self.an)
        self.bn = np.nan_to_num(self.bn)

        self.exterior_computed = True
        return multipoles(mat.wav,self.an,self.bn) 

    def compute_cd(self):
        xvals = self.mat.k*self.r*self.n_b
        for i,x in enumerate(xvals):
            m = (self.mat.eps[i]/self.eps_b)**.5
            mt = m*self.mu_b/self.mat.mu[i]

            jn,jn_p = riccati_1(self.nmax,x)
            jnm,jnm_p = riccati_1(self.nmax,m*x)
            yn,yn_p = riccati_2(self.nmax,x)

            c = (m*jn*yn_p - m*yn*jn_p)/(jnm*yn_p - mt*yn*jnm_p)
            d = (m*jn*yn_p - m*yn*jn_p)/(mt*jnm*yn_p - yn*jnm_p)
            self.cn[:,i] = c[1:]
            self.dn[:,i] = d[1:]
        
        self.cn = np.nan_to_num(self.cn)
        self.dn = np.nan_to_num(self.dn)

        self.interior_computed = True

    def E_field(self, k_index, Nmax = None):
        if not Nmax: Nmax = self.nmax

        def E(r, theta, phi):
            for n in range(1,Nmax+1):
                VSH = vector_spherical_harmonics(n,1)
                En = 1j**n*(2n+1)/(n*(n+1))


