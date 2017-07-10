"""
mie_sphere calculates the scattering coefficients of a sphere using Mie theory
"""
import numpy as np
from miepy.scattering import multipoles
from miepy.special_functions import riccati_1,riccati_2,vector_spherical_harmonics

class sphere:
    def __init__(self, nmax, mat, r, eps_b=1, mu_b=1):
        """Determine an, bn coefficients for a sphere

               nmax  = maximum number of orders to use
               mat   = material object
               r     = radius
               eps_b = background permittivity
               mu_b  = background permeability """


        self.n_b = (eps_b)**.5
        self.mat = mat
        self.r = r
        self.eps_b = eps_b
        self.mu_b = mu_b
        self.nmax = nmax
        self.energy = mat.energy

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
        return multipoles(self.mat.wav, self.n_b, self.an, self.bn) 

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
        """Return an electric field function E(r,theta,phi) for a given wavenumber index"""
        if not Nmax: Nmax = self.nmax
        if not self.interior_computed: self.compute_cd()
        if not self.exterior_computed: self.scattering()

        def E_func(r, theta, phi):
            E = np.zeros(shape = [3] + list(r.shape), dtype=np.complex)
            id_inside = r <= self.r

            for n in range(1,Nmax+1):
                En = 1j**n*(2*n+1)/(n*(n+1))

                an = self.an[n-1, k_index]
                bn = self.bn[n-1, k_index]
                cn = self.cn[n-1, k_index]
                dn = self.dn[n-1, k_index]

                k = self.mat.k[k_index]*self.eps_b**.5
                VSH = vector_spherical_harmonics(n,3)
                E[:,~id_inside] += En*(1j*an*VSH.N_e1n(k)(r[~id_inside],theta[~id_inside],phi[~id_inside])  \
                                - bn*VSH.M_o1n(k)(r[~id_inside],theta[~id_inside],phi[~id_inside]))

                k = self.mat.k[k_index]*self.mat.eps[k_index]**.5
                VSH = vector_spherical_harmonics(n,1)
                E[:,id_inside] += En*(cn*VSH.M_o1n(k)(r[id_inside],theta[id_inside],phi[id_inside])  \
                                - 1j*dn*VSH.N_e1n(k)(r[id_inside],theta[id_inside],phi[id_inside]))

            return E

        return E_func

    def H_field(self, k_index, Nmax = None):
        """Return a magnetic field function H(r,theta,phi) for a given wavenumber index"""
        if not Nmax: Nmax = self.nmax
        if not self.interior_computed: self.compute_cd()
        if not self.exterior_computed: self.scattering()

        def H_func(r, theta, phi):
            H = np.zeros(shape = [3] + list(r.shape), dtype=np.complex)
            id_inside = r <= self.r

            for n in range(1,Nmax+1):
                En = 1j**n*(2*n+1)/(n*(n+1))

                an = self.an[n-1, k_index]
                bn = self.bn[n-1, k_index]
                cn = self.cn[n-1, k_index]
                dn = self.dn[n-1, k_index]

                k = self.mat.k[k_index]*self.eps_b**.5
                omega = self.mat.k[k_index]  # FIX THIS
                VSH = vector_spherical_harmonics(n,3)
                H[:,~id_inside] += k*En/(omega*self.mu_b)*(1j*bn*VSH.N_o1n(k)(r[~id_inside],theta[~id_inside],phi[~id_inside])  \
                                + an*VSH.M_e1n(k)(r[~id_inside],theta[~id_inside],phi[~id_inside]))

                k = self.mat.k[k_index]*self.mat.eps[k_index]**.5
                mu = self.mat.mu[k_index]
                VSH = vector_spherical_harmonics(n,1)
                H[:,id_inside] += -k*En/(omega*mu)*(dn*VSH.M_e1n(k)(r[id_inside],theta[id_inside],phi[id_inside])  \
                                + 1j*cn*VSH.N_o1n(k)(r[id_inside],theta[id_inside],phi[id_inside]))

            return H

        return H_func
