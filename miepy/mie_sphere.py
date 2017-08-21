"""
mie_sphere calculates the scattering coefficients of a sphere using Mie theory
"""
import numpy as np
import pandas as pd
import miepy
from miepy.special_functions import riccati_1,riccati_2,vector_spherical_harmonics
from miepy.material_functions import constant_material
from collections import namedtuple

class single_mie_sphere:
    def __init__(self, radius, material, wavelength, Lmax, medium=None):
        """Solve traditional Mie theory: a single sphere in x-polarized plane wave illumination
               radius           particle radius
               material         particle material
               wavelength[N]    wavelength(s) to solve the system at
               Lmax             maximum number of orders to use in angular momentum expansion
               medium           material medium (must be non-absorbing; defaults to vacuum)
        """

        self.radius = radius
        self.material = material
        if np.isscalar(wavelength):
            self.wavelength = np.array([wavelength], dtype=float)
        else:
            self.wavelength = np.asarray(wavelength, dtype=float)
        self.Lmax = Lmax
        if medium is None:
            self.medium = constant_material(1.0, 1.0)
        else:
            self.medium = medium
            if (self.medium.eps(self.wavelength).imag != 0).any()  \
                         or (self.medium.mu(self.wavelength).imag != 0).any():
                raise ValueError('medium must be non-absorbing')

        self.Nfreq = len(self.wavelength)

        material_type = namedtuple('material_data', ['wavelength', 'k', 'eps', 'mu', 'n', 'eps_b', 'mu_b', 'n_b'])

        eps   = self.material.eps(self.wavelength)
        mu    = self.material.mu(self.wavelength)
        eps_b = self.medium.eps(self.wavelength)
        mu_b  = self.medium.mu(self.wavelength)
        n_b   = np.sqrt(eps_b*mu_b)
        self.material_data = material_type(
               wavelength = self.wavelength,
               k          = 2*np.pi*n_b/self.wavelength,
               eps        = eps,
               mu         = mu,
               n          = np.sqrt(eps*mu),
               eps_b      = eps_b,
               mu_b       = mu_b,
               n_b        = n_b,
               )
               
        self.an = np.zeros((self.Nfreq, self.Lmax), dtype=np.complex)
        self.bn = np.zeros((self.Nfreq, self.Lmax), dtype=np.complex)
        self.cn = np.zeros((self.Nfreq, self.Lmax), dtype=np.complex)
        self.dn = np.zeros((self.Nfreq, self.Lmax), dtype=np.complex)

        self.scattering_properties = (self.an, self.bn, self.material_data.k)

        self.exterior_computed = False
        self.interior_computed = False
    
    def solve_exterior(self):
        """solve for the exterior of the sphere, the an and bn coefficients"""
        xvals = self.material_data.k*self.radius
        m = (self.material_data.eps/self.material_data.eps_b)**.5
        mt = m*self.material_data.mu_b/self.material_data.mu
        for i,x in enumerate(xvals):
            jn,jn_p = riccati_1(self.Lmax,x)
            jnm,jnm_p = riccati_1(self.Lmax,m[i]*x)
            yn,yn_p = riccati_2(self.Lmax,x)
            a = (mt[i]*jnm*jn_p - jn*jnm_p)/(mt[i]*jnm*yn_p - yn*jnm_p)
            b = (jnm*jn_p - mt[i]*jn*jnm_p)/(jnm*yn_p - mt[i]*yn*jnm_p)
            self.an[i] = a[1:]
            self.bn[i] = b[1:]
        
        self.an = np.nan_to_num(self.an)
        self.bn = np.nan_to_num(self.bn)

        self.exterior_computed = True
        return self.an, self.bn

    def solve_interior(self):
        """solve for the interior of the sphere, the cn and dn coefficients"""
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
        return self.cn, self.dn

    def cross_sections(self):
        """Return the 3 cross-sections: (Scattering, Absorbption, Extinction)"""
        if not self.exterior_computed: self.solve_exterior()
        return miepy.cross_sections(*self.scattering_properties)

    def cross_sections_per_multipole(self):
        """Return the 3 cross-sections per multipole: (Scattering, Absorbption, Extinction)"""
        if not self.exterior_computed: self.solve_exterior()
        return (miepy.scattering_per_multipole(*self.scattering_properties),
                miepy.absorbption_per_multipole(*self.scattering_properties),
                miepy.extinction_per_multipole(*self.scattering_properties))


    def E_field(self, k_index, Nmax = None):
        """Return an electric field function E(r,theta,phi) for a given wavenumber index"""
        if not Nmax: Nmax = self.nmax
        if not self.interior_computed: self.solve_interior()
        if not self.exterior_computed: self.solve_exterior()

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

            return -E

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

            return -H

        return H_func
