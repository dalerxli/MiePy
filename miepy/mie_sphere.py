"""
mie_sphere calculates the scattering coefficients of a sphere using Mie theory
"""
import numpy as np
import pandas as pd
import miepy
from miepy.special_functions import riccati_1,riccati_2,vector_spherical_harmonics
from miepy.material_functions import constant_material

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

        self.material_data = {}
        self.material_data['wavelength'] = self.wavelength
        self.material_data['eps']        = self.material.eps(self.wavelength)
        self.material_data['mu']         = self.material.mu(self.wavelength)
        self.material_data['n']          = np.sqrt(self.material_data['eps']*self.material_data['mu'])
        self.material_data['eps_b']      = self.medium.eps(self.wavelength)
        self.material_data['mu_b']       = self.medium.mu(self.wavelength)
        self.material_data['n_b']        = np.sqrt(self.material_data['eps_b']*self.material_data['mu_b'])
        self.material_data['k']          = 2*np.pi*self.material_data['n_b']/self.wavelength
               
        self.an = np.zeros((self.Nfreq, self.Lmax), dtype=np.complex)
        self.bn = np.zeros((self.Nfreq, self.Lmax), dtype=np.complex)
        self.cn = np.zeros((self.Nfreq, self.Lmax), dtype=np.complex)
        self.dn = np.zeros((self.Nfreq, self.Lmax), dtype=np.complex)

        self.scattering_properties = (self.an, self.bn, self.material_data['k'])

        self.exterior_computed = False
        self.interior_computed = False
    
    def solve_exterior(self):
        """solve for the exterior of the sphere, the an and bn coefficients"""
        xvals = self.material_data['k']*self.radius
        m = (self.material_data['eps']/self.material_data['eps_b'])**.5
        mt = m*self.material_data['mu_b']/self.material_data['mu']
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
        xvals = self.material_data['k']
        m = (self.material_data['eps']/self.material_data['eps_b'])**.5
        mt = m*self.material_data['mu_b']/self.material_data['mu']
        for i,x in enumerate(xvals):
            jn,jn_p = riccati_1(self.Lmax,x)
            jnm,jnm_p = riccati_1(self.Lmax,m[i]*x)
            yn,yn_p = riccati_2(self.Lmax,x)

            c = (m[i]*jn*yn_p - m[i]*yn*jn_p)/(jnm*yn_p - mt[i]*yn*jnm_p)
            d = (m[i]*jn*yn_p - m[i]*yn*jn_p)/(mt[i]*jnm*yn_p - yn*jnm_p)
            self.cn[i] = c[1:]
            self.dn[i] = d[1:]
        
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


    def E_field(self, index=None, Lmax=None):
        """Return an electric field function E(r,theta,phi) for a given wavenumber index"""
        if Lmax is None: Lmax = self.Lmax
        if index is None:
            index = np.s_[:]

        if not self.interior_computed: self.solve_interior()
        if not self.exterior_computed: self.solve_exterior()
        mat = self.material_data

        def E_func(r, theta, phi):
            E = np.zeros(shape = [3] + list(r.shape), dtype=np.complex)
            id_inside = r <= self.radius

            for n in range(1,Lmax+1):
                En = 1j**n*(2*n+1)/(n*(n+1))

                an = self.an[index, n-1]
                bn = self.bn[index, n-1]
                cn = self.cn[index, n-1]
                dn = self.dn[index, n-1]

                k = mat['k'][index]
                VSH = vector_spherical_harmonics(n,3)
                E[:,~id_inside] += En*(1j*an*VSH.N_e1n(k)(r[~id_inside],theta[~id_inside],phi[~id_inside])  \
                                - bn*VSH.M_o1n(k)(r[~id_inside],theta[~id_inside],phi[~id_inside]))

                k = 2*np.pi*mat['n'][index]/self.wavelength[index]
                VSH = vector_spherical_harmonics(n,1)
                E[:,id_inside] += En*(cn*VSH.M_o1n(k)(r[id_inside],theta[id_inside],phi[id_inside])  \
                                - 1j*dn*VSH.N_e1n(k)(r[id_inside],theta[id_inside],phi[id_inside]))

            return -E

        return E_func

    def H_field(self, index=None, Lmax=None):
        """Return a magnetic field function H(r,theta,phi) for a given wavenumber index"""
        if Lmax is None: Lmax = self.Lmax
        if index is None:
            index = np.s_[:]

        if not self.interior_computed: self.solve_interior()
        if not self.exterior_computed: self.solve_exterior()
        mat = self.material_data

        def H_func(r, theta, phi):
            H = np.zeros(shape = [3] + list(r.shape), dtype=np.complex)
            id_inside = r <= self.radius

            for n in range(1,Lmax+1):
                En = 1j**n*(2*n+1)/(n*(n+1))


                an = self.an[index, n-1]
                bn = self.bn[index, n-1]
                cn = self.cn[index, n-1]
                dn = self.dn[index, n-1]

                k = mat['k'][index]
                omega = 2*np.pi/self.wavelength[index]  # FIX THIS
                VSH = vector_spherical_harmonics(n,3)
                H[:,~id_inside] += k*En/(omega*mat['mu_b'][index])*(1j*bn*VSH.N_o1n(k)(r[~id_inside],theta[~id_inside],phi[~id_inside])  \
                                + an*VSH.M_e1n(k)(r[~id_inside],theta[~id_inside],phi[~id_inside]))

                k = 2*np.pi*mat['n'][index]/self.wavelength[index]
                mu = mat['mu'][index]
                VSH = vector_spherical_harmonics(n,1)
                H[:,id_inside] += -k*En/(omega*mu)*(dn*VSH.M_e1n(k)(r[id_inside],theta[id_inside],phi[id_inside])  \
                                + 1j*cn*VSH.N_o1n(k)(r[id_inside],theta[id_inside],phi[id_inside]))

            return -H

        return H_func
