"""
The Generalized Mie Theory (GMT) for a collection of spheres.
"""
import numpy as np
import miepy

class spheres:
    """A collection of N spheres"""
    def __init__(self, position, radius, material):
        """Arguments:
                position[N,3] or [3]        sphere positions
                radius[N] or scalar         sphere radii
                material[N] or scalar       sphere materials
        """
        self.position = np.asarray(np.atleast_2d(position), dtype=float)
        self.radius = np.asarray(np.atleast_1d(radius), dtype=float)
        self.material = np.asarray(np.atleast_1d(material), dtype=np.object)

        if (self.radius.shape[0] == 1):
            self.radius = np.repeat(self.radius[0], self.position.shape[0])

        if (self.material.shape[0] == 1):
            self.material = np.repeat(self.material[0], self.position.shape[0])

        if (self.position.shape[0] != self.radius.shape[0] != self.material.shape[0]):
            raise ValueError("The shapes of position, radius, and material do not match")

class gmt:
    """Solve Generalized Mie Theory: N particles in an arbitray source profile"""
    def __init__(self, spheres, source, wavelength, Lmax, medium=None, interactions=True):
        """Arguments:
               spheres          spheres object specifying the positions, radii, and materials
               source           source object specifying the incident E and H functions
               wavelength[M]    wavelength(s) to solve the system at
               Lmax             maximum number of orders to use in angular momentum expansion (int)
               medium           material medium (must be non-absorbing; default=vacuum)
               interactions     If True, include particle interactions (bool, default=True) 
        """
        self.spheres = spheres
        self.source = source
        self.wavelength = np.asarray(np.atleast_1d(wavelength), dtype=float)
        self.Lmax = Lmax
        self.interactions = interactions

        if medium is None:
            self.medium = miepy.constant_material(1.0, 1.0)
        else:
            self.medium = medium
            if (self.medium.eps(self.wavelength).imag != 0).any()  \
                         or (self.medium.mu(self.wavelength).imag != 0).any():
                raise ValueError('medium must be non-absorbing')

        self.Nfreq = len(self.wavelength)
        self.Nparticles = self.spheres.position.shape[0]

        self.material_data = {}
        self.material_data['wavelength'] = self.wavelength
        self.material_data['eps']        = np.zeros([self.Nparticles, self.Nfreq], dtype=complex) 
        self.material_data['mu']         = np.zeros([self.Nparticles, self.Nfreq], dtype=complex) 
        self.material_data['n']          = np.zeros([self.Nparticles, self.Nfreq], dtype=complex) 
        self.material_data['eps_b']      = self.medium.eps(self.wavelength)
        self.material_data['mu_b']       = self.medium.mu(self.wavelength)
        self.material_data['n_b']        = np.sqrt(self.material_data['eps_b']*self.material_data['mu_b'])
        self.material_data['k']          = 2*np.pi*self.material_data['n_b']/self.wavelength

        self.a = np.zeros([self.Nparticles,self.Nfreq,self.Lmax], dtype=complex)
        self.b = np.zeros([self.Nparticles,self.Nfreq,self.Lmax], dtype=complex)
        self.p = np.zeros([2,self.Nparticles,self.Nfreq], dtype=complex)
        self.q = np.zeros([2,self.Nparticles,self.Nfreq], dtype=complex)

        for i in range(self.Nparticles):
            sphere = miepy.single_mie_sphere(self.spheres.radius[i], self.spheres.material[i],
                        self.wavelength, self.Lmax, self.medium)
            self.material_data['eps'][i] = sphere.material_data['eps']
            self.material_data['mu'][i] = sphere.material_data['mu']
            self.material_data['n'][i] = sphere.material_data['n']
            self.a[i], self.b[i] = sphere.solve_exterior()

        if (self.interactions):
            self._solve_interactions()
    
    def E(self, x, y, z, inc=True):
        """Compute the electric field due to all particles
             
            Arguments:
                x      x position (array-like) 
                y      y position (array-like) 
                z      z position (array-like) 
                inc    Include the incident field (bool, default=True)

            Returns: E[3]
        """
        pass

    def H(self, x, y, z, inc=True):
        """Compute the electric field due to all particles
             
            Arguments:
                x      x position (array-like) 
                y      y position (array-like) 
                z      z position (array-like) 
                inc    Include the incident field (bool, default=True)

            Returns: H[3]
        """
        pass

    def flux_from_particle(self, i, buffer=None, inc=False):
        """Determine the scattered flux from a single particle

            Arguments:
                i         Particle index
                buffer    Distance between sphere radius and sphere calculation (default=)
                inc       Include the incident field (bool, default=False)
            
            Returns: flux[M], M = number of wavelengths
        """
        pass

    def force_on_particle(self, i, buffer=None, inc=True, torque=False):
        """Determine the force on a single particle

            Arguments:
                i         Particle index
                buffer    Distance between sphere radius and sphere calculation (default=)
                inc       Include the incident field (bool, default=False)
                torque    Also calculate the torque (bool, default=False)
            
            Returns: F[3,M] or, if torque is True, (F[3,M],T[3,M]), M = number of wavelengths
        """
        pass

    def flux(self, buffer=None, inc=False):
        """Determine the scattered flux from every particle

            Arguments:
                buffer    Distance between sphere radius and sphere calculation (default=)
                inc       Include the incident field (bool, default=False)
            
            Returns: flux[N,M], N = number of particle, M = number of wavelengths
        """
        pass

    def force(self, buffer=None, inc=True, torque=False):
        """Determine the force on every particle

            Arguments:
                buffer    Distance between sphere radius and sphere calculation (default=)
                inc       Include the incident field (bool, default=False)
                torque    Also calculate the torque (bool, default=False)
            
            Returns: F[3,N,M] or, if torque is True, (F[3,N,M],T[3,N,M]), 
                     N = number of particles, M = number of wavelengths
        """
        pass

    #TODO vectorize for loops. Avoid transpose of position->pass x,y,z to source instead...?
    def _solve_interactions(self):
        pos = self.spheres.position.T
        Einc = self.source.E(pos,self.material_data['k'])
        Einc = Einc[:2,:]
        
        identity = np.zeros(shape = (2, self.Nparticles, 2, self.Nparticles), dtype=np.complex)
        np.einsum('xixi->xi', identity)[...] = 1
        
        for k in range(self.Nfreq):
            MieMatrix = np.zeros(shape = (2, self.Nparticles, 2, self.Nparticles), dtype=np.complex)
            for i in range(self.Nparticles):
                for j in range(self.Nparticles):
                    if i == j: continue
                    pi = self.spheres.position[i]
                    pj = self.spheres.position[j]
                    dji = pi -  pj
                    r_ji = np.linalg.norm(dji)
                    theta_ji = np.arccos(dji[2]/r_ji)
                    phi_ji = np.arctan2(dji[1], dji[0])
                    
                    rhat = np.array([np.sin(theta_ji)*np.cos(phi_ji), np.sin(theta_ji)*np.sin(phi_ji), np.cos(theta_ji)])
                    that = np.array([np.cos(theta_ji)*np.cos(phi_ji), np.cos(theta_ji)*np.sin(phi_ji), -np.sin(theta_ji)])
                    phat = np.array([-np.sin(phi_ji), np.cos(phi_ji), np.zeros_like(theta_ji)])
                    
                    E_func = miepy.scattering.scattered_E(self.a[j,k], self.b[j,k], self.material_data['k'][k])
                    xsol = E_func(r_ji, theta_ji, phi_ji)
                    ysol = E_func(r_ji, theta_ji, phi_ji - np.pi/2)
                    xsol = xsol[0]*rhat + xsol[1]*that + xsol[2]*phat
                    ysol = ysol[0]*rhat + ysol[1]*that + ysol[2]*phat
                    
                    MieMatrix[:,i,0,j] = xsol[:2]
                    MieMatrix[:,i,1,j] = ysol[:2]

            A = identity - MieMatrix
            sol = np.linalg.solve(A.reshape(2*self.Nparticles, 2*self.Nparticles), Einc.reshape(2*self.Nparticles)).reshape(2, self.Nparticles)
            self.p[...,k] = sol
        
if __name__ == "__main__":
    system = gmt(spheres([[0,-50e-9,0],[0,50e-9,0]], 20e-9, miepy.constant_material(2)), 
                  miepy.sources.x_polarized_plane_wave(),
                  600, 2)
    from IPython import embed
    embed()