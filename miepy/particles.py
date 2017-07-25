"""
Particle class for reprententing more general excitations
"""

import numpy as np
from miepy import sphere
from mpl_toolkits.mplot3d import Axes3D

levi = np.zeros((3,3,3))
levi[0,1,2] = levi[1,2,0] = levi[2,0,1] = 1
levi[0,2,1] = levi[1,0,2] = levi[2,1,0] = -1

Ntheta = 30
Nphi = 30

class particle:
    """Sphere with a position"""

    def __init__(self, sp, center, freq_index, source):
        self.center = np.asarray(center)
        self.radius = sp.r
        self.solution = sp
        self.k = sp.mat.k[freq_index]/(sp.eps_b*sp.mu_b)**0.5
        self.E_func = sp.E_field(freq_index)
        self.H_func = sp.H_field(freq_index)
        self.source = source

    def E(self, X, Y, Z, inc=True):
        Xs = X - self.center[0]
        Ys = Y - self.center[1]
        Zs = Z - self.center[2]

        R = np.sqrt(Xs**2 + Ys**2 + Zs**2)
        THETA = np.arccos(Zs/R)
        PHI = np.arctan2(Ys,Xs)
        # PHI[PHI<0] += 2*np.pi

        # incident field
        xhat = np.array([np.sin(THETA)*np.cos(PHI), np.cos(THETA)*np.cos(PHI), -np.sin(PHI)])
        yhat = np.array([np.sin(THETA)*np.sin(PHI), np.cos(THETA)*np.sin(PHI), np.cos(PHI)])

        rhat = np.array([np.sin(THETA)*np.cos(PHI), np.sin(THETA)*np.sin(PHI), np.cos(THETA)])
        that = np.array([np.cos(THETA)*np.cos(PHI), np.cos(THETA)*np.sin(PHI), -np.sin(THETA)])
        phat = np.array([-np.sin(PHI), np.cos(PHI), np.zeros_like(THETA)])

        amp = np.exp(1j*self.k*Zs)

        if inc:
            Einc = self.source.E(np.array([Xs,Ys,Zs]), self.k)
        else:
            Einc = 0
        Ax,Ay,Az = self.source.E(self.center, self.k)

        Escat = Ax*self.E_func(R,THETA,PHI) + Ay*self.E_func(R,THETA,PHI-np.pi/2)
        
        # convert to cartesian
        Etot = Escat[0]*rhat + Escat[1]*that + Escat[2]*phat - Einc
        return Etot

    def H(self, X, Y, Z, inc=True):
        Xs = X - self.center[0]
        Ys = Y - self.center[1]
        Zs = Z - self.center[2]

        R = np.sqrt(Xs**2 + Ys**2 + Zs**2)
        THETA = np.arccos(Zs/R)
        PHI = np.arctan2(Ys,Xs)
        PHI[PHI<0] += 2*np.pi

        # incident field
        xhat = np.array([np.sin(THETA)*np.cos(PHI), np.cos(THETA)*np.cos(PHI), -np.sin(PHI)])
        yhat = np.array([np.sin(THETA)*np.sin(PHI), np.cos(THETA)*np.sin(PHI), np.cos(PHI)])
        rhat = np.array([np.sin(THETA)*np.cos(PHI), np.sin(THETA)*np.sin(PHI), np.cos(THETA)])
        that = np.array([np.cos(THETA)*np.cos(PHI), np.cos(THETA)*np.sin(PHI), -np.sin(THETA)])
        phat = np.array([-np.sin(PHI), np.cos(PHI), np.zeros_like(THETA)])

        if inc:
            Hinc = self.source.H(np.array([Xs,Ys,Zs]), self.k)
        else:
            Hinc = 0
        Ax,Ay,Az = self.source.H(self.center, self.k)

        Hscat = Ay*self.H_func(R,THETA,PHI) + Ax*self.H_func(R,THETA,PHI-np.pi/2)
        
        # convert to cartesian
        Htot = Hscat[0]*rhat + Hscat[1]*that + Hscat[2]*phat - Hinc
        return Htot

    def force(self, other_particles = []):
        r = np.array([self.radius + 1])
        tau = np.linspace(-1,1, Ntheta) 
        theta = np.pi - np.arccos(tau)
        phi = np.linspace(0, 2*np.pi, Nphi)
        R, THETA, PHI = np.meshgrid(r,theta,phi, indexing='ij')

        X = self.center[0] + R*np.sin(THETA)*np.cos(PHI)
        Y = self.center[1] + R*np.sin(THETA)*np.sin(PHI) 
        Z = self.center[2] + R*np.cos(THETA)

        # E and H fields
        E = self.E(X,Y,Z)
        H = self.H(X,Y,Z)

        for p in other_particles:
            E += p.E(X,Y,Z, inc=False)
            H += p.H(X,Y,Z, inc=False)

        E = np.squeeze(E)
        H = np.squeeze(H)
        THETA = np.squeeze(THETA)
        PHI = np.squeeze(PHI)

        # cartesian unit vectors
        xhat = np.array([np.sin(THETA)*np.cos(PHI), np.cos(THETA)*np.cos(PHI), -np.sin(PHI)])
        yhat = np.array([np.sin(THETA)*np.sin(PHI), np.cos(THETA)*np.sin(PHI), np.cos(PHI)])
        zhat = np.array([np.cos(THETA), -np.sin(THETA), np.zeros_like(THETA)])

        # spherical unit vectors
        rhat = np.array([np.sin(THETA)*np.cos(PHI), np.sin(THETA)*np.sin(PHI), np.cos(THETA)])
        that = np.array([np.cos(THETA)*np.cos(PHI), np.cos(THETA)*np.sin(PHI), -np.sin(THETA)])
        phat = np.array([-np.sin(PHI), np.cos(PHI), np.zeros_like(THETA)])

        eps_b = self.solution.eps_b
        mu_b = self.solution.mu_b
        sigma = eps_b*np.einsum('ixy,jxy->ijxy', E, np.conj(E)) \
                + mu_b*np.einsum('ixy,jxy->ijxy', H, np.conj(H)) \
                - 0.5*np.einsum('ij,xy->ijxy', np.identity(3), eps_b*np.sum(np.abs(E)**2, axis=0)) \
                - 0.5*np.einsum('ij,xy->ijxy', np.identity(3), mu_b*np.sum(np.abs(H)**2, axis=0))

        # compute F
        dA = (4*np.pi*r[0]**2/(len(theta)*len(phi)))
        integrand = np.einsum('ijxy,jxy->ixy', sigma, rhat)*dA
        F = np.array([simps_2d(tau, phi, integrand[i].real) for i in range(3)])
        print("Force: {}".format(F))

        # compute T
        integrand = np.einsum('imn,mxy,njxy,jxy->ixy', levi, r[0]*rhat, sigma, rhat)*dA
        T = np.array([simps_2d(tau, phi, integrand[i].real) for i in range(3)])
        print("Torque: {}".format(T))
        return F,T
    
    def flux(self, other_particles = []):
        r = np.array([self.radius + 1])
        tau = np.linspace(-1,1, Ntheta) 
        theta = np.pi - np.arccos(tau)
        phi = np.linspace(0, 2*np.pi, Nphi)
        R, THETA, PHI = np.meshgrid(r,theta,phi, indexing='ij')

        X = self.center[0] + R*np.sin(THETA)*np.cos(PHI)
        Y = self.center[1] + R*np.sin(THETA)*np.sin(PHI) 
        Z = self.center[2] + R*np.cos(THETA)

        # E and H fields
        E = self.E(X,Y,Z, inc=False)
        H = self.H(X,Y,Z, inc=False)

        for p in other_particles:
            E += p.E(X,Y,Z, inc=False)
            H += p.H(X,Y,Z, inc=False)

        E = np.squeeze(E)
        H = np.squeeze(H)
        THETA = np.squeeze(THETA)
        PHI = np.squeeze(PHI)

        # cartesian unit vectors
        xhat = np.array([np.sin(THETA)*np.cos(PHI), np.cos(THETA)*np.cos(PHI), -np.sin(PHI)])
        yhat = np.array([np.sin(THETA)*np.sin(PHI), np.cos(THETA)*np.sin(PHI), np.cos(PHI)])
        zhat = np.array([np.cos(THETA), -np.sin(THETA), np.zeros_like(THETA)])

        # spherical unit vectors
        rhat = np.array([np.sin(THETA)*np.cos(PHI), np.sin(THETA)*np.sin(PHI), np.cos(THETA)])
        that = np.array([np.cos(THETA)*np.cos(PHI), np.cos(THETA)*np.sin(PHI), -np.sin(THETA)])
        phat = np.array([-np.sin(PHI), np.cos(PHI), np.zeros_like(THETA)])

        eps_b = self.solution.eps_b
        mu_b = self.solution.mu_b
        S = np.real(np.einsum('ijk,jxy,kxy->ixy', levi, E, np.conj(H)))

        # compute Flux
        dA = (4*np.pi*r[0]**2/(len(theta)*len(phi)))
        integrand = np.einsum('ixy,ixy->xy', S, rhat)*dA
        Flux = simps_2d(tau, phi, integrand)
        return Flux


class particle_system:
    def __init__(self, bodies, source, eps_b = 1, mu_b = 1, max_reflections = 0):
        self.particles = [particle(sphere(body['Nmax'], body['material'], body['radius'],
                          eps_b, mu_b), body['position'], 0, source) for body in bodies]
        self.source = source
        self.eps_b = eps_b
        self.mu_b = mu_b
        self.max_reflections = max_reflections
        self.Nparticles = len(self.particles)

    def E(self,X,Y,Z, inc = True):
        Efield = self.particles[0].E(X,Y,Z, inc=inc).squeeze()
        Efield += sum([p.E(X,Y,Z, inc = False).squeeze() for p in self.particles[1:]])
        return Efield 

    def H(self,X,Y,Z, inc = True):
        Hfield = self.particles[0].H(X,Y,Z, inc=inc).squeeze()
        Hfield += sum([p.H(X,Y,Z, inc = False).squeeze() for p in self.particles[1:]])
        return Hfield 


    def particle_force(self,i):
        other_particles = (self.particles[j] for j in range(self.Nparticles) if j != i)
        return self.particles[i].force(other_particles)

    def forces(self):
        all_forces  = np.zeros([self.Nparticles, 3])
        all_torques = np.zeros([self.Nparticles, 3])
        for i in range(self.Nparticles):
            F,T = self.particle_force(i)
            all_forces[i]  = F
            all_torques[i] = T

        return all_forces, all_torques

    def center_of_mass(self):
        com = np.zeros(3)
        for particle in self.particles:
            com += particle.center
        com /= self.Nparticles
        return com

    def moment_of_intertia(self, center = None, rho=1):
        if center is None:
            center = self.center_of_mass()
        I = 0
        for particle in self.particles:
            mass = 4/3*np.pi*particle.radius**3*rho
            di = np.linalg.norm(particle.center - center)
            Ii = mass*di**2 + 2/5*mass*particle.radius**2
            I += Ii
        return I

    def net_force(self,center = None, radius = None):
        if center is None:
            center = self.center_of_mass()
        center = np.asarray(center)

        if radius is None:
            radius = 0
            for p in self.particles:
                radius = max(radius, np.linalg.norm(p.center - center) + p.radius)
            radius += 1

        r = np.array([radius])
        tau = np.linspace(-1,1, Ntheta) 
        theta = np.pi - np.arccos(tau)
        phi = np.linspace(0, 2*np.pi, Nphi)
        R, THETA, PHI = np.meshgrid(r,theta,phi, indexing='ij')

        X = center[0] + R*np.sin(THETA)*np.cos(PHI)
        Y = center[1] + R*np.sin(THETA)*np.sin(PHI) 
        Z = center[2] + R*np.cos(THETA)

        # E and H fields
        E = self.E(X,Y,Z)
        H = self.H(X,Y,Z)
        E = np.squeeze(E)
        H = np.squeeze(H)
        THETA = np.squeeze(THETA)
        PHI = np.squeeze(PHI)

        # cartesian unit vectors
        xhat = np.array([np.sin(THETA)*np.cos(PHI), np.cos(THETA)*np.cos(PHI), -np.sin(PHI)])
        yhat = np.array([np.sin(THETA)*np.sin(PHI), np.cos(THETA)*np.sin(PHI), np.cos(PHI)])
        zhat = np.array([np.cos(THETA), -np.sin(THETA), np.zeros_like(THETA)])

        # spherical unit vectors
        rhat = np.array([np.sin(THETA)*np.cos(PHI), np.sin(THETA)*np.sin(PHI), np.cos(THETA)])
        that = np.array([np.cos(THETA)*np.cos(PHI), np.cos(THETA)*np.sin(PHI), -np.sin(THETA)])
        phat = np.array([-np.sin(PHI), np.cos(PHI), np.zeros_like(THETA)])

        eps_b = self.eps_b
        mu_b = self.mu_b
        sigma = eps_b*np.einsum('ixy,jxy->ijxy', E, np.conj(E)) \
                + mu_b*np.einsum('ixy,jxy->ijxy', H, np.conj(H)) \
                - 0.5*np.einsum('ij,xy->ijxy', np.identity(3), eps_b*np.sum(np.abs(E)**2, axis=0)) \
                - 0.5*np.einsum('ij,xy->ijxy', np.identity(3), mu_b*np.sum(np.abs(H)**2, axis=0))

        # compute F
        dA = (4*np.pi*r[0]**2/(len(theta)*len(phi)))
        integrand = np.einsum('ijxy,jxy->ixy', sigma, rhat)*dA
        F = np.array([simps_2d(tau, phi, integrand[i].real) for i in range(3)])
        print("Force: {}".format(F))

        # compute T
        integrand = np.einsum('imn,mxy,njxy,jxy->ixy', levi, r[0]*rhat, sigma, rhat)*dA
        T = np.array([simps_2d(tau, phi, integrand[i].real) for i in range(3)])
        print("Torque: {}".format(T))
        return F,T