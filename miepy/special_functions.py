"""
special_functions defines any additional special functions needed by MiePy
"""

import numpy as np
from scipy import special

def spherical_hn(n, z, derivative=False):
    return special.spherical_jn(n,z,derivative) + 1j*special.spherical_yn(n,z,derivative)

def riccati_1(nmax,x):
    """Riccati bessel function of the 1st kind

       returns (r1, r1'), n=0,1,...,nmax"""

    jn,jnp = special.sph_jn(nmax,x)

    r0 = x*jn
    r1 = jn + x*jnp
    return np.array([r0,r1])

def riccati_2(nmax,x):
    """Riccati bessel function of the 2nd kind

       returns (r2, r2'), n=0,1,...,nmax"""

    jn,jnp,yn,ynp = special.sph_jnyn(nmax,x)
    hn = jn + 1j*yn
    hnp = jnp + 1j*ynp

    r0 = x*hn
    r1 = hn + x*hnp
    return np.array([r0,r1])

def riccati_3(nmax,x):
    """Riccati bessel function of the 3rd kind

       returns (r3, r3'), n=0,1,...,nmax"""

    yn,ynp = special.sph_yn(nmax,x)

    r0 = x*yn
    r1 = yn + x*ynp
    return np.array([r0,r1])

def pi_tau_func(n):
    # if np.sin(theta) == 0: return 0
    lpn = special.legendre(n)
    lpn_p = lpn.deriv()

    def pi_func(theta):
        with np.errstate(divide='ignore', invalid='ignore'):
            pi_func = lpn(np.cos(theta))/np.sin(theta)
            pi_func[pi_func == np.inf] = 0
            pi_func = np.nan_to_num(pi_func)

    def tau_func(theta):
        tau_func = -1*np.sin(theta)*lpn_p(np.cos(theta))

    return pi_func, tau_func 

class vector_spherical_harmonics:
    def __init__(self, n, superscript=1):
        self.pi_func, self.tau_func = pi_tau_func(n)
        self.n = n

        if superscript == 1:
            self.z_func = lambda x: special.spherical_jn(n,x)
            self.zp_func = lambda x: special.spherical_jn(n,x, derivative=True)
        elif superscript == 3:
            self.z_func = lambda x: spherical_hn(n,x)
            self.zp_func = lambda x: spherical_hn(n,x, derivative=True)

    def M_o1n(self, k):
        def f(r, theta, phi):
            theta_comp = np.cos(phi)*self.pi_func(theta)*self.z_func(k*r)
            phi_comp = -1*np.sin(phi)*self.tau_func(theta)*self.z_func(k*r)
            r_comp = np.zeros(shape = theta.shape, dtype=np.complex)
            return np.array([r_comp, theta_comp, phi_comp])
        return f

    def M_e1n(self, k):
        def f(r, theta, phi):
            theta_comp = -1*np.sin(phi)*self.pi_func(theta)*self.z_func(k*r)
            phi_comp = -1*np.cos(phi)*self.tau_func(theta)*self.z_func(k*r)
            r_comp = np.zeros(shape = theta.shape, dtype=np.complex)
            return np.array([r_comp, theta_comp, phi_comp])
        return f

    def N_o1n(self, k):
        def f(r, theta, phi):
            p = k*r
            theta_comp = np.sin(phi)*self.tau_func(theta)*(self.z_func(p) + p*self.zp_func(p))/p
            phi_comp = np.cos(phi)*self.pi_func(theta)*(self.z_func(p) + p*self.zp_func(p))/p
            r_comp = np.sin(phi)*self.n*(self.n+1)*np.sin(theta)*self.pi_func(theta)*self.z_func(p)/p
            return np.array([r_comp, theta_comp, phi_comp])
        return f

    def N_e1n(self, k):
        def f(r, theta, phi):
            p = k*r
            theta_comp = np.cos(phi)*self.tau_func(theta)*(self.z_func(p) + p*self.zp_func(p))/p
            phi_comp = -1*np.sin(phi)*self.pi_func(theta)*(self.z_func(p) + p*self.zp_func(p))/p
            r_comp = np.cos(phi)*self.n*(self.n+1)*np.sin(theta)*self.pi_func(theta)*self.z_func(p)/p
            return np.array([r_comp, theta_comp, phi_comp])
        return f

def riccati_1_single(n,x):
    """Riccati_1, but only a single n value"""
    pre = (np.pi*x/2)**.5
    jn = pre*special.jv(n+0.5,x)
    jnp = jn/(2*x) + pre*special.jvp(n+0.5,x)

    return np.array([jn,jnp])

def riccati_2_single(n,x):
    """Riccati_2, but only a single n value"""
    pre = (np.pi*x/2)**.5
    hn = pre*special.hankel1(n+0.5,x)
    hnp = hn/(2*x) + pre*special.h1vp(n+0.5,x)

    return np.array([hn,hnp])

def riccati_3_single(n,x):
    """Riccati_3, but only a single n value"""
    # pre = (np.pi*x/2)**.5
    # yn = pre*special.yv(n+0.5,x)
    # ynp = yn/(2*x) + pre*special.yvp(n+0.5,x)

    # return np.array([yn,ynp])
    return riccati_2_single(n,x) - riccati_1_single(n,x)


if __name__ == "__main__":
    theta = np.linspace(0,1,5)
    print(pi_tau_func(3,theta))
