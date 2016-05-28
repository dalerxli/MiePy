import numpy as np
import matplotlib.pyplot as plt
from scipy import special

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

class material:
    """Contains eps and mu as function of wavelength"""
    def __init__(self,wav,eps,mu):
        self.Nfreq = len(wav)
        self.wav = wav
        self.eps = eps
        self.mu = mu
        self.k = 2*np.pi/wav

class multipoles:
    """Contains an and bn as function of wavelength"""
    def __init__(self, wav, an, bn):
        self.Nfreq = len(wav)
        self.wav = wav
        self.an = an
        self.bn = bn
        self.k = 2*np.pi/wav

    def mode_scattering(self, mtype, n):
        """Get modal scattering intensity

                mtype = 'e' or 'm'
                n     = order"""

        cn = self.an if mtype=='e' else self.bn
        return 2*np.pi*(2*n+1)*np.abs(cn[n-1])**2/self.k**2

    def scattering_array(self):
        """Get modal scattering intensity for all modes
           Return scat[2,nmax,Nfreq]"""

        nmax = self.an.shape[0]
        scat = np.zeros([2,nmax,self.Nfreq])
        for n in range(1,nmax+1):
            scat[0,n-1] = self.mode_scattering('e', n)
            scat[1,n-1] = self.mode_scattering('m', n)
        return scat

    def scattering(self):
        """Get total scattering intensity using all coefficients an,bn

           Return (scattering, absorbption) amplitudes"""

        nmax = self.an.shape[0]
        nvals = 2*np.arange(1,nmax+1) + 1
        nvals = np.expand_dims(nvals,axis=1)
        sum_val_1 = np.sum(nvals*(np.abs(self.an)**2 + np.abs(self.bn)**2),axis=0)
        sum_val_2 = np.sum(nvals*np.real(self.an+self.bn),axis=0)
        scat = 2*np.pi*sum_val_1/self.k**2
        extinc = 2*np.pi*sum_val_2/self.k**2
        absorb = extinc - scat
        return scat, absorb


def mie_sphere(nmax, mat, r, eps_b=1, mu_b=1):
    """Determine an, bn coefficients for a sphere

           nmax  = maximum number of orders to use
           mat   = material object
           r     = radius
           eps_b = background permittivity
           mu_b  = background permeability

       Returns multipoles object"""

    n_b = (eps_b)**.5
    xvals = mat.k*r*n_b

    an = np.zeros((nmax,mat.Nfreq), dtype=np.complex)
    bn = np.zeros((nmax,mat.Nfreq), dtype=np.complex)

    for i,x in enumerate(xvals):
        m = (mat.eps[i]/eps_b)**.5
        mt = m*mu_b/mat.mu[i]

        jn,jn_p = riccati_1(nmax,x)
        jnm,jnm_p = riccati_1(nmax,m*x)
        yn,yn_p = riccati_2(nmax,x)
        a = (mt*jnm*jn_p - jn*jnm_p)/(mt*jnm*yn_p - yn*jnm_p)
        b = (jnm*jn_p - mt*jn*jnm_p)/(jnm*yn_p - mt*yn*jnm_p)
        an[:,i] = a[1:]
        bn[:,i] = b[1:]
    
    an = np.nan_to_num(an)
    bn = np.nan_to_num(bn)
    return multipoles(mat.wav,an,bn) 


if __name__ == "__main__":
    wav = np.linspace(400,1000,1000)
    eps = 3.7**2*np.ones(1000)
    mu = 1*np.ones(1000)
    diel = material(wav,eps,mu)
    m = mie_sphere(10, diel, 100)

    plt.figure(1)
    C,A = m.scattering()
    CeD = m.mode_scattering('e',1)
    CeQ = m.mode_scattering('e',2)
    CeO = m.mode_scattering('e',3)
    CmD = m.mode_scattering('m',1)
    CmQ = m.mode_scattering('m',2)
    CmO = m.mode_scattering('m',3)
    plt.plot(wav,C,label="Total", linewidth=2)
    plt.plot(wav,CeD,label="eD", linewidth=2)
    plt.plot(wav,CeQ,label="eQ", linewidth=2)
    plt.plot(wav,CmD,label="mD", linewidth=2)
    plt.plot(wav,CmQ,label="mQ", linewidth=2)
    plt.plot(wav,CeO,label="eO", linewidth=2)
    plt.plot(wav,CmO,label="mO", linewidth=2)
    plt.legend()

    plt.figure(2)
    plt.plot(wav,C,label="Scattering", linewidth=2)
    plt.plot(wav,A,label="Absorption", linewidth=2)
    plt.legend()
    plt.show()



