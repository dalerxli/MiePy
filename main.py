import numpy as np
import matplotlib.pyplot as plt
from scipy import special

eps_b = 1
mu_b  = 1
n_b = (eps_b)**.5
eps = -4
mu  = -4
r = 200    #in nm
Nfreq = 1000
wav = np.linspace(400,3000,Nfreq)
k = 2*np.pi/wav

def riccati_1(nmax,x):
    jn,jnp = special.sph_jn(nmax,x)

    r0 = x*jn
    r1 = jn + x*jnp
    return r0,r1

def riccati_2(nmax,x):
    jn,jnp,yn,ynp = special.sph_jnyn(nmax,x)
    hn = jn + 1j*yn
    hnp = jnp + 1j*ynp

    r0 = x*hn
    r1 = hn + x*hnp
    return r0,r1

def get_ab(nmax):
    xvals = k*r*n_b
    m = (eps/eps_b)**.5
    mt = m*mu_b/mu

    an = np.zeros((nmax,Nfreq), dtype=np.complex)
    bn = np.zeros((nmax,Nfreq), dtype=np.complex)

    for i,x in enumerate(xvals):
        jn,jn_p = riccati_1(nmax,x)
        jnm,jnm_p = riccati_1(nmax,m*x)
        yn,yn_p = riccati_2(nmax,x)
        a = (mt*jnm*jn_p - jn*jnm_p)/(mt*jnm*yn_p - yn*jnm_p)
        b = (jnm*jn_p - mt*jn*jnm_p)/(jnm*yn_p - mt*yn*jnm_p)
        an[:,i] = a[1:]
        bn[:,i] = b[1:]
    
    an = np.nan_to_num(an)
    bn = np.nan_to_num(bn)
    return an,bn 

def mode_scattering(cn, n):
    return 2*np.pi*(2*n+1)*np.abs(cn[n-1])**2/k**2


def scattering(an,bn):
    nmax = an.shape[0]
    nvals = 2*np.arange(1,nmax+1) + 1
    nvals = np.expand_dims(nvals,axis=1)
    sum_val_1 = np.sum(nvals*(np.abs(an)**2 + np.abs(bn)**2),axis=0)
    sum_val_2 = np.sum(nvals*np.real(an+bn),axis=0)
    scat = 2*np.pi*sum_val_1/k**2
    extinc = 2*np.pi*sum_val_2/k**2
    absorb = extinc - scat
    return scat, absorb


an,bn = get_ab(10)

plt.figure(1)
C,A = scattering(an,bn)
CeD = mode_scattering(an,1)
CeQ = mode_scattering(an,2)
CmD = mode_scattering(bn,1)
CmQ = mode_scattering(bn,2)
CeO = mode_scattering(an,3)
CmO = mode_scattering(bn,3)
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



