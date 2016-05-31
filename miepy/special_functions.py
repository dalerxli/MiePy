"""
special_functions defines any additional special functions needed by MiePy
"""

import numpy as np
from scipy import special

def riccati_1(nmax,x):
    """Riccati bessel function of the 1st kind

       returns (r1, r1'), n=0,1,...,nmax"""

    jn,jnp = special.sph_jn(nmax,x)

    r0 = x*jn
    r1 = jn + x*jnp
    return np.array([r0,r1])

def riccati_1_single(n,x):
    """Riccati_1, but only a single n value"""
    pre = (np.pi*x/2)**.5
    jn = pre*special.jv(n+0.5,x)
    jnp = jn/(2*x) + pre*special.jvp(n+0.5,x)

    return np.array([jn,jnp])

def riccati_2(nmax,x):
    """Riccati bessel function of the 2nd kind

       returns (r2, r2'), n=0,1,...,nmax"""

    jn,jnp,yn,ynp = special.sph_jnyn(nmax,x)
    hn = jn + 1j*yn
    hnp = jnp + 1j*ynp

    r0 = x*hn
    r1 = hn + x*hnp
    return np.array([r0,r1])


def riccati_2_single(n,x):
    """Riccati_2, but only a single n value"""
    pre = (np.pi*x/2)**.5
    hn = pre*special.hankel1(n+0.5,x)
    hnp = hn/(2*x) + pre*special.h1vp(n+0.5,x)

    return np.array([hn,hnp])

def riccati_3(nmax,x):
    """Riccati bessel function of the 3rd kind

       returns (r3, r3'), n=0,1,...,nmax"""

    yn,ynp = special.sph_yn(nmax,x)

    r0 = x*yn
    r1 = yn + x*ynp
    return np.array([r0,r1])

def riccati_3_single(n,x):
    """Riccati_3, but only a single n value"""
    # pre = (np.pi*x/2)**.5
    # yn = pre*special.yv(n+0.5,x)
    # ynp = yn/(2*x) + pre*special.yvp(n+0.5,x)

    # return np.array([yn,ynp])
    return riccati_2_single(n,x) - riccati_1_single(n,x)

