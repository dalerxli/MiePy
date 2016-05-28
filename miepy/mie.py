import numpy as np
from miepy.scattering import multipoles
from miepy.special_functions import riccati_1,riccati_2

def sphere(nmax, mat, r, eps_b=1, mu_b=1):
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
