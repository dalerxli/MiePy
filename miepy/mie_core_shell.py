"""
mie_core_shell calculates the scattering coefficients of a core-shell structure using Mie theory
"""

import numpy as np
from miepy.scattering import multipoles
from miepy.special_functions import riccati_1,riccati_2,riccati_3

def M_matrix():
    pass

def b_values():
    pass

def get_index(mat):
    return (mat.eps*met.mu)**.5

def core_shell(nmax, mat_in, mat_out,  r_in, r_out, eps_b=1, mu_b=1):
    """Determine an, bn coefficients for a core-shell structure

           nmax      = maximum number of orders to use
           mat_in    = material object for core
           r_in      = radius of core
           mat_out   = material object for shell
           r_out     = radius of core+shll (r_out >= r_in)
           eps_b     = background permittivity
           mu_b      = background permeability

       Returns multipoles object"""

    k = mat_in.k
    Nfreq = len(k)
    nb = (eps_b*mu_b)**.5
    m1 = get_index(mat_in)/nb
    m2 = get_index(mat_out)/nb

    xvals = k*r_in*n_b
    yvals = k*r_out*n_b

    an = np.zeros((nmax,mat.Nfreq), dtype=np.complex)
    bn = np.zeros((nmax,mat.Nfreq), dtype=np.complex)

    for i in range(Nfreq):
        M = M_matrix()
        b = b_values()

        a = (mt*jnm*jn_p - jn*jnm_p)/(mt*jnm*yn_p - yn*jnm_p)
        b = (jnm*jn_p - mt*jn*jnm_p)/(jnm*yn_p - mt*yn*jnm_p)
        an[:,i] = a[1:]
        bn[:,i] = b[1:]
    
    an = np.nan_to_num(an)
    bn = np.nan_to_num(bn)
    return multipoles(mat_in.wav,an,bn) 
