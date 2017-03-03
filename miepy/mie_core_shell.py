"""
mie_core_shell calculates the scattering coefficients of a core-shell structure using Mie theory
"""

import numpy as np
from miepy.scattering import multipoles
from miepy.special_functions import riccati_1_single,riccati_2_single,riccati_3_single
R1 = riccati_1_single
R2 = riccati_2_single
R3 = riccati_3_single

def M_matrix(m1,m2,x,y,mu,mu1,mu2,n):
    
    M = np.zeros([8,8,len(m1)], dtype=np.complex)
    z = np.zeros(len(m1))
    M[0] = np.array([z,z, -m2*R1(n,m1*x)[0],z, m1*R1(n,m2*x)[0],z, -m1*R3(n,m2*x)[0],z])
    M[1] = np.array([z,z,z, m2*R1(n,m1*x)[1],z, -m1*R1(n,m2*x)[1],z, m1*R3(n,m2*x)[1]])
    M[2] = np.array([z,z, mu2*R1(n,m1*x)[1],z, -mu1*R1(n,m2*x)[1],z, mu1*R3(n,m2*x)[1],z])
    M[3] = np.array([z,z,z, -mu2*R1(n,m1*x)[0],z, mu1*R1(n,m2*x)[0],z, -mu1*R3(n,m2*x)[0]])
    M[4] = np.array([-m2*R2(n,y)[1],z,z,z,z, -R1(n,m2*y)[1],z, R3(n,m2*y)[1]])
    M[5] = np.array([z, m2*R2(n,y)[0],z,z, R1(n,m2*y)[0],z, -R3(n,m2*y)[0],z])
    M[6] = np.array([-mu2*R2(n,y)[0],z,z,z,z, -mu*R1(n,m2*y)[0],z, mu*R3(n,m2*y)[0]])
    M[7] = np.array([z, mu2*R2(n,y)[1],z,z, mu*R1(n,m2*y)[1],z, -mu*R3(n,m2*y)[1],z])

    return np.transpose(M, (2,0,1))

def c_values(m2,y,mu2,n):

    z = np.zeros(len(m2))
    c = np.zeros([8,len(m2)], dtype=np.complex)
    c = np.array([z,z,z,z, -m2*R1(n,y)[1], m2*R1(n,y)[0], -mu2*R1(n,y)[0], mu2*R1(n,y)[1]])
    return np.transpose(c)

def get_index(mat):
    return (mat.eps*mat.mu)**.5

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

    xvals = k*r_in*nb
    yvals = k*r_out*nb

    an = np.zeros((nmax,Nfreq), dtype=np.complex)
    bn = np.zeros((nmax,Nfreq), dtype=np.complex)

    for n in range(nmax):
        M = M_matrix(m1,m2,xvals,yvals,mu_b,mat_in.mu,mat_out.mu,n+1)
        c = c_values(m2,yvals,mat_out.mu,n+1)
        sol = np.linalg.solve(M,c)
        a = sol[:,0]
        b = sol[:,1]

        an[n,:] = a
        bn[n,:] = b
    
    an = np.nan_to_num(an)
    bn = np.nan_to_num(bn)
    return multipoles(mat_in.wav, nb, an,bn) 
