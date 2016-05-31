"""
mie_core_shell calculates the scattering coefficients of a core-shell structure using Mie theory
"""

import numpy as np
from miepy.scattering import multipoles
from miepy.special_functions import riccati_1,riccati_2,riccati_3
R1 = riccati_1
R2 = riccati_2
R3 = riccati_3

def M_matrix(m1,m2,x,y,mu,mu1,mu2,n):
    
    M = np.zeros([8,8], dtype=np.complex)
    M[0] = np.array([0,0, -m2*R1(n,m1*x)[0][n],0, m1*R1(n,m2*x)[0][n],0, -m1*R3(n,m2*x)[0][n],0])
    M[1] = np.array([0,0,0, m2*R1(n,m1*x)[1][n],0, -m1*R1(n,m2*x)[1][n],0, m1*R3(n,m2*x)[1][n]])
    M[2] = np.array([0,0, mu2*R1(n,m1*x)[1][n],0, -mu1*R1(n,m2*x)[1][n],0, mu1*R3(n,m2*x)[1][n],0])
    M[3] = np.array([0,0,0, -mu2*R1(n,m1*x)[0][n],0, mu1*R1(n,m2*x)[0][n],0, -mu1*R3(n,m2*x)[0][n]])
    M[4] = np.array([-m2*R2(n,y)[1][n],0,0,0,0, -R1(n,m2*y)[1][n],0, R3(n,m2*y)[1][n]])
    M[5] = np.array([0, m2*R2(n,y)[0][n],0,0, R1(n,m2*y)[0][n],0, -R3(n,m2*y)[0][n],0])
    M[6] = np.array([-mu2*R2(n,y)[0][n],0,0,0,0, -mu*R1(n,m2*y)[0][n],0, mu*R3(n,m2*y)[0][n]])
    M[7] = np.array([0, mu2*R2(n,y)[1][n],0,0, mu*R1(n,m2*y)[1][n],0, -mu*R3(n,m2*y)[1][n],0])

    return M

def c_values(m2,y,mu2,n):

    c = np.zeros(8, dtype=np.complex)
    c = np.array([0,0,0,0, -m2*R1(n,y)[1][n], m2*R1(n,y)[0][n], -mu2*R1(n,y)[0][n], mu2*R1(n,y)[1][n]])
    return c

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

    for i in range(Nfreq):
        for n in range(nmax):
            M = M_matrix(m1[i],m2[i],xvals[i],yvals[i],mu_b,mat_in.mu[i],mat_out.mu[i],n+1)
            c = c_values(m2[i],yvals[i],mat_out.mu[i],n+1)
            sol = np.linalg.solve(M,c)
            a = sol[0]
            b = sol[1]

            an[n,i] = a
            bn[n,i] = b
    
    an = np.nan_to_num(an)
    bn = np.nan_to_num(bn)
    return multipoles(mat_in.wav,an,bn) 
