"""
scattering defines class multipoles, used to analyze scattering & absorption
properties of some object given its an, bn coefficients
"""

import numpy as np
import matplotlib.pyplot as plt
import miepy


label_map = {(0,0): 'eD', (0,1): 'eQ', (0,2): 'eO',
         (1,0): 'mD', (1,1): 'mQ', (1,2): 'mO'}

def get_label(i,n):
    """Get multipole label.
            i = 0 (electric), 1(magnetic)
            n = 1,2,3... (order)"""

    mtype = 'e' if i == 0 else 'm'
    if n < 3:
        label = label_map[(i,n)]
    else:
        label = mtype + str(2**(n+1))
    return label

class multipoles:
    """Contains an and bn as function of wavelength
       Used to calculate scattering/absorption properties"""

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

    def scattering_array(self, nmax=None):
        """Get modal scattering intensity for all modes
           Return scat[Nfreq,2,nmax]"""

        if nmax == None:
            nmax = np.inf
        nmax = min(self.an.shape[0], nmax)

        scat = np.zeros([self.Nfreq,2,nmax])
        for n in range(1,nmax+1):
            scat[:,0,n-1] = self.mode_scattering('e', n)
            scat[:,1,n-1] = self.mode_scattering('m', n)
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


    def plot_scattering_modes(self, nmax): 
        """Plot scattering due to each mode up to nmax"""

        modes = self.scattering_array()
        m_nmax = modes.shape[1]
        nmax = min([nmax, m_nmax])

        for i,mtype in enumerate(('e','m')):
            for n in range(nmax):
                label = get_label(i,n)
                plt.plot(self.wav, modes[:,i,n], linewidth=2, label=label)

        plt.legend()

    def save(self, filename, nmax = 0, delimiter='\t'):
        wav = self.wav
        S,A = self.scattering()
        modes = self.scattering_array(nmax)

        out_data = [wav,S,A]
        header = "Wavelength (nm){0}Scattering{0}Absorption"
        for i in range(modes.shape[1]):
            for j in range(modes.shape[2]):
                out_data.append(modes[:,i,j])
                header += "{0}" + get_label(i,j)

        header = header.format(delimiter)
        out = np.array(out_data).T
        miepy.array_io.save(filename, out, delimiter, header)

