"""
array_io defines functions to save and load numpy arrays in a format convenient for MiePy.

Any numpy array can be saved or loaded, but more commonly called functions would be 
load_material and save_material in materials, and multipoles.save in scattering.
"""
import os
import numpy as np

def save(filename, out, delimiter='\t', header=""):
    """Save numpy array 'out' at filename. If filename is .npy, file is binary, otherwise text"""

    file_extension = os.path.splitext(filename)[1]
    if file_extension == ".npy":
        np.save(filename, out)
    else:
        np.savetxt(filename, out, fmt = "%e", delimiter=delimiter, header=header)

def load(filename):
    """Load numpy array at filename. Accepts text or .npy files
       Return numpy array"""

    file_extension = os.path.splitext(filename)[1]
    if file_extension == ".npy":
        return np.load(filename)
    else:
        return np.loadtxt(filename)
