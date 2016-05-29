"""
MiePy
=======
Python module to calcuate scattering coefficients of a plane wave incident on a sphere or core-shell structure using Mie theory
"""


#main submodules
from . import scattering
from . import materials
from . import mie_sphere
from .mie_sphere import sphere
from .materials import material, load_material
