"""
MiePy
=======
Python module to calcuate scattering coefficients of a plane wave incident on a sphere or core-shell structure using Mie theory
"""

#main submodules
from . import scattering
from . import materials
from . import mie_sphere
from . import mie_core_shell
from . import array_io

from .mie_sphere import sphere
from .mie_core_shell import core_shell
from .materials import material, load_material

