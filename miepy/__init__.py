"""
MiePy
=======
Python module to calcuate scattering coefficients of a plane wave incident on a sphere or core-shell structure using Mie theory
"""

#main submodules
from . import mie_sphere
from . import mie_core_shell
from . import particles
from . import sources
from . import scattering
from . import material_functions
from . import materials

from .mie_sphere import single_mie_sphere
from .mie_core_shell import single_mie_core_shell
from .material_functions import material
from .particles import particle, particle_system
from .scattering import scattering_per_multipole, absorbption_per_multipole, \
                        extinction_per_multipole, cross_sections