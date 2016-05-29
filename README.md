MiePy
==============
MiePy is a Python module to calculate scattering and absorption properties of spheres and core-shell structures. Mie theory is used, following the procedure in "Absorption and Scattering of Light by Small Particles" by Bohren & Huffman.

The material can be specified by a frequency dependent permittivity and permeability.


Images
--------------
In the image below, scattering intensity is shown for a 100 nm radius dielectric sphere with refractive index n = 3.7. Individual contributions from different multipoles are also shown (eD = electric dipole, etc.).

<p align="center">
  <img src="images/sphere_scattering.png?raw=true" width="600">
</p>


Dependencies
--------------
Required Python modules: numpy, scipy, matplotlib, sympy (for core-shell)

For installation of these, see https://www.scipy.org/install.html


License
--------------
MiePy is licensed under the terms of the MIT license.
