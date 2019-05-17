.. _features:

Features and supported software
===============================

APDFT does not perform any QM calculations itself, but rather uses well-tested existing codes. However, the generation of the (non-standard) input files and the calculation of all possible target molecule properties is done by APDFT.

Gaussian
--------

If you have a license for `Gaussian <http://gaussian.com/>`_, APDFT can calculate the following levels of theory, all of which can be combined with all Gaussian basis sets from the `EMSL Basis Set Exchange <https://www.basissetexchange.org/>`_.

+---------------------------+-----------+-----------------+
| Level of theory           | Energies  | Dipole Moments  |
+===========================+===========+=================+
| HF                        | Yes       | Yes             |
+---------------------------+-----------+-----------------+
| LDA                       | Yes       | Yes             |
+---------------------------+-----------+-----------------+
| PBE                       | Yes       | Yes             |
+---------------------------+-----------+-----------------+
| PBE0                      | Yes       | Yes             |
+---------------------------+-----------+-----------------+
| CCSD                      | Yes       | Yes             |
+---------------------------+-----------+-----------------+

Adding support for additional software packages
===============================================

To check whether it is possible to do APDFT with your favourite QM code, two requirements need to be checked.

APDFT requires a QM package to perform calculations with fractional nuclear charges but with a fixed basis set. This means that changing the nuclear charge of a site must not alter the basis set used. While some codes support this directly, some others require a workaround where you initialise the QM calculation with your reference molecule just to add point charges at the sites of the nuclei to counteract the Coulombic interaction between nuclei and electron density. Either of the methods is equivalent. Typically, it is advisable to compare energies of e.g. the Nitrogen dimer at the CO basis set and CO with CO basis set evaluated by defining an Nitrogen dimer input together with two point charges of +1 and -1, respectively.

The other requirement is that the QM code is able to write out the resulting electron density on a integration grid which only depends on nuclear coordinates but not nuclear charge. The dependence on nuclear coordinates is almost universal since integration grids need to model the core region of the electron density particularly well. It is important however, that a rectangular grid like in cube-files is not sufficient. APDFT requires to evaluate integrals over the change in electron density in space. Rectangular grids are not suitable, since they either are unnecessarily large or feature a resolution which is by far too low. The integration grid must not depend on the actual nuclear charges, since APDFT combines the self-consistent electron density of multiple runs and requires a identical integration grid for that purpose. In an ideal world, the grid coordinates can be specified to the QM code, but any fixed integration grid will do. If it is not possible to specify the required grid points to the QM code, an acceptable alternative is having the QM code list the grid points together with their weights and the corresponding electron density in the output. Another possibilty to fulfill this requirement is a MOLDEN file which contains the basis functions and the coefficients such that APDFT can calculate the spatial electron density.