.. _features:

Features and supported software
===============================

APDFT does not perform any QM calculations itself, but rather uses well-tested existing codes. However, the generation of the (non-standard) input files and the calculation of all possible target molecule properties is done by APDFT.

.. include:: generated-featurematrix.rst

PySCF
-----
Recommended for testing. Free to use for both academic and commercial use, already installed if you install APDFT, so it works out-of-the-box. For large scale production runs, other codes might be more efficient.

Gaussian (G09)
--------------

If you have a license for `Gaussian <http://gaussian.com/>`_, APDFT can calculate the following levels of theory, all of which can be combined with all Gaussian basis sets from the `EMSL Basis Set Exchange <https://www.basissetexchange.org/>`_.

MRCC
----
This code is free to use for academic purposes (after registration). `MRCC <https://www.mrcc.hu/>`_ allows many high-level CC evaluations at very decent speed.


Adding support for additional software packages
===============================================

To check whether it is possible to do APDFT with your favourite QM code, two requirements need to be checked.

APDFT requires a QM package to perform calculations with fractional nuclear charges but with a fixed basis set. This means that changing the nuclear charge of a site must not alter the basis set used. While some codes support this directly, some others require a workaround where you initialise the QM calculation with your reference molecule just to add point charges at the sites of the nuclei to counteract the Coulombic interaction between nuclei and electron density. Either of the methods is equivalent. Typically, it is advisable to compare energies of e.g. the Nitrogen dimer at the CO basis set and CO with CO basis set evaluated by defining an Nitrogen dimer input together with two point charges of +1 and -1, respectively.

The other requirement is that the QM code is able to evaluate

.. math::

    \int_\Omega dr \frac{\rho(\mathbf{r})}{|\mathbf{r}-\mathbf{R}_I|}

either by writing out the resulting electron density on a integration grid which only depends on nuclear coordinates but not nuclear charge or by evaluating this integral directly, which is the preferred method.

In case a grid is needed, the dependence on nuclear coordinates is almost universal since integration grids need to model the core region of the electron density particularly well. It is important however, that a rectangular grid like in cube-files is not sufficient. APDFT requires to evaluate integrals over the change in electron density in space. Rectangular grids are not suitable, since they either are unnecessarily large or feature a resolution which is by far too low. The integration grid must not depend on the actual nuclear charges, since APDFT combines the self-consistent electron density of multiple runs and requires a identical integration grid for that purpose. In an ideal world, the grid coordinates can be specified to the QM code, but any fixed integration grid will do. If it is not possible to specify the required grid points to the QM code, an acceptable alternative is having the QM code list the grid points together with their weights and the corresponding electron density in the output. Another possibilty to fulfill this requirement is a MOLDEN file which contains the basis functions and the coefficients such that APDFT can calculate the spatial electron density.

Required interface
------------------

To add support for your favourite QM program, you need a new calculator class. This is done by adding a corresponding file in :py:mod:`apdft.calculator`, where a new class inherits from :class:`apdft.calculator.Calculator`. This class needs to implement the following functions:

- `get_input` for the generation of the corresponding input file. If necessary, there is a folder for input file templates that can help keep the code clean.
- `get_epn` to obtain the electronic electrostatic potential at each nucleus for each calculation.

Furthermore, the attribute `_methods` of the class needs to hold a dictionary of all supported levels of theory. Afterwards, the supported code needs to be registered in :class:`apdft.settings.CodeEnum` such that the user can select it via the command line interface. Finally, the command line handler in :func:`commandline.mode_energies` needs a selection branch for the new calculator.

Reference values
----------------

To see whether the electronic electrostatic potential at the nuclei is evaluated correctly, here are some reference numbers for the case of the Nitrogen dimer with a bond distance of 1 Angstrom.

+-----------------+-------------------+-------------------+---------------------+
| Nuclear charges | EPN site 1 [a.u.] | EPN site 2 [a.u.] | Total Energy [a.u.] |
+-----------------+-------------------+-------------------+---------------------+
| **Gaussian** (analytical evaluation), CCSD/6-31G                              |
+-----------------+-------------------+-------------------+---------------------+
| 7.00, 7.00      | 21.98896747469    | 21.98896747469    | -109.0418572        |
+-----------------+-------------------+-------------------+---------------------+
| 7.05, 7.00      | 22.05431347468    | 21.98919533522    | -109.9577300        |
+-----------------+-------------------+-------------------+---------------------+
| 6.95, 7.00      | 21.92294147469    | 21.98846761415    | -108.1292686        |
+-----------------+-------------------+-------------------+---------------------+
| 7.05, 7.05      | 22.05426133522    | 22.05426133522    | -110.8722842        |
+-----------------+-------------------+-------------------+---------------------+
| 6.95, 6.95      | 21.92217561415    | 21.92217561415    | -107.2153887        |
+-----------------+-------------------+-------------------+---------------------+
| **MRCC** (numerical evaluation), CCSD/6-31G                                   |
+-----------------+-------------------+-------------------+---------------------+
| 7.00, 7.00      | 21.98870584174    | 21.98870584174    | -109.0398258        |
+-----------------+-------------------+-------------------+---------------------+
| 7.05, 7.00      | 22.05398872497    | 21.98892949523    | -109.9557470        |
+-----------------+-------------------+-------------------+---------------------+
| 6.95, 7.00      | 21.92274858697    | 21.98820947247    | -108.1271854        |
+-----------------+-------------------+-------------------+---------------------+
| 7.05, 7.05      | 22.05393336588    | 22.05393336588    | -110.8703496        |
+-----------------+-------------------+-------------------+---------------------+
| 6.95, 6.95      | 21.92198579936    | 21.92198579936    | -107.2132536        |
+-----------------+-------------------+-------------------+---------------------+
| **PySCF** (analytical evaluation), CCSD/6-31G                                 |
+-----------------+-------------------+-------------------+---------------------+
| 7.00, 7.00      | 21.99292179017    | 21.99292179017    | -109.0418571        |
+-----------------+-------------------+-------------------+---------------------+
| 7.05, 7.00      | 22.05794809640    | 21.99345693140    | -109.9577300        |
+-----------------+-------------------+-------------------+---------------------+
| 6.95, 7.00      | 21.92718607388    | 21.99212675698    | -108.1292685        |
+-----------------+-------------------+-------------------+---------------------+
| 7.05, 7.05      | 22.05821304142    | 22.05821304140    | -110.8722841        |
+-----------------+-------------------+-------------------+---------------------+
| 6.95, 6.95      | 21.92613340256    | 21.92613340258    | -107.2153886        |
+-----------------+-------------------+-------------------+---------------------+
