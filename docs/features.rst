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
