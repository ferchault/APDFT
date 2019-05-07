.. _features:

Features and supported software
===============================

APDFT does not perform any QM calculations itself, but rather uses well-tested existing codes. However, the generation of the (non-standard) input files and the calculation of all possible target molecule properties is done by APDFT.

+---------------------------+----------------------+-------------------+
| Level of theory           | Energies             | Dipole Moments    |
+---------------------------+----------------------+-------------------+
| HF                        | Yes, [#gaussian]_    | Yes, [#gaussian]_ |
| LDA                       | Yes, [#gaussian]_    | Yes, [#gaussian]_ |
| PBE                       | Yes, [#gaussian]_    | Yes, [#gaussian]_ |
| PBE0                      | Yes, [#gaussian]_    | Yes, [#gaussian]_ |
| CCSD                      | Yes, [#gaussian]_    | Yes, [#gaussian]_ |
+---------------------------+----------------------+-------------------+

All calculations can be combined with all Gaussian basis sets from the `EMSL Basis Set Exchange <https://www.basissetexchange.org/>`_ can be used.

.. rubric:: Footnotes

.. [#gaussian] Using `Gaussian <http://gaussian.com/>`_