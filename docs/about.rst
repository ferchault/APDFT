APDFT
=====

APDFT is a software to allow quantum-chemistry calculations of many isoelectronic molecules at once rather than evaluating them one-by-one. This is achieved through *Alchemical Perturbation Density Functional Theory* [APDFT]_ where the change in external potential between molecules is treated as perturbation. This concept works just as fine for post-HF methods.

All gaussian basis sets from the `EMSL Basis Set Exchange <https://www.basissetexchange.org/>`_ and a variety of methods (HF, LDA, PBE, PBE0, CCSD) are supported. APDFT does not reinvent the wheel but leverages other QM software in the background. Currently, we support Gaussian as only backend. For more details, please see :ref:`features`.

.. Note::
   APDFT is under development. While the software is ready to use, the API may be subject to change.


.. [APDFT] https://doi.org/10.1103/PhysRevResearch.2.023220
