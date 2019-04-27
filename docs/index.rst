multiQM
=======

multiQM is a software to allow quantum-chemistry calculations of many isoelectronic molecules at once rather than evaluating them one-by-one. This is achieved through *Alchemical Perturbation Density Functional Theory* [APDFT]_ where the change in external potential between molecules is treated as perturbation.

All gaussian basis sets from the `EMSL Basis Set Exchange <https://www.basissetexchange.org/>`_ and a variety of methods (HF, LDA, PBE, PBE0, CCSD) are supported. multiQM does not reinvent the wheel but leverages other QM software in the background. Currently, we support Gaussian as only backend.

.. Note::
   multiQM is under development. While the software is ready to use, the API may be subject to change.


.. [APDFT] https://arxiv.org/abs/1809.01647


.. toctree::
   :maxdepth: 2

   mqm
   multiQM

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
