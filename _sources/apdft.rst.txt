Implementation
==============

In the context of APDFT, we evaluate energy differences by

.. math::

    \Delta E = \int_\Omega d\mathbf{r} \Delta V(\mathbf{r})\tilde\rho(\mathbf{r})

where :math:`\tilde\rho` is obtained by means of finite difference evaluations and the change in external potential :math:`\Delta V` is given by the Coulomb term

.. math::

    \Delta V(\mathbf{r}) = \sum_I \frac{\Delta Z_I}{|\mathbf{r} - \mathbf{R_I}|} = \sum_I  \Delta V_I(\mathbf{r})

running over all nuclei :math:`I`. :math:`\tilde\rho` in fact is a linear combination of individual electron densities for fractional nuclear charges, so we can write is as follows

.. math::

    \tilde\rho(\mathbf{r}) = \sum_i \alpha_i\rho_i(\mathbf{r})

where the coefficients :math:`\alpha_i` are determined by the finite difference scheme employed. 

Energies
--------

Together with the above equations for :math:`\Delta V_I`, the expression for the change in energy can be re-cast

.. math::

    \Delta E &= \int_\Omega d\mathbf{r} \left(\sum_I \frac{\Delta Z_I}{|\mathbf{r} - \mathbf{R_I}|}\right) \left(\sum_i \alpha_i\rho_i(\mathbf{r})\right) \\
             &= \sum_I\sum_i \Delta Z_I\alpha_i \int_\Omega d\mathbf{r} \frac{\rho_i(\mathbf{r})}{|\mathbf{r} - \mathbf{R_I}|}

which is much more efficient to implement, since the (costly) evaluation of the integral in space only needs to be done once for every combination of density :math:`\rho_i` and nucleus :math:`I`. The latter term corresponds to the electrostatic potential at the nucleus (EPN) which can be calculated analytically in many codes (e.g. psi4).

This can be further simplified as an outer product, i.e. :math:`\mathbf{\Delta Z}=\{\Delta Z_I\}`:

.. math::

    \Delta E = \sum_{jk} \left(\left[\mathbf{\Delta Z}\otimes \mathbf{\alpha}\right]\circ \mathbf{Q}\right)_{jk}

where :math:`\mathbf{Q}_{Ii}` is just the electrostatic potential of density :math:`i` at nucleus :math:`I`. This formulation is numerically efficient to implement, since it only requires the QM codes to export :math:`\mathbf{Q}` instead of an explicit density grid. Moreover, since the density matrix does not need to be exported, interfacing is more reliable because no basis function ordering is relevant to the results.

Density Moments
---------------
The electronic component of the charge distribution moments can be obtained from a linear combination of the same quantities evaluated for the individual single point calculations from the finite difference scheme. For example, for the dipole moment, we have

.. math::

    \mu_i = \int_\Omega d\mathbf{r} \rho(\mathbf{r})\mathbf{r}_i

Similarly to the energy, the target density :math:`\rho_t` can be build from coefficients of the individual calculations in the context of the finite difference scheme. Note that these coefficients :math:`\beta_i` are not the same as for the energy expression.

.. math::

    \rho_t(\mathbf{r}) = \sum_i \beta_i\rho_i(\mathbf{r})

For the dipole moment, this results in a linear combination of the dipole moments of the individual single point calculations

.. math::

    \mu_i = \sum_i \beta_i \int_\Omega d\mathbf{r} \rho_i(\mathbf{r}) \mathbf{r}_i

Note that this only applies to the electronic component of the charge distribution moments.

Ionic Forces
------------
Similarly to dipole moments, the expression for the ionic force allows linear combinations:

.. math::

    \mathbf{F}_I &= Z_I\int_\Omega d\mathbf{r} \rho_t(\mathbf{r})\frac{\mathbf{r} - \mathbf{R}_I}{|\mathbf{r} - \mathbf{R}_I|^3}\\
                &=Z_I\sum_i \beta_i\int_\Omega d\mathbf{r} \rho_i(\mathbf{r})\frac{\mathbf{r} - \mathbf{R}_I}{|\mathbf{r} - \mathbf{R}_I|^3}\\


Calculators
-----------

.. automodule:: apdft.calculator
    :members:
    :undoc-members:
    :show-inheritance:

Gaussian
^^^^^^^^

.. automodule:: apdft.calculator.gaussian
    :members:
    :undoc-members:
    :show-inheritance:

MRCC
^^^^

.. automodule:: apdft.calculator.gaussian
    :members:
    :undoc-members:
    :show-inheritance:

Derivatives
-----------

.. automodule:: apdft.Derivatives
    :members:
    :undoc-members:
    :show-inheritance:

Physics-related functions
-------------------------

.. automodule:: apdft.physics
    :members:
    :undoc-members:


Math-related functions
----------------------

.. automodule:: apdft.math
    :members:
    :undoc-members:

APDFT logic
-----------

.. automodule:: apdft
    :members:
    :undoc-members:
    :show-inheritance:

Command Line Interface
----------------------

.. automodule:: apdft.commandline
    :members:
    :undoc-members:
    :show-inheritance:

Settings
--------

.. automodule:: apdft.settings
    :members:
    :undoc-members:
    :show-inheritance: