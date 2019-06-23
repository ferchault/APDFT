Usage
=====

Getting Started
---------------

In the following, the simplest possible workflow is described. More complex topics are discussed afterwards.

First, we need the reference molecule. In the current directory, which is treated as working directory by apdft, create a XYZ file of any name. In this example, we will call it *molecule.xyz*. Once this file is available, call

.. code-block:: bash

    apdft energies molecule.xyz

which will build a set of input folders and a configuration file, :ref:`apdft.conf <apdft_conf>`. The latter contains a complete list of all configuration options the current version of apdft understands. apdft itself does not perform any QM calculations, but rather relies on well-tested software packages like Gaussian or MRCC to do so. The file :ref:`commands.sh <commands_sh>` contains the required program calls for each QM calculation. Running it will perform all required calculations and ensure availability of the appropriate output files:

.. code-block:: bash

    . commands.sh

With the QM calculations completed, apdft can use the results to predict energies of related molecules:

.. code-block:: bash

    apdft energies

which will use the QM results to predict energies of target molecules. The results are then stored in :ref:`energies.csv <energies_csv>`.


Input Files and Output Files
----------------------------

.. _apdft_conf:

apdft.conf
    The configuration file. Is automatically generated upon the first invocation of apdft.

.. _commands_sh:

commands.sh
    All required calls to QM software like Gaussian and MRCC to obtain the energies and electron densities apdft needs. This file assumes standard paths for each of the software package. If you system differs from that for any reason, you might need to adjust environment variables accordingly. Each line contains all required commands for a single calculation, therefore they can be executed independently from each other.

.. _folder_order:

order-*
    Folders containing the generated input files for the QM software packages. The folder structure needs to be kept the same.

Running on a Compute Cluster
----------------------------

Since a (modern) python installation is not necessarily common on compute clusters, we assume that apdft is run on a local workstation while the expensive QM calculations are run on a compute cluster. In practice, most of the calculations will be too heavy for a single machine, in particular as APDFT benefits from performing the reference calculation at a higher level of theory. The input files can easily be generated on a regular machine, as this is not computationally expensive. 

To move the heavy QM calculations to a more powerful environment, all :ref:`order-* <folder_order>` folders and the :ref:`commands.sh <commands_sh>` file should be copied to a compute cluster. Most likely, you want to run the individual calculations as separate jobs in a queueing system. Each line in that file contains the workload of one job and is independent of all other lines in the file. Once the calculations have been completed, all newly created files and folders have to be transferred back to the initial machine for post-processing.

