Installation
============

APDFT does not perform calculations itself. You will need access to one of the supported QM codes as listed [here]. However, it is not required to install APDFT on the same machine as the QM code. This separation is often useful since the calculations need to run on a compute cluster while it is more convenient to do pre-processing interactively on a workstation. This is particularly true if the version of python on the compute cluster is not recent enough.

Installing APDFT
----------------

First, clone APDFT from github:

.. code ::

    git clone https://github.com/ferchault/APDFT.git
    cd APDFT

APDFT requires python 3.6 or newer. If that is met, all dependencies can be installed as follows:

.. code ::
    
    pip3 install jinja2 basis_set_exchange Cython numpy scipy h5py pyscf cclib structlog colorama

Since one of the dependencies, orbkit, is not pre-packaged, it needs to be installed manually. To that end, please run

.. code ::

    cd dep; . install.sh; cd ../../../

which will download and install the latest version of orbkit.

Finally, the paths need to be set correctly:

.. code ::

    echo 'export PATH="'$(realpath src)':$PATH"' >> ~/.bashrc
    echo 'export PYTHONPATH="'$(realpath src)':$PYTHONPATH"' >> ~/.bashrc
    source ~/.bashrc

Now you are ready to run *apdft.py* on the command line. Have a look at :ref:`Getting Started` to see an example.