#!/bin/bash
test -d orbkit && exit 0
mkdir orbkit
cd orbkit
wget https://github.com/orbkit/orbkit/archive/cython.zip
unzip cython.zip
mv orbkit-cython orbkit
export ORBKITPATH=$PWD/orbkit
cd $ORBKITPATH
python3 setup.py build_ext --inplace clean
