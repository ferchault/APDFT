#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 16:22:58 2020

@author: misa
"""

import numpy as np

import qml
import qml.fchl

import sys
sys.path.insert(0,'/home/misa/APDFT/prototyping/atomic_energies/')
import qml_interface as qmi

data, molecule_size = qmi.load_alchemy_data(qmi.wrapper_alch_data())

local_reps = qmi.generate_atomic_representations(data, molecule_size)

    
savebase = '/home/misa/APDFT/prototyping/atomic_energies/results/analyse_learning/coulomb/'
sigmas = np.logspace(-1, 4, 12).tolist()

for sigma in sigmas:
    print('Computing kernel for sigma = {}'.format(sigma))
    out = qml.kernels.gaussian_kernel_symmetric(local_reps, sigma)

    print('Writing kernel for sigma = {} to disk'.format(sigma))
    np.savetxt(savebase+'full_kernel_alchoff_sig{}'.format(sigma), out)
    
