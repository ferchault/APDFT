#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 17:33:48 2020

@author: misa

calculate FCHL kernels for different sigma values
"""

import numpy as np

import qml
import qml.fchl

import sys
sys.path.insert(0,'/home/misa/APDFT/prototyping/atomic_energies/')
import qml_interface as qmi
import qml_interface2 as qmi2

data, molecule_size = qmi.load_alchemy_data(qmi.wrapper_alch_data())
reps =[]
for i in zip(data, molecule_size):
    rep_fchl = qml.fchl.generate_representation(i[0][:, 1:4], i[0][:,0], max_size=i[1])
    reps.append(rep_fchl)
    
savebase = '/home/misa/APDFT/prototyping/atomic_energies/results/analyse_learning/FCHL/'
sigmas = np.logspace(-1, 4, 12).tolist()

#sigmas = [432.8761281083057, 10000]
for sigma in sigmas:
    print('Computing kernel for sigma = {}'.format(sigma))
    out = qmi2.generate_fchl_atomic_kernel(reps, molecule_size, sigma)

    print('Writing kernel for sigma = {} to disk'.format(sigma))
    np.savetxt(savebase+'full_kernel_sig{}'.format(sigma), out)