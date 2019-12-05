#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 11:25:12 2019

@author: misa

can be used to generate learning curves as long as hyperparameters are provided

hypar: specifies file where hyperparameters can be found

"""

import qml
import numpy as np

import sys
sys.path.insert(0, '/home/misa/APDFT/prototyping/atomic_energies/')
import alchemy_tools as alch
import qml_interface as qi

def save(set_sizes, error, std, name):
    big_array = np.array([set_sizes, error, std]).T
#    big_array = np.flip(big_array, axis=0)
    fname = '/home/misa/APDFT/prototyping/atomic_energies/results/analyse_learning/'+name
    np.savetxt(fname, big_array, delimiter='\t')

# load data (with mic)
paths=qi.wrapper_alch_data()
alchemy_data, molecule_size = qi.load_alchemy_data(paths)

# prepare representations and labels
local_reps = qi.generate_atomic_representations(alchemy_data, molecule_size)
local_labels = qi.generate_label_vector(alchemy_data, molecule_size.sum(), value='atomic')

# read optimized hyperparameters
hypar = '/home/misa/APDFT/prototyping/atomic_energies/results/analyse_learning/mic/optimized_hyperpar_atomic_mic.txt'
with open(hypar, 'r') as f:
    lines = f.readlines()
sigma = float(lines[2].split('\t')[0])
lam = float(lines[2].split('\t')[1])

# define number of training points for which MAE is calculated
set_sizes = np.logspace(0,9, 10, base=2).astype(int)

# calculate error for every training point size
error_cv = np.zeros(len(set_sizes))
error_std = np.zeros(len(set_sizes))
for idx, tr_size in enumerate(set_sizes):
        error_cv[idx], error_std[idx] = qi.crossvalidate(local_reps, local_labels, molecule_size, tr_size, sigma, lam, molecule=False, num_cross=10)



#if global_ == True:
#    
#    # optimized hyperparameters
#    hypar = '/home/misa/APDFT/prototyping/atomic_energies/results/analyse_learning/optimized_hyperparameters_global_rep.txt'
#    with open(hypar, 'r') as f:
#        lines = f.readlines()
#    sigma = float(lines[2].split('\t')[0])
#    lam = float(lines[2].split('\t')[1])
#    error_cv = np.zeros(len(set_sizes))
#    error_std = np.zeros(len(set_sizes))
#
#    # global data
#    global_reps = qi.wrapper_global_representations(alchemy_data, molecule_size, rep_par='coulomb')
#    global_labels = qi.generate_label_vector(alchemy_data, len(global_reps), value='atomisation_global')
#    
#    
#    for idx, tr_size in enumerate(set_sizes):
#        error_cv[idx], error_std[idx] = qi.crossvalidate_new(global_reps, global_labels, molecule_size, tr_size, sigma, lam, local=False, num_cross=50)