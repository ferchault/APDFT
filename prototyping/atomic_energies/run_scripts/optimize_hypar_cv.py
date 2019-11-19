#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 15:43:49 2019

@author: misa

performs hyperparameter optimization, the labels can be specified in pl
"""

import sys
sys.path.insert(0, '/home/misa/APDFT/prototyping/atomic_energies/')

import qml_interface as qi

base='/home/misa/APDFT/prototyping/atomic_energies/results/slice_ve38/'
pl = ['alch_pot', 'atomic', 'atomisation']
tr_set = 512
num_cv = 10
results = []

for p in pl:
    # load data into list, count number of atoms per molecule
    paths=qi.wrapper_alch_data()
    alchemy_data, molecule_size = qi.load_alchemy_data(paths)
    
    # local data
    local_reps = qi.generate_atomic_representations(alchemy_data, molecule_size)
    local_labels = qi.generate_label_vector(alchemy_data, molecule_size.sum(), value=p)
    
    opt_sigma, opt_lambda, min_error, std = qi.optimize_hypar_cv(local_reps, local_labels, tr_set, molecule_size, num_cv)
    
    together = (opt_sigma, opt_lambda, min_error, std.mean())
    results.append(together)

file = '/home/misa/APDFT/prototyping/atomic_energies/results/analyse_learning/optimized_hyperpar_'

for idx,el in enumerate(results):
    with open(file+pl[idx]+'_mic.txt', 'w') as f:
        f.write('optimized hyperparameters for label: {}; training set size: {}; number of crossvalidations: {}\n'.format(pl[idx], tr_set, num_cv))
        f.write('opt_sigma \t opt_lambda \t mean_error \t std\n')
        f.write('{}\t{}\t{}\t{}'.format(el[0], el[1], el[2], el[3]))