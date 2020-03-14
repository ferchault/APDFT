#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 19:57:25 2019

@author: misa
"""

import qml
import numpy as np

import sys
sys.path.insert(0, '/home/misa/APDFT/prototyping/atomic_energies/')
import alchemy_tools as alch
import qml_interface as qi


# load data into list, count number of atoms per molecule
paths=qi.wrapper_alch_data()
alchemy_data, molecule_size = qi.load_alchemy_data(paths)

# local data
local_reps = qi.generate_atomic_representations(alchemy_data, molecule_size)
local_labels = qi.generate_label_vector(alchemy_data, molecule_size.sum(), value='atomisation')
local_labels_shifted = qi.shift_by_mean_energy(local_reps, local_labels)

set_sizes = np.logspace(9, 0, 10, base=2).astype(int)
# optimized hyperparameters
hypar = '/home/misa/APDFT/prototyping/atomic_energies/results/analyse_learning/optimized_hyperparameters.txt'
with open(hypar, 'r') as f:
    lines = f.readlines()
sigma = float(lines[3].split('\t')[0])
lam = float(lines[3].split('\t')[1])
error_cv = np.zeros(len(set_sizes))
error_std = np.zeros(len(set_sizes))

for idx, tr_size in enumerate(set_sizes):
    error_cv[idx], error_std[idx] = qi.crossvalidate(local_reps, local_labels, molecule_size, tr_size, sigma, lam, molecule=True, num_cross=10)
    

#tr_points = np.logspace(9, 0, 10, base=2).astype(int)
#h = 'training_points\t mean_error\t std'
#p='/home/misa/APDFT/prototyping/atomic_energies/results/analyse_learning/'
#
#a = np.array([set_sizes, error_cv, error_std])
#np.savetxt(p+'learning_curves.tab', a.T, delimiter='\t', header=h)


def differenet_labels():
    # load data into list, count number of atoms per molecule
    paths=qi.wrapper_alch_data()
    alchemy_data, molecule_size = qi.load_alchemy_data(paths)
    
    # local data
    local_reps = qi.generate_atomic_representations(alchemy_data, molecule_size)
    local_labels = qi.generate_label_vector(alchemy_data, molecule_size.sum(), value='atomisation')
    local_labels_shifted = qi.shift_by_mean_energy(local_reps, local_labels)
    
    set_sizes = np.logspace(9, 0, 10, base=2).astype(int)
    # optimized hyperparameters
    hypar = '/home/misa/APDFT/prototyping/atomic_energies/results/analyse_learning/optimized_hyperparameters.txt'
    with open(hypar, 'r') as f:
        lines = f.readlines()
    sigma = float(lines[3].split('\t')[0])
    lam = float(lines[3].split('\t')[1])
    error_cv = np.zeros(len(set_sizes))
    error_std = np.zeros(len(set_sizes))
    
    for idx, tr_size in enumerate(set_sizes):
        error_cv[idx], error_std[idx] = qi.crossvalidate(local_reps, local_labels, molecule_size, tr_size, sigma, lam, molecule=False, num_cross=10)


def choose_different_lambdas():
    """
    out of the lambda values 0,0.2,0.4,0.6,0.8,1 select only 5 and calculate
    the calculate the ML error to estimate how much the error will improve if more
    lambda values are added
    """
    base='/home/misa/APDFT/prototyping/atomic_energies/results/slice_ve38/'
    pl = ['paths_no_8', 'paths_no_15', 'paths_no_23', 'paths_no_30']
    results = []
    
    for p in pl:
    
        # load data into list, count number of atoms per molecule
        paths=qi.wrapper_alch_data(path=base+p)
        alchemy_data, molecule_size = qi.load_alchemy_data(paths)
        
        # local data
        local_reps = qi.generate_atomic_representations(alchemy_data, molecule_size)
        local_labels = qi.generate_label_vector(alchemy_data, molecule_size.sum(), value='atomisation')
        local_labels_shifted = qi.shift_by_mean_energy(local_reps, local_labels)
        
        set_sizes = np.logspace(9, 0, 10, base=2).astype(int)
        # optimized hyperparameters
        hypar = '/home/misa/APDFT/prototyping/atomic_energies/results/analyse_learning/optimized_hyperparameters.txt'
        with open(hypar, 'r') as f:
            lines = f.readlines()
        sigma = float(lines[3].split('\t')[0])
        lam = float(lines[3].split('\t')[1])
        error_cv = np.zeros(len(set_sizes))
        error_std = np.zeros(len(set_sizes))
        
        for idx, tr_size in enumerate(set_sizes):
            error_cv[idx], error_std[idx] = qi.crossvalidate(local_reps, local_labels, molecule_size, tr_size, sigma, lam, molecule=False, num_cross=10)
            
        results.append((error_cv, error_std))
    
    tr_points = np.logspace(9, 0, 10, base=2).astype(int)
    h = 'training_points\t mean_error\t std'
    
    p='/home/misa/APDFT/prototyping/atomic_energies/results/analyse_learning/'
    pl = ['no_8', 'no_15', 'no_23', 'no_30']
    
    for i in range(len(results)):
        a = np.array([tr_points, results[i][0],results[i][1]])
        np.savetxt(p+pl[i]+'.tab',a.T, delimiter='\t', header=h)