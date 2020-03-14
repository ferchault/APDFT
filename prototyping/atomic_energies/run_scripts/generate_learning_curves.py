#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 11:25:12 2019

@author: misa

can be used to generate learning curves as long as hyperparameters are provided

hypar: specifies file where hyperparameters can be found

standard values: 10-fold crossvalidation, error per atom not error per molecule
check LABEL_VALUE
"""

import numpy as np

import sys
sys.path.insert(0, '/home/misa/APDFT/prototyping/atomic_energies/')
import qml_interface as qi

def save(set_sizes, error, std, fname):
    big_array = np.array([set_sizes, error, std]).T
#    big_array = np.flip(big_array, axis=0)
    np.savetxt(fname, big_array, delimiter='\t', header='set_size\t error \t error_std')

# load data (with mic)
paths=qi.wrapper_alch_data()
alchemy_data, molecule_size = qi.load_alchemy_data(paths)

# prepare representations and labels
LABEL_VALUE = 'atomisation'
DELTA = True
local_reps = qi.generate_atomic_representations(alchemy_data, molecule_size)
local_labels = qi.generate_label_vector(alchemy_data, molecule_size.sum(), value=LABEL_VALUE)
if DELTA:
    # delta Learning
    # divide indices in groups depending on charge
    charges = qi.generate_label_vector(alchemy_data, molecule_size.sum(), 'charge')
    part_charges = qi.partition_idx_by_charge(charges)
    # get mean atomisation energy per charge
    mean_atomisation = dict.fromkeys(list(set(charges)),0)
    for k in part_charges:
        mtmp = local_labels[part_charges[k]]
        mean_atomisation[k] = mtmp.mean()      
    delta_labels = qi.get_label_delta(mean_atomisation, np.arange(molecule_size.sum()), alchemy_data, molecule_size)
    local_labels = local_labels - delta_labels

# read optimized hyperparameters
#HYPAR_PATH = '/home/misa/APDFT/prototyping/atomic_energies/results/analyse_learning/mic/optimized_hyperpar_atomic_mic.txt'
#with open(HYPAR_PATH, 'r') as f:
#    lines = f.readlines()
sigma =  151.99110829529332# float(lines[2].split('\t')[0])
lam = 1e-06 #float(lines[2].split('\t')[1])

# define number of training points for which MAE is calculated
set_sizes = np.logspace(0,9, 10, base=2).astype(int)
#set_sizes = [512]
# calculate error for every training point size
error_cv = np.zeros(len(set_sizes))
error_std = np.zeros(len(set_sizes))
for idx, tr_size in enumerate(set_sizes):
        error_cv[idx], error_std[idx] = qi.crossvalidate(local_reps, local_labels, molecule_size, tr_size, sigma, lam, molecule=True, num_cross=10)



## load data (with mic)
#paths=qi.wrapper_alch_data()
#alchemy_data, molecule_size = qi.load_alchemy_data(paths)
## optimized hyperparameters
##hypar = '/home/misa/APDFT/prototyping/atomic_energies/results/analyse_learning/optimized_hyperparameters_global_rep.txt'
##with open(hypar, 'r') as f:
##    lines = f.readlines()
#sigma = 151.99110829529332#float(lines[2].split('\t')[0])
#lam = 1e-03#float(lines[2].split('\t')[1])
#set_sizes = [512]
#error_cv = np.zeros(len(set_sizes))
#error_std = np.zeros(len(set_sizes))
#
## global data
#global_reps = qi.wrapper_global_representations(alchemy_data, molecule_size, rep_par='coulomb')
#global_labels = qi.generate_label_vector(alchemy_data, len(global_reps), value='atomisation_global')
#
#
#for idx, tr_size in enumerate(set_sizes):
#    error_cv[idx], error_std[idx] = qi.crossvalidate_new(global_reps, global_labels, molecule_size, tr_size, sigma, lam, local=False, num_cross=10)