#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 15:33:00 2020

@author: misa
"""
import qml.fchl
import numpy as np

def string_mult(a,b):
    product = np.empty((a.shape[0], b.shape[0]), dtype='object')
    for row in range(len(a)):
        for col in range(len(b)):
            pr = a[row] + ', ' + b[col]
            product[row][col] = pr
    return(product)

def generate_fchl_atomic_kernel(reps, molecule_size, sigma):
    """
    produces atomic kernel for all atoms in all molecules
    reps: list of representations for all molecules
    """
    tot_atom_number = molecule_size.sum()
    
    atomic_kernel = np.empty((tot_atom_number, tot_atom_number))
    
    for idx1 in range(len(reps)):
        
        # pick representation at idx1
        # build atomic kernel elements with all representations > idx1-1
        
        for idx2 in range(idx1, len(reps)):
            two_particle_kernel = qml.fchl.get_atomic_kernels(reps[idx1], reps[idx2], [sigma])[0]#string_mult(reps[idx1], reps[idx2])#
            
            rowindex_start = molecule_size[0:idx1].sum()
            colindex_start = molecule_size[0:idx2].sum()
            
            for ri in zip(range(0, two_particle_kernel.shape[0]), range(rowindex_start, rowindex_start+molecule_size[idx1])):
                for ci in zip(range(0, two_particle_kernel.shape[1]), range(colindex_start, colindex_start+molecule_size[idx2])):
#                    print(two_particle_kernel[ri[0], ci[0]])
                    atomic_kernel[ri[1], ci[1]] = two_particle_kernel[ri[0], ci[0]]
                    atomic_kernel[ci[1], ri[1]] = atomic_kernel[ri[1], ci[1]]
            
            
    return(atomic_kernel)