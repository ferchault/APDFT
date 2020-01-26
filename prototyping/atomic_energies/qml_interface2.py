#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 15:33:00 2020

@author: misa
"""
import qml.fchl
import numpy as np
import qml_interface as qmi

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
    
    # pick random sub matrix from kernel for training
#class Data_Splitter(object):
#    """
#    splits data in training and test set
#    """    

def optimize_regularizer(kernel, labels, molecule_size, tr_size = 512, num_cross=10):
    """
    kernel generation for different sigma values can be expensive in CPU time and memory demand
    therefore, it is done elsewhere
    """
    ## repeat for different sigma values
    

    # crossvalidate the following
    lams = np.logspace(-15, 0, 16).tolist()
    mean_errors_cv = np.zeros(len(lams))
    std_errors_cv = np.zeros(len(lams))
    for n in range(num_cross):
        
        # split kernel in training and validation
        global_tr, global_test = qmi.get_indices(len(molecule_size), tr_size)
        tr_ind = qmi.get_local_idx(global_tr, molecule_size)
        test_ind = qmi.get_local_idx(global_test, molecule_size)
    
        tr_kernel, test_kernel = qmi.split_kernel(kernel, tr_ind, test_ind)
        tr_label = labels[tr_ind]
        test_label = labels[test_ind]
        
        # repeat for different lambda values
        lams_error = np.zeros(len(lams))
        lams_error_std = np.zeros(len(lams))
        for idx_lams, l in enumerate(lams):
            # get coefficients
            reg_kernel = tr_kernel + np.identity(len(tr_kernel))*l
            coeffs = qml.math.cho_solve(reg_kernel, tr_label)
            
            # predict
            pred_label = np.dot(test_kernel, coeffs)
    
            # compute error
            error = np.abs(pred_label - test_label)
            mean_error = error.mean()
            std_error = error.std()
            
            # save error
            lams_error[idx_lams] = mean_error
            lams_error_std[idx_lams] = std_error
            
        mean_errors_cv +=  lams_error
        std_errors_cv += lams_error_std
            
    return(lams, lams_error/num_cross, lams_error_std/num_cross)