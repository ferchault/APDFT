#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 16:21:09 2019

@author: misa
"""

import numpy as np

def integrate_lambda_density(density_arrays, lam_vals, method='trapz'):
    """
    calculates \int d\lambda \rho(\lambda, vec{r}) numerically
    density_arrays: list of densities at different lambda values
    lam_vals: the lambda values of the densities
    method: integration method
    """
    
    # reshape the densities into vectors and stores them in a 2D-numpy array
    # dimensions: # of densities, # of gridpoints per density
    shape = density_arrays[0].shape
    reshaped_densities = reshape_densities(density_arrays)
    
    # integration
    if method == 'trapz':
        averaged_density = np.trapz(reshaped_densities, x=lam_vals, axis=0)
    else:
        raise Exception("Unknown integration method!")
        
    # reshaping back to 3D-grid
    averaged_density = np.reshape(averaged_density, shape)
    
    return(averaged_density)
    
def reshape_densities(density_arrays):
    """
    takes a list of densities where each density is represented on a 3D grid;
    reshapes the densities into vectors and stores all vectors in a 2D-numpy array
    density_arrays: list of densities at different lambda values
    """
    # ensure that densities have the same shape
    check_shapes(density_arrays)
    
    # create empty numpy array for storage of reshaped densities
    # dimensions: # of densities, # of gridpoints per density
    new_shape = 1
    for el in density_arrays[0].shape:
        new_shape *= el
    new_density_arrays = np.zeros( (len(density_arrays), new_shape) )
    
    # do the reshaping
    for idx, dens in enumerate(density_arrays):
        new_density_arrays[idx] = dens.reshape(new_shape)
    return(new_density_arrays)
    
    
def check_shapes(density_arrays):
    """
    checks if the densities have the same shape
    """
    prev_shape = None
    for idx, density in enumerate(density_arrays):
        if idx == 0:
            prev_shape = density.shape
        assert prev_shape == density.shape, "Densities have not the same shape!"
        prev_shape = density.shape