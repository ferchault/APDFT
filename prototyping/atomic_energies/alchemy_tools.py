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
    checks if each array in a set (list) has the same shape
    density_arrays: list of density
    """
    prev_shape = None
    for idx, density in enumerate(density_arrays):
        if idx == 0:
            prev_shape = density.shape
        assert prev_shape == density.shape, "Densities have not the same shape!"
        prev_shape = density.shape
        
def calculate_alchemical_potential(density, meshgrid, pos_nuc):
    """
    calculates the alchemical potential (see: arXiv:1907.06677v1, J. Chem. Phys. 125, 154104 (2006) )
    density: electron density of the system (numpy array)
    meshgrid: points in space at which the density is given (output of numpy.meshgrid())
    pos_nuc: position of the nucleus for which the alchemical potential is calculated (numpy array)
    """
    # calculate distance between gridpoints and position of nucleus
    
    dist_gpt_nuc = np.sqrt(np.power(meshgrid[0]-pos_nuc[0], 2) + np.power(meshgrid[1]-pos_nuc[1], 2) + np.power(meshgrid[2]-pos_nuc[2], 2))
    
    # density already scaled
    return((density/dist_gpt_nuc).sum())
    
def get_atomic_energies(density, nuclei, meshgrid):
    """
    returns the atomic energies of the compound
    nuclei: charge and coordinates of nuclei in compound
    """
        
    # calculate atomic energies for every atom in compound
    atomic_energies = []
    for nuc in nuclei:
        alch_pot = calculate_alchemical_potential(density, meshgrid, nuc[1:4])
        atomic_en = nuc[0]*alch_pot
        atomic_energies.append(atomic_en)
    
    return(atomic_energies)
    
    
    
    
    
    
    
    
    
    
    
    
    