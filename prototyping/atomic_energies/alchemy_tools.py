#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 16:21:09 2019

@author: misa
"""

import numpy as np
from parse_cube_files import CUBE

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
    
    #dist_gpt_nuc = np.sqrt(np.power(meshgrid[0]-pos_nuc[0], 2) + np.power(meshgrid[1]-pos_nuc[1], 2) + np.power(meshgrid[2]-pos_nuc[2], 2))
    meshgrid_xyz = np.vstack([_.flatten() for _ in meshgrid]).T
    dist_gpt_nuc = np.linalg.norm(meshgrid_xyz - pos_nuc, axis=1)
    
    # density already scaled
    return(-(density.flatten()/dist_gpt_nuc).sum())
    #return(-(density/dist_gpt_nuc).sum())
    
def calculate_atomic_energies(density, nuclei, meshgrid):
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
    
def transfer_atomic_energies(at_en_alch, total_en, free_at_en):
    """
    returns atomic contributions to the atomisation energy
    
    1) shift of the atomic energies calculated from the lambda-averaged 
       alchemical potential (at_en_alch) to the total energy (total_en) at some level of theory
        - at_en_alch are relative to the free electron gas with pbc
        - total_en is the total energy of the compund e.g. calculated with B3LYP
        - after the shift the sum of the atomic energies is equal to total_en
        
    2) get the atomisation energy
       - from each atomic energy the energy of the corresponding free atom is subtracted
    
    at_en_dec: atomic energies calculated from lambda averaged alchemical potential
    total_en: total energy of the system at the desired level of theory
    free_at_en: energies of single atoms at the desired level of theory
    """
    
    # shift at_en_alch to give total_en
    num_atoms = len(at_en_alch)
    total_en_alch = at_en_alch.sum()
    shift = (total_en - total_en_alch) / num_atoms
    at_en_alch_shifted = at_en_alch + shift
    
    # calculate contribution of every atom to the atomisation energy
    atomisation_energies = at_en_alch_shifted - free_at_en
    
    return(atomisation_energies)
    
def atomic_energy_decomposition(cube_files, save=False):
    
    # sort cube-files
    
    # read data from cube files
    scaled_densities = [] # density at every grid point scaled by its volume
    nuclei = None # charge and positions of nuclei
    gpts = None # grid points where the density is given
    for idx, file in enumerate(cube_files):
        cube_obj = CUBE(file)
        cube_obj.scale()
        scaled_densities.append(cube_obj.data_scaled)
        
        if idx == 0: # read these values only from one cube file because they are always the same
            nuclei = cube_obj.atoms
            gpts = cube_obj.get_grid()
            
    
    # integrate density with respect to lambda
    # where to get lambda-values?
    av_dens = integrate_lambda_density(scaled_densities, lam_vals)
    
    # calculate alchemical potentials and atomic energies
    atomic_energies, alch_pots = calculate_atomic_energies(av_dens, nuclei, gpts)
    
    if save:
        # write atomic energies and alchemical potentials to file
    else:
        return(atomic_energies, alch_pots)
    
        
    

    
    
    
    
    
    
    
    
    
    
    