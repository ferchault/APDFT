#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 16:21:09 2019

@author: misa
"""
import os
import numpy as np
import scipy
import glob

import sys
sys.path.insert(0, '/home/misa/APDFT/prototyping/atomic_energies/')

from parse_cube_files import CUBE
from explore_qml_data import get_property
from explore_qml_data import get_free_atom_data
from explore_qml_data import get_num_val_elec

def load_cube_data(paths_cubes):
    """
    returns the data necessary to calculate the atomic energies from the cube-files
    for different lambda values
    
    paths_cubes: paths to cubes files
    densities: densities given in different cube files
    lam_vals: lambda value for cube file, obtained by parsing filename
    nuclei: charges and coordinates of the nuclei
    gpts: the gridpoints where density values are given
    """
    
    densities = []
    lam_vals = []
    nuclei = None # nuclear charges and their positions
    gpts = None # gridpoints where density values are given
    h_matrix = np.zeros((3,3)) # needed for the calculation of the distance of the nuclei to the gridpoints with MIC
    
    for idx, path in enumerate(paths_cubes):
        cube = CUBE(path)
        
        densities.append(cube.data_scaled) # density
        
        fname = os.path.split(path)[1] # lambda value
        fname = fname.split('.')[0]
        total_ve = get_num_val_elec(cube.atoms[:, 0])
        lam = float(fname[3:])/total_ve
        lam_vals.append(lam)
        
        if idx==len(paths_cubes)-1:
            nuclei = cube.atoms
            gpts = cube.get_grid()
            h_matrix = [cube.X*cube.NX, cube.Y*cube.NY, cube.Z*cube.NZ]
    
    return(np.array(lam_vals), np.array(densities), nuclei, gpts, h_matrix)

def integrate_lambda_density(densities, lam_vals, method='trapz'):
    """
    calculates the integral over lambda numerically
    param scaled_densities: valence electron densities corresponding to the lambda-values in lam_vals
    type scaled densities: numpy array, shape (len(lam_vals), shape(gpts))
    lam_vals: the lambda values of the densities
    method: integration method
    """
        
    if method == 'trapz':
        averaged_density = np.trapz(densities, x=lam_vals, axis=0)
    elif method == 'cspline':
        poly_obj = scipy.interpolate.CubicSpline(lam_vals, densities, axis=0, bc_type=('clamped', 'not-a-knot'))
        averaged_density = poly_obj.integrate(0, 1)
    else:
        raise Exception("Unknown integration method!")
   
    return(averaged_density)
    
def distance_MIC2(pos_nuc, meshgrid, h_matrix):
    """
    calculates the distance between the position of the nucleus and the nearest image of a gridpoint
    works so far only for cubic symmetry
    
    pos_nuc: position of nucleus
    meshgrid: meshgrid containing x,y,z components of every gridpoint
    h_matrix: needed for calculation of MIC distance
    :return: distance between pos_nuc and every gridpoint
    :rtype: numpy array of shape meshgrid.shape
    """
    
    hinv = np.linalg.inv(h_matrix)
    a_t = np.dot(hinv, pos_nuc)
    
    # calculate product of h_matrix and grid componentwise
    b_t_x = hinv[0][0]*meshgrid[0]
    b_t_y = hinv[1][1]*meshgrid[1]
    b_t_z = +hinv[2][2]*meshgrid[2]
    
    t_12_x = b_t_x - a_t[0]
    t_12_y = b_t_y - a_t[1]
    t_12_z = b_t_z - a_t[2]
    
    t_12_x -= np.round(t_12_x)
    t_12_y -= np.round(t_12_y)
    t_12_z -= np.round(t_12_z)
    
    x = np.power(h_matrix[0][0]*t_12_x, 2)
    y = np.power(h_matrix[1][1]*t_12_y, 2)
    z = np.power(h_matrix[2][2]*t_12_z, 2)
    
    return(np.sqrt(x+y+z))
       
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
            
def nuclear_repulsion(charges, positions):
    """
    decomposition of the nuclear repulsion in atomic contributions
    
    charges: charges of the nuclei
    type charges: numpy array
    positions: positions of the nuclei
    type: numpy array
    return: nuclear repulsion decomposed into atomic contributions
    type: numpy array
    """
    
    atomic_nuc_rep = np.zeros(len(charges))
    
    for i, val_i in enumerate(charges):
        rep_tmp = 0.0
        
        # sum up repulsion term
        for j, val_j in enumerate(charges):
            if j != i:
                rep_tmp += val_j/(np.linalg.norm(positions[i]-positions[j]))
    
        atomic_nuc_rep[i] = rep_tmp*val_i/2
        
    return(atomic_nuc_rep)
    
        
def calculate_atomic_energies(density, nuclei, meshgrid, h_matrix):
    """
    performs intgegration over electron density with minimum image convention

    density: the lambda averaged electron density
    nuclei: charge and coordinates of nuclei in compound
    meshgrid: rid which contains the coordinates for which the electron densities  are given
    h_matrix: H matrix with the cell vectors as columns
    return: the alchemical potentials and atomic energies of the compound if the input the lambda averaged density
    """
        
    # calculate atomic energies for every atom in compound
    atomic_energies = np.empty(nuclei.shape[0])
    alch_pots = np.empty(nuclei.shape[0])
    for idx, nuc in enumerate(nuclei):
        distance_nuc_grid = distance_MIC2(nuc[1:4], meshgrid, h_matrix) # calculate distance of gpts to nucleus with MIC
        alch_pot = -(density/distance_nuc_grid).sum() # integrate over position of electron density
        alch_pots[idx] = alch_pot
        atomic_energies[idx] = nuc[0]*alch_pot
    
    return(atomic_energies, alch_pots)
        
def atomic_energy_decomposition(lam_vals, scaled_densities, nuclei, gpts, h_matrix, intgr_method='trapz'):
    """
    returns charge and position of nuclei, alchemical potentials and atomic energies for atoms in compound
   
    param lam_vals: lambda-values
    type lam_vals: numpy array, shape (1,)
    param scaled_densities: valence electron densities corresponding to the lambda-values in lam_vals
    type scaled densities: numpy array, shape (len(lam_vals), shape(gpts))
    param nuclei: charge and positions of the nuclei
    type nuclei: numpy array shape (number nuclei, 4)
    param gpts: grid which contains the coordinates for which the electron densities  are given
    type gpts: numpy array, shape (number of pts x-direction, number of pts y-direction, number of pts z-direction)
    h_matrix: H matrix with the cell vectors as columns
    type h_matrix: numpy array, shape (3, 3)
    param intgr_method: method used for integration over lambda
    type intgr_method: string
   """
   
    # integrate density with respect to lambda
    av_dens = integrate_lambda_density(scaled_densities, lam_vals, method=intgr_method)
    
    # calculate alchemical potentials and atomic energies by integrating over r
    atomic_energies, alch_pots = calculate_atomic_energies(av_dens, nuclei, gpts, h_matrix)
    
    return(nuclei, atomic_energies, alch_pots)
    
def calculate_atomisation_energies(ae_alch, total_en, free_at_en):
    """
    returns atomic contributions to the atomisation energy
    
    1) shifts the atomic energies calculated from the lambda-averaged 
       alchemical potential (at_en_alch) to the total energy (total_en) at some level of theory
        - ae_alch are relative to the free electron gas with pbc
        - total_en is the total energy of the compund e.g. calculated with B3LYP
        - after the shift the sum of the atomic energies is equal to total_en
        
    2) get the atomisation energy
       - from each atomic energy the energy of the corresponding free atom is subtracted
    
    at_en_dec: atomic energies calculated from lambda averaged alchemical potential
    total_en: total energy of the system at the desired level of theory
    free_at_en: energies of the single atoms for the corresponding alchemical potential
    """
    
    # shift at_en_alch to give total_en
    total_en_alch = ae_alch.sum()
    num_atoms = len(ae_alch)
    shift = (total_en - total_en_alch) / num_atoms
    ae_alch_shifted = ae_alch + shift
    
    # calculate contribution of every atom to the atomisation energy
    atomisation_energies = ae_alch_shifted - free_at_en
    
    return(atomisation_energies)
    
def write_atomisation_energies(dirs):
    """
    return atomisation energies for compounds given in dirs
    dirs: path to compounds
    """
    for comp_path in dirs:
        # get lambda = 0
        path_ueg = '/home/misa/APDFT/prototyping/atomic_energies/results/slice_ve38/ueg/ueg.cube'
        cube_files = [(path_ueg, 0.0)] # stores tuples of paths and lambda values
        # get paths and almbda_values
        ve = np.array([8, 15, 23, 30, 38])
        
        for num_ve in ve:
            path_tmp = 'cube-files/ve_' + str(num_ve) + '.cube'
            cube_files.append( (os.path.join(comp_path, path_tmp), num_ve/38) )
#        print('##############################')
#        [print(el) for el in cube_files]
        
        # calculate atomic energies in LDA relative to UEG with pbc
        nuclei, atomic_energies, alch_pots = atomic_energy_decomposition(cube_files, intgr_method='cspline')
        
        # total energy from B3LYP
        comp_name = comp_path.split('/')[len(comp_path.split('/'))-2]
        total_energy = get_property(os.path.join('/home/misa/datasets/qm9', comp_name + '.xyz'), 'U0')
        # energies of the free atoms in qm9
        free_atoms = get_free_atom_data()
        # free atom energy for every atom in compound
        free_en = get_free_atom_energies(nuclei[:,0], free_atoms)
        
        atomisation_energies = calculate_atomisation_energies(atomic_energies, total_energy, free_en)
        
        # write atomic energies and alchemical potentials to file
        store = np.array([nuclei[:,0], nuclei[:,1], nuclei[:,2], nuclei[:,3], alch_pots, atomic_energies, atomisation_energies]).T
        header = 'charge\t x_coord\t y_coord\t z_coord\t alchemical_potential\t atomic_energies\t atomisation_energies'
        save_dir = os.path.join(comp_path, 'atomic_energies_cspline.txt')
        np.savetxt(save_dir, store, delimiter='\t', header = header)
        
def write_atomisation_energies_new(dirs):
    """
    return atomisation energies for compounds given in dirs
    dirs: path to compounds
    """
    for comp_path in dirs:
        # get lambda = 0
        path_ueg = '/home/misa/APDFT/prototyping/atomic_energies/results/slice_ve38/ueg/ueg.cube'
        cube_files = [(path_ueg, 0.0)] # stores tuples of paths and lambda values
        # get paths and almbda_values
        paths = glob.glob(comp_path+'/*.cube')
        paths.sort()
        
        for path in paths:
            num_ve = float(path.split('/')[-1].split('.')[0].split('_')[1])/38.0
            cube_files.append([path, num_ve])
#        print('##############################')
#        [print(el) for el in cube_files]
        
        # calculate atomic energies in LDA relative to UEG with pbc
        nuclei, atomic_energies, alch_pots = atomic_energy_decomposition(cube_files, intgr_method='trapz')
        
        # total energy from B3LYP
        comp_name = comp_path.split('/')[len(comp_path.split('/'))-2]
        total_energy = get_property(os.path.join('/home/misa/datasets/qm9', comp_name + '.xyz'), 'U0')
        # energies of the free atoms in qm9
        free_atoms = get_free_atom_data()
        # free atom energy for every atom in compound
        free_en = get_free_atom_energies(nuclei[:,0], free_atoms)
        
        atomisation_energies = calculate_atomisation_energies(atomic_energies, total_energy, free_en)
        
        # write atomic energies and alchemical potentials to file
        store = np.array([nuclei[:,0], nuclei[:,1], nuclei[:,2], nuclei[:,3], alch_pots, atomic_energies, atomisation_energies]).T
        header = 'charge\t x_coord\t y_coord\t z_coord\t alchemical_potential\t atomic_energies\t atomisation_energies'
        save_dir = os.path.join(comp_path, 'many_lambdas.txt')
        np.savetxt(save_dir, store, delimiter='\t', header = header)
        
        
def test_impact_lambda(dirs):
    """
    calculate integrals while neglecting one lambda value to estimate error of to little lambda values
    return atomisation energies for compounds given in dirs
    dirs: path to compounds
    """
    for comp_path in dirs:
        # get lambda = 0
        path_ueg = '/home/misa/APDFT/prototyping/atomic_energies/results/slice_ve38/ueg/ueg.cube'
        cube_files = [(path_ueg, 0.0)] # stores tuples of paths and lambda values
        # get paths and almbda_values
        ve = np.array([8, 15, 23, 30, 38])
        
        for num_ve in ve:
            path_tmp = 'cube-files/ve_' + str(num_ve) + '.cube'
            cube_files.append( (os.path.join(comp_path, path_tmp), num_ve/38) )
#        print('##############################')
#        [print(el) for el in cube_files]
        no_30 = [cube_files[0], cube_files[1], cube_files[2], cube_files[3], cube_files[5]]
        no_23 = [cube_files[0], cube_files[1], cube_files[2], cube_files[4], cube_files[5]]
        no_15 = [cube_files[0], cube_files[1], cube_files[3], cube_files[4], cube_files[5]]
        no_8 = [cube_files[0], cube_files[2], cube_files[3], cube_files[4], cube_files[5]]
        cube_sets = [no_8, no_15, no_23, no_30]
        cube_set_names = ['no_8', 'no_15', 'no_23', 'no_30']
        
        for idx,cube_file_set in enumerate(cube_sets):
            # calculate atomic energies in LDA relative to UEG with pbc
            nuclei, atomic_energies, alch_pots = atomic_energy_decomposition(cube_file_set)
            
            # total energy from B3LYP
            comp_name = comp_path.split('/')[len(comp_path.split('/'))-2]
            total_energy = get_property(os.path.join('/home/misa/datasets/qm9', comp_name + '.xyz'), 'U0')
            # energies of the free atoms in qm9
            free_atoms = get_free_atom_data()
            # free atom energy for every atom in compound
            free_en = get_free_atom_energies(nuclei[:,0], free_atoms)
            
            atomisation_energies = calculate_atomisation_energies(atomic_energies, total_energy, free_en)
            
            # write atomic energies and alchemical potentials to file
            store = np.array([nuclei[:,0], nuclei[:,1], nuclei[:,2], nuclei[:,3], alch_pots, atomic_energies, atomisation_energies]).T
            header = 'charge\t x_coord\t y_coord\t z_coord\t alchemical_potential\t atomic_energies\t atomisation_energies'
            save_dir = os.path.join(comp_path, cube_set_names[idx]+'.txt')
            np.savetxt(save_dir, store, delimiter='\t', header = header)

def get_free_atom_energies(nuc, e_free):
    """
    returns np.array with energy of the free atoms for every element of nuc
    
    nuc: list of nuclear charges
    e_free: energy of free atoms used in qm9 as dict {element_symbol:energy}
    """
    
    energy_free_atom = np.zeros(len(nuc))
    
    for idx, n in enumerate(nuc):
        
        if int(n)==1:
            energy_free_atom[idx] = e_free['H']
        elif int(n) == 6:
            energy_free_atom[idx] = e_free['C']
        elif int(n) == 7:
            energy_free_atom[idx] = e_free['N']
        elif int(n) == 8:
            energy_free_atom[idx] = e_free['O']
        elif int(n) == 9:
            energy_free_atom[idx] = e_free['F']
        else:
            raise Exception("Element not in qm9")
            
    return(energy_free_atom)

    
    
    
    
    
    
    
    
    
    
    