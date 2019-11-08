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
    
    return(lam_vals, densities, nuclei, gpts)

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
    elif method == 'cspline':
        poly_obj = scipy.interpolate.CubicSpline(lam_vals, reshaped_densities, axis=0, bc_type=('clamped', 'not-a-knot'))
        averaged_density = poly_obj.integrate(0, 1)
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
    meshgrid_xyz = np.vstack([_.flatten() for _ in meshgrid]).T
    dist_gpt_nuc = np.linalg.norm(meshgrid_xyz - pos_nuc, axis=1)
    
    # density already scaled
    return(-(density.flatten()/dist_gpt_nuc).sum())
    
def calculate_atomic_energies(density, nuclei, meshgrid):
    """
    returns the atomic energies of the compound
    nuclei: charge and coordinates of nuclei in compound
    """
        
    # calculate atomic energies for every atom in compound
    atomic_energies = np.empty(nuclei.shape[0])
    alch_pots = np.empty(nuclei.shape[0])
    for idx, nuc in enumerate(nuclei):
        alch_pot = calculate_alchemical_potential(density, meshgrid, nuc[1:4])
        alch_pots[idx] = alch_pot
        atomic_energies[idx] = nuc[0]*alch_pot
    
    return(atomic_energies, alch_pots)
        
def atomic_energy_decomposition(cube_files, intgr_method='trapz', save_dir=None):
    """
    returns charge and position of nuclei, alchemical potentials and atomic energies for atoms in compound
   
    cube_files: list of tuples
    tuple[0] = path to cube file and tuple[1] = lambda value for cube-file
        assumed order of elements: lowest lambda value has index 0 
        and highest lambda value has highest index
    save_dir: path to file where output is written; if None atomic energies and 
        alchemical potentials are returned as a tuple
        if given charge, coordinates, alchemical potentials and atomic energies
        are written to file
   """
   
   # read data from cube files
    scaled_densities = [] # density scaled by the volume of the gridpoints
    lam_vals = [] # lambda values of the densities
    nuclei = None # charge and positions of nuclei
    gpts = None # grid points where the density is given
    for idx, file in enumerate(cube_files):
        cube_obj = CUBE(file[0])
        cube_obj.scale()
        scaled_densities.append(cube_obj.data_scaled)
        lam_vals.append(file[1])
        
        if idx == len(cube_files)-1: # read these values only from one cube file because they are always the same
            nuclei = np.array(cube_obj.atoms)
            gpts = cube_obj.get_grid()
            
    
    # integrate density with respect to lambda
    # where to get lambda-values?
    av_dens = integrate_lambda_density(scaled_densities, lam_vals, method=intgr_method)
    
    # calculate alchemical potentials and atomic energies
    atomic_energies, alch_pots = calculate_atomic_energies(av_dens, nuclei, gpts)
    
    if save_dir:
        # write atomic energies and alchemical potentials to file
        store = np.array([nuclei[:,0], nuclei[:,1], nuclei[:,2], nuclei[:,3], alch_pots, atomic_energies]).T
        header = 'charge\t x_coord\t y_coord\t z_coord\t alchemical_potential\t atomic_energies'
        np.savetxt(save_dir, store, delimiter='\t', header = header)

    return(nuclei, atomic_energies, alch_pots)
    
def calculate_atomisation_energies(ae_alch, total_en, free_at_en):
    """
    returns atomic contributions to the atomisation energy
    
    1) shift of the atomic energies calculated from the lambda-averaged 
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

    
    
    
    
    
    
    
    
    
    
    