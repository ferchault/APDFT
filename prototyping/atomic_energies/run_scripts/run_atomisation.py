#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 16:19:34 2019

@author: misa

calculates the atomisation energies for the compounds specified in dirs
needs following cube-files: 've_8.cube', 've_15.cube', 've_23.cube', 've_30.cube', 've_38.cube'
"""

import sys
sys.path.insert(0, '/home/misa/git_repositories/APDFT/prototyping/atomic_energies/')
sys.path.insert(0, '/home/misa/git_repositories/APDFT/prototyping/atomic_energies/hitp')

import os
os.chdir('/home/misa/APDFT/prototyping/atomic_energies/results/slice_ve38')

import numpy as np
import re

import alchemy_tools2 as at
from explore_qml_data import get_free_atom_data
import utils_qm as uqm


# paths to the compounds
dirs = uqm.read_list('/home/misa/APDFT/prototyping/atomic_energies/results/slice_ve38/batch3')

for compound_path in dirs[101:]:
    print(f'Calculation energy in {compound_path}', flush=True)
    # paths to the cube files
    base = compound_path
    base = os.path.join(base, 'cube-files')
    cubes = ['ve_08.cube', 've_15.cube', 've_23.cube', 've_30.cube', 've_38.cube']
    for i in range(len(cubes)):
        cubes[i] = os.path.join(base, cubes[i])
        
    path_to_ueg = '/home/misa/APDFT/prototyping/atomic_energies/results/slice_ve38/ueg/ve_00.cube'
    cubes.insert(0, path_to_ueg)
    # load 
    lam_vals, densities, nuclei, gpts, h_matrix = at.load_cube_data(cubes)
    
    # atomic energy decomposition
    nuclei, atomic_energies_with_repulsion, atomic_energies, alch_pots = at.atomic_energy_decomposition(lam_vals, densities, nuclei, gpts, h_matrix)
    #print(atomic_energies_with_repulsion)
    #print(atomic_energies)
    
    # calculation of atomisation energy
    # read total energy B3LYP from xyz-file in qm9 database
    comp_name = re.search('dsgdb9nsd_......', base)[0] # get the compound name from the path
    total_energy = at.get_property(os.path.join('/home/misa/datasets/qm9/', comp_name + '.xyz'), 'U0')
    
    # read energies B3LYP of free atoms
    free_atoms_dict = get_free_atom_data()
    # free atom energy for every atom in compound
    free_atom_energies = at.get_free_atom_energies(nuclei[:,0], free_atoms_dict)
    
    
    # calculate atomic atomisation energies
    atomisation_energies = at.calculate_atomisation_energies(atomic_energies, total_energy, free_atom_energies) # without nuclear repulsion
    #atomisation_energies_rep = at.calculate_atomisation_energies(atomic_energies_with_repulsion, total_energy, free_atom_energies) # with nuclear repulsion
    
    # write atomic energies and alchemical potentials to file
    store = np.array([nuclei[:,0], nuclei[:,1], nuclei[:,2], nuclei[:,3], alch_pots, atomic_energies, atomisation_energies]).T
    header = 'charge\t x_coord\t y_coord\t z_coord\t alchemical_potential\t atomic_energies\t atomisation_energies'
    save_dir = os.path.join(compound_path, 'atomic_energies_with_mic.txt')
    np.savetxt(save_dir, store, delimiter='\t', header = header)
    
    # if atomic energies with repulsion are calculated
    # write atomic energies and alchemical potentials to file
    #store = np.array([nuclei[:,0], nuclei[:,1], nuclei[:,2], nuclei[:,3], alch_pots, atomic_energies, atomisation_energies, atomisation_energies_rep]).T
    #header = 'charge\t x_coord\t y_coord\t z_coord\t alchemical_potential\t atomic_energies\t atomisation_energies\t atomisation_energies_repulsion'
    #save_dir = os.path.join(compound_path, 'atomic_energies_with_mic_repulsion.txt')
    #np.savetxt(save_dir, store, delimiter='\t', header = header)
