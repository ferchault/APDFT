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
from find_converged import concatenate_files



# paths to the compounds
dirs = concatenate_files(['/home/misa/APDFT/prototyping/atomic_energies/results/slice_ve38/paths_atomic_energies'])
#dirs = dirs_all[123:]

for compound_path in dirs:
    print(f'Calculation energy in {compound_path}', flush = True)
    # paths to the cube files
    base = compound_path
    base = base + 'cube-files/'
    cubes = ['ve_38.cube']
    for i in range(len(cubes)):
        cubes[i] = base + cubes[i]
        
    #path_to_ueg = '/home/misa/APDFT/prototyping/atomic_energies/results/slice_ve38/ueg/ve_00.cube'
    #cubes.insert(0, path_to_ueg)
    # load 
    lam_vals, densities, nuclei, gpts, h_matrix = at.load_cube_data(cubes)
    
    atomic_energies_with_repulsion, atomic_energies, alch_pots = at.calculate_atomic_energies(densities[0], nuclei, gpts, h_matrix, intg_method = 'sum')
        
    # write atomic energies and alchemical potentials to file
    store = np.array([nuclei[:,0], nuclei[:,1], nuclei[:,2], nuclei[:,3], alch_pots]).T
    header = 'charge\t x_coord\t y_coord\t z_coord\t alchemical_potential'
    save_dir = os.path.join(compound_path, 'alchpot_lam1.txt')
    np.savetxt(save_dir, store, delimiter='\t', header = header)