#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 16:19:34 2019

@author: misa

calculates the atomisation energies for the compounds specified in dirs
needs following cube-files: 've_8.cube', 've_15.cube', 've_23.cube', 've_30.cube', 've_38.cube'
"""

import sys
sys.path.insert(0, '/home/misa/APDFT/prototyping/atomic_energies/')
sys.path.insert(0, '/home/misa/APDFT/prototyping/atomic_energies/hitp')

import os
os.chdir('/home/misa/APDFT/prototyping/atomic_energies/results/slice_ve38')

import numpy as np
import re

import alchemy_tools2 as at
from explore_qml_data import get_free_atom_data
from find_converged import concatenate_files
import utils_qm as uq

# load average density of free atoms
base_path = '/home/misa/APDFT/prototyping/atomic_energies/results/slice_ve38/free_atoms/'
densities_free_atoms = {1.0:None, 4.0:None, 5.0:None, 6.0:None}
elements = ['H', 'C', 'N', 'O']
for el, k in zip(elements, densities_free_atoms):
    densities_free_atoms[k] = uq.load(base_path + 'av_dens_{el}')

# paths to the compounds
dirs = concatenate_files(['/home/misa/APDFT/prototyping/atomic_energies/results/slice_ve38/paths_atomic_energies'])

dirs = ['/home/misa/APDFT/prototyping/atomic_energies/results/slice_ve38/dsgdb9nsd_001212/']
for compound_path in dirs:
    # paths to the cube files
    base = compound_path
    base = base + 'cube-files/'
    cubes = ['ve_8.cube', 've_15.cube', 've_23.cube', 've_30.cube', 've_38.cube']
    for i in range(len(cubes)):
        cubes[i] = base + cubes[i]
        
    path_to_ueg = '/home/misa/APDFT/prototyping/atomic_energies/results/slice_ve38/ueg/ve_00.cube'
    cubes.insert(0, path_to_ueg)
    # load 
    lam_vals, densities, nuclei, gpts, h_matrix = at.load_cube_data(cubes)
    
    # atomic energy decomposition
    nuclei, atomic_energies_with_repulsion, atomic_energies, alch_pots = at.atomic_energy_decomposition(lam_vals, densities, nuclei, gpts, h_matrix)
    
    # calculate alchemical potential of free atoms at position of every nucleus
    alch_pot_free = []
    for atom_I in nuclei:
        alch_pot_free_I = 0
        for atom_J in nuclei:
            s = np.array([10.0, 10, 10]) - atom_J[1:4]
            RI_prime = atom_I[1:4] + s
            dist = at.distance_MIC2(RI_prime, meshgrid, h_matrix)
            
            nuc_charge = atom_J[0]
            density_fa = densities_free_atoms[nuc_charge]
            elec_pot = -(density_fa/dist).sum()
            alch_pot_free_I += elec_pot
        alch_pot_free.append(alch_pot_free_I)
        
    alch_pot_free = np.array(alch_pot_free)
    alch_pot_bind = molecule[:, 4] - alch_pot_free
    atomic_atomisation_pbc = alch_pot_bind*nuclei[:,0]
    
    # write atomic energies and alchemical potentials to file
    store = np.array([nuclei[:,0], nuclei[:,1], nuclei[:,2], nuclei[:,3], alch_pots, alch_pot_free, alch_pot_bind, atomic_atomisation_pbc]).T
    header = 'charge\t x_coord\t y_coord\t z_coord\t alchemical_potential\t alch_pot_free\t alch_pot_bind\t atomic_atomisation_pbc'
    save_dir = os.path.join(compound_path, 'atomic_binding_energies_no_fit.txt')
    np.savetxt(save_dir, store, delimiter='\t', header = header)