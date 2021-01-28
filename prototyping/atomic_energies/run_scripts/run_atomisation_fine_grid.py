#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 16:19:34 2019

@author: misa

calculates the atomisation energies for fine grid
"""

import sys
sys.path.insert(0, '/home/misa/git_repositories/APDFT/prototyping/atomic_energies/')
sys.path.insert(0, '/home/misa/git_repositories/APDFT/prototyping/atomic_energies/hitp')

import os
import glob
import numpy as np
import re

import alchemy_tools2 as at
from explore_qml_data import get_free_atom_data

#############################################
#                Set before running         #
#############################################

def get_paths(directory):
    # load data from cube files
    # paths_cubes = ['/home/misa/APDFT/prototyping/atomic_energies/results/slice_ve38/ueg/ve_00.cube'] # at avl24
    paths_cubes = ['/home/misa/projects/Atomic-Energies/data/ueg_reference/ueg/box20/ve_00.cube'] # at avl51
    paths2 = glob.glob(directory+'/cube-files/ve*')
    paths2.sort()
    paths_cubes.extend(paths2)
    return(paths_cubes)

#paths_cubes = get_paths('/home/misa/APDFT/prototyping/atomic_energies/results/slice_ve38/dsgdb9nsd_003886')
COMPOUND_PATH = sys.argv[1]
print(COMPOUND_PATH, flush=True)
cube_paths = get_paths(COMPOUND_PATH)
print(cube_paths, flush = True)
lam_vals, densities, nuclei, gpts, h_matrix = at.load_cube_data(cube_paths)

# atomic energy decomposition
nuclei, atomic_energies_with_repulsion, atomic_energies, alch_pots = at.atomic_energy_decomposition(lam_vals, densities, nuclei, gpts, h_matrix)
#print(atomic_energies_with_repulsion)
#print(atomic_energies)

# calculation of atomisation energy
# read total energy B3LYP from xyz-file in qm9 database
comp_name = re.search('dsgdb9nsd_......', COMPOUND_PATH)[0] # get the compound name from the path
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
save_dir = os.path.join(COMPOUND_PATH, 'atomic_energies_fine_grid.txt')
np.savetxt(save_dir, store, delimiter='\t', header = header)
