#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 16:19:34 2019

@author: misa

calculates the atomisation energies for the compounds specified in dirs
sets density outside of cuoff_radius to zero
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

def get_integrals_box(density, gpts, nuclei, radii):
    """
    return value of integrals for different vdW for one boxsize
    """
    estat_int = []
    for r in radii:
        estat_int.append( get_integral_radius(density, gpts, nuclei, r) )
    return(np.array(estat_int))
        
def get_integral_radius(density, gpts, nuclei, radius):
    """
    return value of integral for one multiple of the vdW radii
    """
    
    # set density to zero outside of integration radius
    density_bound = generate_bound_density(density, nuclei, radius, gpts)

    # calculate integral for new density
    estat_int = at.calculate_atomic_energies(density_bound, nuclei, gpts, h_matrix)
    
    return(estat_int)
    
def generate_bound_density(density, nuclei, radius, gpts):
    # reshape into vectors
    density_flattened = density.flatten()
    gpts_flattened = at.meshgrid2vector(gpts)
    
    # get density grid points lying within the weighted vdW radii
    all_idc_in = get_all_idc_in(nuclei, radius, gpts_flattened)
    # get indices of all gridpoints as flattened 1D-array
    all_idc = np.arange(len(density_flattened))
     # get density grid points lying out of the weighted vdW radii
    all_idc_out = at.get_all_idc_out(all_idc_in, all_idc)
    
    # set density out of weighted vdW radii to zero
    density_flattened[all_idc_out] = 0.0
    # reshape flattened density to grid
    density_bound = density_flattened.reshape((density.shape))
    return(density_bound)
        
def get_all_idc_in(nuclei, radius, gpts_flattened):
    all_idc_in = []
    for n in nuclei:
        r_vdW = lookup_vdW(n[0])
        all_idc_in.append(at.get_idc_rvdW(np.array([n[1:4]]), r_vdW*radius, gpts_flattened))
    return(all_idc_in)
   
from ase.units import Bohr
def lookup_vdW(Z, unit = 'Bohr'):
    """
    return van der Waals radius for element
    from Bondi J. Phys. Chem 68, 1964
    """
    
    if Z == 1:
        r_w = 1.2
    elif Z == 6:
        r_w = 1.7
    elif Z == 7:
        r_w = 1.55
    elif Z == 8:
        r_w = 1.52
    else:
        raise Exception("r_w not implemented for Z = {}".format(Z))
        
    if unit == 'Bohr':
        r_w /= Bohr
    return(r_w)

# cutoff radius (multiples of bohr radius)
cradius = 3.0

# paths to the compounds
dirs = concatenate_files(['/home/misa/APDFT/prototyping/atomic_energies/results/slice_ve38/paths_atomic_energies'])
#dirs = concatenate_files(['/home/misa/APDFT/prototyping/atomic_energies/results/slice_ve38/restart_cals'])

#dirs = ['/home/misa/APDFT/prototyping/atomic_energies/results/slice_ve38/dsgdb9nsd_001212/']
for compound_path in dirs:
    print("entering directory {}".format(compound_path))
    # paths to the cube files
    base = compound_path
    base = base + 'cube-files/'
    cubes = ['ve_8.cube', 've_15.cube', 've_23.cube', 've_30.cube', 've_38.cube']
    for i in range(len(cubes)):
        cubes[i] = base + cubes[i]
        
    path_to_ueg = '/home/misa/APDFT/prototyping/atomic_energies/results/slice_ve38/ueg/ve_00.cube'
    cubes.insert(0, path_to_ueg)

    # load
    print("loading density")
    lam_vals, densities, nuclei, gpts, h_matrix = at.load_cube_data(cubes)
    # set density to zero at lambda = 0.0 since we treat system with infinitely large box
    densities[0].fill(0.0)
    
    # cut density
    print("cutting density")
    for i,d in enumerate(densities):
        dens_b = generate_bound_density(d, nuclei, cradius, gpts)
        densities[i] = dens_b
    
    # atomic energy decomposition
    print("calculate atomic energies")
    nuclei, atomic_energies_with_repulsion, atomic_energies, alch_pots = at.atomic_energy_decomposition(lam_vals, densities, nuclei, gpts, h_matrix)
    #print(atomic_energies_with_repulsion)
    #print(atomic_energies)
    # calculation of atomisation energy
    
    print("calculate atomisation energies")
    # read total energy B3LYP from xyz-file in qm9 database
    comp_name = re.search('dsgdb9nsd_......', base)[0] # get the compound name from the path
    total_energy = at.get_property(os.path.join('/home/misa/datasets/qm9/', comp_name + '.xyz'), 'U0')
    
    # read energies B3LYP of free atoms
    free_atoms_dict = get_free_atom_data()
    # free atom energy for every atom in compound
    free_atom_energies = at.get_free_atom_energies(nuclei[:,0], free_atoms_dict)
    
    
    # calculate atomic atomisation energies
    atomisation_energies = at.calculate_atomisation_energies(atomic_energies, total_energy, free_atom_energies)
    atomisation_energies_rep = at.calculate_atomisation_energies(atomic_energies_with_repulsion, total_energy, free_atom_energies)
    
    # write atomic energies and alchemical potentials to file
    print("save results")
    store = np.array([nuclei[:,0], nuclei[:,1], nuclei[:,2], nuclei[:,3], alch_pots, atomic_energies, atomisation_energies, atomisation_energies_rep]).T
    header = 'charge\t x_coord\t y_coord\t z_coord\t alchemical_potential\t atomic_energies\t atomisation_energies\t atomisation_energies_repulsion'
    save_dir = os.path.join(compound_path, f'atomic_energies_cutoff_{cradius}.txt')
    np.savetxt(save_dir, store, delimiter='\t', header = header)
