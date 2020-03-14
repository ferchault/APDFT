#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 16:59:12 2020

@author: misa
"""
import numpy as np

import sys
sys.path.insert(0, '/home/misa/APDFT/prototyping/atomic_energies/')
import alchemy_tools2 as at


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
    
# get path to density file
#basepath = [sys.argv[1]+'/']
main = '/home/misa/APDFT/prototyping/atomic_energies/results/slice_ve38/'
comps = ['dsgdb9nsd_001212', 'dsgdb9nsd_003712', 'dsgdb9nsd_003886', 'dsgdb9nsd_002626', 'dsgdb9nsd_014656','dsgdb9nsd_000228', 'dsgdb9nsd_002025','dsgdb9nsd_002900','dsgdb9nsd_001442','dsgdb9nsd_003727']
paths = [main + p + '/' for p in comps]

for basepath in paths:
    print(basepath)
    
    # generate density and all the other variables
#    dens, nuclei, gpts, h_matrix = at.load_vasp_dens([basepath[0]+'CHG'])
    lam,dens, nuclei, gpts, h_matrix = at.load_cube_data([basepath +'cube-files/'+'ve_8.cube'])
    
    # calculate :
    # a) density within certain radius to nuclei
    # b) integral of density over distance to nuclei
    energies = []
    alch_pots = []
    radii = [0.5, 1, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 7.0]
    #radii = [0.5, 1, 1.5]
    for r in radii:
        print('generating bound density radius = {}'.format(r))
        dens_b = generate_bound_density(dens, nuclei, r, gpts)[0]
        print('calculating atomic energy')
        estat_int = at.calculate_atomic_energies(dens_b, nuclei, gpts, h_matrix)
        
        energies.append(estat_int[1])
        alch_pots.append(estat_int[2])
    #    np.save(basepath[0]+'energies_radius', energies, allow_pickle=False)
        
    #    # save density projections for certain density
    #    pr01 = dens_b.sum(axis=(0,1))
    #    pr02 = dens_b.sum(axis=(0,2))
    #    pr12 = dens_b.sum(axis=(1,2))
    #    
    #    main = basepath[0] + 'radius{}_'.format(r)
    #    f0 = main + 'pr0.txt'
    #    f1 = main + 'pr1.txt'
    #    f2 = main + 'pr2.txt'
    #    
    #    f01 = main + 'pr01.txt'
    #    f02 = main + 'pr02.txt'
    #    f12 = main + 'pr12.txt'
    #    
    #    files = [f0, f1, f2, f01, f02, f12]
    #    axis = [0, 1, 2, (0,1), (0,2), (1,2)]
    #    
    #    for s in zip(files, axis):
    #        pr = dens_b.sum(axis=s[1])
    #        np.save(s[0], pr, allow_pickle=False)
    
    energies = np.array(energies)
    alch_pots = np.array(alch_pots)
    print('saving results')
    np.save(basepath+'radius_energies', energies, allow_pickle=False)
    np.save(basepath+'radius_alchpots', alch_pots, allow_pickle=False)

    
    

    
    
    
    
    
    