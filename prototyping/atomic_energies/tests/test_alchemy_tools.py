#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 13:16:00 2019

@author: misa
"""

import numpy as np

import sys
sys.path.insert(0, '/home/misa/APDFT/prototyping/atomic_energies/')

import alchemy_tools as at
from parse_cube_files import CUBE

#def test_calculate_atomic_energies():
    

def test_atomic_energy_decomposition():
    
    # generate reference data
    ve38 = CUBE(r'/home/misa/APDFT/prototyping/atomic_energies/results/test_calculations/lambda_integrals/dsgdb9nsd_002626/DENS_38.cube')
    ve30 = CUBE(r'/home/misa/APDFT/prototyping/atomic_energies/results/test_calculations/lambda_integrals/dsgdb9nsd_002626/DENS_30.cube')
    ve23 = CUBE(r'/home/misa/APDFT/prototyping/atomic_energies/results/test_calculations/lambda_integrals/dsgdb9nsd_002626/DENS_23.cube')
    ve15 = CUBE(r'/home/misa/APDFT/prototyping/atomic_energies/results/test_calculations/lambda_integrals/dsgdb9nsd_002626/DENS_15.cube')
    ve8 = CUBE(r'/home/misa/APDFT/prototyping/atomic_energies/results/test_calculations/lambda_integrals/dsgdb9nsd_002626/DENS_8.cube')
    ve0 = CUBE(r'/home/misa/APDFT/prototyping/atomic_energies/results/test_calculations/lambda_integrals/free_electron_gas.cube')
    
    lambda_densities = [ve0, ve8, ve15, ve23, ve30, ve38]
    density_arrays = []
    for dens in lambda_densities:
        dens.scale()
        density_arrays.append(dens.data_scaled)
    lam_vals = [0, 8/38, 15/38, 23/38, 30/38, 1]
    
    av_dens = at.integrate_lambda_density(density_arrays, lam_vals)
    
    nuclei = ve38.atoms
    meshgrid = ve38.get_grid()
    atomic_energies, al_pot = at.calculate_atomic_energies(av_dens, nuclei, meshgrid)
    
    # input data
    p38 = '/home/misa/APDFT/prototyping/atomic_energies/results/test_calculations/lambda_integrals/dsgdb9nsd_002626/DENS_38.cube'
    p30 = '/home/misa/APDFT/prototyping/atomic_energies/results/test_calculations/lambda_integrals/dsgdb9nsd_002626/DENS_30.cube'
    p23 = '/home/misa/APDFT/prototyping/atomic_energies/results/test_calculations/lambda_integrals/dsgdb9nsd_002626/DENS_23.cube'
    p15 = '/home/misa/APDFT/prototyping/atomic_energies/results/test_calculations/lambda_integrals/dsgdb9nsd_002626/DENS_15.cube'
    p8 = '/home/misa/APDFT/prototyping/atomic_energies/results/test_calculations/lambda_integrals/dsgdb9nsd_002626/DENS_8.cube'
    p0 = '/home/misa/APDFT/prototyping/atomic_energies/results/test_calculations/lambda_integrals/free_electron_gas.cube'
    cubes = [(p0, 0), (p8, 8/38), (p15, 15/38), (p23, 23/38), (p30, 30/38), (p38, 1)]
    
    # run test function
    atomic_energies_test, al_pot_test = at.atomic_energy_decomposition(cubes)
    
    #comparison
    t1 = np.allclose(atomic_energies, atomic_energies_test)
    t2 = np.allclose(al_pot, al_pot_test)
    
    return(t1, t2)