#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 14:09:36 2020

@author: misa
"""

import numpy as np

import sys
sys.path.insert(0, '/home/misa/APDFT/prototyping/atomic_energies/')
import glob

import alchemy_tools2 as at
from parse_density_files import CUBE

#def test_calculate_atomic_energies():

import unittest

class TestAlchemyTools2(unittest.TestCase):

    def test_nuclear_repulsion(self):
        # reference data
        
        # hydrogen molecule
        nuclei_h2 = np.array([[1, 0, 0, 0], [1, 1.398397333119327,0,0]])
        rep_ref_h2 = np.array([0.5*1/1.398397333119327,0.5*1/1.398397333119327])
        # execute
        rep_test_h2 = at.nuclear_repulsion(nuclei_h2[:,0], nuclei_h2[:, 1:4])
        # test
        self.assertTrue(np.array_equal(rep_ref_h2, rep_test_h2))
        
        
        # dsgdb9nsd_001212
        path = '/home/misa/APDFT/prototyping/atomic_energies/results/slice_ve38/dsgdb9nsd_001212/atomic_energies_with_mic.txt'
        data_ref = np.loadtxt(path)
        
        # calculate element in middle
        atom_position = data_ref[5][1:4]
        atom_charge = data_ref[5][0]
        
        repulsion_ref = 0
        for i in range(len(data_ref)):
            if i != 5:
                other_charge = data_ref[i][0]
                other_position = data_ref[i][1:4]
                repulsion_ref += atom_charge*other_charge/np.linalg.norm(atom_position - other_position)
        repulsion_ref /= 2
        # execute
        repulsion_test_dsgdb9nsd_001212 = at.nuclear_repulsion(data_ref[:,0], data_ref[:, 1:4])
        
        #self.assertTrue(np.array_equal(repulsion_ref, repulsion_test_dsgdb9nsd_001212[5]))
        self.assertTrue(np.allclose(repulsion_ref, repulsion_test_dsgdb9nsd_001212[5]))
        
        
#    def test_atomic_energy_decomposition(self):
#        
#        # generate reference data
#        ve38 = CUBE(r'/home/misa/APDFT/prototyping/atomic_energies/results/test_calculations/lambda_integrals/dsgdb9nsd_002626/DENS_38.cube')
#        ve30 = CUBE(r'/home/misa/APDFT/prototyping/atomic_energies/results/test_calculations/lambda_integrals/dsgdb9nsd_002626/DENS_30.cube')
#        ve23 = CUBE(r'/home/misa/APDFT/prototyping/atomic_energies/results/test_calculations/lambda_integrals/dsgdb9nsd_002626/DENS_23.cube')
#        ve15 = CUBE(r'/home/misa/APDFT/prototyping/atomic_energies/results/test_calculations/lambda_integrals/dsgdb9nsd_002626/DENS_15.cube')
#        ve8 = CUBE(r'/home/misa/APDFT/prototyping/atomic_energies/results/test_calculations/lambda_integrals/dsgdb9nsd_002626/DENS_8.cube')
#        ve0 = CUBE(r'/home/misa/APDFT/prototyping/atomic_energies/results/test_calculations/lambda_integrals/free_electron_gas.cube')
#        
#        lambda_densities = [ve0, ve8, ve15, ve23, ve30, ve38]
#        density_arrays = []
#        for dens in lambda_densities:
#            dens.scale()
#            density_arrays.append(dens.data_scaled)
#        lam_vals = [0, 8/38, 15/38, 23/38, 30/38, 1]
#        
#        av_dens = at.integrate_lambda_density(density_arrays, lam_vals)
#        
#        nuclei = ve38.atoms
#        meshgrid = ve38.get_grid()
#        atomic_energies, al_pot = at.calculate_atomic_energies(av_dens, nuclei, meshgrid)
#        
#        # input data
#        p38 = '/home/misa/APDFT/prototyping/atomic_energies/results/test_calculations/lambda_integrals/dsgdb9nsd_002626/DENS_38.cube'
#        p30 = '/home/misa/APDFT/prototyping/atomic_energies/results/test_calculations/lambda_integrals/dsgdb9nsd_002626/DENS_30.cube'
#        p23 = '/home/misa/APDFT/prototyping/atomic_energies/results/test_calculations/lambda_integrals/dsgdb9nsd_002626/DENS_23.cube'
#        p15 = '/home/misa/APDFT/prototyping/atomic_energies/results/test_calculations/lambda_integrals/dsgdb9nsd_002626/DENS_15.cube'
#        p8 = '/home/misa/APDFT/prototyping/atomic_energies/results/test_calculations/lambda_integrals/dsgdb9nsd_002626/DENS_8.cube'
#        p0 = '/home/misa/APDFT/prototyping/atomic_energies/results/test_calculations/lambda_integrals/free_electron_gas.cube'
#        cubes = [(p0, 0), (p8, 8/38), (p15, 15/38), (p23, 23/38), (p30, 30/38), (p38, 1)]
#        
#        # run test function
#        atomic_energies_test, al_pot_test = at.atomic_energy_decomposition(cubes)
#        
#        #comparison
#        t1 = np.allclose(atomic_energies, atomic_energies_test)
#        t2 = np.allclose(al_pot, al_pot_test)
#        
#        return(t1, t2)
#    
#    def test_write_atomisation_energies(self):
#        # data from analysis/lambda_integrals.ipny
#        ref_part=np.loadtxt('/home/misa/APDFT/prototyping/atomic_energies/results/test_calculations/lambda_integrals/output')
#        #    ref_charge = ref[:,0]
#        #    ref_x = ref[:,1]
#        #    ref_y = ref[:,2]
#        #    ref_z = ref[:,3]
#        #    ref_alch = ref[:, 4]
#        #    ref_adec = ref[:, 5]
#        
#        ref_ae = np.array([23.53488468, -0.07943982, -0.76277164, -1.152305  ,  0.15735545,
#            2.43826624,  6.33442572, -4.00594386, -4.31814562, -4.25600311,
#           -4.39080099, -4.40580498, -4.14793128, -4.14118554, -3.15357225])
#        ref = np.empty( ( len(ref_part), len(ref_part.T)+1 ) )
#        for i in range(0, len(ref.T)):
#            if i < len(ref.T)-1:
#                ref[:, i] = ref_part[:, i] 
#            else:
#                ref[:, i] = ref_ae
#        # input data was generated with run_atomisation.py
#        result=np.loadtxt('/home/misa/APDFT/prototyping/atomic_energies/results/slice_ve38/dsgdb9nsd_002626/atomic_energies.txt')
#        for idx in range(0, len(result.T)):
#            print(np.allclose(ref[:, idx],result[:, idx]))
#        
#    def test_load_cube_data(self):
#        # paths to cube-files
#        paths_cubes = ['/home/misa/APDFT/prototyping/atomic_energies/results/slice_ve38/dsgdb9nsd_003886/cube-files/ve_04.cube',
#                         '/home/misa/APDFT/prototyping/atomic_energies/results/slice_ve38/dsgdb9nsd_003886/cube-files/ve_08.cube',
#                         '/home/misa/APDFT/prototyping/atomic_energies/results/slice_ve38/dsgdb9nsd_003886/cube-files/ve_15.cube',
#                         '/home/misa/APDFT/prototyping/atomic_energies/results/slice_ve38/dsgdb9nsd_003886/cube-files/ve_22.cube']
#        
#        # reference data
#        lam_vals_ref = [4/38, 8/38, 15/38, 22/38]
#        
#        densities_ref = []
#        for cube in paths_cubes:
#            tmp = CUBE(cube)
#            densities_ref.append(tmp.data_scaled)
#        
#        cube_obj = CUBE(paths_cubes[len(paths_cubes)-1])
#        nuclei_ref = cube_obj.atoms
#        gpts_ref = cube_obj.get_grid()
#        
#        # execute
#        lam_vals, densities, nuclei, gpts = at.load_cube_data(paths_cubes)
#        
#        # test lam_vals
#        self.assertListEqual(lam_vals_ref, lam_vals)
#        # test densities
#        for i in range(len(densities)):
#            self.assertTrue(np.array_equal(densities_ref[i], densities[i]))
#        # test nuclei
#        self.assertTrue(np.array_equal(nuclei_ref, nuclei))
#        self.assertTrue(np.array_equal(gpts_ref, gpts))
        
        
if __name__ == '__main__':
    unittest.main()
        
        
        
        
        
        
        
        
        
        
        
        
        
        

    