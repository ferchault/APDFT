#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 20:26:04 2019

@author: misa
"""

import sys
sys.path.insert(0, '/home/misa/APDFT/prototyping/atomic_energies/')
import qml_interface as qi
import unittest
import numpy as np
import qml
import qml.distance

class TestQmlInterface(unittest.TestCase):

    def test_wrapper_alch_data(self):     
        # reference data
        path = '/home/misa/APDFT/prototyping/atomic_energies/results/slice_ve38/finished_abs'
        paths = []
        with open(path, 'r') as f:
            for line in f:
                paths.append(line.rstrip('\n'))
        # execute
        paths_qi = qi.wrapper_alch_data()
    
        # test
        self.assertCountEqual(paths, paths_qi)
    
    def test_load_alchemy_data(self):
        # reference data
        paths = qi.wrapper_alch_data()
        
        alchemy_data = []
        molecule_size = np.zeros(len(paths), dtype=np.intc)
        for idx, path in enumerate(paths):
            alch = np.loadtxt(path)
            molecule_size[idx] = len(alch[:,0])
            alchemy_data.append(alch)
            
        # execute
        alch_qi, mol_qi = qi.load_alchemy_data(paths)
        
        # test
        for idx in range(0, len(alch_qi)):
            self.assertTrue(np.array_equal(alchemy_data[idx], alch_qi[idx]))
        self.assertTrue(np.array_equal(molecule_size, mol_qi))

    def test_generate_atomic_representations(self):
        # reference data
        paths = qi.wrapper_alch_data()
        alchemy_data, molecule_size = qi.load_alchemy_data(paths)
        # generate all representations
        max_size = np.amax(molecule_size)
        largest_rep = int(max_size*(max_size+1)/2)
        full_matrix = np.zeros((np.sum(molecule_size), largest_rep))
        start = 0
        for idx, molecule in enumerate(alchemy_data):
        #     rep = qml.representations.generate_atomic_coulomb_matrix(molecule[:,0], molecule[:,[1,2,3]], size=max_size, sorting='distance', interaction_cutoff=1000, interaction_decay=-1, central_cutoff=1000, central_decay=-1)
            rep = qml.representations.generate_atomic_coulomb_matrix(molecule[:,0], molecule[:,[1,2,3]], size=max_size, sorting='distance')
        
            for idx2 in range(0, len(rep)):
                full_matrix[start+idx2] = rep[idx2]
            start += len(rep)
        
        # execute
        full_rep_qi = qi.generate_atomic_representations(alchemy_data, molecule_size)
        # test
        self.assertTrue(np.array_equal(full_matrix, full_rep_qi))
        
    def test_calculate_distances(self):
        # reference data
        paths = qi.wrapper_alch_data()
        alchemy_data, molecule_size = qi.load_alchemy_data(paths)
        full_matrix = qi.generate_atomic_representations(alchemy_data, molecule_size)
        
        dist_shape = int(len(full_matrix)*(len(full_matrix)+1)/2)
        start=0
        width = len(full_matrix)
        dist = np.empty(dist_shape)
        for idx in range(0, len(full_matrix)):
            dist[start:start+width] = np.linalg.norm(full_matrix[idx]-full_matrix[idx:], axis=1)
            start = start+width
            width -= 1
        
        # execute
        dist_qi = qi.calculate_distances(full_matrix)
        # test
        self.assertTrue(np.allclose(dist, dist_qi))

    def test_generate_label_vector(self):
        # reference data
        paths = qi.wrapper_alch_data()
        alchemy_data, molecule_size = qi.load_alchemy_data(paths)
        full_matrix = qi.generate_atomic_representations(alchemy_data, molecule_size)

        energies = np.zeros(len(full_matrix))
        start = 0
        for idx, mol in enumerate(alchemy_data):
            length = len(alchemy_data[idx][:,6])
            energies[start:length+start] = alchemy_data[idx][:,6]
            start += length
            
        # execute
        label_qi = qi.generate_label_vector(alchemy_data, molecule_size.sum())
        # test
        self.assertTrue(np.array_equal(energies, label_qi))
        
    def test_get_local_idx(self):
        #######################################################################
        #                    select representation
        #######################################################################
        
        # reference data
        paths=qi.wrapper_alch_data()
        alchemy_data, molecule_size = qi.load_alchemy_data(paths)
        max_size = np.amax(molecule_size)
        idx = [3, 120, 435]
        
        molecule = alchemy_data[idx[0]]
        rep_ref0 = qml.representations.generate_atomic_coulomb_matrix(molecule[:,0], molecule[:,[1,2,3]], size=max_size, sorting='distance')
        
        molecule = alchemy_data[idx[1]]
        rep_ref1 = qml.representations.generate_atomic_coulomb_matrix(molecule[:,0], molecule[:,[1,2,3]], size=max_size, sorting='distance')
        
        molecule = alchemy_data[idx[2]]
        rep_ref2 = qml.representations.generate_atomic_coulomb_matrix(molecule[:,0], molecule[:,[1,2,3]], size=max_size, sorting='distance')
        
        ref_rows = molecule_size[idx[0]]+molecule_size[idx[1]]+molecule_size[idx[2]]
        rep_ref = np.empty((ref_rows, int(max_size*(max_size+1)/2)))
        rep_ref[0:molecule_size[idx[0]]] = rep_ref0
        rep_ref[molecule_size[idx[0]] : molecule_size[idx[0]]+molecule_size[idx[1]] ] = rep_ref1
        rep_ref[molecule_size[idx[0]]+molecule_size[idx[1]] : molecule_size[idx[0]]+molecule_size[idx[1]]+molecule_size[idx[0]] ] = rep_ref2
        
        full_local_rep = qi.generate_atomic_representations(alchemy_data, molecule_size)
        # execute
        local_idx = qi.get_local_idx(idx, molecule_size)
        rep_test = full_local_rep[local_idx]
        
        # test
        self.assertTrue(np.array_equal(rep_ref, rep_test))
        
        #######################################################################
        #                    select kernel
        #######################################################################
        
        # reference
        kernel_ref = qml.kernels.gaussian_kernel(rep_ref, rep_ref, 1000)
        
        full_local_kernel = qml.kernels.gaussian_kernel(full_local_rep, full_local_rep, 1000)
        local_idx = qi.get_local_idx(idx, molecule_size)
        kernel_test = qi.select_sub_matrix(full_local_kernel, local_idx, local_idx)
        
        self.assertTrue(np.array_equal(kernel_ref, kernel_test))
        
        
    def test_optimize_hyperpar(self):
        # reference data
        paths=qi.wrapper_alch_data()
        # load data into list, count number of atoms per molecule
        alchemy_data, molecule_size = qi.load_alchemy_data(paths)
        local_reps = qi.generate_atomic_representations(alchemy_data, molecule_size)
        
        local_labels = qi.generate_label_vector(alchemy_data, molecule_size.sum())
        
        training_set_size = 100
        global_idc = qi.get_indices(len(alchemy_data), training_set_size)
        local_idc = qi.get_local_idx(global_idc[0], molecule_size), qi.get_local_idx(global_idc[1], molecule_size)
        
        rep = local_reps[local_idc[0]], local_reps[local_idc[1]]
        labels = local_labels[local_idc[0]], local_labels[local_idc[1]] # labels for training and validation
        sigmas = np.logspace(-1, 4, 12) #14)
        lams = np.logspace(-15, 0, 16)#16)
        
        # execute
        out = qi.optimize_hypar(rep, labels, sigmas, lams)
        results = out[0][np.where(np.amin(out[0][:,2])==out[0])[0]] # best sigma, lambda, mean error
        results = results.reshape(3)
        
        # generate reference from results
        tr_kernel_ref = qml.kernels.gaussian_kernel(rep[0], rep[0], results[0])
        val_kernel_ref = qml.kernels.gaussian_kernel(rep[1], rep[0], results[0])
        
        tr_kernel_ref = tr_kernel_ref + np.identity(len(tr_kernel_ref))*results[1]
        coeffs_ref = qml.math.cho_solve(tr_kernel_ref, labels[0])
            
        # validation
        val_errors_ref = np.abs( np.dot(val_kernel_ref, coeffs_ref) - labels[1] ) 
        val_err_mean_ref = val_errors_ref.mean()
        
        # test
        self.assertTrue( np.array_equal(val_err_mean_ref, results[2]) )
        self.assertTrue( np.array_equal(coeffs_ref, out[1]) )
        self.assertTrue( np.array_equal(val_errors_ref, out[2]) )
        
        
if __name__ == '__main__':
    unittest.main()