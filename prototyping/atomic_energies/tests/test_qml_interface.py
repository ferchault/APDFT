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


if __name__ == '__main__':
    unittest.main()