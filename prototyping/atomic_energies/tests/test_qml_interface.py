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
    
#    def test_load_alchemy_data():
#        # reference data
#        path = '/home/misa/APDFT/prototyping/atomic_energies/results/slice_ve38/finished_paths'
#        paths = []
#        with open(path, 'r') as f:
#            for line in f:
#                paths.append(line.split())
#        # matrix for all representations
#        alchemy_data = []
#        molecule_size = np.zeros(len(paths), dtype=np.intc)
#        for idx, path in enumerate(paths):
#            alch = np.loadtxt(path[1])
#            molecule_size[idx] = len(alch[:,0])
#            alchemy_data.append(alch)
#        
#        max_size = np.amax(molecule_size)
#        
    
if __name__ == '__main__':
    unittest.main()