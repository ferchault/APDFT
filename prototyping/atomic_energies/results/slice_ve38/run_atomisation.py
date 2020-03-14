#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 16:19:34 2019

@author: misa
"""

import sys
sys.path.insert(0, '/home/misa/APDFT/prototyping/atomic_energies/')
sys.path.insert(0, '/home/misa/APDFT/prototyping/atomic_energies/hitp')

<<<<<<< HEAD
from alchemy_tools import write_atomisation_energies
=======
from alchemy_tools import write_atomisation_energies, test_impact_lambda
>>>>>>> 51f21a2... unmerged error
import os
os.chdir('/home/misa/APDFT/prototyping/atomic_energies/results/slice_ve38')
from find_converged import concatenate_files

dirs = concatenate_files(['/home/misa/APDFT/prototyping/atomic_energies/results/slice_ve38/paths_atomic_energies'])

<<<<<<< HEAD
write_atomisation_energies(dirs)
=======
#write_atomisation_energies(dirs)
test_impact_lambda(dirs)
>>>>>>> 51f21a2... unmerged error
