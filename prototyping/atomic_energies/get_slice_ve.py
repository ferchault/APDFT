#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 13:34:20 2019

@author: misa
"""

import qml
import numpy as np
from matplotlib import pyplot as plt
import os

# calculates number of valence electrons based on nuclear charges
def get_num_val_elec(nuclear_charges):
    num_val = 0
    for charge in nuclear_charges:
        el = 0
        if charge <=2:
            num_val += charge
        elif charge >= 3 and charge <= 10:
            el = charge - 2
            num_val += el
        elif charge >= 11 and charge <= 18:
            el = charge - 10
            num_val += el
        else:
            assert('Cannot calculate number of valence electrons!')
    return(num_val)



# build path to data
d = '/home/misa/datasets/qm9/'
path_list = [os.path.join(d, f) for f in os.listdir(d)]
path_list.sort()

#calculate number of valence electrons for every compound
compounds = []
num_val = np.empty(len(path_list), dtype=int)
for idx, file in enumerate(path_list):
    mol = qml.Compound(xyz=file)
    num_val[idx] = get_num_val_elec(mol.nuclear_charges)
    
    if num_val[idx] == 38:
        compounds.append(file)
        
with open('/home/misa/APDFT/prototyping/atomic_energies/results/val_el_38.txt', 'w') as f:
    for item in compounds:
        f.write("%s\n" % item)
f.close()

