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

# plot results 
occurences = np.bincount(num_val)
electron_number = np.linspace(0, 56, 57)
plt.bar(electron_number, occurences)
plt.xlabel('# of valence electrons')
plt.ylabel('# of molecules')

#fig, ax = plt.subplots(1, 2)
#ax[0].bar(electron_number, occurences)
#ax[0].set_xlabel('# of valence electrons')
#ax[0].set_ylabel('# of molecules')
#ax[1].bar(electron_number, np.log10(occurences))
#ax[1].set_xlabel('# of valence electrons')
#ax[1].set_ylabel(r'$\log_{10}$ (# of molecules)')