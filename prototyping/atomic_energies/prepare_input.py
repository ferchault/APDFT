#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 18:36:56 2019

@author: misa
"""

import qml
import numpy as np
from matplotlib import pyplot as plt
import os
from shutil import copyfile

pp_dir = '/home/misa/software/PP_LIBRARY/'
calc_dir = '/home/misa/APDFT/prototyping/atomic_energies/results/calculations/test/'

# shifts set of coordinates so that centroid is at centroid_final
def get_centroid(coordinates_initial, centroid_final):
    centroid_initial = np.mean(coordinates_initial, axis=0)
    shift = centroid_final - centroid_initial
    return(coordinates_initial+shift)

def copy_pp_file(idx, atomsym):
    filename_parent = atomsym + '_SG_LDA'
    filename = atomsym + str(idx) + '_SG_LDA'
    pp_path = os.path.join(pp_dir, filename_parent)
    calc_path = os.path.join(calc_dir, filename)
    copyfile(pp_path, calc_path)

def generate_pp_file(idx, atomsym, atom_charge, lam):
    copy_pp_file(idx, atomsym)
    


path = '/home/misa/datasets/qm9/dsgdb9nsd_000226.xyz'

molecule = qml.Compound(xyz=path)
box_center = np.array([7.5, 7.5, 7.5])
centered_coordinates = get_centroid(molecule.coordinates, box_center)

pp_path = generate_pp_file(1, molecule.atomtypes[1], molecule.nuclear_charges[1], 0)