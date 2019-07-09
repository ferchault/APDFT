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


###############################################################################
#                     copy pp-files to calculation directory                  #
###############################################################################


# shifts set of coordinates so that centroid is at centroid_final
def shift2center(coordinates_initial, centroid_final):
    centroid_initial = np.mean(coordinates_initial, axis=0)
    shift = centroid_final - centroid_initial
    return(coordinates_initial+shift)

# copy pp file for element specified by atomsym from pp-directory into directory for calculation calc_dir
def copy_pp_file(atomsym, filename_dest, calc_dir):
    filename_parent = atomsym + '_SG_LDA' 
    pp_path = os.path.join('/home/misa/software/PP_LIBRARY/', filename_parent)
    calc_path = os.path.join(calc_dir, filename_dest)
    copyfile(pp_path, calc_path)

# creates name of pp-file (atomsymbol followed by index of atom in compound and standard pp-file description)    
def generate_pp_file_name(idx, atomsym):
    return(atomsym + str(idx) + '_SG_LDA')

def generate_pp_file(calc_dir, idx, atomsym):
    pp_file_name_dest = generate_pp_file_name(idx, atomsym)
    copy_pp_file(atomsym, pp_file_name_dest, calc_dir)

def get_pp_files(calc_dir, compound):
    for idx in range(0, len(compound.atomtypes)):
        generate_pp_file(calc_dir, idx, compound.atomtypes[idx])

###############################################################################

#pp_dir = '/home/misa/software/PP_LIBRARY/'
compound = qml.Compound(xyz='/home/misa/datasets/qm9/dsgdb9nsd_014656.xyz')
calc_dir = '/home/misa/APDFT/prototyping/atomic_energies/results/calculations/dsgdb9nsd_014656/boxsize23/'

# generate pp's
get_pp_files(calc_dir, compound)


