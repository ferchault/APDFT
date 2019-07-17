#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 10:55:23 2019

@author: misa
"""
import qml
import numpy as np

from atom_section import generate_atom_section, shift2center
from read_file import read_till_keyword

compound = qml.Compound(xyz='/home/misa/datasets/qm9/dsgdb9nsd_014656.xyz')
boxsize = np.array([20.0, 20, 20])



###############################################################################
#                           generate atom section                             #
###############################################################################
# shift coordinates
boxcenter = boxsize/2
compound.coordinates = shift2center(compound.coordinates, boxcenter)
# generate atom section
at_sec = generate_atom_section(compound)

###############################################################################
#                           read input template                               #
###############################################################################
template_initial_run = '/home/misa/APDFT/prototyping/atomic_energies/input-template/run-1/run.inp'
input_file = read_till_keyword(template_initial_run, 'ATOMS')

###############################################################################
#           merge template and atom section and write new file                #
###############################################################################
input_file.extend(at_sec)
calc_dir = '/home/misa/APDFT/prototyping/atomic_energies/results/calculations/fractional_charge/dsgdb9nsd_014656/initial_run/run.inp'
with open(calc_dir, 'w') as f:
    f.writelines(input_file)