#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 17:41:37 2019

@author: misa
"""

import qml
import numpy as np

def generate_atom_section(compound):
    """
    concantenates inputs for individual atoms to one list where each element is one line of the input file
    """
    atom_section = ['&ATOMS\n']
    for idx in range(0, len(compound.atomtypes)):
        atom = generate_atom(compound.atomtypes[idx], compound.coordinates[idx])
        atom_section.extend(atom)
    atom_section.append('&END')
    return(atom_section)


def generate_atom(atomsym, coordinates):
    """
    prepare the input for one atom:
    the name of the pp-files is is 'element_name' + element symbol '_SG_LDA'
    the coordinates are read from Compund as well (must be shifted to center before)
    """
    line1 = '*' + atomsym + '_SG_LDA FRAC' + '\n'
    line2 = ' LMAX=S\n'
    line3 = ' 1\n'
    line4 = ' ' + str(coordinates[0]) + ' ' + str(coordinates[1]) + ' ' + str(coordinates[2]) + '\n'
    return( [line1, line2, line3, line4] )

def shift2center(coordinates_initial, centroid_final):
    centroid_initial = np.mean(coordinates_initial, axis=0)
    shift = centroid_final - centroid_initial
    return(coordinates_initial+shift)

#input from outside
#compound = qml.Compound(xyz='/home/misa/datasets/qm9/dsgdb9nsd_014656.xyz')
#boxsize = np.array([20.0, 20, 20])
#
## shift coordinates
#boxcenter = boxsize/2
#compound.coordinates = shift2center(compound.coordinates, boxcenter)
## generate atom section
#at_sec = generate_atom_section(compound)