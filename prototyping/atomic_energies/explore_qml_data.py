#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 08:03:20 2019

@author: misa

Set of functions to explore datasets where data entries are qml.compounds
"""

import numpy as np
import qml
from matplotlib import pyplot as plt
import os

###############################################################################
###                                I/O-functions
###############################################################################
def load_compounds(path_list):
    """
    loads qm9 entries given in path_list in Compound objects, returns list with compounds
    """
    compounds = []
    for file in path_list:
        mol = qml.Compound(xyz=file)
        compounds.append(mol)
    return(compounds)

def paths_slice_ve(num_ve):
    """
    input: number of valence electrons
    return: paths to compounds of qm9 with the number of valence electrons specified in the input
    """
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
    return(compounds)

def write_paths_slice_ve(num_ve, path):
    """
    Description:
        writes the paths to qm9 entries with specified number of electrons in a txt-file
    input: 
        num_ve: number of valence electrons
        path:   path to file
    """
    compounds = paths_slice_ve(num_ve)
    with open(path, 'w') as f:
        for item in compounds:
            f.write("%s\n" % item)

###############################################################################
###                                size of molecule
###############################################################################

def find_largest_mol(compounds):
    """
    input list with elements of type qml.compound
    returns compound with the largest distance between geometric center and nucleus
    """
    idx_max = idx_largest_comp(compounds)  
    return(compounds[idx_max])
    
def max_dist_center_nuc(com):
    """
    returns the largest distance between geometric center and nucleus for qml.compound
    """
    centroid = np.mean(com.coordinates, axis=0)
    distance2centroid = np.linalg.norm(centroid - com.coordinates, axis = 1)
    max_dist = np.amax(distance2centroid)
    return( max_dist)
    
def idx_largest_comp(compounds):
    """
    input list with elements of type qml.compound
    returns index of compound with the largest distance between geometric center and nucleus
    """
    distances = np.empty(len(compounds))
    for idx, com in enumerate(compounds):
        distances[idx] = max_dist_center_nuc(com)
    idx_max_dist = np.where( distances == np.amax(distances) )
    return(idx_max_dist[0][0])
    
def max_dist_distribution(compounds):
    """
    input: list with elements of type qml.compound
    returns: largest distance between geometric center and nucleus for every compound
    """
    distances = np.empty(len(compounds))
    for idx, com in enumerate(compounds):
        distances[idx] = max_dist_center_nuc(com)
    return(distances)

###############################################################################
###                                other
###############################################################################

def get_num_val_elec(nuclear_charges):
    """
    calculates total number of valence electrons of molecule up to third row of periodic table
    input: list of nuclear charges
    return: total number of valence electrons
    """
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




