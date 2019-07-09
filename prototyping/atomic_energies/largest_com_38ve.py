#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 08:03:20 2019

@author: misa
"""

import numpy as np
import qml


# load qm9 entries (at given paths) in Compound objects, returns list with compounds
def load_compounds(path_list):
    compounds = []
    for file in path_list:
        mol = qml.Compound(xyz=file)
        compounds.append(mol)
    return(compounds)

def find_largest_mol(compounds):
    
    idx_max = idx_largest_comp(compounds)
    
    return(compounds[idx_max])
    
def max_dist_center_nuc(com):
    centroid = np.mean(com.coordinates, axis=0)
    distance2centroid = np.linalg.norm(centroid - com.coordinates, axis = 1)
    max_dist = np.amax(distance2centroid)
    return( max_dist)
    
def idx_largest_comp(compounds):
    distances = np.empty(len(compounds))
    for idx, com in enumerate(compounds):
        distances[idx] = max_dist_center_nuc(com)
    idx_max_dist = np.where( distances == np.amax(distances) )
    return(idx_max_dist[0][0])
# get paths of slice with 38 ve
path_list = []
with open('/home/misa/APDFT/prototyping/atomic_energies/results/val_el_38.txt') as file:
    path_list = [line.rstrip('\n') for line in file]
    
compounds = load_compounds(path_list)
largest_comp = find_largest_mol(compounds)