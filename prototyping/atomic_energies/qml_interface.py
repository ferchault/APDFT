#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 11:37:43 2019

@author: misa
"""

import qml
import numpy as np

def wrapper_alch_data():
    """
    returns paths to files from file with all directories
    """
    # load paths to data
    path = '/home/misa/APDFT/prototyping/atomic_energies/results/slice_ve38/finished_paths'
    paths = []
    
    with open(path, 'r') as f:
        for line in f:
            paths.append(line.split())
    return(paths)

def load_alchemy_data(paths):
    """
    loads information about molecules into list
    each element contains charge coordinates, alchemical potential, raw atomic energies
    and atomic atomisation energies
    
    paths: list of paths to atomisation energy files
    """
    
    # matrix for all representations
    alchemy_data = []
    molecule_size = np.zeros(len(paths), dtype=np.intc)
    for idx, path in enumerate(paths):
        alch = np.loadtxt(path[1])
        molecule_size[idx] = len(alch[:,0])
        alchemy_data.append(alch)
    
    return(alchemy_data, molecule_size)
    
def generate_atomic_representations(alchemy_data, molecule_size, rep='coulomb'):
    """
    generates the local representations for every atom
    returns a 2D numpy array where every row contains the representation for one atom
    
    alchemy_data: list every element contains the information about the atoms in one molecule
    molecule_size: numpy array, every element is the number of atoms in a molecule (necessary to 
    generate the matrix for storage and get the correct size of the representations)
    rep: representation
    full_matrix: 2D numpy array where every row contains the representation for one atom
    """
    
    full_matrix = np.zeros((np.sum(molecule_size), 210))
    max_size = np.amax(molecule_size)
    start = 0
    for idx, molecule in enumerate(alchemy_data):
        if rep=='coulomb':
            rep = qml.representations.generate_atomic_coulomb_matrix(molecule[:,0], molecule[:,[1,2,3]], size=max_size, sorting='distance', interaction_cutoff=5, interaction_decay=2, central_cutoff=7, central_decay=1)
        for idx2 in range(0, len(rep)):
            full_matrix[start+idx2] = rep[idx2]
        start += len(rep)
    
    return(full_matrix)
    


        
        

