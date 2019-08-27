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
    path = '/home/misa/APDFT/prototyping/atomic_energies/results/slice_ve38/finished_abs'
    paths = []
    with open(path, 'r') as f:
        for line in f:
            paths.append(line.rstrip('\n'))
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
        alch = np.loadtxt(path)
        molecule_size[idx] = len(alch[:,0])
        alchemy_data.append(alch)
    
    return(alchemy_data, molecule_size)
    
def generate_atomic_representations(alchemy_data, molecule_size, rep_par='coulomb'):
    """
    generates the local representations for every atom
    returns a 2D numpy array where every row contains the representation for one atom
    
    alchemy_data: list every element contains the information about the atoms in one molecule
    molecule_size: numpy array, every element is the number of atoms in a molecule (necessary to 
    generate the matrix for storage and get the correct size of the representations)
    rep: representation
    full_matrix: 2D numpy array where every row contains the representation for one atom
    """
    max_size = np.amax(molecule_size)
    size_U = int(max_size*(max_size + 1)/2) # number of elements in upper triangle of representation matrix
    full_matrix = np.zeros((np.sum(molecule_size), size_U))
    
    start = 0
    for idx, molecule in enumerate(alchemy_data):
        if rep_par=='coulomb':
            rep = qml.representations.generate_atomic_coulomb_matrix(molecule[:,0], molecule[:,[1,2,3]], size=max_size, sorting='distance')
        for idx2 in range(0, len(rep)):
            full_matrix[start+idx2] = rep[idx2]
        start += len(rep)
        
    return(full_matrix)
    
def calculate_distances(rep_matrix, norm='l2'):
    """
    calculates the distance of every representation with all representations (including itself)
    returns the distances as a 1D numpy array
    
    rep_matrix: 2D numpy array every row contains the representation for one atom
    dist: distances as a 1D numpy array
    """
    # define output array
    dist_shape = int(len(rep_matrix)*(len(rep_matrix)+1)/2)
    dist = np.empty(dist_shape)
    # indices of distances in matrix
    start=0
    width = len(rep_matrix)
    # calculate distances
    for idx in range(0, len(rep_matrix)):
        if norm=='l2':
            dist[start:start+width] = np.linalg.norm(rep_matrix[idx]-rep_matrix[idx:], axis=1)
        start = start+width
        width -= 1
        
    return(dist)

def generate_label_vector(alchemy_data, num_rep, value='atomisation'):
    """
    extracts the atomic energies from the alchemy files
    returns a 1D numpy array with the atomic energies for all atoms in the training set
    
    alchemy_data: list where every element contains the information about the atoms in one molecule
    value: label (atomisation energy, atomic energy from LDA...)
    num_rep: number of reprentations (number of atoms) in training set
    """
    
    energies = np.zeros(num_rep)
    start = 0
    for idx, mol in enumerate(alchemy_data):
        length = len(alchemy_data[idx][:,6])
        if value == 'atomisation':
            energies[start:length+start] = alchemy_data[idx][:,6]
        start += length 
    
    

