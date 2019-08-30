#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 11:37:43 2019

@author: misa
"""

import qml
import qml.distance
import numpy as np
import qml.kernels
import qml.math

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
    
def wrapper_global_representations(alchemy_data, molecule_size, rep_par='coulomb'):
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
    full_matrix = np.zeros( (len(alchemy_data), size_U) ) # matrix which contains the representations of all molecules
    
    for idx, molecule in enumerate(alchemy_data):
        #generate representation of molecule
        if rep_par=='coulomb':
            rep = qml.representations.generate_coulomb_matrix(molecule[:,0], molecule[:,[1,2,3]], size=max_size, sorting='row-norm')
            full_matrix[idx] = rep
        
    return(full_matrix)
    
def calculate_distances(rep_matrix):
    """
    calculates the distance of every representation with all representations (including itself)
    returns the distances as a 1D numpy array
    
    rep_matrix: 2D numpy array every row contains the representation for one atom
    dist: distances as a 1D numpy array
    """
    
    d2 = qml.distance.l2_distance(rep_matrix, rep_matrix)
    d2_flattened = d2[np.triu_indices(len(rep_matrix))]
    return(d2_flattened)

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
        elif value == 'atomic':
            energies[start:length+start] = alchemy_data[idx][:,5]
        start += length 
    
    return(energies)
        

def get_indices(dim_full_rep, tr_size, test_size=None):
    """
    choose randomly the indices of the representations are used for training and testing
    each index belongs to the representation vector in the matrix that contains all representations as rows
    
    @in
    dim_full_rep_rep: number of representations
    tr_size: size of the training set
    test_size: size of the test set (if not given all representations that are not part of the training set)
    
    @out
    2-tuple: (indices of the representations used for training, indices of the representations used for training)
    """
    
    if test_size == None:
        test_size = dim_full_rep - tr_size
    all_indices = np.arange(dim_full_rep)
    np.random.shuffle(all_indices)
    tr_indices = np.sort(all_indices[0:tr_size])
    test_indices = np.sort(all_indices[tr_size:tr_size+test_size])
    return(tr_indices, test_indices)
    
def get_local_idx(global_idx, molecule_size):
    """
    global_idx gives the indices of representations (molecules) in the matrix that
    contains all the global representations, the function then returns for every molecule
    in global_idx the indices of the local representations (of the atoms) in the full local
    representation matrix
    
    @in
    global_idx: indices of the global representations in the full representation matrix
    molecule_size: list of number of atoms of every representation in the full representation matrix
    
    @out
    returns a lsit with the indices of the representations in the full local representation matrix
    """
    global_idx.sort()
    indices = []
    for idx in global_idx:
        start_idx = np.sum(molecule_size[0:idx])
        length = molecule_size[idx]
        idc_mol = np.arange(start_idx, start_idx+length)
        indices.extend(idc_mol)
    return(indices)
    
def select_sub_matrix(full_matrix, row_ind, col_ind):
    """
    returns all possible combinations of row_ind and col_ind as an 2D array
    
    @in
    full_matrix: the matrix from which as sub matrix will be selected
    row_ind, col_ind: the indices of the rows and cols that will be chosen
    
    @out
    2D numpy array with shape len(row_ind), len(col_ind), the elements have all combinations of 
    row_ind and col_ind as their index 
    """
    tmp = full_matrix[row_ind]
    sub_matrix = tmp[:, col_ind].copy()
    return(sub_matrix)

def split_kernel(full_kernel, tr_indices, test_indices):
    """
    generates a kernel that is function of training representations
    and a kernel where every row contains the distances between one test representation
    and all training representations
    
    @in
    full_kernel: contains the kernel elements for every representation combination
    tr_indices: the indices of the representations used for training
    test_indices: indices of the representations used for training
    
    @out:
    2-tuple: (subset of full kernel containing the combinations of all representations for training, 
              subset of kernel containg the combinations of every test representation with every training representation)
    """
    
    tr_kernel = select_sub_matrix(full_kernel, tr_indices, tr_indices)
    test_kernel = select_sub_matrix(full_kernel, test_indices, tr_indices)
    
    return(tr_kernel, test_kernel)

def test(rep_matrix, tr_ind, test_ind, labels, sigma, lam_val):
    
    # calculate full kernel
    gaussian_kernel = qml.kernels.gaussian_kernel(rep_matrix, rep_matrix, sigma)
    
    # select training and test kernel
    tr_kernel, test_kernel = split_kernel(gaussian_kernel, tr_ind, test_ind)
    
    # labels of training representations
    tr_labels = labels[tr_ind]
    
    # calculate regression coefficents (do not add identity matrix if lam = 0 for stability reasons)
    if lam_val == 0:
        coeffs = qml.math.cho_solve(tr_kernel, tr_labels)
    else:
        mat = tr_kernel + np.identity(len(tr_kernel))*lam_val
        coeffs = qml.math.cho_solve(mat, tr_labels)
    
    # calculate errors
    predicition_errors = np.abs( np.dot(test_kernel, coeffs) - labels[test_ind] )
    mean_pred_error = np.mean(predicition_errors)
    
    training_errors = np.abs( np.dot(mat, coeffs) - labels[tr_ind] )
    mean_tr_error = np.mean(training_errors)
    
    return( (predicition_errors, mean_pred_error), (training_errors, mean_tr_error) )










