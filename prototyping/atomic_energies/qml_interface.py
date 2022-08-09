#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 11:37:43 2019

@author: misa
"""

import qml
import qml.kernels
import qml.math
import qml.fchl


import numpy as np
import sklearn.model_selection as sk

def alchpot_lam1(program):
    files_alch = []
    if program == 'cpmd':
        path = '/home/misa/APDFT/prototyping/atomic_energies/results/slice_ve38/paths_lam1.txt'
    elif program == 'pyscf':
        path = '/home/misa/projects/Atomic-Energies/data/vacuum_reference/qm9_data/alchpots_lam1.txt'
    with open(path, 'r') as f:
        for l in f:
            if not 'dsgdb9nsd_000829' in l:
                files_alch.append(l.strip('\n'))
    labels = []
    for p in files_alch:
        if program == 'pyscf':
            labels.extend(np.load(p))
        elif program == 'cpmd':
            data_full = np.loadtxt(p)
            labels.extend(data_full[:,-1]) # alchpot is the last column
    alch_pots = np.array(labels)
    return(alch_pots)

def crossvalidate(reps, labels, tr_size, sigma, lam_val, num_cv):
    errors = []
    for cv in range(num_cv):
        reps_tr, reps_test, labels_tr, labels_test = sk.train_test_split(reps,labels,train_size=tr_size)
        coeffs = train_kernel(reps_tr, labels_tr, sigma, lam_val)
        labels_predicted = predict_labels(reps_test, reps_tr, sigma, coeffs)
        errors.append((np.abs(labels_predicted - labels_test)).mean())
    errors = np.array(errors)
    return(errors.mean(), errors.std())

def wrapper_alch_data(path='/home/misa/projects/Atomic-Energies/data/ueg_reference/qm9_ve38_lda/energies_ve38_paths'):
    """
    returns paths to files from file with all directories
    """
    # load paths to data
    path = path
    paths = []
    with open(path, 'r') as f:
        for line in f:
            file_path = line.rstrip('\n')
            paths.append(file_path)
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
        molecule_size[idx] = len(alch[:, 0])
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
            rep = qml.representations.generate_atomic_coulomb_matrix(molecule[:,0], molecule[:,[1,2,3]], size=max_size, sorting='row-norm')

            #rep = qml.representations.generate_atomic_coulomb_matrix(molecule[:,0], molecule[:,[1,2,3]], size=max_size, sorting='row-norm', central_cutoff = 15, central_decay = 0.1)
        for idx2 in range(0, len(rep)):
            full_matrix[start+idx2] = rep[idx2]
        start += len(rep)
        
    return(full_matrix)
    
def wrapper_global_representations(alchemy_data, molecule_size, rep_par='coulomb'):
    """
    generates the representations for all molecules
    returns a 2D numpy array where every row contains the representation for one atom
    
    alchemy_data: list every element contains the information about the atoms in one molecule
    molecule_size: numpy array, every element is the number of atoms in a molecule (necessary to 
    generate the matrix for storage and get the correct size of the representations)
    rep: representation
    full_matrix: 2D numpy array where every row contains the representation for one atom
    """
    max_size = np.amax(molecule_size)
    if rep_par == 'coulomb':
        size_U = int(max_size*(max_size + 1)/2) # number of elements in upper triangle of representation matrix
        full_matrix = np.zeros( (len(alchemy_data), size_U) ) # matrix which contains the representations of all molecules
    elif rep_par == 'FCHL':
        full_matrix = np.zeros( (len(alchemy_data), max_size, 5, max_size) )
    
    for idx, molecule in enumerate(alchemy_data):
        #generate representation of molecule
        if rep_par=='coulomb':
            rep = qml.representations.generate_coulomb_matrix(molecule[:,0], molecule[:,[1,2,3]], size=max_size, sorting='row-norm')
            full_matrix[idx] = rep
        elif rep_par=='FCHL':
            rep = qml.fchl.generate_representation(molecule[:,[1,2,3]], molecule[:,0], max_size=molecule_size[idx])
            full_matrix[idx] = rep
        
    return(full_matrix)

def get_dmatrix(reps):
    """
    calculates distance between representations with L2-norm
    input should be 2D-array of representations: row = representation, column: elements of a single representation
    returns symmetric distance matrix
    """
    
    num_reps = len(reps)
    dmatrix = np.zeros((num_reps, num_reps))
    for rows in range(num_reps):
        for cols in range(rows, num_reps):
            dmatrix[rows, cols] = np.linalg.norm(reps[rows]-reps[cols])
            dmatrix[cols, rows] = dmatrix[rows, cols]
    return(dmatrix)

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
        length = len(alchemy_data[idx][:,0])
        if value == 'atomisation':
            energies[start:length+start] = alchemy_data[idx][:,6]
        elif value == 'atomic':
            energies[start:length+start] = alchemy_data[idx][:,5]
        elif value == 'alch_pot':
            energies[start:length+start] = alchemy_data[idx][:,4]
        elif value == 'charge':
            energies[start:length+start] = alchemy_data[idx][:,0]
        elif value == 'atomisation_global':
            energies[idx] = alchemy_data[idx][:, 6].sum()
        else:
            raise Exception( "Unkown label")
        start += length

    
    return(energies)

def get_label_delta(prop_mean, local_idc, data, molecule_size):
    """
    generates baseline for certain label by assigning average label value for every element to every index
    
    prop_mean: dictionary with mean value per element
    local_idc: local indices of atoms
    data: alchemy data
    molecule_size: size of every molecule in data
    """
    label_delta = np.zeros(len(local_idc))
    # generate array contoining the charge of the elements at index local_idc
    # iterate over all these charges, for every charge assign the corresponding mean value at the index in label_delta
    for idx, i in enumerate(get_property_from_local_index(local_idc, data, 'charge', molecule_size)):
        label_delta[idx] = prop_mean[i]
    return(label_delta)

def get_average_property(idc, data, molecule_size, prop):
    """
    returns average label value for every element
    idc: atom indices for which average label value will be calculated
    """
    # get charges, energies
    charges = get_property_from_local_index(idc, data, 'charge', molecule_size)
    prop = get_property_from_local_index(idc, data, prop, molecule_size)
    
    # divide indices of charges into groups
    charges_divided = partition_idx_by_charge(charges)
    # get energies for different groups
    mean_prop = dict.fromkeys(charges_divided.keys(),0)
    for mp in mean_prop:
        mean_prop[mp] = prop[charges_divided[mp]].mean()
    return(mean_prop)

def get_property_from_local_index(local_idc, alchemy_data, prop, molecule_size):
    """
    returns property from local index
    global_idx: index of molecule within all molecules
    idx_in_mol: index of atom in its molecule
    alchemy_data: contains information about atom in molecule and property
    prop: atomic property from alchemy data set
    """
    global_idc = get_global_idx(local_idc, molecule_size)
    idc_in_mol = get_idx_in_molecule(global_idc, local_idc, molecule_size)
    
    prop_local_idc = []
    for idc in zip(global_idc, idc_in_mol):
        prop_local_idc.append(get_property(idc[0], idc[1], alchemy_data, prop))
    return(np.array(prop_local_idc))
    
def get_property(global_idx, idx_in_mol, alchemy_data, prop):
    """
    global_idx: index of molecule within all molecules
    idx_in_mol: index of atom in its molecule
    alchemy_data: contains information about atom in molecule and property
    prop: atomic property from alchemy data set
    """
    
    if prop == 'charge':
        return(alchemy_data[global_idx][idx_in_mol,0])
    elif prop == 'coords':
        return(alchemy_data[global_idx][idx_in_mol,1:4])
    elif prop == 'alch_pot':
        return(alchemy_data[global_idx][idx_in_mol,4])
    elif prop == 'atomic':
        return(alchemy_data[global_idx][idx_in_mol,5])
    elif prop == 'atomisation':
        return(alchemy_data[global_idx][idx_in_mol,6])

def get_indices(dim_full_rep, tr_size, val_size=None):
    """
    choose randomly the indices of the representations are used for training and testing
    each index belongs to the representation vector in the matrix that contains all representations as rows
    
    @in
    dim_full_rep_rep: number of representations
    tr_size: size of the training set
    val_size: size of the test set (if not given all representations that are not part of the training set)
    
    @out
    2-tuple: (indices of the representations used for training, indices of the representations used for training)
    """
    
    if val_size == None:
        val_size = dim_full_rep - tr_size
    all_indices = np.arange(dim_full_rep)
    np.random.shuffle(all_indices)
    tr_indices = np.sort(all_indices[0:tr_size])
    val_indices = np.sort(all_indices[tr_size:tr_size+val_size])
    return(tr_indices, val_indices)
    
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
    returns a list with the indices of the representations in the full local representation matrix
    """
    #global_idx.sort()
    indices = []
    for idx in global_idx:
        start_idx = np.sum(molecule_size[0:idx])
        length = molecule_size[idx]
        idc_mol = np.arange(start_idx, start_idx+length)
        indices.extend(idc_mol)
    return(indices)
    
def get_global_idx(local_idc, molecule_size):
    """
    inverse of get_local_index, return global index in local index is provided
    local_idc: list of local indices
    molecule_size: list with number of atoms in every molecule
    global_idc: index of molecule in alchemy_data for every local index in local_idx
    """
    #local_idc.sort()
    cumulated = []
    for idx in range(len(molecule_size)):
        cumulated.append(np.sum(molecule_size[0:idx+1]))
    
    cumulated=np.array(cumulated)

    global_idc = []
    
    for atom_idx in local_idc:
        out = np.where(cumulated <= atom_idx)
        global_idx = len(out[0])
        assert atom_idx in get_local_idx([global_idx], molecule_size), "Wrong global index"
        global_idc.append(global_idx)
        
    return(np.array(global_idc))
    
def get_idx_in_molecule(global_idx, local_idx, molecule_size):
    """
    returns the index of the atom in its molecule (at which index in data_set for the molecule)
    global_idx: index of molecule
    local_idx: index of atom in list over all molecules
    
    idx_in_mol: index of atom in molecule
    """
    
    idx_in_mol = []
    for idc in zip(global_idx, local_idx):
        idx_in_mol.append(idc[1] - get_local_idx([idc[0]], molecule_size)[0])
    return(np.array(idx_in_mol))
    
    
    
    
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
    sub_matrix = tmp[:, col_ind]#.copy()
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

def crossvalidate_new(reps, labels, molecule_size, tr_set_size, sigma, lam_val, local=True, molecule=False, num_cross=10):
    """
    calculates the mean error for num_cross randomly selected training sets, returns the mean and std of these mean errors
    
    reps: representations of training and validation data
    labels: labels of training and validation data
    molecule_size: the number of atoms for every representation
    tr_set_size: the size of the training set
    sigma: the kernel width
    lam_val: the regularizer
    num_cross: the number of cross-validations
    
    error_crossval: the mean error for every cross-validation run
    """
    
    error_crossval = np.zeros(num_cross)
    
    for idx in range(0, num_cross):
        
        # split data into training and validation set
        idc_tr, idc_val = get_indices(len(molecule_size), tr_set_size)
        
        if local == True:
            local_idc_tr, local_idc_val = get_local_idx(idc_tr, molecule_size), get_local_idx(idc_val, molecule_size)
            rep_splitted_loc = reps[local_idc_tr], reps[local_idc_val] # select the representations
            labels_splitted_loc = labels[local_idc_tr], labels[local_idc_val] # select the labels
        else:
            rep_splitted_loc = reps[idc_tr], reps[idc_val] # select the representations
            labels_splitted_loc = labels[idc_tr], labels[idc_val] # select the labels
        
        # calculate error
        coeffs = train_kernel(rep_splitted_loc[0], labels_splitted_loc[0], sigma, lam_val)
        labels_predicted = predict_labels(rep_splitted_loc[1], rep_splitted_loc[0], sigma, coeffs)
        
        if molecule:
            error_crossval[idx] = calculate_error_atomisation_energy(labels_predicted, molecule_size[idc_val], labels_splitted_loc[1]).mean()
        else:
            error_crossval[idx] = np.abs(labels_predicted - labels_splitted_loc[1]).mean()
#            error_crossval[idx] = (np.abs(labels_predicted - labels_splitted_loc[1])/np.abs(labels_splitted_loc[1])).mean()

    
    return(error_crossval.mean(), error_crossval.std())





def split_data(reps, labels, tr_set_size, molecule_size, local=True):
    # split data into training and validation set
    global_idc_tr, global_idc_val = get_indices(len(molecule_size), tr_set_size)
    if local == True:
        local_idc_tr, local_idc_val = get_local_idx(global_idc_tr, molecule_size), get_local_idx(global_idc_val, molecule_size)
        rep_splitted = reps[local_idc_tr], reps[local_idc_val] # select the representations
        labels_splitted = labels[local_idc_tr], labels[local_idc_val] # select the labels
    else:
        rep_splitted = reps[global_idc_tr], reps[global_idc_val]
        labels_splitted = labels[global_idc_tr], labels[global_idc_val]
    
    return(rep_splitted, labels_splitted)
    
def predict_labels(rep_test, rep_tr, sigma, coeffs):
    """
    predict the labels for given coefficents and training, validation/test set
    """
    kernel = qml.kernels.gaussian_kernel(rep_test, rep_tr, sigma)
    prediction = np.dot(kernel, coeffs)
    return(prediction)

def calculate_error_atomisation_energy(atomic_energies, molecule_size, ref_atomic_energies):
    """
    calculate the total error in atomisation energy if the atomisation energy is build up from 
    atomic contributions
    
    atomic_energies: the predicted atomic energies
    atomsisation_en_predicted: 1D numpy array, the predicted atomisation energies for every molecule
    molecule_size: the size of every molecule for which the atomic energies where predicted
    ref_atomic_energies: the correct atomic energies of the validation/test molecules (the test labels)
    atomisation_en_ref: the correct (total) atomisation energies of the validation/test molecules
    
    error: numpy array 1D errors per molecule
    """

    # sum up atomic energies
    atomisation_en_predicted = np.zeros(len(molecule_size))
    atomisation_en_ref = np.zeros(len(molecule_size))
#    error_rel = np.zeros(len(molecule_size))
    
    start = 0
    for idx, size in enumerate(molecule_size):
        atomisation_en_predicted[idx] = atomic_energies[start:start+size].sum()
        atomisation_en_ref[idx] = ref_atomic_energies[start:start+size].sum()
        
#        err_rel = np.abs(atomic_energies[start:start+size] - ref_atomic_energies[start:start+size])/np.abs(ref_atomic_energies[start:start+size])
#        error_rel[idx] = err_rel.sum()
        
        start += size
    
    # compare to correct values
    error = np.abs(atomisation_en_predicted - atomisation_en_ref)
    
#    error = error_rel
    return(error)
    
def shift_by_mean_energy(reps, labels):
    """
    shifts the atomic energies by the mean atomic energy for the corresponding element
    returns the shifted energies as a copy
    
    reps: the (local) representations
    labels: the (local) labels
    labels_copy: copy of labels, shifted by mean energies
    """
    
    labels_copy = labels.copy()
    
    # get the element (charge) from the representations
    nuc_charges = np.power(reps*2, 1/2.4)[:,0].astype(int)
    
    # shift the energies for each element
    for el in set(nuc_charges):
        idc_el = np.where(nuc_charges==el)
        mean_energy = labels_copy[idc_el].mean()
        labels_copy[idc_el] = labels_copy[idc_el] - mean_energy
        
    return(labels_copy)
    
def train_kernel(rep_tr, labels_tr, sigma, lam_val, kernel_type = "gaussian"):
    """
    return coefficients from representation, labels, sigma and lambda
    
    rep_tr: training representations
    labels_tr: training labels
    sigma: kernel width
    lam_val: regularizer
    """
    if kernel_type == "gaussian":
        tr_kernel = qml.kernels.gaussian_kernel(rep_tr, rep_tr, sigma)
    elif kernel_type == "laplacian":
        tr_kernel = qml.kernels.laplacian_kernel(rep_tr, rep_tr, sigma)
    reg_kernel = tr_kernel + np.identity(len(tr_kernel))*lam_val
    coeffs = qml.math.cho_solve(reg_kernel, labels_tr)
    return(coeffs)

def partition_idx_by_charge(charges):
    """
    partitions the idx in idx_list into groups, every group contains the indices
    with the same charge; for every unique nuclear charge a tuple (charge, list of idx where nuclear_charge == charge)
    is created and the list of these tuples is returned
    
    alchemy_data: output from write_atomisation_energies in alchemy tools
    idx_list: global indices (molecules) that shall be partitioned into groups with the same charge
    """
    
    unique_charges = list(set(charges))
    unique_charges.sort()
    
    charges_partitioned = dict.fromkeys(unique_charges,0)
    for k in charges_partitioned:
        charges_partitioned[k] = np.where(charges == k)
    return(charges_partitioned)